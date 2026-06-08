"""Usage reporting — read-only aggregation of the durable `requests` table (plus
the cost-bearing benchmark/training/compute tables) into a usage report for the
admin **Usage Reports** page.

Why Postgres and not Prometheus: the gateway's Prometheus counters are
process-since-start and the monitoring stack (deploy/monitoring/) is optional and
short-retention. The `requests` table is the always-available, historically
complete source, so the report is built on it — no new instrumentation.

Scope: admins see every app/user; non-admins are forced to their own `owner_id`
(same rule as `/apps/{app_id}/requests`).

Notes / caveats baked into the data:
  * **4xx/5xx is approximate.** The literal HTTP status isn't stored on a request
    row — only the internal lifecycle status. We map completed→success,
    failed/error/timeout→server_error (~5xx), cancelled→client_cancelled (~4xx).
    Genuine 4xx rejected before a row is created (bad model, auth) are not counted.
  * **Tokens are best-effort** from `output["usage"]` (present on non-streaming
    chat/completions); streaming/audio usually lack it → reported via a coverage %.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import current_user
from .db import App, Request as ReqRow, User, get_session

router = APIRouter(prefix="/v1/usage", tags=["usage"])

# Terminal status → outcome class (see module docstring caveat).
_SERVER_ERROR = {"failed", "error", "timeout"}


def _outcome(status: str) -> str:
    if status == "completed":
        return "success"
    if status in _SERVER_ERROR:
        return "server_error"
    if status == "cancelled":
        return "client_cancelled"
    return "pending"


def _tokens(output: Any) -> tuple[int, int, bool]:
    """(prompt_tokens, completion_tokens, had_usage) from a stored vLLM output."""
    if not isinstance(output, dict):
        return 0, 0, False
    usage = output.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, False
    try:
        pin = int(usage.get("prompt_tokens") or 0)
        pout = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return 0, 0, False
    return pin, pout, True


def _fmt_elapsed(sec: float) -> str:
    sec = int(max(0, sec))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m"
    if sec < 86400:
        return f"{sec / 3600:.1f}h"
    return f"{sec / 86400:.1f}d"


def _zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz)
    except Exception:
        return ZoneInfo("UTC")


def _local_midnight(date_str: str, zone: ZoneInfo) -> datetime:
    y, m, d = (int(x) for x in date_str.split("-"))
    return datetime(y, m, d, tzinfo=zone)


# ── Response models ─────────────────────────────────────────────────────────

class UsageSummary(BaseModel):
    total_requests: int
    completed: int
    server_error: int       # ~5xx (failed/error/timeout)
    client_cancelled: int   # ~4xx (cancelled)
    pending: int
    success_rate: Optional[float]
    tokens_in: int
    tokens_out: int
    tokens_total: int
    token_coverage_pct: Optional[float]
    distinct_models: int
    distinct_apps: int
    distinct_users: int
    avg_latency_s: Optional[float]
    p95_latency_s: Optional[float]


class UsageByModel(BaseModel):
    model: str
    requests: int
    completed: int
    server_error: int
    client_cancelled: int
    tokens_total: int
    avg_latency_s: Optional[float]


class UsageByApp(BaseModel):
    app_id: str
    name: str
    requests: int
    completed: int
    server_error: int
    success_rate: Optional[float]
    tokens_total: int


class UsageByUser(BaseModel):
    owner_id: int
    username: str
    requests: int
    tokens_total: int


class UsageByEndpoint(BaseModel):
    endpoint: str
    requests: int
    completed: int
    server_error: int


class UsageTimePoint(BaseModel):
    ts: int                      # unix seconds, bucket start
    label: str
    total: int
    by_model: dict[str, int]
    success: int
    server_error: int
    client_cancelled: int


class UsageDayRequest(BaseModel):
    request_id: str
    app_id: str
    model: str
    endpoint: str
    username: str
    status: str
    outcome: str
    start_time: str
    end_time: str
    elapsed_label: str


class UsageDay(BaseModel):
    date: str
    day_label: str
    requests: int
    completed: int
    server_error: int
    client_cancelled: int
    tokens_total: int
    jobs: list[UsageDayRequest]


class UsageReport(BaseModel):
    period: dict
    scope: str            # "platform" | "owner"
    bucket: str           # "hour" | "day"
    summary: UsageSummary
    by_model: list[UsageByModel]
    by_app: list[UsageByApp]
    by_user: list[UsageByUser]
    by_endpoint: list[UsageByEndpoint]
    time_series: list[UsageTimePoint]
    daily: list[UsageDay]
    models: list[str]
    apps: list[dict]
    users: list[dict]
    note: str


_MAX_DAY_ROWS = 250  # cap per-day request rows so the payload stays bounded


@router.get("/report", response_model=UsageReport)
async def usage_report(
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None, alias="to"),
    tz: str = "UTC",
    app_id: Optional[str] = None,
    owner_id: Optional[int] = None,
    model: Optional[str] = None,    # csv filter
    status: Optional[str] = None,   # csv filter
    bucket: str = "auto",           # auto | hour | day
):
    zone = _zone(tz)
    now_local = datetime.now(zone)

    to_mid = _local_midnight(to, zone) if to else now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    from_mid = _local_midnight(from_, zone) if from_ else (
        (now_local - timedelta(days=29)).replace(hour=0, minute=0, second=0, microsecond=0)
    )
    start_utc = from_mid.astimezone(timezone.utc)
    end_utc = (to_mid + timedelta(days=1)).astimezone(timezone.utc)  # exclusive next-midnight

    is_admin = bool(user.is_admin or user.role == "admin")
    scope = "platform" if is_admin else "owner"
    eff_owner = (owner_id if (is_admin and owner_id is not None) else None) if is_admin else user.id

    # Project to small columns only and extract `model` + token counts in SQL, so
    # we never load the (potentially multi-KB) payload/output JSON blobs into
    # memory. A full-row load over a wide window on a busy platform OOMs the
    # gateway (each row carries the request payload + vLLM output).
    model_col = func.json_extract_path_text(ReqRow.payload, "model").label("model")
    tin_col = func.json_extract_path_text(ReqRow.output, "usage", "prompt_tokens").label("tin")
    tout_col = func.json_extract_path_text(ReqRow.output, "usage", "completion_tokens").label("tout")
    stmt = select(
        ReqRow.request_id, ReqRow.app_id, ReqRow.owner_id, ReqRow.endpoint,
        ReqRow.status, ReqRow.created_at, ReqRow.completed_at,
        model_col, tin_col, tout_col,
    ).where(ReqRow.created_at >= start_utc, ReqRow.created_at < end_utc)
    if eff_owner is not None:
        stmt = stmt.where(ReqRow.owner_id == eff_owner)
    if app_id:
        stmt = stmt.where(ReqRow.app_id == app_id)
    if status:
        wanted = [s.strip() for s in status.split(",") if s.strip()]
        if wanted:
            stmt = stmt.where(ReqRow.status.in_(wanted))
    rows = (await session.execute(stmt.order_by(ReqRow.created_at))).all()

    # Resolve apps + users for labels/model fallback.
    app_ids = {r.app_id for r in rows}
    owner_ids = {r.owner_id for r in rows}
    apps: dict[str, App] = {}
    if app_ids:
        for a in (await session.execute(select(App).where(App.app_id.in_(app_ids)))).scalars().all():
            apps[a.app_id] = a
    users: dict[int, User] = {}
    if owner_ids:
        for u in (await session.execute(select(User).where(User.id.in_(owner_ids)))).scalars().all():
            users[u.id] = u

    def model_of(r) -> str:
        m = r.model
        if not m:
            a = apps.get(r.app_id)
            m = (a.model if a and a.model else None) or "(unknown)"
        return str(m)

    def toks_of(r) -> tuple[int, int, bool]:
        def _i(x) -> int:
            try:
                return int(x) if x not in (None, "") else 0
            except (TypeError, ValueError):
                return 0
        return _i(r.tin), _i(r.tout), (r.tin is not None or r.tout is not None)

    # Full model list (for the filter dropdown) before applying the model filter.
    models_all = sorted({model_of(r) for r in rows})
    model_filter = {m.strip() for m in model.split(",") if m.strip()} if model else None
    if model_filter:
        rows = [r for r in rows if model_of(r) in model_filter]

    # Bucket size for the time-series.
    span_days = max(1, (end_utc - start_utc).days)
    bkt = ("hour" if span_days <= 2 else "day") if bucket == "auto" else ("hour" if bucket == "hour" else "day")
    single_day = span_days <= 1

    # ── Accumulators ────────────────────────────────────────────────────────
    s_completed = s_server = s_cancel = s_pending = 0
    s_tin = s_tout = 0
    tok_have = tok_eligible = 0
    latencies: list[float] = []

    bymodel: dict[str, dict] = defaultdict(lambda: {"requests": 0, "completed": 0, "server_error": 0, "client_cancelled": 0, "tokens": 0, "lat": []})
    byapp: dict[str, dict] = defaultdict(lambda: {"requests": 0, "completed": 0, "server_error": 0, "tokens": 0})
    byuser: dict[int, dict] = defaultdict(lambda: {"requests": 0, "tokens": 0})
    byendpoint: dict[str, dict] = defaultdict(lambda: {"requests": 0, "completed": 0, "server_error": 0})
    ts_buckets: dict[int, dict] = {}
    day_buckets: dict[str, dict] = {}

    for r in rows:
        oc = _outcome(r.status)
        m = model_of(r)
        pin, pout, had = toks_of(r)
        toks = pin + pout
        local = r.created_at.astimezone(zone)

        # latency (terminal + completed_at present)
        lat = None
        if r.completed_at is not None:
            lat = (r.completed_at - r.created_at).total_seconds()
            if lat >= 0:
                latencies.append(lat)

        # summary
        if oc == "success":
            s_completed += 1
            tok_eligible += 1
            if had:
                tok_have += 1
        elif oc == "server_error":
            s_server += 1
        elif oc == "client_cancelled":
            s_cancel += 1
        else:
            s_pending += 1
        s_tin += pin
        s_tout += pout

        # by model
        bm = bymodel[m]
        bm["requests"] += 1
        bm["tokens"] += toks
        if oc == "success":
            bm["completed"] += 1
        elif oc == "server_error":
            bm["server_error"] += 1
        elif oc == "client_cancelled":
            bm["client_cancelled"] += 1
        if lat is not None and lat >= 0:
            bm["lat"].append(lat)

        # by app
        ba = byapp[r.app_id]
        ba["requests"] += 1
        ba["tokens"] += toks
        if oc == "success":
            ba["completed"] += 1
        elif oc == "server_error":
            ba["server_error"] += 1

        # by user
        bu = byuser[r.owner_id]
        bu["requests"] += 1
        bu["tokens"] += toks

        # by endpoint
        be = byendpoint[r.endpoint]
        be["requests"] += 1
        if oc == "success":
            be["completed"] += 1
        elif oc == "server_error":
            be["server_error"] += 1

        # time-series bucket
        if bkt == "hour":
            key_local = local.replace(minute=0, second=0, microsecond=0)
            label = key_local.strftime("%H:%M") if single_day else key_local.strftime("%m-%d %H:00")
        else:
            key_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
            label = key_local.strftime("%m-%d")
        key_ts = int(key_local.timestamp())
        tb = ts_buckets.get(key_ts)
        if tb is None:
            tb = ts_buckets[key_ts] = {"label": label, "total": 0, "by_model": defaultdict(int), "success": 0, "server_error": 0, "client_cancelled": 0}
        tb["total"] += 1
        tb["by_model"][m] += 1
        if oc in ("success", "server_error", "client_cancelled"):
            tb[oc] += 1

        # daily bucket
        date_str = local.strftime("%Y-%m-%d")
        db = day_buckets.get(date_str)
        if db is None:
            db = day_buckets[date_str] = {
                "day_label": local.strftime("%A, %d %B %Y"),
                "requests": 0, "completed": 0, "server_error": 0, "client_cancelled": 0,
                "tokens": 0, "jobs": [],
            }
        db["requests"] += 1
        db["tokens"] += toks
        if oc == "success":
            db["completed"] += 1
        elif oc == "server_error":
            db["server_error"] += 1
        elif oc == "client_cancelled":
            db["client_cancelled"] += 1
        if len(db["jobs"]) < _MAX_DAY_ROWS:
            u = users.get(r.owner_id)
            end_local = r.completed_at.astimezone(zone) if r.completed_at else None
            db["jobs"].append(UsageDayRequest(
                request_id=r.request_id,
                app_id=r.app_id,
                model=m,
                endpoint=r.endpoint,
                username=(u.username if u else str(r.owner_id)),
                status=r.status,
                outcome=oc,
                start_time=local.strftime("%H:%M"),
                end_time=(end_local.strftime("%H:%M") if end_local else ""),
                elapsed_label=(_fmt_elapsed(lat) if lat is not None else ""),
            ))

    total = len(rows)
    terminal = s_completed + s_server + s_cancel
    success_rate = round(100.0 * s_completed / terminal, 1) if terminal else None
    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else None
    p95_lat = None
    if latencies:
        sl = sorted(latencies)
        p95_lat = round(sl[min(len(sl) - 1, int(0.95 * (len(sl) - 1)))], 2)
    token_cov = round(100.0 * tok_have / tok_eligible, 1) if tok_eligible else None

    summary = UsageSummary(
        total_requests=total,
        completed=s_completed,
        server_error=s_server,
        client_cancelled=s_cancel,
        pending=s_pending,
        success_rate=success_rate,
        tokens_in=s_tin,
        tokens_out=s_tout,
        tokens_total=s_tin + s_tout,
        token_coverage_pct=token_cov,
        distinct_models=len({model_of(r) for r in rows}),
        distinct_apps=len({r.app_id for r in rows}),
        distinct_users=len({r.owner_id for r in rows}),
        avg_latency_s=avg_lat,
        p95_latency_s=p95_lat,
    )

    by_model = sorted(
        (UsageByModel(
            model=m, requests=v["requests"], completed=v["completed"],
            server_error=v["server_error"], client_cancelled=v["client_cancelled"],
            tokens_total=v["tokens"],
            avg_latency_s=(round(sum(v["lat"]) / len(v["lat"]), 2) if v["lat"] else None),
        ) for m, v in bymodel.items()),
        key=lambda x: x.requests, reverse=True,
    )

    by_app = sorted(
        (UsageByApp(
            app_id=aid, name=(apps[aid].name if aid in apps else aid),
            requests=v["requests"], completed=v["completed"], server_error=v["server_error"],
            success_rate=(round(100.0 * v["completed"] / (v["completed"] + v["server_error"]), 1)
                          if (v["completed"] + v["server_error"]) else None),
            tokens_total=v["tokens"],
        ) for aid, v in byapp.items()),
        key=lambda x: x.requests, reverse=True,
    )

    by_user = sorted(
        (UsageByUser(
            owner_id=oid, username=(users[oid].username if oid in users else str(oid)),
            requests=v["requests"], tokens_total=v["tokens"],
        ) for oid, v in byuser.items()),
        key=lambda x: x.requests, reverse=True,
    )

    by_endpoint = sorted(
        (UsageByEndpoint(endpoint=ep, requests=v["requests"], completed=v["completed"], server_error=v["server_error"])
         for ep, v in byendpoint.items()),
        key=lambda x: x.requests, reverse=True,
    )

    time_series = [
        UsageTimePoint(
            ts=ts, label=v["label"], total=v["total"], by_model=dict(v["by_model"]),
            success=v["success"], server_error=v["server_error"], client_cancelled=v["client_cancelled"],
        )
        for ts, v in sorted(ts_buckets.items())
    ]

    daily = [
        UsageDay(
            date=d, day_label=v["day_label"], requests=v["requests"], completed=v["completed"],
            server_error=v["server_error"], client_cancelled=v["client_cancelled"],
            tokens_total=v["tokens"], jobs=v["jobs"],
        )
        for d, v in sorted(day_buckets.items())
    ]

    return UsageReport(
        period={"from": start_utc.isoformat(), "to": end_utc.isoformat(),
                "from_date": from_mid.strftime("%Y-%m-%d"), "to_date": to_mid.strftime("%Y-%m-%d")},
        scope=scope,
        bucket=bkt,
        summary=summary,
        by_model=by_model,
        by_app=by_app,
        by_user=by_user,
        by_endpoint=by_endpoint,
        time_series=time_series,
        daily=daily,
        models=models_all,
        apps=[{"app_id": a.app_id, "name": a.name} for a in sorted(apps.values(), key=lambda a: a.name)],
        users=[{"owner_id": u.id, "username": u.username} for u in sorted(users.values(), key=lambda u: u.username)],
        note=("4xx/5xx are approximated from request status (cancelled≈4xx; failed/error/timeout≈5xx) "
              "and exclude requests rejected before a row is created. Tokens are best-effort from response usage."),
    )


# ── Resource spend (benchmark / training / compute $) ───────────────────────

class SpendRow(BaseModel):
    resource_type: str   # benchmark | training | compute
    count: int
    cost_usd: float
    gpu_hours: Optional[float]


class SpendByUser(BaseModel):
    owner_id: int
    username: str
    cost_usd: float


class UsageSpend(BaseModel):
    period: dict
    scope: str
    by_type: list[SpendRow]
    by_user: list[SpendByUser]
    total_cost_usd: float
    has_cost_data: bool
    note: str


@router.get("/spend", response_model=UsageSpend)
async def usage_spend(
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None, alias="to"),
    tz: str = "UTC",
    owner_id: Optional[int] = None,
):
    # Lazy import to avoid any import-order coupling with main.
    from .bench import Benchmark
    from .training_api import TrainingRun
    from .compute import ComputePod

    zone = _zone(tz)
    now_local = datetime.now(zone)
    now_utc = datetime.now(timezone.utc)
    to_mid = _local_midnight(to, zone) if to else now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    from_mid = _local_midnight(from_, zone) if from_ else (
        (now_local - timedelta(days=29)).replace(hour=0, minute=0, second=0, microsecond=0)
    )
    start_utc = from_mid.astimezone(timezone.utc)
    end_utc = (to_mid + timedelta(days=1)).astimezone(timezone.utc)

    is_admin = bool(user.is_admin or user.role == "admin")
    scope = "platform" if is_admin else "owner"
    eff_owner = (owner_id if (is_admin and owner_id is not None) else None) if is_admin else user.id

    def elapsed_h(start: Optional[datetime], end: Optional[datetime]) -> float:
        if start is None:
            return 0.0
        e = end or now_utc
        return max(0.0, (e - start).total_seconds() / 3600.0)

    by_type: dict[str, dict] = {t: {"count": 0, "cost": 0.0, "gpu_h": 0.0} for t in ("benchmark", "training", "compute")}
    by_user: dict[int, float] = defaultdict(float)
    user_ids: set[int] = set()
    has_cost = False

    async def gather(model_cls, rtype, start_attr, end_attr, has_gpu):
        nonlocal has_cost
        st = select(model_cls).where(model_cls.created_at >= start_utc, model_cls.created_at < end_utc)
        if eff_owner is not None:
            st = st.where(model_cls.owner_id == eff_owner)
        for row in (await session.execute(st)).scalars().all():
            rate = getattr(row, "cost_per_hr", None)
            hrs = elapsed_h(getattr(row, start_attr, None) or row.created_at, getattr(row, end_attr, None))
            gpu_count = getattr(row, "gpu_count", 1) if has_gpu else 1
            by_type[rtype]["count"] += 1
            if has_gpu:
                by_type[rtype]["gpu_h"] += hrs * (gpu_count or 1)
            if rate:
                has_cost = True
                cost = hrs * rate
                by_type[rtype]["cost"] += cost
                by_user[row.owner_id] += cost
                user_ids.add(row.owner_id)

    await gather(Benchmark, "benchmark", "started_at", "ended_at", False)
    await gather(TrainingRun, "training", "started_at", "ended_at", True)
    await gather(ComputePod, "compute", "ready_at", "terminated_at", True)

    users: dict[int, User] = {}
    if user_ids:
        for u in (await session.execute(select(User).where(User.id.in_(user_ids)))).scalars().all():
            users[u.id] = u

    total = sum(v["cost"] for v in by_type.values())
    return UsageSpend(
        period={"from_date": from_mid.strftime("%Y-%m-%d"), "to_date": to_mid.strftime("%Y-%m-%d")},
        scope=scope,
        by_type=[SpendRow(resource_type=t, count=v["count"], cost_usd=round(v["cost"], 2),
                          gpu_hours=(round(v["gpu_h"], 1) if v["gpu_h"] else None))
                 for t, v in by_type.items()],
        by_user=sorted((SpendByUser(owner_id=oid, username=(users[oid].username if oid in users else str(oid)),
                                    cost_usd=round(c, 2)) for oid, c in by_user.items()),
                       key=lambda x: x.cost_usd, reverse=True),
        total_cost_usd=round(total, 2),
        has_cost_data=has_cost,
        note=("Cost = elapsed_hours × cost_per_hr (NULL for FakeProvider runs → counted but $0). "
              "Resources counted by creation time within the period."),
    )
