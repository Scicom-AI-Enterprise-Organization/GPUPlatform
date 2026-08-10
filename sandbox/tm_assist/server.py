#!/usr/bin/env python3
"""TM text-assist sandbox — answers the 18 real TM agent tools during an Experiments replay.

Same contract as `sandbox/server.py` (that one is a 6-tool fixture for wiring the
path up; this one is the real TM world), so it drops into
`/experiments/sandboxes/new` the same way:

    Mode          HTTP endpoint
    Endpoint URL  http://127.0.0.1:8078/tool
    Result path   content

    .venv/bin/python sandbox/tm_assist/server.py       # needs kb.json — see build_kb.py

WHY A SANDBOX IS THE THING THAT MAKES THIS CORPUS EVALUABLE
----------------------------------------------------------
Experiments replays ONE request per dataset row. On an agentic corpus like this
one that scores the first turn only (see the root CLAUDE.md caveat). A sandbox is
what answers the model's tool calls, turning one request into the whole
trajectory — so the model must actually search the KB, read what came back, and
answer from it. That is also the only way the grounding metric means anything:
"did the model assert something the tool results don't support" needs real tool
results to compare against.

THREE PROPERTIES THAT MAKE THE NUMBERS TRUSTWORTHY
--------------------------------------------------
1. **Deterministic.** Same (tool, arguments) always gives the same answer. A random
   simulator makes a comparison measure luck instead of model quality. Records are
   derived via sha256 of the identifier — NOT `hash()`, which Python salts per
   process, so it would silently change between restarts.
2. **A frozen clock.** `--today` (default 2026-08-10) anchors every date, so
   `list_available_slots` returns the same 21-day window this month and next. With
   a live clock two runs a week apart would face different worlds.
3. **It refuses to invent customers.** An identifier is only real if it appears in
   the conversation the model was given. Ask about an account nobody mentioned and
   you get the tool's documented empty/not-found shape — never a plausible record.
   This is load-bearing: a model that hallucinates an account number must not be
   rewarded with a confident-looking answer, and an evaluator can't tell the
   difference if the sandbox makes one up.

Response shapes follow the `returns` contract declared for each tool in the
corpus's `tm_text_assist_functions.json` (sen for money, 'A'/'S'/'T' status codes,
dd/mm/yyyy HH:mm:ss for field-force timestamps, letter order-state codes).

FAULT INJECTION — call these tool names to exercise the error paths:

    _http_500   → HTTP 500                    (gateway: sandbox_http_error)
    _not_json   → 200, non-JSON body          (sandbox_bad_response)
    _wrong_key  → 200, JSON without the path  (sandbox_bad_response)
    _slow       → sleeps 30s                  (sandbox_unreachable, set Timeout low)
    _empty      → 200, empty string           (a valid, empty tool result)
    _kb_outage  → kbms_search's documented outage payload, which is a NORMAL 200
                  the model is supposed to handle by falling back to general
                  policy knowledge — not a transport failure

An unknown tool name returns a structured `{"error": "unknown_tool"}` at HTTP 200:
an unknown call is something the MODEL should react to, and a sandbox must never
fabricate a plausible success for a call it cannot answer.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

HERE = Path(__file__).resolve().parent

API_KEY = ""
QUIET = False
TODAY = datetime(2026, 8, 10)          # frozen clock; --today overrides
KB: dict = {"chunks": [], "tags": []}
INDEX: dict[str, list[int]] = {}
IDF: dict[str, float] = {}

# --------------------------------------------------------------------------- #
# Identifier formats (from the corpus's shared_entities schema)
# --------------------------------------------------------------------------- #
ID_PATTERNS = {
    "msisdn": re.compile(r"\b60\d{8,11}\b|\b[\w.+-]+@unifi\b", re.IGNORECASE),
    "account_number": re.compile(r"\b1[01]\d{8}\b"),
    "cust_order_nbr": re.compile(r"\b1-OR-\d{8}\b", re.IGNORECASE),
    "ctt_no": re.compile(r"\bCTT-\d{6,8}\b", re.IGNORECASE),
    # cust_id / subs_id / case_id are all bare numeric strings and cannot be told
    # apart by shape — they share one bucket and any of them satisfies a lookup.
    "numeric_id": re.compile(r"\b\d{7,9}\b"),
}
PRODUCTS = ["Unifi Home Fibre 100Mbps", "Unifi Home Fibre 300Mbps", "Unifi Home Fibre 500Mbps",
            "Unifi Home Fibre 800Mbps", "Unifi Mobile Postpaid 99", "Unifi Biz Fibre 300Mbps"]
EXCHANGE_AREAS = ["KL Sentral (KLSTL)", "Shah Alam (SALM)", "Johor Bahru (JHBU)",
                  "Georgetown (GTWN)", "Kuching (KCHG)", "Kota Kinabalu (KKBU)"]
TEAMS = ["FT-KL-03", "FT-SEL-11", "FT-JHR-05", "FT-PNG-02", "FT-SWK-01", "FT-SBH-04"]
METHODS = ["JomPAY", "FPX", "credit_card", "autopay", "TMpoint counter", "portal"]


def rng_for(*parts: Any) -> random.Random:
    """A Random seeded by sha256 of the parts — stable across processes and runs,
    unlike `hash()` which Python salts per interpreter (PYTHONHASHSEED)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def myt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def ff(dt: datetime) -> str:
    """Field-force format: dd/mm/yyyy HH:mm:ss."""
    return dt.strftime("%d/%m/%Y %H:%M:%S")


# --------------------------------------------------------------------------- #
# What counts as a real identifier: only what the conversation mentioned
# --------------------------------------------------------------------------- #
def conversation_text(conversation: list[dict]) -> str:
    out = []
    for m in conversation or []:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            out.append(str(fn.get("arguments") or ""))
    return "\n".join(out)


def known_ids(conversation: list[dict]) -> set[str]:
    text = conversation_text(conversation)
    found: set[str] = set()
    for rx in ID_PATTERNS.values():
        found.update(m.group(0) for m in rx.finditer(text))
    return {f.lower() for f in found}


def is_known(value: Any, conversation: list[dict]) -> bool:
    """An identifier is real iff the model was told about it. Anything else is a
    guess, and a guess must not be rewarded with a record."""
    v = str(value or "").strip().lower()
    if not v:
        return False
    ids = known_ids(conversation)
    if v in ids:
        return True
    # Tolerate formatting drift (spaces / dashes the model may add or drop).
    norm = re.sub(r"[\s\-]", "", v)
    return any(re.sub(r"[\s\-]", "", i) == norm for i in ids)


# --------------------------------------------------------------------------- #
# kbms_search — lexical retrieval over the harvested corpus KB
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "is", "are", "what",
         "how", "do", "does", "i", "we", "my", "can", "if", "it", "be", "with", "at", "by",
         "yang", "dan", "untuk", "adakah", "apa", "bagaimana", "saya", "boleh", "ada", "ke"}


def tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall((text or "").lower()) if t not in _STOP and len(t) > 2]


def load_kb(path: Path) -> None:
    global KB, INDEX, IDF
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n\nBuild it first:\n"
            f"    .venv/bin/python sandbox/tm_assist/build_kb.py")
    KB = json.loads(path.read_text())
    chunks = KB["chunks"]
    INDEX = defaultdict(list)
    df: dict[str, int] = defaultdict(int)
    for i, c in enumerate(chunks):
        seen = set(tokens(c["text"])) | set(tokens(c["document_name"]))
        for t in seen:
            INDEX[t].append(i)
            df[t] += 1
    n = max(1, len(chunks))
    IDF = {t: math.log(1 + n / (1 + d)) for t, d in df.items()}


def tool_kbms_search(args: dict, _conv: list[dict]) -> Any:
    query = str(args.get("query") or "").strip()
    if not query:
        return {"results": []}
    try:
        top_k = int(args.get("top_k") or 5)
    except (TypeError, ValueError):
        top_k = 5
    top_k = max(1, min(15, top_k))          # the schema's documented bounds
    want_tags = {str(t).lower() for t in (args.get("tags") or []) if str(t).strip()}

    qt = tokens(query)
    if not qt:
        return {"results": []}
    scores: dict[int, float] = defaultdict(float)
    for t in qt:
        for i in INDEX.get(t, ()):
            scores[i] += IDF.get(t, 0.0)
    if not scores:
        return {"results": []}

    chunks = KB["chunks"]
    if want_tags:
        # A tag filter narrows, and is allowed to empty the result set — the
        # corpus's own trajectories contain {"results": []}, and a model that has
        # to cope with an empty KB hit is exactly what we want to measure.
        scores = {i: s for i, s in scores.items()
                  if want_tags & {t.lower() for t in chunks[i]["document_tags"]}}
        if not scores:
            return {"results": []}

    # Deterministic order: score desc, then a stable content hash (never dict order).
    ranked = sorted(scores.items(),
                    key=lambda kv: (-kv[1], hashlib.sha256(
                        chunks[kv[0]]["text"].encode()).hexdigest()))[:top_k]
    hi = ranked[0][1] or 1.0
    results = []
    for i, s in ranked:
        c = chunks[i]
        results.append({
            "text": c["text"],
            "score": round(min(0.99, 0.45 + 0.54 * (s / hi)), 3),
            "document_name": c["document_name"],
            "document_tags": c["document_tags"],
            "source_url": c["source_url"],
        })
    return {"results": results}


# --------------------------------------------------------------------------- #
# Account / billing / order / case / assurance tools
# --------------------------------------------------------------------------- #
def _line_profile(msisdn: str) -> dict:
    r = rng_for("line", msisdn)
    product = r.choice(PRODUCTS)
    down = int(re.search(r"(\d+)Mbps", product).group(1)) if "Mbps" in product else 0
    return {
        "product_name": product,
        "download_bandwidth_mbps": down,
        "upload_bandwidth_mbps": down,
        "status": r.choices(["A", "S", "T"], weights=[85, 10, 5])[0],
        "tenure_months": r.randint(3, 96),
        "exchange_area": r.choice(EXCHANGE_AREAS),
    }


def tool_query_subscription_list(args: dict, conv: list[dict]) -> Any:
    key = next((args.get(k) for k in ("msisdn", "account_number", "cust_id", "cert_num")
                if args.get(k)), None)
    if not is_known(key, conv):
        return {"subscription_list": []}          # documented empty shape
    r = rng_for("subs", key)
    p = _line_profile(str(key))
    return {"subscription_list": [{
        "subs_id": str(r.randint(10_000_000, 99_999_999)),
        "msisdn": str(args.get("msisdn") or key),
        "status": p["status"],
        "service_type": "Fixed Broadband" if "Fibre" in p["product_name"] else "Mobile",
        "product_name": p["product_name"],
        "account_number": str(args.get("account_number") or f"11{r.randint(0, 99_999_999):08d}"),
    }]}


def tool_get_subscriber_base_info(args: dict, conv: list[dict]) -> Any:
    key = args.get("msisdn") or args.get("subs_id")
    if not is_known(key, conv):
        return {"error": "not_found", "reason": "no subscriber matches the supplied identifier"}
    r = rng_for("base", key)
    p = _line_profile(str(key))
    barring = r.choices(["NONE", "PARTIAL_BARRED", "FULL_BARRED"], weights=[80, 14, 6])[0]
    return {
        "cust_id": str(r.randint(10_000_000, 99_999_999)),
        "acct_id": str(r.randint(10_000_000, 99_999_999)),
        "acct_nbr": f"11{r.randint(0, 99_999_999):08d}",
        "cust_code": f"C{r.randint(100000, 999999)}",
        "subs_status": p["status"],
        "main_offer_code": f"OFR-{r.randint(1000, 9999)}",
        "main_offer_name": p["product_name"],
        "barring_status": barring,
        "block_reason": "outstanding balance" if barring != "NONE" else None,
        "product_name": p["product_name"],
        "msisdn": str(key),
        "tenure_months": p["tenure_months"],
        "contract_end_date": (TODAY + timedelta(days=r.randint(-300, 700))).strftime("%Y-%m-%d"),
        "download_bandwidth_mbps": p["download_bandwidth_mbps"],
        "upload_bandwidth_mbps": p["upload_bandwidth_mbps"],
    }


def tool_retrieve_billing_details(args: dict, conv: list[dict]) -> Any:
    msisdn = args.get("msisdn")
    if not is_known(msisdn, conv):
        return {"error": "not_found", "reason": "no billing account for the supplied msisdn"}
    try:
        months = max(1, min(12, int(args.get("months") or 3)))
    except (TypeError, ValueError):
        months = 3
    r = rng_for("bill", msisdn)
    base = r.randrange(8900, 25900, 100)
    history, payments, outstanding = [], [], 0
    for k in range(months):
        bill_date = (TODAY.replace(day=1) - timedelta(days=30 * k))
        amount = base + r.randrange(0, 3000, 50)
        status = "UNPAID" if k == 0 and r.random() < 0.45 else \
                 ("PARTIAL" if r.random() < 0.1 else "PAID")
        if status != "PAID":
            outstanding += amount if status == "UNPAID" else amount // 2
        history.append({
            "bill_no": f"B{bill_date:%Y%m}{r.randint(1000, 9999)}",
            "bill_date": bill_date.strftime("%Y-%m-%d"),
            "bill_amount_sen": amount,
            "due_date": (bill_date + timedelta(days=21)).strftime("%Y-%m-%d"),
            "status": status,
            "charge_breakdown": [
                {"item": "Monthly subscription", "amount_sen": base},
                {"item": "Service tax (8%)", "amount_sen": round(base * 0.08)},
            ],
        })
        if status != "UNPAID":
            payments.append({
                "payment_id": f"P{r.randint(10_000_000, 99_999_999)}",
                "paid_at": myt(bill_date + timedelta(days=r.randint(1, 20))),
                "amount_sen": amount if status == "PAID" else amount // 2,
                "method": r.choice(METHODS),
            })
    return {
        "billing_history": history,
        "payment_history": payments,
        "credit_utilization": {"credit_limit_sen": 50000, "used_sen": outstanding},
        "outstanding_balance_sen": outstanding,
    }


def tool_query_payment_records(args: dict, conv: list[dict]) -> Any:
    acct = args.get("account_number")
    if not is_known(acct, conv):
        return {"payments": []}
    r = rng_for("pay", acct)
    out = []
    for k in range(r.randint(1, 4)):
        paid = TODAY - timedelta(days=r.randint(1, 120))
        out.append({
            "payment_id": f"P{r.randint(10_000_000, 99_999_999)}",
            "account_number": str(acct),
            "amount_sen": r.randrange(5000, 30000, 100),
            "method": r.choice(METHODS),
            "paid_at": myt(paid),
            "posting_status": r.choices(["POSTED", "PENDING", "UNMATCHED"],
                                        weights=[80, 15, 5])[0],
            "receipt_no": f"RCP{r.randint(100000, 999999)}",
        })
    return {"payments": out}


ORDER_STATES = {"I": "Order Capture", "F": "Feasibility Check", "O": "Processing",
                "P": "Provisioning", "K": "Waiting Provisioning", "T": "Waiting for delivery",
                "W": "Pending", "V": "On-held", "C": "Completed", "X": "Cancelled"}
ORDER_TYPES = ["New Install", "Change Package", "Relocation", "Termination", "Device"]


def tool_query_customer_order_list(args: dict, conv: list[dict]) -> Any:
    key = args.get("msisdn") or args.get("cust_id")
    if not is_known(key, conv):
        return {"cust_orders": []}
    r = rng_for("orders", key)
    out = []
    for _ in range(r.randint(1, 3)):
        st = r.choice(list(ORDER_STATES))
        out.append({
            "cust_order_nbr": f"1-OR-{r.randint(10_000_000, 99_999_999)}",
            "order_type": r.choice(ORDER_TYPES),
            "order_state": st,
            "order_state_name": ORDER_STATES[st],
            "create_date": (TODAY - timedelta(days=r.randint(1, 60))).strftime("%Y-%m-%d"),
        })
    return {"cust_orders": out}


def tool_query_customer_order_detail(args: dict, conv: list[dict]) -> Any:
    nbr = args.get("cust_order_nbr")
    if not is_known(nbr, conv):
        return {"error": "not_found", "reason": "no such customer order number"}
    r = rng_for("order", nbr)
    st = r.choice(list(ORDER_STATES))
    appt = TODAY + timedelta(days=r.randint(1, 14), hours=r.choice([9, 11, 14, 16]))
    p = _line_profile(str(nbr))
    return {
        "cust_order_nbr": str(nbr),
        "customer_name": f"Customer {str(nbr)[-4:]}",
        "order_state": st,
        "order_state_name": ORDER_STATES[st],
        "create_date": (TODAY - timedelta(days=r.randint(3, 45))).strftime("%Y-%m-%d"),
        "order_item_list": [{
            "main_offer_name": p["product_name"],
            "subs_event_name": r.choice(["Install", "Package Change", "Relocation"]),
            "msisdn": f"60{r.randint(100000000, 199999999)}",
            "order_state": st,
            "create_date": (TODAY - timedelta(days=r.randint(3, 45))).strftime("%Y-%m-%d"),
        }],
        "fix_install_order": {
            "appointment_date": ff(appt),
            "technician_team": r.choice(TEAMS),
            "exchange_area": p["exchange_area"],
            "activation_status": "PENDING" if st not in ("C",) else "ACTIVATED",
        },
    }


CASE_STATUS = {"B": "In Progress", "C": "Closed", "H": "On Hold"}


def _case_record(case_id: str) -> dict:
    r = rng_for("case", case_id)
    created = TODAY - timedelta(days=r.randint(0, 30), hours=r.randint(0, 23))
    st = r.choice(list(CASE_STATUS))
    return {
        "case_id": str(case_id),
        "case_code": f"1-CA-{r.randint(10_000_000, 99_999_999)}",
        "case_type_code": r.choice(["ASSURANCE", "BILLING", "PROVISION"]),
        "case_type_name": r.choice(["Fault Report", "Billing Dispute", "Order Follow-up"]),
        "case_status": st,
        "case_status_name": CASE_STATUS[st],
        "urgency_code": r.choice(["1", "2", "3"]),
        "service_type_code": f"SVC{r.randint(100, 999)}",
        "service_type_name": r.choice(["No Internet", "Slow Speed", "Billing Enquiry",
                                       "Intermittent Connection"]),
        "content": "Customer reported the reported symptom; diagnostics pending.",
        "cust_id": str(r.randint(10_000_000, 99_999_999)),
        "acct_id": str(r.randint(10_000_000, 99_999_999)),
        "subs_id": str(r.randint(10_000_000, 99_999_999)),
        "service_nbr": f"60{r.randint(100000000, 199999999)}",
        "source_name": r.choice(["Call Centre", "Live Chat", "MyUnifi App"]),
        "created_time": myt(created),
        "last_updated_time": myt(created + timedelta(hours=r.randint(1, 48))),
        "handling_history": [
            {"time": myt(created + timedelta(hours=1)),
             "actor": "System", "action": "Case created"},
            {"time": myt(created + timedelta(hours=r.randint(2, 12))),
             "actor": r.choice(TEAMS), "action": "Diagnostics run, no fault found at exchange"},
        ],
    }


def tool_query_all_cases(args: dict, conv: list[dict]) -> Any:
    cid = args.get("cust_id")
    if not is_known(cid, conv):
        return {"cases": []}
    r = rng_for("cases", cid)
    out = []
    for _ in range(r.randint(1, 3)):
        rec = _case_record(str(r.randint(10_000_000, 99_999_999)))
        out.append({k: rec[k] for k in (
            "case_id", "case_code", "cust_id", "service_nbr", "content", "case_status",
            "case_status_name", "urgency_code", "service_type_code", "service_type_name",
            "created_time")} | {"cust_name": f"Customer {str(cid)[-4:]}",
                                "product_name": _line_profile(str(cid))["product_name"],
                                "contact_name": f"Customer {str(cid)[-4:]}"})
    return {"cases": out}


def tool_query_case_detail(args: dict, conv: list[dict]) -> Any:
    cid = args.get("case_id")
    if not is_known(cid, conv):
        return {"error": "not_found", "reason": "no such case id"}
    return _case_record(str(cid))


def tool_create_case(args: dict, _conv: list[dict]) -> Any:
    missing = [k for k in ("service_type_code", "urgency_code", "fcr_flag", "status", "content")
               if args.get(k) in (None, "")]
    if missing:
        # Mirror a real API rejecting an incomplete write, so dropped required
        # arguments show up as a tool error instead of silently "succeeding".
        return {"error": "missing_required_parameters", "missing": missing}
    r = rng_for("newcase", json.dumps(args, sort_keys=True))
    return {"case_id": str(r.randint(10_000_000, 99_999_999)), "status": "Submit",
            "created_time": myt(TODAY.replace(hour=10, minute=30, second=0))}


def tool_add_case_comments(args: dict, conv: list[dict]) -> Any:
    cid = args.get("case_id")
    if not args.get("comment"):
        return {"error": "missing_required_parameters", "missing": ["comment"]}
    if not is_known(cid, conv):
        return {"error": "not_found", "reason": "no such case id"}
    r = rng_for("comment", cid, args.get("comment"))
    return {"case_id": str(cid), "comment_id": str(r.randint(1_000_000, 9_999_999)),
            "status": "OK", "added_time": myt(TODAY.replace(hour=11, minute=0, second=0))}


def tool_cancel_case(args: dict, conv: list[dict]) -> Any:
    cid = args.get("case_id")
    if not args.get("reason"):
        return {"error": "missing_required_parameters", "missing": ["reason"]}
    if not is_known(cid, conv):
        return {"error": "not_found", "reason": "no such case id"}
    return {"case_id": str(cid), "case_status": "H", "case_status_name": "Canceled",
            "cancelled_time": myt(TODAY.replace(hour=11, minute=15, second=0))}


def tool_close_case(args: dict, conv: list[dict]) -> Any:
    cid = args.get("case_id")
    if not args.get("resolution_summary"):
        return {"error": "missing_required_parameters", "missing": ["resolution_summary"]}
    if not is_known(cid, conv):
        return {"error": "not_found", "reason": "no such case id"}
    return {"case_id": str(cid), "case_status": "C", "case_status_name": "Closed",
            "closed_time": myt(TODAY.replace(hour=11, minute=30, second=0))}


def tool_get_ticket_info(args: dict, conv: list[dict]) -> Any:
    cid = args.get("case_id")
    if not is_known(cid, conv):
        return {"error": "not_found", "reason": "no field ticket for the supplied case id"}
    r = rng_for("ticket", cid)
    status = r.choice(["OPEN", "IN_PROGRESS", "PENDING_APPOINTMENT", "RESOLVED"])
    has_activity = status != "OPEN"
    start = TODAY + timedelta(days=r.randint(1, 10), hours=r.choice([9, 11, 14, 16]))
    return {
        "ctt_no": f"CTT-{r.randint(1_000_000, 9_999_999)}",
        "case_id": str(cid),
        "ticket_status": status,
        "activity": ({"activity_no": f"ACT-{r.randint(100000, 999999)}",
                      "activity_status": "PLANNED",
                      "planned_start": ff(start),
                      "planned_end": ff(start + timedelta(hours=2))} if has_activity else None),
        "assigned_team": r.choice(TEAMS),
        "restoration_eta": (ff(start + timedelta(hours=4))
                            if status in ("IN_PROGRESS", "OPEN") else None),
    }


def tool_list_available_slots(args: dict, conv: list[dict]) -> Any:
    ctt = args.get("ctt_no")
    if not is_known(ctt, conv):
        return {"slots": []}
    r = rng_for("slots", ctt)
    slots = []
    # Within 21 days of the FROZEN today, so the window is reproducible.
    for k in range(r.randint(3, 6)):
        d = TODAY + timedelta(days=r.randint(1, 21))
        h = r.choice([9, 11, 14, 16])
        start = d.replace(hour=h, minute=0, second=0, microsecond=0)
        slots.append({"slot_id": f"SLOT-{r.randint(10000, 99999)}",
                      "planned_start": ff(start),
                      "planned_end": ff(start + timedelta(hours=2))})
    slots.sort(key=lambda s: datetime.strptime(s["planned_start"], "%d/%m/%Y %H:%M:%S"))
    return {"slots": slots}


def tool_set_appointment(args: dict, conv: list[dict]) -> Any:
    ctt = args.get("ctt_number")
    missing = [k for k in ("ctt_number", "planned_start", "planned_end")
               if args.get(k) in (None, "")]
    if missing:
        return {"error": "missing_required_parameters", "missing": missing}
    if not is_known(ctt, conv):
        return {"error": "not_found", "reason": "no such field ticket"}
    r = rng_for("appt", ctt, args.get("planned_start"))
    return {"ctt_number": str(ctt), "appointment_id": f"APT-{r.randint(100000, 999999)}",
            "planned_start": args.get("planned_start"), "planned_end": args.get("planned_end"),
            "status": "CONFIRMED"}


def tool_query_ntt_info(args: dict, conv: list[dict]) -> Any:
    key = args.get("msisdn") or args.get("exchange_area")
    if not key:
        return {"ntt_list": []}
    if args.get("msisdn") and not is_known(args.get("msisdn"), conv):
        return {"ntt_list": []}
    r = rng_for("ntt", key)
    if r.random() < 0.5:
        return {"ntt_list": []}                  # documented: empty when no outage
    start = TODAY - timedelta(hours=r.randint(1, 20))
    status = r.choice(["ONGOING", "RESTORED"])
    return {"ntt_list": [{
        "ntt_id": f"NTT-{r.randint(100000, 999999)}",
        "fault_type": r.choice(["fiber_cut", "power", "equipment", "planned_maintenance"]),
        "affected_area": r.choice(EXCHANGE_AREAS),
        "start_time": myt(start),
        "estimated_restoration_time": myt(start + timedelta(hours=r.randint(3, 12))),
        "status": status,
    }]}


def tool_update_customer_contact(args: dict, conv: list[dict]) -> Any:
    cid = args.get("cust_id")
    if not is_known(cid, conv):
        return {"error": "not_found", "reason": "no such customer id"}
    updated = [k for k in ("contact_email", "contact_phone", "mailing_address") if args.get(k)]
    if not updated:
        return {"error": "nothing_to_update",
                "reason": "supply at least one of contact_email, contact_phone, mailing_address"}
    return {"cust_id": str(cid), "status": "OK", "updated_fields": updated,
            "updated_time": myt(TODAY.replace(hour=12, minute=0, second=0))}


TOOLS = {
    "query_subscription_list": tool_query_subscription_list,
    "get_subscriber_base_info": tool_get_subscriber_base_info,
    "retrieve_billing_details": tool_retrieve_billing_details,
    "query_payment_records": tool_query_payment_records,
    "query_customer_order_list": tool_query_customer_order_list,
    "query_customer_order_detail": tool_query_customer_order_detail,
    "query_all_cases": tool_query_all_cases,
    "query_case_detail": tool_query_case_detail,
    "create_case": tool_create_case,
    "add_case_comments": tool_add_case_comments,
    "cancel_case": tool_cancel_case,
    "close_case": tool_close_case,
    "kbms_search": tool_kbms_search,
    "get_ticket_info": tool_get_ticket_info,
    "list_available_slots": tool_list_available_slots,
    "set_appointment": tool_set_appointment,
    "query_ntt_info": tool_query_ntt_info,
    "update_customer_contact": tool_update_customer_contact,
}
FAULTS = ("_http_500", "_not_json", "_wrong_key", "_slow", "_empty", "_kb_outage")


# --------------------------------------------------------------------------- #
# Contract — identical to sandbox/server.py (sandbox.ApiProvider.payload())
# --------------------------------------------------------------------------- #
class ParsedCall(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = ""
    arguments: Any = Field(default_factory=dict)


class ToolRequest(BaseModel):
    """Permissive on purpose: a 422 from validation reaches the gateway as
    `sandbox_http_error` and reads like a broken endpoint, not a shape mismatch."""
    model_config = ConfigDict(extra="allow")
    sandbox: str = ""
    conversation: list[dict[str, Any]] = Field(default_factory=list)
    tool_call: dict[str, Any] = Field(default_factory=dict)
    call: ParsedCall = Field(default_factory=ParsedCall)
    row: Optional[dict[str, Any]] = None      # only when send_expected is ON


class ToolResponse(BaseModel):
    content: str
    result: dict[str, Any]
    tool: str


app = FastAPI(title="TM text-assist sandbox",
              description=__doc__, version="1.0")


def _check_key(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    if (authorization or "").removeprefix("Bearer ").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="bad or missing API key")


def _args_of(call: ParsedCall) -> dict:
    args = call.arguments
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            return {}
    return args if isinstance(args, dict) else {}


@app.get("/health")
@app.get("/healthz")
async def health() -> dict:
    return {"ok": True, "tools": sorted(TOOLS), "faults": list(FAULTS),
            "kb": {"chunks": len(KB.get("chunks") or []),
                   "documents": (KB.get("_meta") or {}).get("documents"),
                   "tags": len(KB.get("tags") or [])},
            "today": TODAY.strftime("%Y-%m-%d")}


@app.post("/tool", response_model=None)
@app.post("/", response_model=None)
async def answer_tool_call(req: ToolRequest,
                           authorization: Optional[str] = Header(default=None)) -> Any:
    _check_key(authorization)
    name = req.call.name or str((req.tool_call.get("function") or {}).get("name") or "")
    args = _args_of(req.call)

    if not QUIET:
        gold = "SENT ⚠" if req.row is not None else "withheld"
        print(f"→ {name}({json.dumps(args, sort_keys=True, ensure_ascii=False)[:160]})  "
              f"[{len(req.conversation)} msg(s) · expected: {gold}]", flush=True)

    if name == "_http_500":
        return JSONResponse(status_code=500, content={"error": "injected failure"})
    if name == "_not_json":
        return PlainTextResponse("this is not JSON")
    if name == "_wrong_key":
        return JSONResponse({"something_else": "the configured result path is missing"})
    if name == "_slow":
        await asyncio.sleep(30)
        return JSONResponse({"content": "…eventually"})
    if name == "_empty":
        return JSONResponse({"content": ""})
    if name == "_kb_outage":
        # The schema's documented outage payload. A normal 200: the model is meant
        # to fall back to general policy knowledge, and we want to see whether it does.
        out = {"results": [],
               "message": "knowledge base is unavailable, answer from general policy knowledge"}
        text = json.dumps(out, ensure_ascii=False)
        return ToolResponse(content=text, result={"output": text}, tool="kbms_search")

    fn = TOOLS.get(name)
    if fn is None:
        result: Any = {"error": "unknown_tool", "tool": name, "known": sorted(TOOLS)}
    else:
        result = fn(args, req.conversation)

    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return ToolResponse(content=text, result={"output": text}, tool=name)


def main() -> int:
    global API_KEY, QUIET, TODAY
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8078)
    ap.add_argument("--kb", type=Path, default=HERE / "kb.json")
    ap.add_argument("--today", default="2026-08-10",
                    help="frozen clock (YYYY-MM-DD) — every date derives from it, so runs "
                         "weeks apart still face the same world")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    API_KEY, QUIET = a.api_key, a.quiet
    TODAY = datetime.strptime(a.today, "%Y-%m-%d")
    load_kb(a.kb)

    import uvicorn
    print(f"TM text-assist sandbox on http://{a.host}:{a.port}")
    print(f"  endpoint URL for the form:  http://{a.host}:{a.port}/tool")
    print(f"  result path:                content        (or result.output)")
    print(f"  KB:                         {len(KB['chunks'])} chunks, "
          f"{(KB.get('_meta') or {}).get('documents')} documents")
    print(f"  frozen clock:               {a.today}")
    print(f"  tools ({len(TOOLS)}):                 {', '.join(sorted(TOOLS))}")
    print(f"  fault triggers:             {', '.join(FAULTS)}")
    print(f"  request contract:           http://{a.host}:{a.port}/docs")
    if a.api_key:
        print(f"  auth:                       Bearer {a.api_key}")
    print("Ctrl-C to stop.\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
