"""Audit logging — immutable record of every state-changing action.

Call sites use `record(...)`; this module owns the persistence path so individual
features (compute / benchmark / apps) don't have to import the model directly.
Best-effort: if the audit write fails the original action still succeeds — we
never want a logging glitch to break user-visible flows. Failures are logged
to the python logger so they're visible in `kubectl logs`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .db import AuditLog, User, session_factory

logger = logging.getLogger("gateway.audit")


def cost_breakdown(
    started_at: Optional[datetime],
    ended_at: Optional[datetime],
    cost_per_hr: Optional[float],
) -> Optional[dict[str, Any]]:
    """Compute the final accumulated cost for a resource being torn down.

    Returns a dict suitable for inclusion in the audit `details` blob, or
    None if we don't have enough info (no start time, no rate, etc.). When
    `ended_at` is missing we treat the resource as still running and bill
    to now — gives the audit log a useful "spent so far" snapshot at the
    moment of deletion.
    """
    if started_at is None or cost_per_hr is None:
        return None
    end = ended_at or datetime.now(timezone.utc)
    elapsed_s = max(0.0, (end - started_at).total_seconds())
    final_cost = (elapsed_s / 3600.0) * cost_per_hr
    return {
        "final_cost_usd": round(final_cost, 6),
        "duration_s": int(elapsed_s),
        "rate_per_hr": cost_per_hr,
    }


# Canonical action names. Keep these short, dotted, lowercase. The set is
# closed by convention so the admin UI can render filter dropdowns; new
# actions need to be added here on purpose.
ACTIONS = {
    "inference.create",
    "inference.delete",
    "inference.restart",
    "inference.stop",
    "inference.purge",
    "inference.update_autoscaler",
    "benchmark.create",
    "benchmark.delete",
    "compute.create",
    "compute.delete",
    "compute.request",
    "compute.approve",
    "compute.reject",
    "user.role_change",
    "user.permissions_change",
    "user.delete",
    "policy_role.create",
    "policy_role.update",
    "policy_role.delete",
}


async def record(
    actor: User,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    resource_name: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
) -> None:
    """Persist a single audit row. Never raises — best-effort by design."""
    if action not in ACTIONS:
        # Don't drop; we'd rather notice an unregistered action in logs.
        logger.warning("audit: unregistered action %r — recording anyway", action)
    try:
        async with session_factory()() as s:
            row = AuditLog(
                actor_id=actor.id,
                actor_username=actor.username,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                resource_name=resource_name,
                details=details or None,
            )
            s.add(row)
            await s.commit()
    except Exception:
        logger.exception(
            "audit: failed to record %s for actor=%s resource=%s/%s",
            action, actor.username, resource_type, resource_id,
        )
