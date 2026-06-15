"""Resolve cloud-provider credentials for bench / compute / serverless paths.

Replaces direct `os.environ["RUNPOD_API_KEY"]` reads scattered across the
codebase. Fallback chain when caller passes `provider_id=None`:

    1. The requesting user's single owned provider of the right kind (if any)
    2. The gateway-wide env var (RUNPOD_API_KEY / PI_API_KEY)
    3. raise — caller decides whether that's 400 (user-facing) or skip-best-effort

This module never raises HTTPException directly; callers map the RuntimeError
to whatever response shape they need.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import crypto
from .db import Provider

logger = logging.getLogger("gateway.provider_resolve")


@dataclass
class CloudCreds:
    api_key: str
    ssh_priv_pem: Optional[str]    # None when sourced from env (no managed keypair)
    ssh_pub: Optional[str]
    cloud_type: Optional[str]      # populated only by env (RUNPOD_CLOUD_TYPE)
    source: str                    # "provider:<id>" | "env"


_ENV_KEY_BY_KIND = {"runpod": "RUNPOD_API_KEY", "pi": "PI_API_KEY"}


async def resolve_cloud_creds(
    session: AsyncSession,
    provider_id: Optional[str],
    expected_kind: str,
    owner_id: Optional[int] = None,
) -> CloudCreds:
    """Return the credential bundle for a runpod/pi action.

    `owner_id`, when provided, is used by the "user's sole owned provider"
    fallback so we don't pick someone else's account. Pass `None` to skip
    that step entirely and only fall back to env.
    """
    if expected_kind not in _ENV_KEY_BY_KIND:
        raise RuntimeError(f"resolve_cloud_creds: unknown kind {expected_kind!r}")

    # Central kill-switch: when cloud providers are disabled (CAE/CCE), no
    # runpod/pi credentials resolve anywhere — serverless per-app rows, compute,
    # and benchmarks all route through here. try_resolve_cloud_creds() swallows
    # this (RuntimeError) and returns None, so best-effort cost lookups degrade
    # cleanly rather than erroring.
    from .provider import cloud_providers_disabled, CloudProviderDisabled
    if cloud_providers_disabled():
        raise CloudProviderDisabled(expected_kind)

    if provider_id:
        row = await session.get(Provider, provider_id)
        if row is None:
            raise RuntimeError(f"provider {provider_id} not found")
        if row.kind != expected_kind:
            raise RuntimeError(
                f"provider {provider_id} is kind={row.kind}, expected {expected_kind}"
            )
        cfg = row.config or {}
        enc = cfg.get("api_key_enc")
        if not enc:
            raise RuntimeError(f"provider {provider_id} has no stored api_key")
        priv_enc = cfg.get("ssh_priv_enc")
        return CloudCreds(
            api_key=crypto.decrypt(enc),
            ssh_priv_pem=crypto.decrypt(priv_enc) if priv_enc else None,
            ssh_pub=cfg.get("ssh_pub"),
            cloud_type=None,
            source=f"provider:{provider_id}",
        )

    if owner_id is not None:
        result = await session.execute(
            select(Provider).where(
                Provider.owner_id == owner_id,
                Provider.kind == expected_kind,
            )
        )
        rows = list(result.scalars().all())
        if len(rows) == 1:
            row = rows[0]
            cfg = row.config or {}
            enc = cfg.get("api_key_enc")
            if enc:
                priv_enc = cfg.get("ssh_priv_enc")
                return CloudCreds(
                    api_key=crypto.decrypt(enc),
                    ssh_priv_pem=crypto.decrypt(priv_enc) if priv_enc else None,
                    ssh_pub=cfg.get("ssh_pub"),
                    cloud_type=None,
                    source=f"provider:{row.id}",
                )

    env_var = _ENV_KEY_BY_KIND[expected_kind]
    env_key = os.environ.get(env_var, "").strip()
    if env_key:
        return CloudCreds(
            api_key=env_key,
            ssh_priv_pem=None,
            ssh_pub=None,
            cloud_type=os.environ.get("RUNPOD_CLOUD_TYPE") if expected_kind == "runpod" else None,
            source="env",
        )

    raise RuntimeError(
        f"no {expected_kind} credentials — register a provider or set {env_var}"
    )


async def try_resolve_cloud_creds(
    session: AsyncSession,
    provider_id: Optional[str],
    expected_kind: str,
    owner_id: Optional[int] = None,
) -> Optional[CloudCreds]:
    """Best-effort variant — returns None instead of raising. Used by cost
    lookups and other paths where a missing key shouldn't break the run."""
    try:
        return await resolve_cloud_creds(session, provider_id, expected_kind, owner_id)
    except RuntimeError:
        return None
