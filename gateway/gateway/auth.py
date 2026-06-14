"""User auth: register, login, session-token middleware.

- Passwords: bcrypt-hashed in Postgres `users.password_hash`.
- Sessions: random opaque tokens. Stored in Redis as `session:{token}` -> user_id
  with TTL (default 7d). Logout = DEL key.
- Bearer tokens in `Authorization: Bearer <token>` resolve to a User row.
"""
from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import User, get_session, get_user_by_id

SESSION_TTL_S = 7 * 24 * 3600  # 7 days

# Long-lived API keys: `sgpu_<random>`. Distinguished from session tokens by the
# prefix so `current_user` can route to the right validation path.
API_KEY_PREFIX = "sgpu_"
# Don't write last_used_at on every request — only when it's this stale.
API_KEY_LAST_USED_THROTTLE_S = 60


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def mint_api_key() -> str:
    """A fresh plaintext API key. Returned to the user once; only its hash is stored."""
    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except ValueError:
        return False


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


async def create_session(rdb, user_id: int) -> str:
    token = new_session_token()
    await rdb.set(f"session:{token}", str(user_id), ex=SESSION_TTL_S)
    return token


async def revoke_session(rdb, token: str) -> None:
    await rdb.delete(f"session:{token}")


async def resolve_session(rdb, token: str) -> Optional[int]:
    raw = await rdb.get(f"session:{token}")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


async def current_user(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> User:
    """FastAPI dep: resolves Authorization: Bearer <token> -> User. 401 otherwise.

    When AUTH_DISABLED=1, every request acts as the seeded admin user — no
    token required. Intended for demos / dev where security is irrelevant.
    """
    if os.environ.get("AUTH_DISABLED") == "1":
        result = await session.execute(select(User).where(User.is_admin == True).limit(1))
        admin = result.scalar_one_or_none()
        if admin is None:
            raise HTTPException(
                status_code=503,
                detail={"error": "AUTH_DISABLED set but no admin user exists; set ADMIN_USERNAME + ADMIN_PASSWORD env"},
            )
        return admin

    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail={"error": "missing or malformed Authorization header"})
    token = header[len("Bearer "):].strip()

    # API key path: `sgpu_…` validates against the api_keys table (hashed) and
    # authenticates as its owner, inheriting that user's role + section access.
    if token.startswith(API_KEY_PREFIX):
        return await _user_from_api_key(session, token)

    rdb = request.app.state.redis
    user_id = await resolve_session(rdb, token)
    if user_id is None:
        raise HTTPException(status_code=401, detail={"error": "invalid or expired session"})

    user = await get_user_by_id(session, user_id)
    if user is None:
        # Session points at a deleted user; clean up.
        await revoke_session(rdb, token)
        raise HTTPException(status_code=401, detail={"error": "user no longer exists"})
    return user


async def _user_from_api_key(session: AsyncSession, key: str) -> User:
    """Resolve a `sgpu_…` API key to its owner. 401 if unknown/revoked. Updates
    last_used_at at most once per throttle window to avoid a write per request."""
    from .db import ApiKey
    key_hash = hash_api_key(key)
    result = await session.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.revoked_at.is_(None))
    )
    ak = result.scalar_one_or_none()
    if ak is None:
        raise HTTPException(status_code=401, detail={"error": "invalid or revoked API key"})
    user = await get_user_by_id(session, ak.owner_id)
    if user is None:
        raise HTTPException(status_code=401, detail={"error": "API key owner no longer exists"})
    now = datetime.now(timezone.utc)
    last = ak.last_used_at
    if last is not None and last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    if last is None or (now - last) > timedelta(seconds=API_KEY_LAST_USED_THROTTLE_S):
        ak.last_used_at = now
        await session.commit()
    return user


def _has_role(user: User, *roles: str) -> bool:
    return user.role in roles or user.is_admin


async def require_developer(user: User = Depends(current_user)) -> User:
    """User must be at least developer (or admin). Plain 'user' role => 403."""
    if not _has_role(user, "developer", "admin"):
        raise HTTPException(status_code=403, detail={"error": "no platform access — ask an admin to grant developer role"})
    return user


async def require_admin(user: User = Depends(current_user)) -> User:
    """Admin-only endpoints (role management, user list)."""
    if not _has_role(user, "admin"):
        raise HTTPException(status_code=403, detail={"error": "admin only"})
    return user


# Section-level access. Layered on top of role: only `developer` (and `admin`,
# who bypasses) can ever access platform sections, then per-user
# section_permissions narrows it further. Existing developers (pre-migration)
# default to all-true via the migration backfill.
SECTIONS = ("inference", "benchmark", "compute", "datasets", "autotrain", "catalog")


def _disabled_sections() -> set[str]:
    """Sections turned OFF platform-wide via the `DISABLED_SECTIONS` env
    (comma-separated, e.g. `DISABLED_SECTIONS=compute`). A disabled section is
    denied to everyone — admins included — so `/auth/me` reports it false (the web
    hides its nav + pages) and its routes 403. Use this to ship without a surface."""
    raw = os.environ.get("DISABLED_SECTIONS", "")
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


async def has_section(user: User, section: str, session: AsyncSession) -> bool:
    """Effective permission lookup. A platform-wide `DISABLED_SECTIONS` env wins
    over everything; otherwise admins bypass and developers resolve through their
    attached policy role's `sections` map. Tier `user` always denied.
    """
    if section in _disabled_sections():
        return False
    if _has_role(user, "admin"):
        return True
    if not _has_role(user, "developer"):
        return False
    if not user.policy_role_id:
        return False
    from .db import PolicyRole
    role = await session.get(PolicyRole, user.policy_role_id)
    if role is None:
        return False
    return bool((role.sections or {}).get(section, False))


def require_section(section: str):
    """Dependency factory: returns a FastAPI dep that gates on a section."""
    async def _dep(
        user: User = Depends(current_user),
        session: AsyncSession = Depends(get_session),
    ) -> User:
        if not _has_role(user, "developer", "admin"):
            raise HTTPException(
                status_code=403,
                detail={"error": "no platform access — ask an admin to grant developer role"},
            )
        if not await has_section(user, section, session):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": f"no access to {section} — ask an admin to attach a role with {section} access",
                    "section": section,
                },
            )
        return user
    return _dep
