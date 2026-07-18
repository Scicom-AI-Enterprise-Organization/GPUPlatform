"""auth — pure helpers: password hashing, API-key minting, role/section gating.
(The full HTTP auth flow is covered by the integration suite.)"""
from types import SimpleNamespace

from gateway import auth


def test_password_hash_roundtrip():
    h = auth.hash_password("s3cret")
    assert h != "s3cret"
    assert auth.verify_password("s3cret", h)
    assert not auth.verify_password("wrong", h)


def test_verify_password_tolerates_malformed_hash():
    # A corrupted users.password_hash row must fail closed, not 500.
    assert not auth.verify_password("x", "not-a-bcrypt-hash")


def test_api_key_mint_and_hash():
    key = auth.mint_api_key()
    assert key.startswith(auth.API_KEY_PREFIX)
    # sha256 hex digest — stable, so the DB lookup by hash works.
    assert auth.hash_api_key(key) == auth.hash_api_key(key)
    assert len(auth.hash_api_key(key)) == 64
    # Keys are unique.
    assert auth.mint_api_key() != auth.mint_api_key()


def _user(role="user", is_admin=False):
    return SimpleNamespace(role=role, is_admin=is_admin)


def test_role_gate():
    assert auth._has_role(_user("admin"), "admin")
    assert auth._has_role(_user("developer"), "developer", "admin")
    assert not auth._has_role(_user("user"), "developer", "admin")
    # is_admin flag bypasses the role string entirely.
    assert auth._has_role(_user("user", is_admin=True), "developer")


def test_disabled_sections_parsing(monkeypatch):
    monkeypatch.setenv("DISABLED_SECTIONS", " Compute , quantization,, ")
    assert auth._disabled_sections() == {"compute", "quantization"}
    monkeypatch.delenv("DISABLED_SECTIONS")
    assert auth._disabled_sections() == set()


async def test_disabled_section_denied_even_for_admin(monkeypatch):
    monkeypatch.setenv("DISABLED_SECTIONS", "compute")
    admin = _user("admin", is_admin=True)
    # session is never touched on the disabled-section fast path.
    assert not await auth.has_section(admin, "compute", session=None)
    assert await auth.has_section(admin, "inference", session=None)
