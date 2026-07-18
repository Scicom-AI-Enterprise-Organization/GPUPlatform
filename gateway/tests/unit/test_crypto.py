"""crypto — Fernet at-rest secret encryption."""
import pytest

from gateway import crypto


def test_roundtrip():
    ct = crypto.encrypt("ssh-rsa AAAA... secret")
    assert ct != "ssh-rsa AAAA... secret"
    assert crypto.decrypt(ct) == "ssh-rsa AAAA... secret"


def test_ciphertexts_are_salted():
    # Fernet embeds a random IV — equal plaintexts must not produce equal
    # ciphertexts (otherwise the providers table leaks key reuse).
    assert crypto.encrypt("same") != crypto.encrypt("same")


def test_garbage_ciphertext_raises_actionable_error():
    with pytest.raises(RuntimeError, match="PROVIDER_SECRET_KEY"):
        crypto.decrypt("not-a-token")


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("PROVIDER_SECRET_KEY", raising=False)
    crypto._fernet.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="PROVIDER_SECRET_KEY not set"):
            crypto.encrypt("x")
    finally:
        crypto._fernet.cache_clear()  # let later tests re-read the env key
