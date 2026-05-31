"""Integration tests for the storage API (/v1/storage) against the running
gateway, authenticated with a real API key (see conftest).

Admin CRUD + validation + the guarantee that raw credentials are never returned.
Created rows are cleaned up via the `cleanup` fixture. The `/v1/storage/test`
connectivity probe hits S3 / HuggingFace and is out of scope.
"""
from __future__ import annotations

import secrets


def _name(prefix: str) -> str:
    return f"pytest-{prefix}-{secrets.token_hex(4)}"


async def _create_s3(client, cleanup, **extra):
    name = _name("s3")
    body = {"name": name, "kind": "s3", "bucket": "pytest-bucket", **extra}
    r = await client.post("/v1/storage", json=body)
    assert r.status_code == 200, r.text
    st = r.json()
    cleanup.append(f"/v1/storage/{st['id']}")
    return st


async def test_create_list_get(client, cleanup):
    st = await _create_s3(client, cleanup, region="us-east-1", prefix="datasets")
    assert st["kind"] == "s3"
    assert st["bucket"] == "pytest-bucket"
    assert st["region"] == "us-east-1"
    assert st["has_credentials"] is False  # no keys → env fallback
    sid = st["id"]

    r = await client.get("/v1/storage")
    assert r.status_code == 200
    assert sid in [s["id"] for s in r.json()]


async def test_create_with_creds_masked(client, cleanup):
    st = await _create_s3(client, cleanup, access_key_id="AKIATESTKEY", secret_access_key="topsecret123")
    assert st["has_credentials"] is True
    # raw credentials never come back over the wire
    body = (await client.get("/v1/storage")).text
    assert "AKIATESTKEY" not in body
    assert "topsecret123" not in body


async def test_hf_storage_creds_masked(client, cleanup):
    name = _name("hf")
    r = await client.post("/v1/storage", json={"name": name, "kind": "huggingface", "hf_token": "hf_supersecret"})
    assert r.status_code == 200, r.text
    st = r.json()
    cleanup.append(f"/v1/storage/{st['id']}")
    assert st["kind"] == "huggingface"
    assert st["has_credentials"] is True
    assert "hf_supersecret" not in r.text


async def test_create_validation(client):
    # unsupported kind
    assert (await client.post("/v1/storage", json={"name": _name("x"), "kind": "bogus"})).status_code == 400
    # blank name
    assert (await client.post("/v1/storage", json={"name": " ", "kind": "s3", "bucket": "b"})).status_code == 400
    # s3 missing bucket
    assert (await client.post("/v1/storage", json={"name": _name("nb"), "kind": "s3"})).status_code == 400
    # only one half of a credential pair
    assert (
        await client.post("/v1/storage", json={"name": _name("half"), "kind": "s3", "bucket": "b", "access_key_id": "only-this"})
    ).status_code == 400


async def test_duplicate_name(client, cleanup):
    name = _name("dup")
    r1 = await client.post("/v1/storage", json={"name": name, "kind": "s3", "bucket": "b1"})
    assert r1.status_code == 200
    cleanup.append(f"/v1/storage/{r1.json()['id']}")
    r2 = await client.post("/v1/storage", json={"name": name, "kind": "s3", "bucket": "b2"})
    assert r2.status_code == 400


async def test_update_and_delete(client):
    name = _name("u")
    sid = (await client.post("/v1/storage", json={"name": name, "kind": "s3", "bucket": "b"})).json()["id"]

    new = _name("u-renamed")
    r = await client.patch(f"/v1/storage/{sid}", json={"name": new, "prefix": "models", "enabled": False})
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["name"] == new
    assert st["prefix"] == "models"
    assert st["enabled"] is False

    r = await client.delete(f"/v1/storage/{sid}")
    assert r.status_code == 200
    assert sid not in [s["id"] for s in (await client.get("/v1/storage")).json()]


async def test_update_unknown_404(client):
    assert (await client.patch("/v1/storage/store-nope", json={"name": "x"})).status_code == 404
    assert (await client.delete("/v1/storage/store-nope")).status_code == 404
