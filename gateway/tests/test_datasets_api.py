"""Integration tests for the datasets API (/v1/datasets) against the running
gateway, authenticated with a real API key (see conftest).

Exercises the CRUD + validation + per-split column surface that doesn't touch
S3 / HuggingFace / the network. Created rows are cleaned up via the `cleanup`
fixture. Preview / transform / splits / audio hit external services and are out
of scope.
"""
from __future__ import annotations

import secrets


def _name(prefix: str) -> str:
    return f"pytest-{prefix}-{secrets.token_hex(4)}"


async def _create_hf(client, cleanup, repo="pytest-org/example"):
    name = _name("ds")
    r = await client.post("/v1/datasets", json={"name": name, "kind": "hf", "hf_repo": repo})
    assert r.status_code == 200, r.text
    ds = r.json()
    cleanup.append(f"/v1/datasets/{ds['id']}")
    return ds


async def test_create_list_get(client, cleanup):
    ds = await _create_hf(client, cleanup, repo="pytest-org/example")
    assert ds["kind"] == "hf"
    assert ds["hf_repo"] == "pytest-org/example"
    assert ds["id"].startswith("ds-")
    did = ds["id"]

    r = await client.get("/v1/datasets")
    assert r.status_code == 200
    assert did in [d["id"] for d in r.json()]

    r = await client.get(f"/v1/datasets/{did}")
    assert r.status_code == 200
    assert r.json()["id"] == did
    assert r.json()["name"] == ds["name"]


async def test_create_validation(client):
    # unsupported kind
    assert (await client.post("/v1/datasets", json={"name": "x", "kind": "bogus"})).status_code == 400
    # blank name
    assert (await client.post("/v1/datasets", json={"name": "  ", "kind": "hf", "hf_repo": "a/b"})).status_code == 400
    # kind=hf without hf_repo
    assert (await client.post("/v1/datasets", json={"name": _name("nohf"), "kind": "hf"})).status_code == 400
    # kind=upload without storage_id
    assert (await client.post("/v1/datasets", json={"name": _name("noup"), "kind": "upload"})).status_code == 400


async def test_update_columns_and_split_fields(client, cleanup):
    did = (await _create_hf(client, cleanup))["id"]
    r = await client.patch(
        f"/v1/datasets/{did}",
        json={
            "audio_field": "audio_filename",
            "transcription_field": "text",
            "split_fields": {"train": "text", "test": "after"},
        },
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["audio_field"] == "audio_filename"
    assert d["transcription_field"] == "text"
    assert d["split_fields"] == {"train": "text", "test": "after"}

    # {} clears the per-split overrides → null
    r = await client.patch(f"/v1/datasets/{did}", json={"split_fields": {}})
    assert r.status_code == 200
    assert r.json()["split_fields"] is None

    # blank keys/values dropped
    r = await client.patch(f"/v1/datasets/{did}", json={"split_fields": {"train": "text", " ": "x", "test": " "}})
    assert r.status_code == 200
    assert r.json()["split_fields"] == {"train": "text"}


async def test_rename(client, cleanup):
    did = (await _create_hf(client, cleanup))["id"]
    new = _name("renamed")
    r = await client.patch(f"/v1/datasets/{did}", json={"name": new})
    assert r.status_code == 200
    assert r.json()["name"] == new
    # blank name rejected
    assert (await client.patch(f"/v1/datasets/{did}", json={"name": "   "})).status_code == 400


async def test_delete(client):
    name = _name("tmp")
    did = (await client.post("/v1/datasets", json={"name": name, "kind": "hf", "hf_repo": "a/b"})).json()["id"]
    r = await client.delete(f"/v1/datasets/{did}")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert (await client.get(f"/v1/datasets/{did}")).status_code == 404


async def test_get_unknown_404(client):
    assert (await client.get("/v1/datasets/ds-nope")).status_code == 404
