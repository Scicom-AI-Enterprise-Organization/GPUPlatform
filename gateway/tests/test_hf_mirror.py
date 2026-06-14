"""Integration tests for the self-hosted HuggingFace catalog / mirror.

Unlike the other tests in this dir (which go through the web proxy), these drive
the **gateway directly** plus the real `huggingface_hub` library and the `hf` CLI
against the gateway's `/hf` endpoint — exactly how a user would push/pull.

What's covered (model AND dataset):
- whoami-v2 + `hf auth login` / `hf auth whoami`
- push + pull via the Python library (`HfApi.upload_folder` / `snapshot_download`)
- push + pull via the `hf` CLI (`hf upload` / `hf download`)
- create-repo via the library (auto-create)
- LFS (>10 MB) files round-trip + are flagged `lfs` in the manifest
- delete model + delete dataset (via `HfApi.delete_repo`)

Setup: a temp **local** storage is registered through the gateway to host the
repos; `HF_HOME` is redirected to a temp dir so `hf auth login` never touches the
real `~/.cache/huggingface`. Everything is cleaned up on teardown.

Run:  GATEWAY=http://localhost:8080 SGPU_API_KEY=sgpu_… .venv/bin/pytest gateway/tests/test_hf_mirror.py
The suite skips entirely if the gateway is unreachable or the key lacks catalog
access.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import subprocess
import sys
import tempfile

import httpx
import pytest

GATEWAY = os.environ.get("GATEWAY", "http://localhost:8080").rstrip("/")
# Override with SGPU_API_KEY; the default is the key provided for this project.
API_KEY = os.environ.get("SGPU_API_KEY", "sgpu_8SoSHhJzCL9FGjwnwtib5BFo_4EswqXo3z3qkLodfis")
HF_ENDPOINT = f"{GATEWAY}/hf"

# Resolve the `hf` CLI from the active venv first, then PATH.
_HF = os.path.join(os.path.dirname(sys.executable), "hf")
if not os.path.exists(_HF):
    _HF = shutil.which("hf") or "hf"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------- low-level helpers -------------------------------------------


def _gw() -> httpx.Client:
    return httpx.Client(
        base_url=GATEWAY,
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=120.0,
    )


def _api():
    from huggingface_hub import HfApi

    return HfApi(endpoint=HF_ENDPOINT, token=API_KEY)


def _make_files(d: str, *, with_lfs: bool = True) -> None:
    """Write a small repo: regular files (+ a nested one) and one >10 MB LFS file."""
    with open(os.path.join(d, "config.json"), "w") as f:
        f.write('{"model_type":"test","hidden":8}')
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write("---\nlicense: mit\n---\n# test repo\n")
    os.makedirs(os.path.join(d, "sub"), exist_ok=True)
    with open(os.path.join(d, "sub", "tokenizer.json"), "w") as f:
        f.write('{"vocab":["a","b","c"]}')
    if with_lfs:
        with open(os.path.join(d, "weights.bin"), "wb") as f:
            f.write(os.urandom(11 * 1024 * 1024))  # >10 MB → LFS


def _digests(d: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for root, _dirs, files in os.walk(d):
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, d)
            if rel.startswith(".cache"):  # snapshot_download bookkeeping
                continue
            with open(p, "rb") as fh:
                out[rel] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _cli_env(hf_home: str, *, with_token: bool = True) -> dict:
    env = dict(os.environ)
    env["HF_ENDPOINT"] = HF_ENDPOINT
    env["HF_HOME"] = hf_home
    env.pop("HF_TOKEN", None)
    if with_token:
        env["HF_TOKEN"] = API_KEY
    return env


# ---------- fixtures ----------------------------------------------------


@pytest.fixture(scope="session")
def _check():
    """Skip the whole suite unless the gateway is up + the key has catalog access."""
    try:
        with _gw() as c:
            r = c.get("/auth/me")
    except httpx.HTTPError as e:
        pytest.skip(f"gateway not reachable at {GATEWAY} ({e}); set GATEWAY/SGPU_API_KEY")
    if r.status_code != 200:
        pytest.skip(f"API key invalid ({r.status_code}); set SGPU_API_KEY to a valid sgpu_ key")
    if not (r.json().get("sections") or {}).get("catalog"):
        pytest.skip("API key lacks 'catalog' section access")


@pytest.fixture(scope="session")
def hf_home():
    """Isolated HF_HOME so `hf auth login` never clobbers the real token store."""
    d = tempfile.mkdtemp(prefix="sgpu-hf-home-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def storage(_check):
    """A temp local storage registered via the gateway to host test repos."""
    d = tempfile.mkdtemp(prefix="sgpu-hf-store-")
    with _gw() as c:
        r = c.post(
            "/v1/storage",
            json={"name": f"pytest-hf-{secrets.token_hex(4)}", "kind": "local", "path": d},
        )
        assert r.status_code == 200, r.text
        sid = r.json()["id"]
    yield sid
    with _gw() as c:
        c.delete(f"/v1/storage/{sid}")
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def repo_factory(storage):
    """Register repos on the test storage; deletes every created repo on teardown.
    Returns make(repo_type, name=None) -> "ns/name"."""
    created: list[str] = []  # repo ids

    def make(repo_type: str, name: str | None = None) -> str:
        name = name or f"r{secrets.token_hex(4)}"
        with _gw() as c:
            r = c.post(
                "/v1/catalog",
                json={
                    "repo_type": repo_type,
                    "namespace": "pytest",
                    "name": name,
                    "storage_id": storage,
                    "private": True,
                },
            )
            assert r.status_code == 200, r.text
            created.append(r.json()["id"])
        return f"pytest/{name}"

    yield make

    with _gw() as c:
        for rid in created:
            try:
                c.delete(f"/v1/catalog/{rid}?wipe=true")
            except httpx.HTTPError:
                pass


def _repo_record(full_id: str, repo_type: str) -> dict | None:
    """The list-endpoint row for a repo (no file manifest), or None if absent."""
    with _gw() as c:
        rows = c.get(f"/v1/catalog?scope=mine&repo_type={repo_type}").json()
    return next((r for r in rows if r["full_id"] == full_id), None)


def _repo_detail(repo_id: str) -> dict:
    """The detail record (includes the `files` manifest)."""
    with _gw() as c:
        r = c.get(f"/v1/catalog/{repo_id}")
        r.raise_for_status()
        return r.json()


# ---------- whoami / auth ----------------------------------------------


def test_whoami_v2(_check):
    with _gw() as c:
        r = c.get("/hf/api/whoami-v2")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"]  # username
    # `hf auth login` needs both of these or it KeyErrors.
    assert body["auth"]["accessToken"]["role"]
    assert body["auth"]["accessToken"]["displayName"]


def test_auth_login_and_whoami_cli(_check, hf_home):
    env = _cli_env(hf_home, with_token=False)
    r = subprocess.run([_HF, "auth", "login", "--token", API_KEY],
                       env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert "Login successful" in out, out
    w = subprocess.run([_HF, "auth", "whoami"], env=env, capture_output=True, text=True)
    assert w.returncode == 0 and "admin" in (w.stdout + w.stderr), w.stdout + w.stderr


# ---------- python library: push / pull --------------------------------


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
def test_push_pull_library(repo_factory, repo_type):
    repo_id = repo_factory(repo_type)
    api = _api()
    src = tempfile.mkdtemp(prefix="lib-up-")
    dst = tempfile.mkdtemp(prefix="lib-dl-")
    try:
        _make_files(src)
        before = _digests(src)
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=True)
        api.upload_folder(folder_path=src, repo_id=repo_id, repo_type=repo_type)

        from huggingface_hub import snapshot_download

        snapshot_download(repo_id, repo_type=repo_type, endpoint=HF_ENDPOINT,
                          token=API_KEY, local_dir=dst)
        assert _digests(dst) == before, "round-trip mismatch (library)"

        # the >10 MB file is stored + flagged as LFS in the manifest
        rec = _repo_record(repo_id, repo_type)
        assert rec is not None
        detail = _repo_detail(rec["id"])
        wb = next((f for f in (detail["files"] or []) if f["path"] == "weights.bin"), None)
        assert wb is not None and wb["lfs"] is True
    finally:
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(dst, ignore_errors=True)


# ---------- hf CLI: push / pull -----------------------------------------


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
def test_push_pull_cli(repo_factory, hf_home, repo_type):
    repo_id = repo_factory(repo_type)
    env = _cli_env(hf_home)
    src = tempfile.mkdtemp(prefix="cli-up-")
    dst = tempfile.mkdtemp(prefix="cli-dl-")
    try:
        _make_files(src)
        before = _digests(src)
        up = subprocess.run(
            [_HF, "upload", repo_id, src, "--repo-type", repo_type],
            env=env, capture_output=True, text=True,
        )
        assert up.returncode == 0, up.stdout + up.stderr
        dl = subprocess.run(
            [_HF, "download", repo_id, "--repo-type", repo_type, "--local-dir", dst],
            env=env, capture_output=True, text=True,
        )
        assert dl.returncode == 0, dl.stdout + dl.stderr
        assert _digests(dst) == before, "round-trip mismatch (CLI)"
    finally:
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(dst, ignore_errors=True)


# ---------- create repo via the HF library ------------------------------


def test_create_repo_via_library(_check):
    """create_repo with no prior /v1/catalog registration auto-creates the repo
    on the caller's default storage."""
    api = _api()
    name = f"auto{secrets.token_hex(4)}"
    repo_id = f"pytest/{name}"
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    try:
        rec = _repo_record(repo_id, "model")
        assert rec is not None, "create_repo did not register the repo"
        assert rec["repo_type"] == "model"
    finally:
        # always clean up — this repo lands on the default storage, not the
        # test fixture's, so repo_factory won't catch it.
        leftover = _repo_record(repo_id, "model")
        if leftover:
            with _gw() as c:
                c.delete(f"/v1/catalog/{leftover['id']}?wipe=true")


# ---------- delete model + dataset (via the HF library) -----------------


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
def test_delete_via_library(repo_factory, repo_type):
    repo_id = repo_factory(repo_type)
    api = _api()
    # put a file there so there's something to wipe
    src = tempfile.mkdtemp(prefix="del-")
    try:
        with open(os.path.join(src, "config.json"), "w") as f:
            f.write("{}")
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
        api.upload_folder(folder_path=src, repo_id=repo_id, repo_type=repo_type)
        assert _repo_record(repo_id, repo_type) is not None
        # delete via the HF API (POST /hf/api/repos/delete)
        api.delete_repo(repo_id, repo_type=repo_type)
        assert _repo_record(repo_id, repo_type) is None, "repo still present after delete_repo"
    finally:
        shutil.rmtree(src, ignore_errors=True)


# ---------- access control ----------------------------------------------


def test_pull_requires_auth(repo_factory):
    """A private repo can't be read without a valid token (anonymous → 401)."""
    repo_id = repo_factory("model")
    _api().create_repo(repo_id, repo_type="model", exist_ok=True)
    # no Authorization header
    r = httpx.get(f"{HF_ENDPOINT}/api/models/{repo_id}", timeout=30.0)
    assert r.status_code in (401, 404), r.status_code
