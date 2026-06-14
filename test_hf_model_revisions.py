#!/usr/bin/env python3
"""Test the GPUPlatform self-hosted HuggingFace mirror's **multiple revisions** —
push the same repo at several named, overwriteable branches (`main`,
`checkpoint-v1`, …), then read each one back and confirm they're independent.

A repo created by pushing through the mirror is *versioned*: each branch is its
own current file set (push to `main` overwrites main; push to `checkpoint-v1`
overwrites that branch — no immutable commit history). Resolve a revision by
branch name OR commit sha.

Run:
    .venv/bin/python test_hf_revisions.py

Override the defaults with env vars:
    HF_ENDPOINT=https://your-gateway/hf \
    HF_TOKEN=sgpu_xxx \
    REPO=admin/my-rev-test \
    .venv/bin/python test_hf_revisions.py
"""
import os
import tempfile

# HF_ENDPOINT MUST be set before importing huggingface_hub — read once at import.
ENDPOINT = os.environ.setdefault("HF_ENDPOINT", "http://localhost:8080/hf")
# The mirror has no Xet/hf_transfer — force plain HTTP.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("SGPU_API_KEY")
    or "sgpu_8SoSHhJzCL9FGjwnwtib5BFo_4EswqXo3z3qkLodfis"  # your platform API key
)
REPO = os.environ.get("REPO", "admin/hf-revisions-test")
# Export the token so the module-level helpers (list_repo_refs / snapshot_download)
# authenticate too — private repos 401 without it.
os.environ["HF_TOKEN"] = TOKEN

from huggingface_hub import HfApi, list_repo_refs, snapshot_download  # noqa: E402

print(f"endpoint : {ENDPOINT}")
print(f"repo     : {REPO}\n")

api = HfApi(endpoint=ENDPOINT, token=TOKEN)

# Each revision is the SAME filenames with DIFFERENT contents — so reading a
# revision back must return exactly that revision's bytes (proves no collision).
REVISIONS = {
    "main":          {"config.json": '{"revision": "main"}',          "weights.txt": "WEIGHTS-main"},
    "checkpoint-v1": {"config.json": '{"revision": "checkpoint-v1"}', "weights.txt": "WEIGHTS-ckpt1"},
    "checkpoint-v2": {"config.json": '{"revision": "checkpoint-v2"}', "weights.txt": "WEIGHTS-ckpt2"},
}


def _folder(files: dict[str, str]) -> str:
    d = tempfile.mkdtemp()
    for fname, content in files.items():
        with open(os.path.join(d, fname), "w") as f:
            f.write(content)
    return d


# 1) push each revision (create the branch first for non-main)
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
for rev, files in REVISIONS.items():
    if rev != "main":
        api.create_branch(REPO, branch=rev, exist_ok=True)
    api.upload_folder(repo_id=REPO, folder_path=_folder(files), revision=rev,
                      commit_message=f"push {rev}")
    print(f"  ✓ pushed  {rev}  ({len(files)} files)")
print()

# 2) list the branches on the hub
refs = list_repo_refs(REPO)
names = sorted(b.name for b in refs.branches)
print(f"branches on the hub: {names}")
assert names == sorted(REVISIONS), f"expected {sorted(REVISIONS)}, got {names}"

# 3) read each revision back — by branch name AND by its commit sha
for b in refs.branches:
    rev, sha = b.name, b.target_commit
    for selector in (rev, sha):  # resolve by name, then by sha
        d = snapshot_download(REPO, revision=selector, local_dir=tempfile.mkdtemp())
        got = {f: open(os.path.join(d, f)).read() for f in REVISIONS[rev]}
        assert got == REVISIONS[rev], f"{rev} via {selector!r}: expected {REVISIONS[rev]}, got {got}"
    print(f"  {rev:14} sha={sha[:12]}  config.json={got['config.json']!r}  ✓ (by name + by sha)")

print("\nALL GOOD ✅  — each revision pushes + pulls back its own bytes, independently.")
print(f"\nView it at:  /models/{REPO}")
