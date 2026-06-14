#!/usr/bin/env python3
"""Push a SIMPLE dataset to the GPUPlatform mirror at multiple revisions (branches).

One flat dataset (a list of rows) per branch — no subsets, no splits. Each branch
is its own overwriteable snapshot; pull any of them back with `revision=`.

Run:
    .venv/bin/python test_hf_dataset_revisions.py

Override the defaults with env vars:
    HF_ENDPOINT=https://your-gateway/hf  HF_TOKEN=sgpu_xxx  REPO=admin/my-rev-ds \
    .venv/bin/python test_hf_dataset_revisions.py
"""
import os

# HF_ENDPOINT MUST be set before importing huggingface_hub / datasets.
ENDPOINT = os.environ.setdefault("HF_ENDPOINT", "http://localhost:8080/hf")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # mirror has no Xet
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("SGPU_API_KEY")
    or "sgpu_8SoSHhJzCL9FGjwnwtib5BFo_4EswqXo3z3qkLodfis"  # your platform API key
)
REPO = os.environ.get("REPO", "admin/hf-dataset-revisions-test")
os.environ["HF_TOKEN"] = TOKEN  # so the module-level helpers authenticate too

from datasets import Dataset, load_dataset          # noqa: E402
from huggingface_hub import HfApi, list_repo_refs    # noqa: E402

print(f"endpoint : {ENDPOINT}")
print(f"repo     : {REPO}\n")

api = HfApi(endpoint=ENDPOINT, token=TOKEN)

# One simple flat dataset per revision (same shape, different rows).
REVISIONS = {
    "main":          [{"id": i, "text": f"main row {i}"} for i in range(5)],
    "checkpoint-v1": [{"id": i, "text": f"v1 row {i}"} for i in range(3)],
    "checkpoint-v2": [{"id": i, "text": f"v2 row {i}"} for i in range(4)],
}

# 1) push each revision (branch first for non-main)
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
for rev, rows in REVISIONS.items():
    if rev != "main":
        api.create_branch(REPO, repo_type="dataset", branch=rev, exist_ok=True)
    Dataset.from_list(rows).push_to_hub(REPO, revision=rev, commit_message=f"push {rev}")
    print(f"  ✓ pushed  {rev}  ({len(rows)} rows)")
print()

# 2) list the branches
refs = list_repo_refs(REPO, repo_type="dataset")
names = sorted(b.name for b in refs.branches)
print(f"branches on the hub: {names}")
assert names == sorted(REVISIONS), f"expected {sorted(REVISIONS)}, got {names}"

# 3) read each revision back and confirm it's that revision's rows
for rev, rows in REVISIONS.items():
    ds = load_dataset(REPO, revision=rev, split="train", token=TOKEN)
    print(f"  {rev:14} rows={ds.num_rows}  first={ds[0]}")
    assert ds.num_rows == len(rows), f"{rev}: expected {len(rows)} rows, got {ds.num_rows}"
    assert ds[0]["text"] == rows[0]["text"], f"{rev}: wrong rows ({ds[0]})"

print("\nALL GOOD ✅  — a simple dataset pushes + loads back per revision, independently.")
print(f"\nView it at:  /datasets/hosted/{REPO}")
