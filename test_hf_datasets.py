#!/usr/bin/env python3
"""Test the GPUPlatform self-hosted HuggingFace mirror via the `datasets` library
— push multiple **subsets (configs)**, each with **train + test splits**, across
**multiple revisions (branches)**, then read every revision back independently.

A repo created by pushing through the mirror is `versioned=True`, so named
branches (`main`, `v2`, …) are independent + overwriteable. `push_to_hub`
auto-creates the branch (`create_branch(exist_ok=True)`) when `revision=` names
one that doesn't exist yet.

Run:
    .venv/bin/python test_hf_datasets.py

Override the defaults with env vars:
    HF_ENDPOINT=https://your-gateway/hf \
    HF_TOKEN=sgpu_xxx \
    REPO=admin/my-test \
    .venv/bin/python test_hf_datasets.py
"""
import os

# HF_ENDPOINT MUST be set before importing huggingface_hub / datasets — the
# endpoint is read once at import time.
ENDPOINT = os.environ.setdefault("HF_ENDPOINT", "http://localhost:8080/hf")
TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("SGPU_API_KEY")
    or "sgpu_8SoSHhJzCL9FGjwnwtib5BFo_4EswqXo3z3qkLodfis"  # your platform API key
)
REPO = os.environ.get("REPO", "admin/hf-subsets-test")

from datasets import Dataset, get_dataset_config_names, load_dataset  # noqa: E402

print(f"endpoint : {ENDPOINT}")
print(f"repo     : {REPO}\n")

# Two subsets (configs), each with a train + test split. The row text embeds the
# revision name so we can prove each branch reads back its OWN data, and the row
# counts differ per revision so an accidental cross-branch read is obvious.
SUBSETS = ("alpha", "beta")
SPLITS = ("train", "test")
# revision -> per-split row count (same for both subsets, keeps it simple).
REVISIONS = {
    "main": {"train": 5, "test": 2},
    "v2": {"train": 8, "test": 3},
}


def rows_for(rev: str, config: str, split: str, n: int):
    return [{"id": i, "text": f"{rev}/{config}/{split} {i}"} for i in range(n)]


# 1) push each revision × subset × split with `datasets`
for rev, counts in REVISIONS.items():
    for config in SUBSETS:
        for split in SPLITS:
            rows = rows_for(rev, config, split, counts[split])
            Dataset.from_list(rows).push_to_hub(
                REPO, config_name=config, split=split, revision=rev, token=TOKEN
            )
            print(f"  ✓ pushed  {rev:>4}  {config}/{split}  ({len(rows)} rows)")
    print()

# 2) read every revision back independently
for rev, counts in REVISIONS.items():
    configs = sorted(get_dataset_config_names(REPO, revision=rev, token=TOKEN))
    print(f"[{rev}] subsets on the hub: {configs}")
    assert configs == sorted(SUBSETS), f"[{rev}] expected {sorted(SUBSETS)}, got {configs}"
    for config in configs:
        ds = load_dataset(REPO, name=config, revision=rev, token=TOKEN)
        got = {s: ds[s].num_rows for s in ds}
        print(f"  {config}: splits={sorted(ds.keys())}  rows={got}")
        assert set(ds.keys()) == {"train", "test"}, (
            f"[{rev}] {config}: expected train+test, got {list(ds.keys())}"
        )
        for split in SPLITS:
            assert ds[split].num_rows == counts[split], (
                f"[{rev}] {config}/{split}: expected {counts[split]} rows, got {ds[split].num_rows}"
            )
            # prove the rows came from THIS revision, not another branch
            sample = ds[split][0]["text"]
            assert sample.startswith(f"{rev}/{config}/{split}"), (
                f"[{rev}] {config}/{split}: row leaked from another revision: {sample!r}"
            )
    print()

print("ALL GOOD ✅  — subsets + train/test splits push & load per-revision via the mirror.")
print(f"\nView it at:  /datasets/hosted/{REPO}  (use the revision selector to switch branches)")
