#!/usr/bin/env python3
"""Test the GPUPlatform self-hosted HuggingFace mirror via the `datasets` library
— push multiple **subsets (configs)**, each with **train + test splits**, then
read them back.

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

# Two subsets (configs), each with a train + test split.
SUBSETS = {
    "alpha": {
        "train": [{"id": i, "text": f"alpha train {i}"} for i in range(5)],
        "test": [{"id": i, "text": f"alpha test {i}"} for i in range(2)],
    },
    "beta": {
        "train": [{"id": i, "text": f"beta train {i}"} for i in range(4)],
        "test": [{"id": i, "text": f"beta test {i}"} for i in range(3)],
    },
}

# 1) push each subset+split with `datasets`
for config, splits in SUBSETS.items():
    for split, rows in splits.items():
        Dataset.from_list(rows).push_to_hub(REPO, config_name=config, split=split, token=TOKEN)
        print(f"  ✓ pushed  {config}/{split}  ({len(rows)} rows)")
print()

# 2) read them back
configs = sorted(get_dataset_config_names(REPO, token=TOKEN))
print(f"subsets on the hub: {configs}")
assert configs == sorted(SUBSETS), f"expected {sorted(SUBSETS)}, got {configs}"
for config in configs:
    ds = load_dataset(REPO, name=config, token=TOKEN)
    print(f"  {config}: splits={sorted(ds.keys())}  "
          f"rows={ {s: ds[s].num_rows for s in ds} }")
    assert set(ds.keys()) == {"train", "test"}, f"{config}: expected train+test, got {list(ds.keys())}"

print("\nALL GOOD ✅  — subsets + train/test splits push & load via the mirror.")
print(f"\nView it at:  /datasets/hosted/{REPO}")
