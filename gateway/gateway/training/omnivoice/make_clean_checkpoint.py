#!/usr/bin/env python3
"""Turn a training checkpoint into a clean, self-contained inference model dir.

OmniVoice `save_checkpoint` writes an HF-format model (config.json + safetensors
+ text tokenizer) *plus* Accelerate training state (optimizer.bin, scheduler.bin,
random_states_*.pkl) into checkpoint-{step}/.  For inference / HF upload we want
only the model + tokenizers, and we additionally **bundle the Higgs audio
tokenizer** under `audio_tokenizer/` so `OmniVoice.from_pretrained` loads fully
offline instead of re-downloading `eustlb/higgs-audio-v2-tokenizer`.

Usage:
    python make_clean_checkpoint.py --ckpt exp/.../checkpoint-123 --out exp/clean
"""
import argparse
import glob
import os
import shutil

# Accelerate training-state files to exclude from the clean copy.
_DROP_EXACT = {"optimizer.bin", "scheduler.bin", "scaler.pt", "train_config.json",
               "initial_config.json"}
_DROP_GLOBS = ["random_states_*.pkl", "rng_state*", "*.pkl"]


def _is_dropped(name):
    if name in _DROP_EXACT:
        return True
    for g in _DROP_GLOBS:
        if glob.fnmatch.fnmatch(name, g):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--higgs", default="eustlb/higgs-audio-v2-tokenizer")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = set(os.listdir(args.ckpt))
    have_safetensors = any(n.endswith(".safetensors") for n in names)

    for name in sorted(names):
        src = os.path.join(args.ckpt, name)
        if os.path.isdir(src):
            continue
        if _is_dropped(name):
            continue
        # If safetensors exist, skip a redundant pytorch_model.bin duplicate.
        if have_safetensors and name in ("pytorch_model.bin", "pytorch_model.bin.index.json"):
            continue
        shutil.copy2(src, os.path.join(args.out, name))
        print(f"[clean] copied {name}", flush=True)

    # Bundle the Higgs audio tokenizer (best-effort; from_pretrained can also
    # re-download it if this fails).
    dst = os.path.join(args.out, "audio_tokenizer")
    try:
        if os.path.isdir(os.path.join(args.ckpt, "audio_tokenizer")):
            shutil.copytree(os.path.join(args.ckpt, "audio_tokenizer"), dst, dirs_exist_ok=True)
        else:
            from huggingface_hub import snapshot_download
            snap = snapshot_download(args.higgs)
            shutil.copytree(snap, dst, dirs_exist_ok=True)
        print(f"[clean] bundled audio_tokenizer/ from {args.higgs}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[clean] WARNING: could not bundle audio_tokenizer ({e}); "
              f"from_pretrained will re-download it", flush=True)

    print(f"[clean] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
