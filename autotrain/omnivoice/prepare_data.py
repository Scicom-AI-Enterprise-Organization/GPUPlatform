#!/usr/bin/env python3
"""Prepare the Scicom-intl/TM-Voice dataset for OmniVoice finetuning.

The HF dataset ships audio as two zips that must be unzipped, plus a parquet of
metadata (columns: speaker, filename_audio, text). This script:

  1. Downloads english.zip + Mandarin_2026-06-12.zip + the parquet (HF token).
  2. Unzips the audio into <out_dir>/audio_raw/.
  3. Resolves each parquet `filename_audio` to an extracted .wav (exact path,
     then basename fallback). Unresolved rows are dropped with a warning.
  4. Splits TRAIN/TEST *per speaker*: hold out `--n_test` files per speaker
     (seeded), the rest is train.  (TM_English / TM_Mandarin -> 50 test each.)
  5. Emits OmniVoice-format manifests under <out_dir>:
       - train.jsonl       {id, audio_path, text, language_id}      (training)
       - dev.jsonl         {id, audio_path, text, language_id}      (=test set; eval-loss)
       - eval_test.jsonl   {id, text, ref_audio, ref_text, language_id}
            ref_audio/ref_text = a random SAME-SPEAKER train clip (voice prompt),
            for `omnivoice-infer-batch` zero-shot voice-cloning eval.

language_id is "en" for TM_English, "zh" for TM_Mandarin.
Run on the pod (the zips are ~3.6 GB).  HF_TOKEN must be in the env.
"""
import argparse
import json
import os
import re
import random
import zipfile

SPEAKER_LANG = {"TM_English": "en", "TM_Mandarin": "zh"}


def log(m):
    print(m, flush=True)


def cjk_count(s):
    """Number of CJK ideographs — used to pick the Hanzi text variant over pinyin."""
    return sum(1 for c in s if "一" <= c <= "鿿")


def sanitize_id(speaker, filename_audio):
    """Filesystem-safe unique id (used as the output wav name in infer_batch)."""
    stem = os.path.splitext(os.path.basename(filename_audio))[0]
    raw = f"{speaker}__{stem}"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", raw).strip("_")


def build_basename_index(root):
    """basename(lower) -> fullpath for every .wav under root (fallback resolver)."""
    idx = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                idx.setdefault(f.lower(), os.path.join(dirpath, f))
    return idx


def resolve_audio(raw_dir, basename_idx, filename_audio):
    exact = os.path.join(raw_dir, filename_audio)
    if os.path.isfile(exact):
        return exact
    hit = basename_idx.get(os.path.basename(filename_audio).lower())
    return hit  # may be None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Scicom-intl/TM-Voice")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--zips", nargs="+",
                    default=["english.zip", "Mandarin_2026-06-12.zip"])
    ap.add_argument("--parquet", default="data/train-00000-of-00001.parquet")
    ap.add_argument("--n_test", type=int, default=50, help="held-out test files PER speaker")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_download", action="store_true",
                    help="reuse already-downloaded/extracted files in out_dir")
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    os.makedirs(args.out_dir, exist_ok=True)
    raw_dir = os.path.join(args.out_dir, "audio_raw")
    os.makedirs(raw_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    # 1+2. Download + unzip audio
    if not args.skip_download:
        for z in args.zips:
            log(f"[prep] downloading {z} ...")
            zp = hf_hub_download(args.repo, z, repo_type="dataset", token=token,
                                 local_dir=os.path.join(args.out_dir, "dl"))
            log(f"[prep] unzipping {z} -> {raw_dir}")
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(raw_dir)
    pq = hf_hub_download(args.repo, args.parquet, repo_type="dataset", token=token,
                         local_dir=os.path.join(args.out_dir, "dl"))
    df = pd.read_parquet(pq)
    log(f"[prep] parquet rows={len(df)} speakers={dict(df['speaker'].value_counts())}")

    # 3. Resolve audio paths
    basename_idx = build_basename_index(raw_dir)
    log(f"[prep] indexed {len(basename_idx)} extracted .wav files")
    rows, missing = [], 0
    for _, r in df.iterrows():
        spk = str(r["speaker"])
        path = resolve_audio(raw_dir, basename_idx, str(r["filename_audio"]))
        if path is None:
            missing += 1
            continue
        rows.append({
            "id": sanitize_id(spk, str(r["filename_audio"])),
            "audio_path": os.path.abspath(path),
            "text": str(r["text"]).strip(),
            "language_id": SPEAKER_LANG.get(spk, "en"),
            "speaker": spk,
        })
    if missing:
        log(f"[prep] WARNING: {missing} rows had no resolvable audio (dropped)")

    # Dedup to ONE row per audio file. TM-Voice ships each Mandarin clip twice —
    # once as pinyin romanization, once as Hanzi — so we keep the variant with the
    # most CJK characters (the Hanzi one). This prevents the two variants of a file
    # landing on opposite sides of the split (audio leakage) and avoids training/
    # scoring on raw pinyin (use_pinyin_ratio=0; Whisper outputs Hanzi for zh).
    best = {}
    for x in rows:
        k = x["audio_path"]
        if k not in best or cjk_count(x["text"]) > cjk_count(best[k]["text"]):
            best[k] = x
    n_before = len(rows)
    rows = list(best.values())
    log(f"[prep] deduped {n_before} rows -> {len(rows)} unique audio files "
        f"(kept Hanzi over pinyin for zh)")

    # de-dup ids defensively (should already be unique after file-dedup)
    seen, uniq = set(), []
    for x in rows:
        if x["id"] in seen:
            x["id"] = f"{x['id']}_{len(uniq)}"
        seen.add(x["id"])
        uniq.append(x)
    rows = uniq

    # 4. Per-speaker split
    by_spk = {}
    for x in rows:
        by_spk.setdefault(x["speaker"], []).append(x)
    rng = random.Random(args.seed)
    train, test = [], []
    for spk, items in sorted(by_spk.items()):
        items = items[:]
        rng.shuffle(items)
        n = min(args.n_test, len(items) - 1)  # keep >=1 in train
        test.extend(items[:n])
        train.extend(items[n:])
        log(f"[prep] {spk}: total={len(items)} -> test={n} train={len(items)-n}")

    # 5a. train.jsonl + dev.jsonl (dev == test set, for eval-loss logging)
    def write_jsonl(path, items, keys):
        with open(path, "w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps({k: x[k] for k in keys}, ensure_ascii=False) + "\n")
        log(f"[prep] wrote {path} ({len(items)})")

    train_keys = ["id", "audio_path", "text", "language_id"]
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train, train_keys)
    write_jsonl(os.path.join(args.out_dir, "dev.jsonl"), test, train_keys)

    # 5b. eval_test.jsonl: each test item gets a random same-speaker TRAIN clip as ref
    train_by_spk = {}
    for x in train:
        train_by_spk.setdefault(x["speaker"], []).append(x)
    rng2 = random.Random(args.seed + 1)
    eval_rows = []
    for x in test:
        pool = train_by_spk.get(x["speaker"]) or [t for t in train]
        ref = rng2.choice(pool)
        eval_rows.append({
            "id": x["id"],
            "text": x["text"],
            "ref_audio": ref["audio_path"],
            "ref_text": ref["text"],
            "language_id": x["language_id"],
        })
    with open(os.path.join(args.out_dir, "eval_test.jsonl"), "w", encoding="utf-8") as f:
        for x in eval_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    log(f"[prep] wrote {os.path.join(args.out_dir, 'eval_test.jsonl')} ({len(eval_rows)})")

    log(f"[prep] DONE  train={len(train)} test/dev={len(test)} "
        f"(per-speaker test={args.n_test}, seed={args.seed})")


if __name__ == "__main__":
    main()
