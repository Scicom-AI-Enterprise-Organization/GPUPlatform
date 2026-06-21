#!/usr/bin/env python3
"""Sequential, single-process audio tokenizer for OmniVoice finetuning.

Drop-in replacement for `omnivoice.scripts.extract_audio_tokens` that produces
the **identical** WebDataset shard layout the trainer reads
(`omnivoice.data.dataset.WebDatasetReader`/`SampleDecoder`), but WITHOUT the
ProcessPoolExecutor + 24-worker DataLoader machinery. That machinery deadlocks
on a single-GPU pod (27 spawn processes contend on the shared Manager queue /
CUDA init and never make progress). Here we load the Higgs codec once and encode
each clip on the GPU in a simple loop — fast enough for a few-thousand clips.

Output under <out_dir> (== TOKEN_DIR/<split>):
  audios/shard-%06d.tar   # webdataset: "{id}.npy" = int16 codes, shape (8, T)
  txts/shard-%06d.jsonl   # one JSON/line: {"id","text","language_id",...,"num_tokens"}
  data.lst                # "<tar_abspath> <jsonl_abspath> <count> <duration>"
"""
import argparse
import io
import json
import os

import numpy as np
import torch
import webdataset as wds
from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel

from omnivoice.utils.audio import load_audio

SR = 24_000  # HIGGS_INPUT_SAMPLE_RATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer_path", default="eustlb/higgs-audio-v2-tokenizer")
    ap.add_argument("--samples_per_shard", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out_dir, "audios"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "txts"), exist_ok=True)

    print(f"[tok] loading Higgs codec {a.tokenizer_path} on {a.device} ...", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(a.tokenizer_path)
    # Load on CPU then move to GPU (avoid device_map / accelerate dispatch).
    model = HiggsAudioV2TokenizerModel.from_pretrained(a.tokenizer_path).eval().to(a.device)
    print("[tok] codec loaded", flush=True)

    rows = [json.loads(l) for l in open(a.input_jsonl, encoding="utf-8") if l.strip()]
    print(f"[tok] {len(rows)} samples -> {a.out_dir}", flush=True)

    manifest = []
    state = {"idx": 0, "tar": None, "jf": None, "cnt": 0, "dur": 0.0}

    def open_shard():
        i = state["idx"]
        state["tar"] = wds.TarWriter(os.path.join(a.out_dir, "audios", f"shard-{i:06d}.tar"))
        state["jf"] = open(os.path.join(a.out_dir, "txts", f"shard-{i:06d}.jsonl"), "w", encoding="utf-8")
        state["cnt"] = 0
        state["dur"] = 0.0

    def close_shard():
        i = state["idx"]
        state["tar"].close()
        state["jf"].close()
        if state["cnt"] > 0:
            manifest.append((
                os.path.abspath(os.path.join(a.out_dir, "audios", f"shard-{i:06d}.tar")),
                os.path.abspath(os.path.join(a.out_dir, "txts", f"shard-{i:06d}.jsonl")),
                state["cnt"], state["dur"]))
        state["idx"] += 1

    open_shard()
    done = failed = 0
    for r in rows:
        path = r.get("audio_path")
        if not path or not os.path.exists(path):
            failed += 1
            continue
        try:
            wav = np.asarray(load_audio(path, SR)).squeeze()  # 1-D float32
            inputs = fe(raw_audio=wav, sampling_rate=SR, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                codes = model.encode(inputs["input_values"]).audio_codes.squeeze(0)  # (8, T)
            assert codes.ndim == 2 and codes.size(0) == 8, f"bad codes shape {tuple(codes.shape)}"
            arr = codes.to(torch.int16).cpu().numpy()
            buf = io.BytesIO()
            np.save(buf, arr)
            state["tar"].write({"__key__": r["id"], "npy": buf.getvalue()})
            rec = {k: v for k, v in r.items() if k != "speaker"}
            rec["num_tokens"] = int(arr.shape[1])
            state["jf"].write(json.dumps(rec, ensure_ascii=False) + "\n")
            state["cnt"] += 1
            state["dur"] += float(wav.shape[-1]) / SR
            done += 1
            if state["cnt"] >= a.samples_per_shard:
                close_shard()
                open_shard()
            if done % 100 == 0:
                print(f"[tok] {done}/{len(rows)}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[tok] FAIL {r.get('id')}: {e}", flush=True)

    close_shard()  # flush last (possibly empty) shard
    with open(os.path.join(a.out_dir, "data.lst"), "w", encoding="utf-8") as mf:
        for t, j, c, d in manifest:
            mf.write(f"{t} {j} {c} {d:.3f}\n")
    print(f"[tok] DONE done={done} failed={failed} shards={len(manifest)} "
          f"-> {os.path.join(a.out_dir, 'data.lst')}", flush=True)


if __name__ == "__main__":
    main()
