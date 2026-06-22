#!/usr/bin/env python3
"""Score several wav dirs for CER (Whisper) + MOS (UTMOSv2), loading each model
ONCE. Used to find the lowest num_step that drops neither intelligibility (CER)
nor naturalness (MOS). Usage:
  score_dirs.py --test_list eval_test.jsonl --dirs s32=served_baseline s16=served_comp_s16 ...
"""
import argparse
import json
import os
from collections import defaultdict

_WL = {"en": "english", "zh": "chinese"}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_list", default="data/eval_test.jsonl")
    ap.add_argument("--dirs", nargs="+", required=True, help="label=dir ...")
    ap.add_argument("--asr_model", default="openai/whisper-large-v3")
    ap.add_argument("--out", default="opt/score_dirs.json")
    a = ap.parse_args()

    items = read_jsonl(a.test_list)
    dirs = [d.split("=", 1) for d in a.dirs]

    import librosa, torch
    from jiwer import cer
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    print("[score] loading Whisper ...", flush=True)
    m = AutoModelForSpeechSeq2Seq.from_pretrained(a.asr_model, torch_dtype=torch.float16).to("cuda")
    proc = AutoProcessor.from_pretrained(a.asr_model)
    pipe = pipeline("automatic-speech-recognition", model=m, tokenizer=proc.tokenizer,
                    feature_extractor=proc.feature_extractor, torch_dtype=torch.float16, device="cuda")

    def cer_of(d):
        per = defaultdict(list)
        for it in items:
            w = os.path.join(d, f"{it['id']}.wav")
            t = (it.get("text") or "").strip()
            if not t or not os.path.isfile(w):
                continue
            gk = {"task": "transcribe"}
            if it.get("language_id") in _WL:
                gk["language"] = _WL[it["language_id"]]
            try:
                audio, _ = librosa.load(w, sr=16000)
                r = pipe({"raw": audio, "sampling_rate": 16000}, return_timestamps=True, generate_kwargs=gk)
                per[it.get("language_id") or "?"].append(min(1.0, float(cer(t, r["text"].strip()))))
            except Exception as e:  # noqa: BLE001
                print(f"  cer skip {w}: {e}")
        allv = [x for v in per.values() for x in v]
        return (sum(allv) / len(allv) if allv else None), {k: sum(v)/len(v) for k, v in per.items()}

    cer_res = {label: cer_of(d) for label, d in dirs}
    del m, pipe; torch.cuda.empty_cache()

    print("[score] loading UTMOSv2 ...", flush=True)
    import utmosv2
    um = utmosv2.create_model(pretrained=True)

    def mos_of(d):
        vals = []
        for it in items:
            w = os.path.join(d, f"{it['id']}.wav")
            if not os.path.isfile(w):
                continue
            try:
                vals.append(float(um.predict(input_path=w, num_repetitions=3)))
            except Exception as e:  # noqa: BLE001
                print(f"  mos skip {w}: {e}")
        return sum(vals) / len(vals) if vals else None

    mos_res = {label: mos_of(d) for label, d in dirs}

    res = {label: {"cer": cer_res[label][0], "cer_per_lang": cer_res[label][1], "mos": mos_res[label]}
           for label, _ in dirs}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n{'config':>10} {'CER':>8} {'MOS':>8}")
    print("-" * 28)
    for label, _ in dirs:
        c, mo = res[label]["cer"], res[label]["mos"]
        print(f"{label:>10} {c:>8.4f} {mo:>8.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
