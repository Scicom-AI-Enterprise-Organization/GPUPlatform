#!/usr/bin/env python3
"""CER + MOS evaluation for OmniVoice-generated speech.

Mirrors the gateway TTS eval methodology
(`gateway/gateway/training/tts/tts_eval.py`):

  CER  — Whisper-large-v3 transcribes each generated wav; char error rate vs the
         reference text via `jiwer.cer`, capped at 1.0.  Whisper language is set
         per-sample from `language_id` (en/zh).  (jiwer.cer is character-level,
         which is the right unit for both English and Mandarin.)
  MOS  — predicted naturalness via UTMOSv2 (`utmosv2.create_model(pretrained=True)
         .predict(input_path=..., num_repetitions=5)`), same as the gateway.

Inputs: a directory of generated wavs named `{id}.wav` (the `omnivoice-infer-batch
--res_dir` output) + the eval_test.jsonl that produced them (fields id, text,
language_id).  Writes overall + per-language CER/MOS to --out.

Run in an env with: transformers, torch, librosa, jiwer, soundfile, utmosv2.
"""
import argparse
import json
import os
from collections import defaultdict


def log(m):
    print(m, flush=True)


def read_test_list(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_WHISPER_LANG = {"en": "english", "zh": "chinese"}


def score_cer(items, wav_dir, asr_model, extension="wav"):
    import librosa
    import torch
    from jiwer import cer
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    log(f"[eval] CER: loading ASR {asr_model} ...")
    m = AutoModelForSpeechSeq2Seq.from_pretrained(asr_model, torch_dtype=torch.float16).to("cuda")
    proc = AutoProcessor.from_pretrained(asr_model)
    pipe = pipeline("automatic-speech-recognition", model=m, tokenizer=proc.tokenizer,
                    feature_extractor=proc.feature_extractor, torch_dtype=torch.float16, device="cuda")

    per_lang = defaultdict(list)
    per_item = []
    for it in items:
        text = (it.get("text") or "").strip()
        if not text:
            continue
        wav = os.path.join(wav_dir, f"{it['id']}.{extension}")
        if not os.path.isfile(wav):
            log(f"[eval] CER: missing wav {wav}; skip")
            continue
        lang = it.get("language_id")
        gen_kw = {"task": "transcribe"}
        if lang in _WHISPER_LANG:
            gen_kw["language"] = _WHISPER_LANG[lang]
        try:
            audio, _ = librosa.load(wav, sr=16000)
            out = pipe({"raw": audio, "sampling_rate": 16000},
                       return_timestamps=True, generate_kwargs=gen_kw)
            s = min(1.0, float(cer(text, out["text"].strip())))
            per_lang[lang or "?"].append(s)
            per_item.append({"id": it["id"], "language_id": lang, "cer": s,
                             "hyp": out["text"].strip()})
        except Exception as e:  # noqa: BLE001
            log(f"[eval] CER skip {wav}: {e}")
    return per_lang, per_item


def score_mos(items, wav_dir, extension="wav"):
    import utmosv2

    log("[eval] MOS: loading UTMOSv2 ...")
    model = utmosv2.create_model(pretrained=True)
    per_lang = defaultdict(list)
    per_item = {}
    for it in items:
        wav = os.path.join(wav_dir, f"{it['id']}.{extension}")
        if not os.path.isfile(wav):
            continue
        try:
            mos = float(model.predict(input_path=wav, num_repetitions=5))
            per_lang[it.get("language_id") or "?"].append(mos)
            per_item[it["id"]] = mos
        except Exception as e:  # noqa: BLE001
            log(f"[eval] MOS skip {wav}: {e}")
    return per_lang, per_item


def _avg(xs):
    return (sum(xs) / len(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True, help="dir of generated {id}.wav (infer_batch --res_dir)")
    ap.add_argument("--test_list", required=True, help="eval_test.jsonl (id, text, language_id)")
    ap.add_argument("--methods", default="cer,mos")
    ap.add_argument("--asr_model", default=os.environ.get("EVAL_ASR_MODEL", "openai/whisper-large-v3"))
    ap.add_argument("--extension", default="wav")
    ap.add_argument("--out", default="eval_results.json")
    a = ap.parse_args()

    items = read_test_list(a.test_list)
    methods = [m.strip() for m in a.methods.split(",") if m.strip() in ("cer", "mos")]
    results = {"num_test": len(items), "methods": methods}

    if "cer" in methods:
        cer_lang, cer_items = score_cer(items, a.wav_dir, a.asr_model, a.extension)
        results["cer"] = _avg([s for v in cer_lang.values() for s in v])
        results["cer_per_language"] = {k: _avg(v) for k, v in sorted(cer_lang.items())}
        results["cer_items"] = cer_items
    if "mos" in methods:
        mos_lang, mos_items = score_mos(items, a.wav_dir, a.extension)
        results["mos"] = _avg([s for v in mos_lang.values() for s in v])
        results["mos_per_language"] = {k: _avg(v) for k, v in sorted(mos_lang.items())}
        results["mos_items"] = mos_items

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary = {k: results[k] for k in ("num_test", "cer", "cer_per_language",
                                       "mos", "mos_per_language") if k in results}
    log(f"[eval] DONE -> {a.out}\n{json.dumps(summary, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
