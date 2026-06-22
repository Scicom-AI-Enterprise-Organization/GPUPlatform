#!/usr/bin/env python3
"""A/B: base OmniVoice vs the TM-Voice finetune, on the SAME test items.

Answers "did the finetune actually change the audio toward the TM speakers?"
For each held-out test item we have base-generated and finetune-generated wavs
(same text + same reference prompt). We score, for BOTH:

  CER     — Whisper-large-v3 + jiwer.cer (per-language)         [intelligibility]
  sim_ref — speaker-embedding cosine(gen, reference prompt)      [clones the prompt voice]
  sim_gt  — speaker-embedding cosine(gen, ground-truth TM clip)  [matches the real target speaker]

Speaker embeddings: SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb), the standard
zero-shot-TTS speaker-similarity model. Prints a base-vs-finetune table + JSON.

Inputs:
  --base_dir / --ft_dir : {id}.wav dirs from `omnivoice-infer-batch` (base / finetune)
  --test_list           : eval_test.jsonl  (id, text, ref_audio, language_id)
  --gt_list             : dev.jsonl         (id -> audio_path = ground-truth TM clip)
"""
import argparse
import json
import os
from collections import defaultdict


def log(m):
    print(m, flush=True)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


_WL = {"en": "english", "zh": "chinese"}


def score_cer(items, wav_dir, pipe):
    import librosa
    from jiwer import cer
    out = {}
    for it in items:
        wav = os.path.join(wav_dir, f"{it['id']}.wav")
        text = (it.get("text") or "").strip()
        if not text or not os.path.isfile(wav):
            continue
        gen_kw = {"task": "transcribe"}
        if it.get("language_id") in _WL:
            gen_kw["language"] = _WL[it["language_id"]]
        try:
            audio, _ = librosa.load(wav, sr=16000)
            r = pipe({"raw": audio, "sampling_rate": 16000}, return_timestamps=True,
                     generate_kwargs=gen_kw)
            out[it["id"]] = min(1.0, float(cer(text, r["text"].strip())))
        except Exception as e:  # noqa: BLE001
            log(f"[ab] CER skip {wav}: {e}")
    return out


def build_embedder():
    import torch
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:  # noqa: BLE001
        from speechbrain.pretrained import EncoderClassifier
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir="/root/.cache/sb-ecapa",
                                         run_opts={"device": dev})
    import librosa
    cache = {}

    def emb(path):
        if path in cache:
            return cache[path]
        wav, _ = librosa.load(path, sr=16000)
        t = torch.tensor(wav).unsqueeze(0)
        with torch.no_grad():
            e = clf.encode_batch(t).squeeze().detach().cpu()
        cache[path] = e
        return e
    return emb


def cos(a, b):
    import torch
    return float(torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0),
                                                        b.flatten().unsqueeze(0))[0])


def score_sim(items, wav_dir, id2ref, id2gt, emb):
    sim_ref, sim_gt = {}, {}
    for it in items:
        wav = os.path.join(wav_dir, f"{it['id']}.wav")
        if not os.path.isfile(wav):
            continue
        try:
            e = emb(wav)
            if id2ref.get(it["id"]) and os.path.isfile(id2ref[it["id"]]):
                sim_ref[it["id"]] = cos(e, emb(id2ref[it["id"]]))
            if id2gt.get(it["id"]) and os.path.isfile(id2gt[it["id"]]):
                sim_gt[it["id"]] = cos(e, emb(id2gt[it["id"]]))
        except Exception as ex:  # noqa: BLE001
            log(f"[ab] sim skip {wav}: {ex}")
    return sim_ref, sim_gt


def agg(d, items):
    """overall + per-language mean over ids present in d."""
    lang = {it["id"]: it.get("language_id", "?") for it in items}
    per = defaultdict(list)
    allv = []
    for i, v in d.items():
        allv.append(v)
        per[lang.get(i, "?")].append(v)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {"all": mean(allv), **{k: mean(v) for k, v in sorted(per.items())}, "n": len(allv)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--ft_dir", required=True)
    ap.add_argument("--test_list", required=True)
    ap.add_argument("--gt_list", required=True)
    ap.add_argument("--asr_model", default="openai/whisper-large-v3")
    ap.add_argument("--out", default="ab_results.json")
    a = ap.parse_args()

    items = read_jsonl(a.test_list)
    id2ref = {x["id"]: x.get("ref_audio") for x in items}
    id2gt = {x["id"]: x.get("audio_path") for x in read_jsonl(a.gt_list)}

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    log(f"[ab] loading ASR {a.asr_model} ...")
    m = AutoModelForSpeechSeq2Seq.from_pretrained(a.asr_model, torch_dtype=torch.float16).to("cuda")
    proc = AutoProcessor.from_pretrained(a.asr_model)
    pipe = pipeline("automatic-speech-recognition", model=m, tokenizer=proc.tokenizer,
                    feature_extractor=proc.feature_extractor, torch_dtype=torch.float16, device="cuda")
    log("[ab] CER base ..."); cer_b = score_cer(items, a.base_dir, pipe)
    log("[ab] CER ft ...");   cer_f = score_cer(items, a.ft_dir, pipe)
    del m, pipe; torch.cuda.empty_cache()

    log("[ab] loading speaker embedder (ECAPA) ...")
    emb = build_embedder()
    log("[ab] sim base ..."); ref_b, gt_b = score_sim(items, a.base_dir, id2ref, id2gt, emb)
    log("[ab] sim ft ...");   ref_f, gt_f = score_sim(items, a.ft_dir, id2ref, id2gt, emb)

    res = {
        "n_items": len(items),
        "cer":     {"base": agg(cer_b, items), "finetune": agg(cer_f, items)},
        "sim_ref": {"base": agg(ref_b, items), "finetune": agg(ref_f, items)},
        "sim_gt":  {"base": agg(gt_b, items),  "finetune": agg(gt_f, items)},
    }
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)

    def row(name, metric, lo_is_better=False):
        b = res[metric]["base"]["all"]; f = res[metric]["finetune"]["all"]
        arrow = "↓" if lo_is_better else "↑"
        delta = (f - b) if (b is not None and f is not None) else None
        log(f"  {name:22s}({arrow})  base={b:.4f}  finetune={f:.4f}  Δ={delta:+.4f}")
    log("\n===== BASE vs FINETUNE (overall) =====")
    row("CER", "cer", lo_is_better=True)
    row("spk-sim to reference", "sim_ref")
    row("spk-sim to groundtruth", "sim_gt")
    log("\nfull (per-language) -> " + a.out)
    log(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
