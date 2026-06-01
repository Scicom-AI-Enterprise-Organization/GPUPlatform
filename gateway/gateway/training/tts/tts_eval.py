"""TTS evaluation — synthesize the test set with the finetuned Qwen3+NeuCodec
model, then score the generated audio with any combination of:

  cer         — ASR the generated audio (Whisper), char error rate vs. reference text
  mos         — predicted naturalness MOS via UTMOSv2 (faster-UTMOSv2)
  similarity  — TitaNet speaker-embedding cosine vs. the reference (ground-truth) audio

Scorer APIs mirror the Scicom Multilingual-TTS eval scripts verbatim:
  - calculate_cer.py        → AutoModelForSpeechSeq2Seq + jiwer.cer (capped at 1)
  - calculate_mos.py        → utmosv2.create_model(pretrained=True).predict(input_path=…, num_repetitions=5)
  - calculate_similarity.py → titanet_vectors.load() → model(a, lens)[1] → sklearn cosine_similarity

Input is a pre-packed ChiniDataset (kind=tts_packed): each record is one or more
multipacked utterances `<|im_start|>speaker: TEXT<|speech_start|><|s_..|>…<|im_end|>`.
For each utterance we take the prompt up to `<|speech_start|>`, let the model
generate the speech tokens, NeuCodec-decode them → generated wav; the reference
speech tokens (already in the record) decode → reference wav; the text is the
reference transcript. Emits one `@@METRIC {json}` line per method + `@@DONE`.

Run in the TTS venv (built by --deps-only). Heavy deps (utmosv2/titanet/whisper)
are imported lazily so only the selected methods load their model.
"""
import argparse
import json
import os
import re
import sys

# Match the special tokens the packer wrote (see pack_stage1.py).
_SPEECH_START = "<|speech_start|>"
_IM_END = "<|im_end|>"
_IM_START = "<|im_start|>"
_SPEECH_TOK = re.compile(r"<\|s_(\d+)\|>")


def log(m):
    print(m, flush=True)


def emit(tag, obj):
    print(f"@@{tag} {json.dumps(obj)}", flush=True)


# ---------------------------------------------------------------------------
# Read packed records → per-utterance (prompt_ids, ref_speech_codes, ref_text)
# ---------------------------------------------------------------------------
def _read_packed(local_dir):
    """Yield each multipacked record's input_ids (the chinidataset shards)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chinidataset import StreamingDataset

    ds = StreamingDataset(local=local_dir)
    for i in range(len(ds)):
        rec = dict(ds[i])
        ids = rec.get("input_ids")
        mask = rec.get("attention_mask")  # per-utterance lengths
        if ids is None:
            continue
        ids = [int(x) for x in list(ids)]
        mask = [int(x) for x in list(mask or [len(ids)])]
        pos = 0
        for length in mask:
            yield ids[pos:pos + length]
            pos += length


def _utterance_parts(tokenizer, utt_ids):
    """Decode one utterance → (prompt_text incl. <|speech_start|>, ref_speech_codes, ref_text).
    Returns None if it isn't a well-formed `…TEXT<|speech_start|>…<|s_N|>…` utterance."""
    text = tokenizer.decode(utt_ids, skip_special_tokens=False)
    if _SPEECH_START not in text:
        return None
    left, right = text.split(_SPEECH_START, 1)
    right = right.split(_IM_END, 1)[0]
    ref_codes = [int(n) for n in _SPEECH_TOK.findall(right)]
    if not ref_codes:
        return None
    # reference transcript = the text after "speaker:" in the prompt half.
    ref_text = left.replace(_IM_START, "").strip()
    if ":" in ref_text:
        ref_text = ref_text.split(":", 1)[1].strip()
    prompt_text = left + _SPEECH_START
    return prompt_text, ref_codes, ref_text


# ---------------------------------------------------------------------------
# Generation: prompt → speech tokens → NeuCodec wav
# ---------------------------------------------------------------------------
def _decode_neucodec(neucodec, codes, sr_out=24000):
    """NeuCodec codes (list[int]) → mono waveform (numpy float32) + sample rate.
    Inverse of convert_neucodec's `encode_code`/`fsq_codes[0,0]`."""
    import torch

    with torch.no_grad():
        fsq = torch.tensor(codes, dtype=torch.long).reshape(1, 1, -1).to(neucodec.device)
        wav = neucodec.decode_code(fsq)  # [1, 1, T]
    return wav.squeeze().detach().cpu().float().numpy(), getattr(neucodec, "sample_rate", sr_out)


def generate_pairs(model_dir, packed_dir, out_dir, max_samples, max_new_tokens=2048):
    """Synthesize each eval utterance + materialise gen/ref wavs. Returns a list
    of {gen, ref, text} (paths + reference transcript)."""
    import numpy as np  # noqa: F401
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer
    from neucodec import NeuCodec

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qwen3_tts_flash import Model  # the trainer's Qwen3 wrapper

    os.makedirs(out_dir, exist_ok=True)
    log(f"[eval] loading model {model_dir} + tokenizer + NeuCodec …")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = Model.from_pretrained(model_dir, torch_dtype=torch.bfloat16).cuda().eval()
    neucodec = NeuCodec.from_pretrained("neuphonic/neucodec").eval()
    neucodec = neucodec.cuda() if torch.cuda.is_available() else neucodec
    im_end_id = tokenizer.convert_tokens_to_ids(_IM_END)

    pairs = []
    for idx, utt in enumerate(_read_packed(packed_dir)):
        if len(pairs) >= max_samples:
            break
        parts = _utterance_parts(tokenizer, utt)
        if parts is None:
            continue
        prompt_text, ref_codes, ref_text = parts
        try:
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                out = model.generate(
                    prompt_ids, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=0.8, top_p=0.95, eos_token_id=im_end_id,
                )
            gen_ids = out[0, prompt_ids.shape[-1]:].tolist()
            gen_codes = [int(n) for n in _SPEECH_TOK.findall(tokenizer.decode(gen_ids, skip_special_tokens=False))]
            if not gen_codes:
                log(f"[eval] sample {idx}: model produced no speech tokens; skip")
                continue
            gen_wav, sr = _decode_neucodec(neucodec, gen_codes)
            ref_wav, _ = _decode_neucodec(neucodec, ref_codes)
            gen_path = os.path.join(out_dir, f"{len(pairs)}_gen.wav")
            ref_path = os.path.join(out_dir, f"{len(pairs)}_ref.wav")
            sf.write(gen_path, gen_wav, int(sr))
            sf.write(ref_path, ref_wav, int(sr))
            pairs.append({"gen": gen_path, "ref": ref_path, "text": ref_text})
            log(f"[AUTOTRAIN_PROGRESS] step=tts_eval_gen processed={len(pairs)} total={max_samples}")
        except Exception as e:  # noqa: BLE001
            log(f"[eval] sample {idx} generation failed: {e}")
    log(f"[eval] generated {len(pairs)} sample(s) → {out_dir}")
    return pairs


# ---------------------------------------------------------------------------
# Scorers (verbatim APIs from the Scicom eval repos)
# ---------------------------------------------------------------------------
def score_cer(pairs, asr_model, language):
    import torch
    from jiwer import cer
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    log(f"[eval] CER: loading ASR {asr_model} …")
    m = AutoModelForSpeechSeq2Seq.from_pretrained(asr_model, torch_dtype=torch.float16).to("cuda")
    proc = AutoProcessor.from_pretrained(asr_model)
    pipe = pipeline("automatic-speech-recognition", model=m, tokenizer=proc.tokenizer,
                    feature_extractor=proc.feature_extractor, torch_dtype=torch.float16, device="cuda")
    scores = []
    for p in pairs:
        if not (p.get("text") or "").strip():
            continue
        try:
            kw = {"language": language} if language else {}
            txt = pipe(p["gen"], **kw)["text"]
            s = min(1.0, float(cer(p["text"].strip(), txt.strip())))
            scores.append(s)
        except Exception as e:  # noqa: BLE001
            log(f"[eval] CER skip {p['gen']}: {e}")
    return sum(scores) / len(scores) if scores else None


def score_mos(pairs):
    import utmosv2

    log("[eval] MOS: loading UTMOSv2 …")
    model = utmosv2.create_model(pretrained=True)
    scores = []
    for p in pairs:
        try:
            scores.append(float(model.predict(input_path=p["gen"], num_repetitions=5)))
        except Exception as e:  # noqa: BLE001
            log(f"[eval] MOS skip {p['gen']}: {e}")
    return sum(scores) / len(scores) if scores else None


def score_similarity(pairs):
    import librosa
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    from titanet_vectors import load as load_titanet

    log("[eval] similarity: loading TitaNet …")
    model = load_titanet().cuda().eval()
    _ = model.to(torch.float16)

    def vec(path):
        audio, _ = librosa.load(path, sr=16000)
        with torch.no_grad():
            a = torch.from_numpy(audio).unsqueeze(0).cuda().to(torch.float16)
            lens = torch.tensor([a.shape[-1]]).cuda()
            return model(a, lens)[1].cpu().float().numpy()

    scores = []
    for p in pairs:
        try:
            scores.append(float(cosine_similarity(vec(p["gen"]), vec(p["ref"]))[0, 0]))
        except Exception as e:  # noqa: BLE001
            log(f"[eval] similarity skip {p['gen']}: {e}")
    return sum(scores) / len(scores) if scores else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="finetuned TTS checkpoint dir")
    ap.add_argument("--eval_dir", required=True, help="packed ChiniDataset shards (the test set)")
    ap.add_argument("--out_dir", default="/tmp/tts_eval_audio")
    ap.add_argument("--methods", default="cer", help="comma list: cer,mos,similarity")
    ap.add_argument("--max_samples", type=int, default=64)
    ap.add_argument("--asr_model", default=os.environ.get("EVAL_ASR_MODEL", "openai/whisper-large-v3"))
    ap.add_argument("--language", default=os.environ.get("EVAL_LANGUAGE", "") or None)
    a = ap.parse_args()

    methods = [m.strip() for m in a.methods.split(",") if m.strip() in ("cer", "mos", "similarity")]
    if not methods:
        log("[eval] no eval methods selected; nothing to do")
        return

    pairs = generate_pairs(a.model_dir, a.eval_dir, a.out_dir, a.max_samples)
    if not pairs:
        # Best-effort: training already succeeded — never emit @@ERROR (it would
        # mark the run failed). Just log and exit cleanly.
        log("[eval] produced no audio to score — skipping")
        return

    results = {"samples": len(pairs)}
    if "cer" in methods:
        results["cer"] = score_cer(pairs, a.asr_model, a.language)
    if "mos" in methods:
        results["mos"] = score_mos(pairs)
    if "similarity" in methods:
        results["similarity"] = score_similarity(pairs)

    # Only @@METRIC — the gateway routes {tts_eval:…} into result_json.tts_eval.
    # (No @@DONE/@@ERROR: that's the training run's, set by the gateway.)
    emit("METRIC", {"tts_eval": results})
    log(f"[eval] done: {results}")


if __name__ == "__main__":
    main()
