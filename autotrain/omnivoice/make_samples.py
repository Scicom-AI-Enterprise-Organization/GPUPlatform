#!/usr/bin/env python3
"""Build a small voice-cloning demo set: per speaker, one reference clip + N
held-out sentences to synthesize in that speaker's voice with the finetuned model.

Emits samples_test.jsonl (for `omnivoice-infer-batch`) and copies each speaker's
reference wav. Output wav ids are nice names: <speaker>_sample{i}. Uses a ~3-10s
reference (OmniVoice's recommended prompt length) when one is available.
"""
import argparse
import json
import os
import shutil


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def pick_reference(rows):
    """A clip in the 3-10s range if possible (best prompt), else the first."""
    import soundfile as sf
    best = None
    for r in rows:
        try:
            d = sf.info(r["audio_path"]).duration
        except Exception:  # noqa: BLE001
            continue
        if 3.0 <= d <= 10.0:
            return r, d
        if best is None:
            best = (r, d)
    return best if best else (rows[0], None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_jsonl", default="samples_test.jsonl")
    ap.add_argument("--ref_dir", default="samples_ref")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()

    train = read_jsonl(os.path.join(a.data_dir, "train.jsonl"))
    dev = read_jsonl(os.path.join(a.data_dir, "dev.jsonl"))  # held-out texts
    os.makedirs(a.ref_dir, exist_ok=True)

    by_spk_train, by_spk_dev = {}, {}
    for r in train:
        by_spk_train.setdefault(r["language_id"], []).append(r)
    for r in dev:
        by_spk_dev.setdefault(r["language_id"], []).append(r)

    out = []
    manifest = {}
    speaker_name = {"en": "TM_English", "zh": "TM_Mandarin"}
    for lang in sorted(by_spk_train):
        spk = speaker_name.get(lang, lang)
        ref, dur = pick_reference(by_spk_train[lang])
        ref_dst = os.path.join(a.ref_dir, f"{spk}_reference.wav")
        shutil.copy2(ref["audio_path"], ref_dst)
        texts = [r for r in by_spk_dev.get(lang, []) if r["text"].strip()][: a.n]
        for i, t in enumerate(texts, 1):
            sid = f"{spk}_sample{i}"
            out.append({
                "id": sid, "text": t["text"],
                "ref_audio": os.path.abspath(ref["audio_path"]),
                "ref_text": ref["text"], "language_id": lang,
            })
            manifest[f"{sid}.wav"] = {"speaker": spk, "language": lang, "text": t["text"]}
        manifest[f"{spk}_reference.wav"] = {"speaker": spk, "language": lang,
                                            "role": "reference voice prompt",
                                            "text": ref["text"],
                                            "duration_s": round(dur, 1) if dur else None}
        print(f"[samples] {spk}: ref={os.path.basename(ref['audio_path'])} "
              f"({dur if dur else '?'}s), {len(texts)} sentences", flush=True)

    with open(a.out_jsonl, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    json.dump(manifest, open(os.path.join(a.ref_dir, "manifest.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[samples] wrote {a.out_jsonl} ({len(out)} items) + refs/manifest in {a.ref_dir}", flush=True)


if __name__ == "__main__":
    main()
