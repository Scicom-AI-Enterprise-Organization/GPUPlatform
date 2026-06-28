# autotrain · whisper — Whisper ASR fine-tune (standalone)

A self-contained Whisper fine-tuning job (read `../CLAUDE.md` for the shared autotrain rules:
RunPod key in `../.env`, etc.). **Full fine-tune** (`Seq2SeqTrainer`, full weights — *not* LoRA
like the sibling jobs), multilingual call-centre ASR for **en / ms / zh**.

> Distinct from the gateway's production path `gateway/gateway/training/whisper_finetune.py`.
> This dir is an offline experiment; the two share no code.

## Flow

```
raw transcripts ──prepare-mosaic.ipynb──▶ Mosaic MDS {audio_filename, text} ──▶ HF ──whisper.py──▶ model
   (cleaning.py: clean → detect lang → format)        (text already in Whisper prompt format)
```

1. **`cleaning.py`** — the text contract. Pure functions, no heavy deps; the notebook (and any
   other consumer) import from here so there is one implementation.
   > **Vendored copy:** the gateway's production trainer
   > `gateway/gateway/training/whisper_finetune.py` ships to a pod with no gateway imports, so it
   > carries a *verbatim mirror* of these functions (`_prepare_texts` inlines `format_whisper`,
   > and its `detect_language` tolerates a missing model). **Edit both** when you change cleaning
   > or detection.
   - `whisper_textcleaning(text)` — standardize + clean: NFKC, fold CJK/full-width punctuation +
     curly quotes/dashes → ASCII, strip invisibles, drop `[...]`/`(...)`, canonicalize fillers
     (`ok/okay`→`OK`, nasal hesitations→`herm`), collapse repeated/stranded punctuation, fix
     spacing. CJK ideographs are preserved.
   - `chinese_ratio(text)` — CJK ideographs ÷ non-whitespace chars (`_CJK` = Unified + Ext-A +
     Compatibility ranges).
   - `detect_language(text, lang_model, chinese_threshold=0.5)` — **`zh` when `chinese_ratio` ≥
     threshold** (checked first — the fastText model has no `zh` label); else fastText
     `mesolitica/fasttext-language-detection-bahasa-en`: `bahasa`→`ms`, anything else→`en`.
   - `format_whisper(text, lang_model, task='transcribe')` — clean → detect → wrap; returns
     `None` if empty after cleaning (caller emits the silence target). Does **not** strip
     domain markers — do that first (see below).

2. **`prepare-mosaic.ipynb`** — builds + uploads the Mosaic dataset from a HF source
   (`Scicom-intl/emgs-recording-2025-10-13`).
   - **Junk filter**: skip rows whose uppercase-ASCII ratio ≥ `0.2` (garbage / all-caps IDs).
   - Per row: drop call-centre end markers `CALL ENDS` / `CALL ENDED`, then `format_whisper(...)`.
   - **Silence/empty** (no speech, or text empty after cleaning → `format_whisper` returns `None`):
     emit an empty target under **each** language in `languages = ['en','ms','zh']` so the model
     learns silence→nothing per language token.
   - Raw inputs look like Label-platform exports (`*-tasks.json` here, gitignored: keys
     `audio_filename, transcription, reviewed, created_at, last_updated_at, …`).

3. **`whisper.py`** — `Seq2SeqTrainer` full fine-tune. Reads a Mosaic `LocalDataset` folder **or**
   a `.json` list (`train_dataset_name`), each item `{audio_filename, text}` where `text` is the
   **already-formatted** Whisper string. Tokenizes with `add_special_tokens=False`, drops rows
   over `max_label_length` (384) tokens, custom collator pads + masks prompt tokens with `-100`.
   Run: `python whisper.py config.json` (HfArgumentParser) or CLI args.

## The Whisper target format

```
<|startoftranscript|><|LANG|><|transcribe|><|notimestamps|> TEXT<|endoftext|>
```

`LANG ∈ {en, ms, zh}`. No timestamps. `whisper.py` adds `<|x.xx|>` timestamp tokens +
`<|transcribeprecise|>` to the tokenizer and `resize_token_embeddings`, but this dataset uses the
`<|notimestamps|>` form.

## Gotchas

- **Code-switching vs the 0.5 zh threshold.** Heavily code-switched utterances (Chinese matrix +
  English loanwords like "Incoming"/"aspect") can fall *below* 50% CJK chars and get tagged
  `en`/`ms`. Example: `啊，不行。Incoming，… aspect …` cleans to ~42% Chinese → not `zh`. If recall
  on zh matters more than precision, lower `chinese_threshold` (e.g. 0.3) when calling
  `detect_language`.
- **Clean before detect.** Always run text through `whisper_textcleaning` (via `format_whisper`)
  before language detection — fold full-width punctuation so `chinese_ratio` counts ideographs,
  not punctuation, and so fastText sees normalized latin text.
- **Strip domain markers first.** `format_whisper` is generic; remove dataset-specific tags
  (`CALL ENDS`, etc.) before calling it.
