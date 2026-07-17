# Claude guide — Datasets UI (`web/src/app/(app)/datasets/`)

The dataset section of the console. A **Dataset** is a named pointer to `{audio, transcription}`
(or chat-`messages`) rows; it does **not** copy the data — rows are read live from the source.
The web layer is thin: pages render server-side via `gateway.*` (see `web/src/lib/gateway.ts`),
and client cards mutate through the proxy `PATCH/POST /api/proxy/v1/datasets/{id}` then
`router.refresh()`. The data model + business logic live in the Python gateway
(`gateway/gateway/datasets_api.py`, `dataset_transform.py`, `db.py::Dataset`).

## `kind` (the source discriminator)

`upload` / `s3` (metadata file in S3) · `hf` (HuggingFace repo) · `llm` / `llm_packed` /
`llm_dpo_packed` / `tts_packed` (LLM/TTS sources) · **`label`** (live import from a Label-platform
project). `web/src/lib/types.ts` (`DatasetRecord`, `CreateDatasetRequest`, `UpdateDatasetRequest`)
is the contract — keep it in sync with the gateway's pydantic models of the same name.

**Chat vs DPO (preference) datasets.** A chat source (`kind=llm`, or `hf`/`upload` with a
`messages_field`) packs to `llm_packed` for SFT. Setting `rejected_field` (the columns card's
**Preference (DPO)** mode) makes it a preference dataset: `messages_field` = the **chosen** column,
`rejected_field` = the **rejected** column. `Pack for LLM` with **objective=dpo** then produces
`llm_dpo_packed` (chosen/rejected pairs, whole pairs per bin) for DPO training. The row browser
renders a DPO source as chosen ✓ / rejected ✗ pairs (`DpoRowItem`) and a packed DPO dataset shows
its **preference-pair** count + a per-pair decode.

## Pages & cards

- `new/dataset-form.tsx` — register a dataset (one branch per `kind`). The `label` branch collects
  the project URL, a token (pasted `lpat_…` or a global secret), the review-status filter, and the
  **timestamp cutoff** (see below).
- `[datasetId]/dataset-detail.tsx` — tabbed detail. Editable cards PATCH then `router.refresh()`:
  - `columns-card.tsx` — audio/transcription/speaker/messages column mapping. For a chat-only
    dataset it also has a **Chat (SFT) / Preference (DPO)** mode toggle: DPO mode maps a **chosen**
    (= `messages_field`) + **rejected** (`rejected_field`) column; saving `rejected_field` flips the
    dataset into DPO mode (row viewer → pairs; Pack for LLM defaults to objective=dpo).
  - `label-import-card.tsx` — **`kind=label` only**: edit the review status + timestamp cutoff
    post-registration (re-counts rows on save). Mirrors `columns-card`'s edit/save/inline-error
    pattern (no toasts; errors render as `text-destructive`).
  - `transformation-card.tsx` / `transform-card.tsx` — materialise a `label`/`hf` source to an
    HF repo or S3 (the **export** path; honours the cutoff). For a `kind=s3` audio dataset the
    Transform tab adds a **Normalize transcription** mode (see below).
  - `normalize-card.tsx` — **`kind=s3` only**: LLM-normalize the transcription column (see below).
  - `row-browser.tsx` — paged preview; include/exclude rows from training.
  - `hf-mirror-card.tsx` — publish an S3 dataset to the self-hosted HF mirror.
- `merge/` — merge ≥2 `label` datasets into one audio dataset.

## Transcription normalization (`kind=s3` → new `kind=s3`, LLM respelling)

The Transform tab on a `kind=s3` audio dataset offers **Normalize transcription** (`normalize-card.tsx`
→ `POST /v1/datasets/{id}/normalize-transcription`). It rewrites the `transcription` column with a
constrained LLM respelling pass (particle/filler spellings la/lah, ya/ye, Malay affix spacing, zh
spacing — *without* changing what was said), ported from
`ucc_ai_research/speech/stt/llm_normalize_experiment.py` into `gateway/gateway/dataset_normalize.py`
(prompt/few-shots/whitelist copied verbatim — edit both together). The LLM is any OpenAI-compatible
chat endpoint (the card defaults to the gateway's own `…/proxy/for-agentic/v1` + `google/gemma-4-31b-it`).
Orchestration is `dataset_transform.start_normalize` / `_run_normalize` (mirrors the other transforms:
background task, `transform_status`/`transform_log`, cancellable, `[AUTOTRAIN_PROGRESS]` markers).

Design decisions that are **load-bearing** (don't regress):
- **Metadata-only, audio NOT copied.** The new dataset's metadata references the SAME S3 audio via
  presigned URLs of `{base}/audio/{basename}` — the reader (`_read_s3_metadata_rows`) never downloads
  and the writer (`_write_normalized_metadata`) never copies. Presigned https (7-day) is deliberate:
  the whisper trainer re-fetches by key (`whisper_finetune._s3_url_key_for_bucket`) so expiry never
  bites, and preview presigns fresh — matching `_materialise_s3`'s convention. **Do NOT write `s3://`
  URIs** — `whisper_finetune._download_audio_s3` doesn't handle that scheme (would break training).
- **New metadata goes in its OWN sub-folder** `{base}/normalized-{hex}/metadata.csv`, NOT the shared
  source folder. This is a purge-safety requirement: `_dataset_storage_prefix` returns the metadata's
  folder for `kind=s3`, so `DELETE ?purge=true` deletes that folder — a sub-folder scopes it to just
  the one CSV, leaving the shared `{base}/audio/` (and the source dataset) intact. Verified: purge
  removed 1 object, source audio still served.
- **Two guards, fail-safe.** `dataset_normalize.validate_edits` (deterministic, always on) structurally
  proves only whitelisted respells + affix joins happened (no add/delete/reorder/renumber, no CJK
  romanization). The **LLM judge is OFF by default** — it's noisy (hallucinates violations on valid
  respells, ~⅓ false-reject in testing) and only adds marginal safety over the deterministic guard. A
  no-op normalization (LLM returns identical text) short-circuits both guards. A rejected/errored row
  keeps its ORIGINAL transcription. `limit` (0/blank = all) supports cheap trial runs.

## `kind=label` import filter (review status + timestamp cutoff)

A label dataset streams rows from the Label platform's
`GET /api/projects/{id}/export.v1.jsonl` (see `docs/LABEL_PLATFORM.md`). Two filters scope what
is pulled, on **every** read (preview, transform/export, merge):

- **`label_status`** → export `status` (`approved` (default) | `rejected` | `not_reviewed` | `all`).
- **`label_updated_until`** → export `updated_until`: an **inclusive point-in-time cutoff** on each
  task's `last_updated_at` (latest of its creation, edit, or annotation). Only tasks finalized
  at/before the instant are imported. `null` → no upper bound. Stored as a UTC ISO-8601 string on
  the Dataset; **set on `/datasets/new`, editable from the Import-filter card.**

UI ↔ storage timezone convention: the cutoff is a `<input type="datetime-local">` (browser-local
wall clock) converted to a UTC instant with `new Date(value).toISOString()` on save, and back with
`isoToLocalInput()` (in `label-import-card.tsx`) when editing. The form shows the resolved UTC
value as a hint so there's no ambiguity. An empty value clears the cutoff (gateway treats `""` as
"clear", `null`/absent as "leave unchanged").
