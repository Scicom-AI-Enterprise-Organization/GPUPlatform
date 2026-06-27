# Claude guide — Datasets UI (`web/src/app/(app)/datasets/`)

The dataset section of the console. A **Dataset** is a named pointer to `{audio, transcription}`
(or chat-`messages`) rows; it does **not** copy the data — rows are read live from the source.
The web layer is thin: pages render server-side via `gateway.*` (see `web/src/lib/gateway.ts`),
and client cards mutate through the proxy `PATCH/POST /api/proxy/v1/datasets/{id}` then
`router.refresh()`. The data model + business logic live in the Python gateway
(`gateway/gateway/datasets_api.py`, `dataset_transform.py`, `db.py::Dataset`).

## `kind` (the source discriminator)

`upload` / `s3` (metadata file in S3) · `hf` (HuggingFace repo) · `llm` / `llm_packed` /
`tts_packed` (LLM/TTS sources) · **`label`** (live import from a Label-platform project).
`web/src/lib/types.ts` (`DatasetRecord`, `CreateDatasetRequest`, `UpdateDatasetRequest`) is the
contract — keep it in sync with the gateway's pydantic models of the same name.

## Pages & cards

- `new/dataset-form.tsx` — register a dataset (one branch per `kind`). The `label` branch collects
  the project URL, a token (pasted `lpat_…` or a global secret), the review-status filter, and the
  **timestamp cutoff** (see below).
- `[datasetId]/dataset-detail.tsx` — tabbed detail. Editable cards PATCH then `router.refresh()`:
  - `columns-card.tsx` — audio/transcription/speaker/messages column mapping.
  - `label-import-card.tsx` — **`kind=label` only**: edit the review status + timestamp cutoff
    post-registration (re-counts rows on save). Mirrors `columns-card`'s edit/save/inline-error
    pattern (no toasts; errors render as `text-destructive`).
  - `transformation-card.tsx` / `transform-card.tsx` — materialise a `label`/`hf` source to an
    HF repo or S3 (the **export** path; honours the cutoff).
  - `row-browser.tsx` — paged preview; include/exclude rows from training.
  - `hf-mirror-card.tsx` — publish an S3 dataset to the self-hosted HF mirror.
- `merge/` — merge ≥2 `label` datasets into one audio dataset.

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
