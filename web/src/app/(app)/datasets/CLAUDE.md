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

**⚠ Two DPO corpus shapes — `prompt_field` is what tells them apart.** `DpoRowItem` and
`llm_pack._normalize_pair` must agree here or the viewer and the packer disagree about the data:

- **shared-prefix** (ultrafeedback style) — chosen/rejected are FULL conversations agreeing on the
  prompt turns. `prompt_field` is unset; the prompt is their **common prefix**.
- **continuation** (agentic pairs) — a `prompt_field` column holds the prompt turns and
  chosen/rejected hold **only the turns after** them. Such a pair diverges at turn 0, so the common
  prefix is *empty* and the prompt **cannot** be inferred — it must come from the column. (This is
  exactly what silently broke: with `prompt_field` unmapped the row viewer rendered no prompt at
  all and summarised each row by the first *assistant* turn.) Both sides tolerate a corpus that
  ALSO repeats the prompt turns at the head of chosen/rejected.

**Scored vs environment turns.** `DpoRowItem` renders the chosen/rejected tails with
`<ChatBubbles markScored>`: non-assistant turns are dimmed + tagged **not scored**, and each panel
headers a `N turns · M scored` count. An agentic completion interleaves the policy's assistant
turns with `role: tool` results, and the packer masks those out of the loss
(`llm_pack.tokenize_pair(mask_env=True)`) — a flat render overstates what the pair trains. ⚠ This is
a **role-level approximation of a token-level mask**: the real boundary is the template's
`{% generation %}` span, and on a template that carries no mask the packer scores the whole
completion while this UI still shows tool turns as "not scored". The packed-dataset decode
(`PackedRowItem`) is the one that reflects the TRUE per-token mask.

`prompt_field` is a **stored dataset column mapping** (`Dataset.prompt_field`, columns card →
`PATCH`), not just a pack-time argument: the row browser reads it, and `POST /datasets/{id}/pack-llm`
falls back to it when the request omits `prompt_field`. A continuation corpus registered without it
still *packs* (`llm_pack` accepts it per-request) but previews wrong — set it on the dataset.

## Pages & cards

- `new/dataset-form.tsx` — register a dataset (one branch per `kind`). The `label` branch collects
  the project URL, a token (pasted `lpat_…` or a global secret), the review-status filter, and the
  **timestamp cutoff** (see below).
- `[datasetId]/dataset-detail.tsx` — tabbed detail. Editable cards PATCH then `router.refresh()`:
  - `columns-card.tsx` — audio/transcription/speaker/messages column mapping, plus (kind=hf) the
    **Subsets** scope (see below). For a chat-only
    dataset it also has a **Chat (SFT) / Preference (DPO)** mode toggle: DPO mode maps a **chosen**
    (= `messages_field`) + **rejected** (`rejected_field`) column, plus an optional **prompt**
    (`prompt_field`) column for the continuation shape above; saving `rejected_field` flips the
    dataset into DPO mode (row viewer → pairs; Pack for LLM defaults to objective=dpo).
  - `label-import-card.tsx` — **`kind=label` only**: edit the review status + timestamp cutoff
    post-registration (re-counts rows on save). Mirrors `columns-card`'s edit/save/inline-error
    pattern (no toasts; errors render as `text-destructive`).
  - `transformation-card.tsx` / `transform-card.tsx` — materialise a `label`/`hf` source to an
    HF repo or S3 (the **export** path; honours the cutoff). For a `kind=s3` audio dataset the
    Transform tab adds a **Normalize transcription** mode (see below).
  - `normalize-card.tsx` — **`kind=s3` only**: LLM-normalize the transcription column (see below).
  - `row-browser.tsx` — paged preview; include/exclude rows from training; **Download** button
    (see below).
  - `hf-mirror-card.tsx` — publish an S3 dataset to the self-hosted HF mirror.
- `merge/` — merge ≥2 `label` datasets into one audio dataset.

## Download (`GET /v1/datasets/{id}/download` → zip)

The row browser's **Download** button streams a zip of `audio/…` + `metadata.json`, scoped to
**exactly what the browser is showing**: `?split=` (subset) + `?speaker=`, `included_only=1` by
default (un-ticked rows are skipped). `kind=s3|upload` audio datasets only — an HF source is
already downloadable as its repo and a packed one holds token shards, not files.

- **The zip is self-contained.** Each metadata row keeps its original columns but the audio column
  is rewritten to the in-zip relative path (`audio/<basename>`), never the presigned URL — those
  expire, and a downloaded folder must still work offline.
- **The default with a `split` column is `splits[0]`, not "everything"** — that's what the preview
  shows when nothing is ticked, so the button must mean the same. Multi-subset selection disables
  the button rather than silently downloading one of them.
- **Streamed, never staged on disk**: `_ZipSink` is an unseekable sink, so `zipfile` falls back to
  data descriptors and each member is yielded as it's fetched. A member whose fetch fails is
  recorded in `metadata.json`'s `errors[]` (row's audio → `null`) instead of aborting a multi-GB
  download. Basename collisions get the row index appended — collapsing them would drop rows.
- **Fetch in parallel, write in order.** A 12k-row subset is 12k small round-trips — latency-bound,
  not bandwidth-bound. Downloads run on a `ThreadPoolExecutor` over ONE pooled `httpx.Client`
  (`_fetch_url_bytes(url, client=…)`; a client per call was a TLS handshake per clip) while the zip
  writer stays single-threaded and in file order, so the archive is byte-identical to the serial
  one. ⚠ It must stay a **sliding window** (`_DOWNLOAD_WINDOW`, submit one more per completion) —
  mapping the pool over every row would buffer the whole dataset in the gateway's heap. Tune with
  `DATASET_DOWNLOAD_WORKERS` (16) / `DATASET_DOWNLOAD_WINDOW` (64). Measured: ~9.7 MB/s sustained
  on the train split, ~2.3× the serial path on a 50-row one.

## `kind=hf` subset scope (`hf_subsets`)

A HF repo can hold several configs, and a dataset usually wants only some of them (e.g. the two
audio configs of a repo that also ships text-only chat ones). `Dataset.hf_subsets` is that scope —
set on `/datasets/new?source=hf` (comma-separated free text, before the repo has been introspected)
and edited afterwards on the **columns card** as a checkbox multiselect. It bounds the row browser,
the per-split column pickers, and which files a transform downloads.

- **`GET /{id}/splits` is filtered to the scope; the editor needs `?all=1`.** Filtered output can
  only narrow a scope, never widen it back, so the picker fetches `?all=1` (every subset, each
  flagged `in_scope`) and derives BOTH lists from that one call. `?all=1` also still answers when
  the scope matches nothing — that's precisely the state the editor exists to fix, so it must not
  inherit the default listing's error.
- **Selection is by canonical label, never typed here.** The stored value may be a bare config name
  (`synthetic`) that expands to several labels, so "did the user change it" compares against the
  RESOLVED baseline captured at load — comparing against the raw stored value would rewrite
  `["synthetic"]` into explicit labels just for opening the card.
- **Ticking every subset saves `[]` (no scope), not the full list**, so a config added upstream
  later is picked up rather than silently excluded.
- **Saving a changed scope clears `split_fields`.** The per-split transcription picks were seeded
  from the outgoing split list; keeping them would write a mapping keyed by labels the dataset no
  longer has. The card says so before you save.

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
