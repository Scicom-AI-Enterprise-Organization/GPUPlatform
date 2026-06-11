# Label platform — HTTP API reference

The **Label platform** is a separate Next.js app (source: `/home/husein/ssd3/Label`,
dev host `http://localhost:3002`). The gateway integrates with it in two directions:

- **Read** (existing): a `kind=label` Dataset pulls labelled rows via
  `GET /api/projects/{id}/export.v1.jsonl` — see `gateway/gateway/datasets_api.py`
  (`_label_token`, `_label_export_rows`).
- **Write** (autotrain TTS → human eval): after a successful TTS run the gateway
  creates a recording+MOS project and seeds it with synthesized clips — see
  `gateway/gateway/training_api.py` (`_create_label_project_for_run`,
  `_run_tts_label_export_ssh`) + `gateway/gateway/training/tts/tts_label_export.py`.

## Auth

`Authorization: Bearer lpat_…` — a **personal access token (PAT)** that carries its
owner's role. There is **no separate admin/global token**; a PAT inherits the role of
the user who minted it. The bearer scheme accepts **only** values starting with `lpat_`
(anything else falls through to the session cookie and is ignored for API callers).

Roles (`admin` | `manager` | `qa` | `labeller`), required per call:
- **Create project** → `admin`
- **Update storage / import tasks** → `admin` (storage also allows `manager`)
- **Update project settings (MOS axes, etc.)** → `admin` | `manager`
- **List / export** → any authenticated org member

Mint a PAT: UI → tokens, or `POST /api/tokens {name, expiresInDays?}` → `{token, …}`
(plaintext shown once). Project IDs are **bare UUIDs** (not `proj_…`).

## Endpoints used by the integration

```
POST  /api/projects                  {name, description?, type, labeling_mode?}  → {project:{id,…}}
PUT   /api/projects/{id}/storage      {provider, bucket, region, prefix, endpoint?, access_key, secret_key}
PATCH /api/projects/{id}             {mos_enabled?, mos_axes?, restrict_to_assigned?, hide_approved_from_labeller?}
POST  /api/projects/{id}/tasks        {tasks: TaskRow[]}                          → {imported:N}
GET   /api/projects/{id}/export.v1.jsonl?status=approved|rejected|not_reviewed|all
```

`type ∈ transcription | recording | preference | red_teaming | ocr | human_mos | pipeline`.
`labeling_mode ∈ text | word_aligned | segment_aligned` (transcription only).

### Storage (audio/image types require it)

`provider ∈ s3 | gcs | azure | local | sftp`. For S3: `{bucket, region, prefix, endpoint?,
access_key, secret_key}`. Send `secret_key: "••••••••"` to keep the existing secret.
Storage is required before importing tasks for `transcription`/`recording`/`ocr`.

### Tasks (per-type row shape)

| type           | required                         | optional                        |
|----------------|----------------------------------|---------------------------------|
| transcription  | `audio_filename`                 | `transcription` (initial text)  |
| recording      | `transcription` (read-aloud prompt) | `audio_filename` (attached audio) |
| ocr            | `image_filename`                 | `ocr_data {boxes:[{x,y,w,h,text}]}` |
| preference     | `preference_data {prompt, responses[], chosen?, rejected?}` | |
| red_teaming    | `red_team_data {initial_prompt | conversation[]}` | |
| human_mos      | `human_mos_data {messages:[{role,content}]}` (last = assistant) | `model` |

> **The tasks POST endpoint does NOT validate per-type required fields** (only the UI's
> import parser does) — send correct shapes yourself.

### MOS on a recording project

Recording projects can also collect 1–5 MOS ratings: `PATCH /api/projects/{id}` with
`{mos_enabled: true, mos_axes: ["Naturalness","Intelligibility","Noise"]}` (PATCH allows
`mos_axes` for `recording` and `human_mos`). Ratings land in `tasks.mos_scores` and are
emitted in the export. **Caveat:** the recording UI still shows a "Re-record" button next
to the MOS panel — there is no rate-only lock; rely on labeller discipline.

## The audio-filename ↔ storage rule (important)

A task's `audio_filename` is a **bare relative object key** under the project storage prefix.
The two resolution paths disagree on the prefix:

- **Proxy playback** (UI) resolves `key = <storage.prefix><audio_filename>`.
- **S3 export** presigns the **bare `audio_filename`** (prefix NOT prepended).

So for a prefixed S3 bucket these diverge. **To make both work: set the project storage
`prefix: ""` and put the FULL object key in each task's `audio_filename`.** The gateway's
TTS export does exactly this — it uploads WAVs to `…/training-runs/<run_id>/tts-label/NNNN.wav`
and configures the Label storage with `prefix:""`, so proxy key == export key.

## Export (round-trip)

`GET /api/projects/{id}/export.v1.jsonl` — **only `transcription` + `recording`** projects;
NDJSON, one line per task with non-empty `audio_filename`. Each line:
`{id, audio_url, audio_filename, transcription, reviewed, review_comment, word_timings,
segments, pii_entities, mos_scores, annotated_by, reviewed_by, created_at}`. For S3 storage
`audio_url` is a 1h presigned GET; otherwise an internal proxy URL (needs the PAT to fetch).
Headers: `X-Total-Tasks`, `X-Schema-Version: v1`, `ETag` (supports `If-None-Match` → 304).
