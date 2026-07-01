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
GET   /api/projects/{id}/export.v1.jsonl?status=approved|rejected|not_reviewed|all[&updated_since=ISO][&updated_until=ISO]
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
segments, pii_entities, mos_scores, annotated_by, reviewed_by, created_at, last_updated_at}`.
For S3 storage `audio_url` is a 1h presigned GET; otherwise an internal proxy URL (needs the
PAT to fetch). Headers: `X-Total-Tasks`, `X-Schema-Version: v1`, `ETag` (supports
`If-None-Match` → 304).

**Filters** (query params): `status` (`approved` (default) | `rejected` | `not_reviewed` |
`all`), and a `last_updated_at` range — `updated_since` (≥, incremental pulls) and
`updated_until` (≤, **point-in-time snapshot**). `last_updated_at` is the latest of a task's
creation, any edit, or any annotation. The cutoff filters server-side, so `X-Total-Tasks` and
the gateway's pagination stay accurate. A `kind=label` Dataset stores its cutoff in
`Dataset.label_updated_until` (ISO-8601 UTC; set on `/datasets/new`, editable from the dataset's
**Import filter** card) and the gateway forwards it as `updated_until` on every preview /
transform / merge read (`_label_export_rows`, `_label_pairs`).

## Gateway API — trigger a TTS label export (write side)

Drive the **post-train TTS → Label** export programmatically against the **gateway** (these are
gateway routes, auth = a `sgpu_…` API key; not the Label platform's `lpat_…`). The run must be a
finished (`status=done`) **TTS** run with a model artifact.

```
POST /v1/training-runs/{run_id}/label-export          → {"status":"started"}
POST /v1/training-runs/{run_id}/label-export/cancel    → {"status":"cancelled"}
```

`label-export` synthesizes N clips from the trained model, uploads them to the run's S3, creates a
Label **recording+MOS** project, imports the clips as tasks, and round-trips a `kind=label`
Dataset. It runs in the **background** — progress streams to the run's Logs, and
`result_json.label_projects[]` (+ `label_project` for the first) carry the created project(s).
Re-POSTing supersedes an in-flight export. **Pre-flight:** the gateway first does a `GET
{base_url}/api/projects` with the token and **fails fast (502) before provisioning any pod / VM**
if the platform is unreachable or the token is rejected.

Body (all optional — unset fields fall back to the run's stored config):

| field | meaning |
|-------|---------|
| `base_url` / `base_url_secret` | Label platform URL (pasted, or a Secrets-page key) |
| `token` / `token_secret` | admin `lpat_…` (pasted, encrypted at rest, or a Secrets key) |
| `project_name`, `samples`, `mos_axes[]` | project name, #clips, 1–5 MOS rating axes |
| `speakers[]`, `per_speaker`, `speaker_prefix` | voices to use; one project per speaker (each from that speaker's own clips); prefix transcript with the speaker name |
| `reject_keywords[]` | drop text samples containing any phrase (case-insensitive, spacing-agnostic) |
| `tts_codec` | NeuCodec decoder: `neucodec` (upstream neuphonic, 24 kHz — default) or `neucodec-44k` (Scicom 44k-d20 fork, 44.1 kHz). Same speech tokens, so either decodes the model — just output fidelity. |
| `run_on` (`cloud`\|`vm`), `provider_id` | where to synthesize — a fresh RunPod pod (needs a RunPod `provider_id`) or a registered VM. Omitted → the run's own box (the auto post-train export reuses the still-alive training pod). |
| `gpu_type`, `gpu_count`, `secure_cloud`, `disk_gb`, `volume_gb`, `visible_devices`, `venv_path` | cloud-pod hardware + the TTS venv path |

**Automatic export:** enabling *"Create labelling project after training"* on the run sets
`config.label_export=true`; the gateway then runs this same flow at finalize (reusing the live
training pod — no new pod). `cancel` stops a running export (kills the box-side synth, tears down
any pod it spawned, clears the `running` state); a gateway restart also reconciles a stuck
`running` export to `failed`.
