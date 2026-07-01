# Autotrain automation

Config-driven, API-only pipeline that:

1. **imports** each Label-platform project → a `kind=label` dataset
2. **transforms** each to S3 (with a per-dataset held-out test split) → a `kind=s3` audio dataset
3. **merges** all transformed datasets into ONE combined audio dataset
4. **fine-tunes** each model on the merged dataset (RunPod Secure H100 SXM)

Everything runs through the gateway HTTP API — no gateway source imports.

## Run

```bash
# one-time: create your config from the template (config.yaml is git-ignored)
cp automation/config.yaml.example automation/config.yaml   # then edit api_key etc.

# from the repo root, with the repo venv (has httpx + pyyaml)
.venv/bin/python automation/run_pipeline.py

# override the import cutoff for the whole run
.venv/bin/python automation/run_pipeline.py --cutoff "2026-07-01 11:59PM"

# see what it would do without touching anything
.venv/bin/python automation/run_pipeline.py --dry-run

# launch training but don't block waiting for it
.venv/bin/python automation/run_pipeline.py --no-run-watch
```

## Config (`config.yaml`)

| key | meaning |
|-----|---------|
| `gateway_url` / `api_key` | gateway to hit (overridable via `--gateway-url` / `--api-key` / `AUTOTRAIN_API_KEY`) |
| `storage_id` | kind=s3 storage for the audio datasets; blank → auto-pick first enabled S3 |
| `label_base_url` | Label platform host (e.g. `https://….aies.scicom.dev`) |
| `label_token_secret` | gateway global-secret key holding the `lpat_` token (e.g. `LABEL_PROD`) |
| `label_status` | `approved` (default) / `rejected` / `not_reviewed` / `all` |
| `cutoff` | default import cutoff; naive times use `timezone_offset` |
| `timezone_offset` | offset applied to a cutoff with no zone (e.g. `+08:00`) |
| `datasets[]` | `name`, `project_id`, and `test_split_pct` **or** `test_split_count` (omit both = no test set); optional per-dataset `cutoff`, `label_status` |
| `merge` | `enabled`, `name` |
| `train` | shared hyperparameters (see below) |
| `models[]` | `base_model` + any per-model overrides of the `train` block |

### `train` block

`task_type` (`asr`), `max_epochs`, `patience` (early stop), `augment_techniques`
(`all` = the 8 techniques telephone/noise/dropout/gain/pitch/speed/reverb/bandpass,
or a list), `augment_prob`, `warmup_steps`, `batch_size`, `grad_accum`,
`learning_rate` (omit → gateway default 1e-5), `lr_scheduler_type`, `gpu_type`,
`gpu_count`, `secure_cloud`, `provider_id` (blank → auto-pick first enabled RunPod
provider), `data_center_id`.

## Cutoff

`--cutoff` (or `config.cutoff`) accepts ISO-8601 (`2026-07-01T23:59:00+08:00`) or
friendly forms (`2026-07-01 11:59PM`, `2026-07-01 23:59`). A value with no timezone
is interpreted in `timezone_offset`. Precedence: **per-dataset `cutoff` > `--cutoff`
> `config.cutoff`**. The cutoff is a point-in-time snapshot: only Label tasks last
updated at/before it are imported.

## Idempotency / adding more later

A JSON state file (`state/<config-stem>.json`) records every created resource.
Re-running skips work that's already done. To add a 4th dataset (different link,
cutoff, test split) or another model, just append to `config.yaml` and re-run:

- existing datasets → skipped
- the new dataset → imported + transformed
- the merge → re-run (its source set changed) → a new merged dataset
- models → new runs launched against the new merged dataset

Changing a dataset's `project_id`/`cutoff`/test-split re-imports + re-transforms
that one (a new `kind=label` + `kind=s3` dataset; the old rows are left intact).

Pass `--fresh` to ignore existing state and recreate everything.

## Notes

- **Splits**: `kind=s3` sources keep their `split` column through the merge, and
  the Whisper trainer evaluates on the combined `test` rows — so `patience` early
  stopping works on the merged test set.
- **Local + RunPod**: training runs SSH *into* the pod, so the localhost↔RunPod
  reachback issue that affects serverless workers does **not** apply here — a local
  gateway can train on real RunPod pods. Each run spawns a real, billed pod.
- **Timeouts**: `--transform-timeout` (per transform/merge, default 3h),
  `--watch-timeout` (training watch, default 6h). Interrupting the watch does not
  stop the runs — they finish server-side.
