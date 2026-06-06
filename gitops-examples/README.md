# GitOps manifests for the GPU Platform

Declare platform resources as YAML in a git repo and let the gateway reconcile the
live state to match. **Git is the source of truth** — on each sync the gateway
creates/updates declared resources and (when `prune` is on) **deletes** any it
previously created that you've since removed.

Connect a repo in the UI under **GitOps → Add repository** (admin only), or:

```
POST /v1/gitops    { "name": "...", "url": "https://github.com/org/repo.git", "branch": "main", "path": "manifests/" }
POST /v1/gitops/{id}/sync          # force a reconcile now
```

Triggers: background auto-poll (per-repo `poll_interval`, default 300s), the
**Sync now** button, and a push **webhook** at `POST /v1/gitops/webhook`
(set a webhook secret on the repo; the gateway verifies the GitHub-style
`X-Hub-Signature-256` HMAC).

## Manifest format

Every `*.yaml` / `*.yml` file under the repo (or its `path`) is scanned; a file
may hold multiple `---` documents. Each document:

```yaml
kind: <App | Storage | Dataset | Provider | Benchmark | TrainingRun>
name: <stable identity — unique per kind within the repo>
generation: 1        # optional; bump to re-submit a Benchmark/TrainingRun job
spec:                # fields map 1:1 to the resource's create API
  ...
```

`spec` fields are exactly the resource's create-request fields (see each example
below). The manifest `name` is the GitOps identity used to match, update and prune
— it is **not** the platform id (those are generated, e.g. `ds-…`, `prov-…`).

### Cross-references by name

`provider_id`, `storage_id`, `dataset_id`, `test_dataset_id` and
`pack_source_dataset_id` may reference **another manifest's `name`** in the same
repo — the reconciler resolves it to that resource's live platform id. A value
that doesn't match a managed name is passed through as a literal id (so you can
still reference pre-existing, non-GitOps resources). Resources apply in dependency
order: Provider → Storage → Dataset → App / Benchmark / TrainingRun.

### Secrets — never commit them

Manifests reference **Secrets** (admin Secrets page / `GlobalEnv`) by key; the
gateway resolves the value in-memory at apply time and encrypts it at rest. Use:

| Resource | Secret reference field | resolves into |
|---|---|---|
| Provider (vm)  | `spec.vm.private_key_secret`     | the SSH private key |
| Provider (api) | `spec.api.api_key_secret`        | the RunPod/PI API key |
| Storage (s3)   | `spec.access_key_id_secret`, `spec.secret_access_key_secret` | S3 creds |
| Storage (hf)   | `spec.hf_token_secret`           | HF token |
| Dataset (label)| `spec.label_token_secret`        | labeling-platform token |

App/Benchmark/TrainingRun pull `HF_TOKEN`, `WANDB_API_KEY`, etc. from the org-wide
Secrets automatically — no per-manifest reference needed.

## Behaviour notes

- **Jobs** (`Benchmark`, `TrainingRun`) are **submit-once-by-name**: created on
  first sync, never auto-re-run. To run again, rename or bump `generation`.
- **Storage** and **Dataset** changes are applied in place (PATCH). Dataset PATCH
  only covers metadata fields (name/description/audio_prefix/fields/speaker) —
  changing the *source* (storage/kind/uri) isn't applied in place.
- **App** and **Provider** have no PATCH, so a spec change is applied as
  **delete + recreate**. An App keeps its id (the name); a Provider gets a new
  `prov-…` id — manifests that reference it by name auto-rebind on the same sync.
- Real GPU spend: applying an `App` (with a cloud provider) or a job spawns real
  pods. Test with `PROVIDER=fake` or no-spend kinds first.
