# Claude guide — Quantization UI (`web/src/app/(app)/quantization/`)

The llm-compressor quantization section of the console. A **QuantizationJob** pulls an LLM from
Hugging Face, compresses it on a VM/RunPod pod, uploads the compressed-tensors model to S3, and
optionally pushes it back to HF (or the self-hosted mirror). The web layer is thin: server pages
fetch via `gateway.*` (`web/src/lib/gateway.ts`), client components mutate through
`/api/proxy/v1/quantization-jobs/*`. Business logic lives in the Python gateway
(`gateway/gateway/quantization_api.py` — see ITS CLAUDE.md section for backend gotchas).

## The one rule: mirror Autotrain

This section was built as a 1:1 visual sibling of Autotrain — the user explicitly wants the two
consistent. **When changing anything here, diff against the corresponding autotrain file first**;
when autotrain's patterns evolve, this section should follow. The mapping:

| Here | Mirrors |
|---|---|
| `page.tsx` + `quantization-list.tsx` | `autotrain/page.tsx` + `autotrain-list.tsx` (status pills, search/filter/sort, server pagination via `_page`) |
| `new/quantization-form.tsx` | `autotrain/new/training-form.tsx` (FormShell scrollspy, icon-Card `Section`, two-column `Grid` + `FieldWrap`, in-form `<h1>`, FormFooter summary hint) |
| `[jobId]/quantization-detail.tsx` | `autotrain/[runId]/training-detail.tsx` (full-width header band `bg-sidebar/40` with inline rename + KPI row incl. live `CostKpi`, URL-driven `?tab=` line tabs, dark `terminal-block` logs, `JsonView` config with prune toggle, confirm-dialog actions) + `hf-export-tab.tsx` (Destination card with **HF storage picker**, Run-on toggle, submit button OUTSIDE the card right-aligned, Cancel-export while running, status card) |

Tabs: **Overview / Logs / Files / Config / Export to HF** (HF tab only when done + artifact).

## Contracts

- `web/src/lib/types.ts` — `QuantizationJobRecord` / `CreateQuantizationJobRequest` /
  `QuantizationSchemesResponse` / `QuantizationResult` must stay in sync with the pydantic models
  of the same shape in `quantization_api.py`.
- The **scheme dropdown is server-driven**: the form fetches `GET /v1/quantization-jobs/schemes`
  (`{schemes: {id: {label, needs_calibration}}, calib_dataset_kinds}`) — `needs_calibration`
  toggles the whole Calibration-dataset section. Don't hardcode scheme ids in the form beyond the
  per-scheme knob visibility (`w8a8-int8` → SmoothQuant strength; `w4a16`/`w8a8-int8` → GPTQ
  dampening).
- Calibration dataset picker filters `CALIB_KINDS` (`hf`/`llm`/`upload`/`s3`) — keep in sync with
  the gateway's `_CALIB_DATASET_KINDS`. `kind=llm` flips the text-column field to a
  messages-column field (chat template applied worker-side).
- Live updates: detail polls `GET /{id}` every 5s while non-terminal (or while
  `result_json.hf_export.status === "running"`), and streams logs via SSE
  (`quantizationLogsStreamUrl` → `/logs/stream`). `result_json.progress` `{stage, percent}` drives
  the header phase spinner + Overview progress bar.
- Section access: key `quantization` — `Section` (me.ts) / `SectionKey` (types.ts) /
  roles-manager / organization table / user-profile `SECTION_LABEL` maps, sidebar item in
  `console/sidebar.tsx`. Page-side check is `me.is_admin || sections?.quantization` (the sidebar
  item is not platform-gated, like Autotrain's).

## HF export tab specifics

- The **HF storage picker** (kind=huggingface storages) supplies the push token — which may be a
  global-secret reference — and any custom endpoint (the self-hosted mirror). No paste-a-token
  field in this tab (matches autotrain).
- **Run on** (Gateway / Job's VM) only renders when `job.provider_kind === "vm"` — RunPod jobs'
  pods are torn down, so only the gateway push exists for them. Pushing to the own-mirror from a
  VM is rejected server-side (400) — the box can't reach it.
