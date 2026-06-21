# autotrain — standalone LoRA finetune jobs (shared guide)

Each subdirectory is a self-contained training job (NOT part of the gateway), finetuning a
large model with a custom LoRA on a packed dataset via PyTorch FSDP2 on RunPod H100s:

- **`gemma4/`** — Gemma-4 31B (dense, bf16). Custom per-layer `dynamic_attention` (head_dim
  512 SDPA + 256 FA3). See `gemma4/CLAUDE.md`.
- **`minimax-m2/`** — MiniMax-M2 230B/10B FP8 MoE. Custom FP8-dequant LoRA (`lora.py`), stock
  attention. See `minimax-m2/CLAUDE.md`.

Read the per-project `CLAUDE.md` for the model-specific design. The rules below apply to **every**
job under `autotrain/`.

## RunPod API key lives in `../.env` (autotrain root) — always use it

The `RUNPOD_API_KEY` is in **`autotrain/.env`**, i.e. **`../.env`** relative to each job
directory (`gemma4/`, `minimax-m2/`). There is no `.env` inside the job dirs. Always source the
key from there:

```bash
runpodctl config --apiKey "$(grep -E '^RUNPOD_API_KEY=' ../.env | cut -d= -f2-)"
```

**`HF_TOKEN`** (gated/large downloads) and **`WANDB_API_KEY`** (`--wandb` metrics) now also live in
**`autotrain/.env`** (i.e. `../.env`) alongside `RUNPOD_API_KEY` — source them all from there:

```bash
export HF_TOKEN="$(grep -E '^HF_TOKEN=' ../.env | cut -d= -f2-)"
export WANDB_API_KEY="$(grep -E '^WANDB_API_KEY=' ../.env | cut -d= -f2-)"
```

(`HF_TOKEN` may still be present in `gateway/.env` or the shell env; `../.env` is the canonical
place for these jobs. On the pod, `hf auth login --token "$HF_TOKEN"` once deps are installed.)

## Always compare logits when adding/altering a custom implementation

Any time you write or change a **custom forward path** that replaces a stock transformers one —
a registered attention (`dynamic_attention`), a LoRA wrapper that swaps the base matmul
(`LinearLoRA`, fused expert LoRA), an on-the-fly dequant, a packed/varlen collator — you MUST add
and run a **`compare_logits.py`** that checks the custom path against **default transformers** on
the real model **before** any expensive training run. This is non-negotiable: a wrong-but-runnable
forward (e.g. gemma4's float-vs-bool mask leak) silently trains garbage and burns GPU hours.

The invariant to assert:

- **A no-op customization must reproduce the default logits.** If the change is supposed to be
  numerically identical (zero-initialised LoRA `B`, a mathematically-equivalent attention), then
  with the adapter at init the model's **next-token argmax must match** the stock model and the
  logit vectors must be ~equal (cosine ~1). Where the base path itself changes numerics (e.g. FP8
  `w8a8` kernel → bf16 dequant `w8a16`), expect only FP8 quantization noise: argmax still matches,
  cosine > 0.99.
- **Wiring sanity**: poke a non-zero value into the "no-op" adapter and confirm the logits now
  *change* — proves the match was because the delta is zero, not because the custom path is
  silently disconnected.

Examples: `gemma4/compare_logits.py` (dynamic_attention vs default attention) and
`minimax-m2/compare_logits.py` (LoRA(B=0) dequant path vs an independent bf16 reference — the stock
MiniMax-M2 FP8 inference forward NaNs, so it can't be the reference; see that job's CLAUDE.md). The CPU unit
tests (`test_attention.py`, `test_lora.py`) prove correctness + grads cheaply; `compare_logits.py`
confirms it end-to-end on the actual weights on the pod. Run it before training.

## RunPod workflow (shared)

`runpodctl` (`pod create` with `--gpu-id "NVIDIA H100 80GB HBM3"`, `--min-cuda-version 13.0`,
`--terminate-after`), torch 2.12 + the FA3 prebuilt wheel (CUDA backend picked from the host
driver), `NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0`. Full commands + gotchas in each job's
`CLAUDE.md`. **Billing stops only on `runpodctl pod delete <pod-id>` — always terminate.**

**ALWAYS pin the US region: pass `--country-code US` to `runpodctl pod create`** (hard rule, user
requested). Without it RunPod may place the pod anywhere (e.g. IN), which the user doesn't want.

**ALWAYS work under `/` (the container disk), NEVER `/workspace`.** This is a hard rule (the user
has reinforced it). `/workspace` is the RunPod network volume — slow. Put **everything** there:
the job dir (e.g. `/root/autotrain-mm2`, `/root/eval`) AND the HF cache
(`HF_HOME=/root/.cache/huggingface`, the default) on the fast container-disk overlay. `scp` job
files to `/root/...`, not `/workspace/...`. Size the pod with a big **`--container-disk-in-gb`**
(e.g. 600 for MiniMax-M2's ~230GB) rather than a volume.
(Auth/download: `runpodctl config --apiKey` + `hf auth login --token` both read from `../.env`.)

**ALWAYS serve vLLM with `--gpu-memory-utilization 0.85` (hard rule, user requested).** Never higher
(no 0.9/0.92) — leaves headroom and, on the shared tm H20, room for other users + avoids tipping
contended GPUs into OOM. Applies to every `vllm serve` (eval / inference), RunPod or H20.

## The `tm` H20 VM (scicom prod box) — alternative to RunPod, no per-hour billing

A persistent **8× H20-3e** box (~**144 GB/GPU**, ~1 TB RAM, CUDA 13.0) — the quickest path for big
runs (no pod spin-up, models pre-cached) and it has the **VRAM headroom RunPod H100s (80 GB) lack**.
The 144 GB/GPU removes the unshardable activation-memory ceiling (O(seq×layers)) that caps
long-context LoRA training on 80 GB cards — e.g. gemma4 FA4 128k OOMs on 4× H100 but fits here.

```bash
ssh -i ../../scicom root@8.222.165.68 -p 1024      # the `scicom` key = GPUPlatform/scicom (mode 600)
```

- **Shared box** (other users' jobs/tmux live here): **GPUs 0–5 usually free; 6,7 have ~131 GB
  orphaned** (other PID namespace — not killable from our container; don't use them). Pin
  `CUDA_VISIBLE_DEVICES` to the GPUs you're given and **don't hog all 8**.
- **Everything under `/share`** (4.7 T shared disk, but watch free space — it runs ~90% full):
  - venv per job: `uv venv /share/<job>-venv --python 3.12` (NOT `--system`). `uv` is at
    `/usr/local/bin/uv`. ⚠ **Never touch `/share/vllm-venv`** (hand-built fleet venv).
  - HF cache: `export HF_HOME=/share/huggingface` (1.3 T; gemma-4-31B, Mistral-Small-4, etc. already
    cached) + **`HF_HUB_DISABLE_XET=1`** (Xet stalls big pulls here).
  - job dir `/share/autotrain-<job>`; **detach long jobs** (`nohup`/`tmux`) — survives ssh drop.
- **Copy in:** `scp -i ../../scicom -P 1024 *.py *.sh root@8.222.165.68:/share/autotrain-<job>/`.
- Auth: `RUNPOD_API_KEY`/`HF_TOKEN` still come from `../.env`; scp `~/.netrc` to `/root/.netrc` for
  `--wandb`. Per-job verified recipes: `mistral-small/CLAUDE.md` ("H20 node") and `gemma4/CLAUDE.md`
  ("FA4 … on the tm H20").

## Network/transfer benchmarks (HF-Xet ↔ S3 ↔ tm) — why tm downloads are slow

S3 creds for staging live in **`../.env`** (added 2026-06-21): `S3_ACCESS_KEY`, `S3_SECRET`,
`S3_BUCKET_NAME` (`huseinlabel-app-test`), `S3_REGION` (`ap-southeast-5`, AWS **Kuala Lumpur** — IAM
user `aies-label`). It's a plain AWS S3 bucket near the tm box. The user added these to test whether
staging models through S3-KL beats the tm VM's slow ~20 MB/s HuggingFace pulls. **Verdict: it does
NOT — the tm VM's own international uplink (~10–20 MB/s) is the bottleneck, not HF's US location.**

Verified 2026-06-21 (1 GB random file for S3 legs; `Qwen/Qwen3-Embedding-4B` ≈ 8.06 GB, `xetEnabled=True`,
`hf_xet` 1.5.1 for HF leg). All RunPod pods are US-region, `rclone` v1.58.1; tm `rclone` v1.60.1:

- **HF-Xet download (US instance ← HuggingFace)** — *blazing*, and strongly **CPU/core-bound**:
  - CPU pod (US-MD-1, 2 vCPU): **288 MiB/s** (8 GB in 27 s)
  - RTX 5090 (US, 16 vCPU): **1419 MiB/s** (8 GB in 5.4 s)
  - H100 SXM (US, 24 vCPU): **1093 MiB/s** (8 GB in 7 s)
  → HF itself is NOT slow; tm's ~20 MB/s is tm's link. A multi-core US box pulls HF fastest.
- **S3 upload (US instance → S3-KL)** — *path-limited, instance-independent* (~same on CPU/5090/H100):
  default `rclone` **~20 MiB/s**, tuned (`--s3-upload-concurrency 16 --s3-chunk-size 32M`) **~55 MiB/s** (2.7×).
- **S3 download (S3-KL → tm VM)** — *the killer*: default `rclone` **4.2 MiB/s**, `--multi-thread-streams 16`
  **12.3 MiB/s**, 32 streams **5.1 MiB/s** (high variance/congestion). md5 verified. Same ~10–20 MB/s
  ceiling tm hits against HF, despite KL being geographically adjacent → tm's egress is throttled,
  not the source/region. The existing `HF_HUB_DISABLE_XET=1` tip (Xet "stalls" on tm) is really this
  slow link; Xet's chunk negotiation just amplifies it. **Staging via S3 won't rescue tm downloads.**

Gotchas baked into the recipe: `--country-code US` is **ignored for CPU pods** (lands in EUR) and GPU
`pod create` honors only the **first** `--data-center-ids` entry → loop valid US DCs (`US-KS-2,US-KS-3,
US-GA-1,US-GA-2,US-NC-1,US-CA-2,US-TX-1,US-TX-3,US-TX-4,US-IL-1,US-WA-1,US-DE-1,US-MD-1`) one-at-a-time
until one has stock. CPU pods cap container disk at **20 GB**. CPU pod = `runpodctl pod create
--compute-type cpu --template-id runpod-ubuntu-2204` (`runpod/base`, has `rclone`+sshd, injects account
SSH keys — `runpodctl ssh add-key` first). Every S3 op needs **`--s3-no-check-bucket`** (IAM user lacks
`s3:CreateBucket` → 403 otherwise). `mawk`'s `%d` caps at 2147483647 — compute throughput from the
float byte count, not `%d`.
