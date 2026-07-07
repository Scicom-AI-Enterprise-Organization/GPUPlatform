# Claude guide — worker-agent (`worker-agent/worker_agent/`)

Area-specific gotchas for the GPU worker (vLLM runner + multi-model fleet scheduler). **Loads
automatically when you edit files here.** Cross-cutting dev setup lives in the repo-root `CLAUDE.md`;
the gateway-side counterparts (reverse tunnel, benchmaq) live in `gateway/gateway/CLAUDE.md`.

### Serving Whisper / audio (ASR) on the serverless fleet

The fleet serves **Whisper** via the OpenAI-compatible **`/v1/audio/transcriptions`** and
**`/v1/audio/translations`** (global + scoped `/{app_id}/v1/audio/...`). The job queue is
JSON-only, so the gateway base64s the uploaded clip into the payload and the **worker rebuilds
the multipart** request for vLLM (`worker-agent/worker_agent/main.py` `handle()`). Use it from an
endpoint's **Playground → mode: Audio transcription** (a Chat/Embedding/Audio toggle on the Playground tab —
not a separate tab; its model dropdown lists *every* member so any name works, and it keeps its own
per-browser transcription history alongside the chat history), or:
`curl -F file=@clip.mp3 -F model=<model> -H "Authorization: Bearer sgpu_…" $GW/<app_id>/v1/audio/transcriptions`

Three things silently break audio if missing — check these first when transcription fails:
- **Gateway venv needs `python-multipart`** (now in `gateway/pyproject.toml`). Without it FastAPI
  refuses to boot once the `Form`/`File` routes exist — the gateway crashes on start.
- **vLLM venv needs `librosa soundfile resampy av`** (the `vllm[audio]` set — all pip, **NO system
  ffmpeg**; vLLM decodes via soundfile + PyAV). `resampy` is the sneaky one: without it any
  non-16 kHz clip fails as `"Invalid or unsupported audio file"` while a 16 kHz WAV slips through
  (looks like an mp3/codec bug — it isn't). `launcher.ensure_audio_deps()` auto-installs these for
  any audio member on (re)provision.
- **Tag ASR models** with the per-member **"Audio / ASR (Whisper)" checkbox** (create form +
  Overview → Models → Edit) → stores `task: "transcription"`. Required for ASR finetunes whose name
  doesn't contain "whisper" (a name heuristic auto-covers `whisper-*`); the worker installs audio
  deps when `task=="transcription"` OR the name matches.

To add Whisper to an existing fleet: **Overview → Models → Edit → add the member (TP=1), check
Audio, Save** — that re-provisions and re-ships the worker-agent + ensures the audio deps. (A
member needs a GPU slot; small fleets time-share via sleep/wake. Sleep-mode works fine with Whisper.)

### Huawei Ascend NPU (vllm-ascend) — auto-detected, verified on the TM 910B3 box

The worker serves NPUs with **no endpoint-level device flag**: `launcher.is_ascend()`
(= `/dev/davinci_manager` exists) switches venv creation to **python 3.11** (`uv venv --python`,
Euler boxes ship 3.9), overlays the **CANN env** (captured once from `set_env.sh` of
ascend-toolkit + nnal/atb via `bash -c 'source …; env -0'`) onto every uv install AND vLLM
launch, and pins members with **`ASCEND_RT_VISIBLE_DEVICES`** instead of CUDA_VISIBLE_DEVICES.
Verified e2e on the TM box (8× 910B3, CANN 8.5.2): endpoint `npu-qwen3` through the jump-host
reverse tunnel. Gotchas that each cost a debug cycle:
- **`--enable-sleep-mode` is SKIPPED on Ascend** — vllm-ascend's CaMeM allocator
  (`aclrtMallocPhysical`) OOMs on hosts without hugepages (npu-smi shows `0/0`). So NPU fleet
  members stay resident: size fleets to fit; the sleep/wake eviction path can't run.
- **`vllm_install_args` is now MULTI-LINE** (each line = its own sequential `uv pip install`,
  each with its own leading NAME=VALUE env tokens) because vllm and vllm-ascend pin
  **conflicting torch versions** — one resolve is unsatisfiable, sequential installs work.
- **The proven CANN 8.5.x recipe** (the form's "Insert Ascend" preset): line 1
  `VLLM_TARGET_DEVICE=empty vllm==0.18.0`; line 2 `vllm-ascend==0.18.0` + the
  `mirrors.huaweicloud.com/ascend/repos/pypi{,/variant}` extra indexes; line 3
  `setuptools<81 z3-solver==4.12.2.0` (torchair needs `pkg_resources` — removed in
  setuptools 81; newer z3 wheels need GLIBCXX_3.4.29 the Euler host lacks). ⚠ Do NOT bump to
  0.22.1rc1 on a CANN 8.5 box: its wheels target CANN 9.0 — graph mode dies on a missing
  `qkv_rmsnorm_rope` op and eager crashes in broken triton kernels.
- **"Failed to import Triton kernels … No module named 'triton.…'" is BENIGN** on vllm-ascend
  (an optional-kernel probe; serving works without it). It used to trip the `No module named`
  fatal-marker scan and kill healthy boots — `has_fatal_error`/`read_failure_reason` now skip
  lines matching `_BENIGN_ERROR_RE`.
- Host prereqs (one-time, on the box): CANN toolkit AND **NNAL** (`libatb.so` — installed
  8.5.2 from ascend-repo.obs.cn-east-2.myhuaweicloud.com on 2026-07-07); without NNAL,
  torch_npu dies loading ATB ops.

### VM endpoints self-bootstrap their vLLM venv (`venv_path` no longer needs to pre-exist)

A multi-model VM/RunPod worker now **builds its own vLLM venv on boot** — name a `venv_path` that
doesn't exist and the worker installs `uv` (if absent), `uv venv`s the path (mkdir parents), and
installs vLLM, *streaming* the install to the `__worker__` log. Logic in
`worker-agent/.../multi/launcher.py` (`ensure_venv`/`ensure_build_tools`/`run_pre_script`) +
`scheduler.start()`. Knobs (create form **and** Overview → Models → Edit; stored on `App`):
- **vLLM version** (`vllm_version`) — `uv pip install vllm==X --torch-backend=auto`. **Default `0.23.0`**
  (the form pre-fills it / falls back to it; ⚠ 0.23.0 needs the prometheus fix below).
- **vLLM install args** (`vllm_install_args`) — a full `uv pip install` arg string used *verbatim*
  (overrides the version), for nightlies/custom CUDA/**git forks**. The form has an **Insert nightly
  (cu130)** button. A **leading `NAME=VALUE` token is an install-subprocess env var** (shell-style),
  parsed off in `ensure_venv` before the pip args — e.g. `VLLM_USE_PRECOMPILED=1 git+…@ref` reuses
  precompiled vLLM binaries (fast, no CUDA toolchain). The `(?!=)` guard means `vllm==0.23.0` is NOT
  mistaken for an env assignment.
- **Custom vLLM fork (git)** — form-only affordance (URL + ref + "precompiled" checkbox) that *composes*
  `vllm_install_args` to `git+<url>@<ref>` (+ `--torch-backend=auto`); precompiled prepends
  `VLLM_USE_PRECOMPILED=1`, unchecking it builds CUDA from source. A **"Use Gemma-4 FA4 fork"** preset
  one-clicks the [vllm-gemma4-fa4-cute](https://github.com/Scicom-AI-Enterprise-Organization/vllm-gemma4-fa4-cute)
  fork: sets a **dedicated venv** (`/share/vllm-gemma4-fa4-venv` — NOT the shared marker-less
  `/share/vllm-venv`, which the worker refuses to touch so the fork would silently never install), adds
  `--attention-backend FLASH_ATTN_CUTE` to every member, and appends the 0.23.0 prometheus fix to
  `pre_script`. ⚠ Do **NOT** add a `cutlass-dsl` pin — the fork declares its own `nvidia-cutlass-dsl==4.5.2`
  (the README's "cutlass-dsl==4.4.2" is stale; the PyPI name is `nvidia-cutlass-dsl`, imports as `cutlass`),
  and any conflicting pin makes uv unsatisfiable. POD-VERIFIED on the tm H20 (precompiled git install → 13G
  venv, `import vllm`+`cutlass` OK) AND **e2e through the local gateway** (deployed proxy endpoint
  `gemma4-fa4-fork`, gemma-4-31b TP=1 on GPU 6, vLLM log `Using AttentionBackendEnum.FLASH_ATTN_CUTE`
  + `kv cache block size to 80`, served a chat completion over the gateway proxy).
- **Pre-launch script** (`pre_script`) — shell run once after the venv is ready, before launch, with
  `{venv}/bin` on PATH + VIRTUAL_ENV set. The form has **+ Install DeepGEMM** and **+ Fix vLLM 0.23.0
  prometheus** buttons. Runs under `bash` so `bash <(curl … install_deepgemm.sh)` works.

A `{venv}/.sgpu_vllm_spec` marker means: skip if the spec is unchanged, reinstall if you change it,
**never touch a venv with no marker** (hand-built ones like the tm fleet's `/share/vllm-venv`).

Two Blackwell (B300, sm_103) gotchas, both handled: flashinfer JIT-builds kernels at runtime so the
venv needs `ninja`+`cmake` AND `{venv}/bin` on PATH (`launch_member` prepends it) — else
`FileNotFoundError: ninja`. And vLLM **0.23.0** ships a `prometheus_fastapi_instrumentator` that 500s
*every* route incl. `/health` (`'_IncludedRouter' object has no attribute 'path'`) → worker never goes
healthy; workaround is a `pre_script` `uv pip install -U "prometheus-fastapi-instrumentator>=7"`. After
repeated re-provisions, free orphaned GPU memory with `POST /apps/{id}/workers/purge` (a self-exited
vLLM the per-PID cleanup misses → "No available memory for the cache blocks" on the next launch).
