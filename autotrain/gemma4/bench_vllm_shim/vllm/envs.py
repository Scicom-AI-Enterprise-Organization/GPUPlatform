"""Minimal stand-in for vllm.envs — only the symbols triton_unified_attention reads.

Faithful to upstream defaults: VLLM_BATCH_INVARIANT is off unless the env var is set,
exactly as vllm/envs.py resolves it.
"""
import os

VLLM_BATCH_INVARIANT = bool(int(os.environ.get("VLLM_BATCH_INVARIANT", "0")))
