"""Minimal stand-in for vllm.v1.kv_cache_interface.KVQuantMode — the IntEnum the kernel
branches on. Values copied verbatim from vllm/v1/kv_cache_interface.py."""
from enum import IntEnum


class KVQuantMode(IntEnum):
    NONE = 0
    FP8_PER_TENSOR = 1  # per-tensor scales (current fp8 path)
    INT8_PER_TOKEN_HEAD = 2  # per-token-head dynamic scales for int8
    FP8_PER_TOKEN_HEAD = 3  # per-token-head dynamic scales for fp8
