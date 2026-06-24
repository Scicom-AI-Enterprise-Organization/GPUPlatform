"""Minimal stand-in for vllm.triton_utils — re-exports the real triton + triton.language,
matching what the kernel imports (`from vllm.triton_utils import tl, triton`)."""
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453

__all__ = ["triton", "tl", "LOG2E", "LOGE2"]
