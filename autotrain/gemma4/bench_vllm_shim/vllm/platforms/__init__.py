"""Minimal stand-in for vllm.platforms.current_platform — only the methods the
triton_unified_attention kernel touches, with bodies copied from vllm's CUDA platform:

  * fp8_dtype()                  -> torch.float8_e4m3fn   (module-load: torch.finfo(...))
  * is_device_capability_family  -> (cap.to_int()//10)==(family//10)  (line 877 tuning gate)
  * device_type / fp8 attrs      -> for parity with the upstream object

On an H20 (SM 9.0) is_device_capability_family(100) is False — exactly as on a real
H20 vLLM, so the Blackwell-only `tuned_large_head` path stays disabled.
"""
import torch


class _CudaPlatform:
    device_type = "cuda"

    @classmethod
    def fp8_dtype(cls):
        return torch.float8_e4m3fn

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(device_id)
        return major * 10 + minor  # "to_int()" form used below

    @classmethod
    def is_device_capability_family(cls, capability: int, device_id: int = 0) -> bool:
        cur = cls.get_device_capability(device_id)
        if cur is None:
            return False
        return (cur // 10) == (capability // 10)


current_platform = _CudaPlatform()
