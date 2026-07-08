// GPU-type label suggestions for benchmark hardware fields on runs where the
// platform can't detect the GPU itself (ingress / Slurm — nothing is spawned).
// Full RunPod-style names so manually-stated runs group with pod-run benches in
// stats / aggregate and for external consumers (the GPU calculator).
//
// Shared by the create form (`benchmark/new/benchmark-form.tsx`) and the
// post-run editor (`benchmark/[id]/tabs/parameters.tsx`) so both offer the
// exact same list.
export const GPU_TYPE_SUGGESTIONS = [
  "NVIDIA H20",
  "NVIDIA H100 80GB HBM3",
  "NVIDIA H200",
  "NVIDIA B200",
  "NVIDIA B300",
  "NVIDIA A100 80GB PCIe",
  "NVIDIA A100-SXM4-80GB",
  "NVIDIA L40S",
  "NVIDIA L4",
  "NVIDIA RTX A6000",
  "NVIDIA GeForce RTX 4090",
  "NVIDIA GeForce RTX 5090",
  "AMD Instinct MI300X",
  "Ascend 910B3",
] as const;
