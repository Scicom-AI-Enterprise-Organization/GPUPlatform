// Shared RunPod GPU catalog — used by the serverless deploy form
// (`serverless/new/inference-form.tsx`) and the Autotrain "Try it" compute picker
// (`autotrain/[runId]/tryit-compute.tsx`). Values match
// runpod_provider._GPU_NAME_MAP. vramGb feeds the dynamic capacity hint
// (recomputed with the GPU count). Catalog last reviewed: 2026-05.
export type GpuChoice = { value: string; label: string; group: string; vramGb: number };

export const GPU_CHOICES: GpuChoice[] = [
  // Datacenter — current gen
  { value: "B200", label: "B200 (180 GB)", group: "Datacenter — current", vramGb: 180 },
  { value: "H200", label: "H200 (141 GB)", group: "Datacenter — current", vramGb: 141 },
  { value: "H100", label: "H100 SXM (80 GB)", group: "Datacenter — current", vramGb: 80 },
  { value: "H100-PCIe", label: "H100 PCIe (80 GB)", group: "Datacenter — current", vramGb: 80 },
  { value: "H100-NVL", label: "H100 NVL (94 GB)", group: "Datacenter — current", vramGb: 94 },
  { value: "MI300X", label: "MI300X (192 GB)", group: "Datacenter — current", vramGb: 192 },
  // Datacenter — Ampere
  { value: "A100", label: "A100 PCIe (80 GB)", group: "Datacenter — Ampere", vramGb: 80 },
  { value: "A100-SXM", label: "A100 SXM (80 GB)", group: "Datacenter — Ampere", vramGb: 80 },
  { value: "A100-40G", label: "A100 (40 GB)", group: "Datacenter — Ampere", vramGb: 40 },
  { value: "A40", label: "A40 (48 GB)", group: "Datacenter — Ampere", vramGb: 48 },
  { value: "A10", label: "A10 (24 GB)", group: "Datacenter — Ampere", vramGb: 24 },
  // Datacenter — Ada
  { value: "L40S", label: "L40S (48 GB)", group: "Datacenter — Ada", vramGb: 48 },
  { value: "L40", label: "L40 (48 GB)", group: "Datacenter — Ada", vramGb: 48 },
  { value: "L4", label: "L4 (24 GB)", group: "Datacenter — Ada", vramGb: 24 },
  // Workstation
  { value: "RTX6000-Ada", label: "RTX 6000 Ada (48 GB)", group: "Workstation", vramGb: 48 },
  { value: "A6000", label: "RTX A6000 (48 GB)", group: "Workstation", vramGb: 48 },
  { value: "A5000", label: "RTX A5000 (24 GB)", group: "Workstation", vramGb: 24 },
  { value: "A4000", label: "RTX A4000 (16 GB)", group: "Workstation", vramGb: 16 },
  // Consumer
  { value: "RTX5090", label: "RTX 5090 (32 GB)", group: "Consumer", vramGb: 32 },
  { value: "RTX4090", label: "RTX 4090 (24 GB)", group: "Consumer", vramGb: 24 },
  { value: "RTX3090Ti", label: "RTX 3090 Ti (24 GB)", group: "Consumer", vramGb: 24 },
  { value: "RTX3090", label: "RTX 3090 (24 GB)", group: "Consumer", vramGb: 24 },
];

export const GPU_COUNT_CHOICES = [1, 2, 4, 8] as const;

// Estimate the largest model that fits given VRAM budget. Rough formula
// calibrated against vLLM defaults at moderate context (~4-8k) and small
// batches; assumes ~45% of total VRAM is consumed by KV cache, activations,
// and framework overhead, leaving ~55% for weights.
//
//   weights_budget = total_vram * 0.55
//   FP16: 2 bytes/param  → max_B = budget / 2
//   4-bit (AWQ/GPTQ ~0.6 effective): max_B = budget / 0.6
//
// Real-world capacity drops fast at long context — these are upper bounds.
export function capacityHint(vramPerGpu: number, count: number): string {
  const total = vramPerGpu * count;
  const weightsBudget = total * 0.55;
  const fp16B = weightsBudget / 2;
  const q4B = weightsBudget / 0.6;
  const fp16Str = fp16B >= 100 ? `${Math.round(fp16B / 10) * 10}B` : `${Math.round(fp16B)}B`;
  const q4Str = q4B >= 100 ? `${Math.round(q4B / 10) * 10}B` : `${Math.round(q4B)}B`;
  const totalStr = total >= 100 ? `${Math.round(total)} GB` : `${total} GB`;
  const tpHint = count === 1 ? "" : ` · TP=${count} sharding`;
  return `${totalStr} VRAM${tpHint} · fits ~${fp16Str} FP16 / ~${q4Str} 4-bit (KV-cache budgeted)`;
}

/** Best-effort map a stored gpu_type (e.g. "NVIDIA L40S", "L40S") to a catalog
 * `value` ("L40S"). Used to seed the Try-it picker from a run's training GPU.
 * Returns null when nothing matches (caller falls back to a default). */
export function gpuTypeToChoice(gpuType: string | null | undefined): string | null {
  const t = (gpuType || "").trim();
  if (!t) return null;
  const exact = GPU_CHOICES.find((c) => c.value.toLowerCase() === t.toLowerCase());
  if (exact) return exact.value;
  // "NVIDIA L40S" → match the catalog value as a whole word in the string.
  const norm = t.toLowerCase().replace(/nvidia|geforce|rtx/g, "").replace(/\s+/g, " ").trim();
  const byContains = GPU_CHOICES.find((c) => {
    const v = c.value.toLowerCase().replace(/rtx/g, "").replace(/[-\s]+/g, "");
    return v && norm.replace(/[-\s]+/g, "").includes(v);
  });
  return byContains?.value ?? null;
}
