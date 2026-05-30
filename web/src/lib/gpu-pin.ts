// Helpers for per-model GPU pinning on multi-model VM fleets. Shared by the
// create form (serverless/new) and the Overview per-model editor. The gateway
// re-validates everything authoritatively (gateway/main.py
// _normalize_member_gpu_indices); these just give fast client-side feedback +
// suggested layouts that mirror the gateway's auto-packer.

/** Parse a "0,1,2,3,4,5,6" visible_devices string into [0,1,2,3,4,5,6]. */
export function parsePhys(visibleDevices: string): number[] {
  return visibleDevices
    .split(",")
    .map((x) => x.trim())
    .filter((x) => /^\d+$/.test(x))
    .map(Number);
}

/** Parse a per-model "0,1,2,3" pin. Returns null when blank (→ auto-assign), or
 *  throws Error(friendly message) on non-numeric / wrong-count / duplicate ids.
 *  `tp` is the model's tensor-parallel size; the pin must name exactly tp ids. */
export function parseGpuIds(raw: string, tp: number, label: string): number[] | null {
  const parts = raw.split(",").map((x) => x.trim()).filter(Boolean);
  if (!parts.length) return null;
  if (parts.some((x) => !/^\d+$/.test(x))) {
    throw new Error(`${label}: GPU ids must be numbers, e.g. 0,1,2,3`);
  }
  const ids = parts.map(Number);
  if (ids.length !== tp) {
    throw new Error(`${label}: ${ids.length} GPU id(s) but TP=${tp} — they must match`);
  }
  if (new Set(ids).size !== ids.length) {
    throw new Error(`${label}: duplicate GPU id in ${raw}`);
  }
  return ids;
}

/** Mirror the gateway's auto-packer: pack each model's `tp` GPUs round-robin
 *  into tp-wide slots carved from `phys`. Returns one "0,1,2,3" suggestion per
 *  tp (treating every model as auto, so it's a stable hint independent of any
 *  overrides the user has typed). Empty string when there are no GPUs to pack. */
export function suggestPacking(tps: number[], phys: number[]): string[] {
  const universe = phys.length;
  const cursor: Record<number, number> = {};
  return tps.map((rawTp) => {
    const tp = Math.max(1, rawTp);
    if (!universe) return "";
    const nSlots = Math.max(1, Math.floor(universe / tp));
    const si = (cursor[tp] ?? 0) % nSlots;
    cursor[tp] = si + 1;
    return phys.slice(si * tp, si * tp + tp).join(",");
  });
}
