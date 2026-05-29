// Shared parsing + loading for vllm-bench-serve result.json files.
//
// Both the per-benchmark Results tab and the cross-benchmark Compare view read
// the same gateway-built aggregate `result.json` (one fetch, ~44 KB) and fall
// back to the per-config files when a run hasn't finalized yet. Keep the row
// shape + parse rules here so the two views can't drift.

import { gateway } from "@/lib/gateway";

/** One sweep cell: result.json top-level metrics + filename-extracted dims. */
export type Row = {
  filename: string;
  input_len: number;
  output_len: number;
  num_prompts: number;
  concurrency: number;
  duration_s: number | null;
  output_throughput: number | null;
  request_throughput: number | null;
  total_token_throughput: number | null;
  mean_ttft_ms: number | null;
  median_ttft_ms: number | null;
  p99_ttft_ms: number | null;
  mean_tpot_ms: number | null;
  median_tpot_ms: number | null;
  p99_tpot_ms: number | null;
  mean_itl_ms: number | null;
  median_itl_ms: number | null;
  p99_itl_ms: number | null;
  mean_e2el_ms: number | null;
  median_e2el_ms: number | null;
  p99_e2el_ms: number | null;
};

export type StatMode = "median" | "p99" | "mean";

// Monochrome series palette — same index keeps the same shade across the four
// metric charts so they read together. Inside the colour rule (no decorative
// hue, only status/availability gets colour elsewhere).
export const LINE_COLORS = [
  "#18181b", // zinc-900
  "#3f3f46", // zinc-700
  "#52525b", // zinc-600
  "#71717a", // zinc-500
  "#a1a1aa", // zinc-400
  "#27272a", // zinc-800
  "#d4d4d8", // zinc-300
  "#e4e4e7", // zinc-200
];

export function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function parseFilenameDims(name: string): {
  input_len: number;
  output_len: number;
  num_prompts: number;
  concurrency: number;
} {
  // benchmaq emits filenames like:
  //   sgpu-qwen-quick_qwen-quick_in256_out128_p50_c4_56c405.json
  const m = name.match(/_in(\d+)_out(\d+)_p(\d+)_c(\d+)/);
  return {
    input_len: m ? parseInt(m[1], 10) : 0,
    output_len: m ? parseInt(m[2], 10) : 0,
    num_prompts: m ? parseInt(m[3], 10) : 0,
    concurrency: m ? parseInt(m[4], 10) : 0,
  };
}

export function rowFromJson(filename: string, json: Record<string, unknown>): Row {
  const dims = parseFilenameDims(filename);
  return {
    filename,
    input_len: dims.input_len,
    output_len: dims.output_len,
    num_prompts: (num(json.num_prompts) ?? dims.num_prompts) || 0,
    concurrency: (num(json.max_concurrency) ?? dims.concurrency) || 0,
    duration_s: num(json.duration),
    output_throughput: num(json.output_throughput),
    request_throughput: num(json.request_throughput),
    total_token_throughput: num(json.total_token_throughput),
    mean_ttft_ms: num(json.mean_ttft_ms),
    median_ttft_ms: num(json.median_ttft_ms),
    p99_ttft_ms: num(json.p99_ttft_ms),
    mean_tpot_ms: num(json.mean_tpot_ms),
    median_tpot_ms: num(json.median_tpot_ms),
    p99_tpot_ms: num(json.p99_tpot_ms),
    mean_itl_ms: num(json.mean_itl_ms),
    median_itl_ms: num(json.median_itl_ms),
    p99_itl_ms: num(json.p99_itl_ms),
    mean_e2el_ms: num(json.mean_e2el_ms),
    median_e2el_ms: num(json.median_e2el_ms),
    p99_e2el_ms: num(json.p99_e2el_ms),
  };
}

export function statPick(
  r: Row,
  metric: "ttft" | "tpot" | "itl" | "e2el",
  mode: StatMode,
): number | null {
  const k = `${mode}_${metric}_ms` as keyof Row;
  const v = r[k];
  return typeof v === "number" ? v : null;
}

export function bestBy(
  rows: Row[],
  pick: (r: Row) => number | null,
  lower = false,
): Row | null {
  let best: Row | null = null;
  let bestV: number | null = null;
  for (const r of rows) {
    const v = pick(r);
    if (v == null) continue;
    if (bestV == null || (lower ? v < bestV : v > bestV)) {
      best = r;
      bestV = v;
    }
  }
  return best;
}

export function fmt(v: number | null, digits: number): string {
  if (v == null) return "—";
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  return v.toFixed(digits);
}

/** Load every sweep cell for a benchmark. Fast path: the gateway-built
 * aggregate `result.json` (one fetch). Falls back to the per-config files when
 * the aggregate doesn't exist yet (run still in flight). All fetches go through
 * the gateway content proxy (same-origin) — presigned S3 URLs are CORS-blocked.
 * Throws only if the file listing itself fails; individual bad files are
 * skipped. */
export async function fetchBenchRows(benchId: string): Promise<Row[]> {
  try {
    const aggR = await fetch(gateway.benchmarkFileContentUrl(benchId, "result.json"));
    if (aggR.ok) {
      const agg = (await aggR.json()) as { results?: Array<Record<string, unknown>> };
      const list = Array.isArray(agg.results) ? agg.results : [];
      if (list.length > 0) {
        return list.map((e) => rowFromJson(String(e.file ?? ""), e));
      }
    }
  } catch {
    // fall through to per-config files
  }

  const files = await gateway.listBenchmarkFiles(benchId);
  const jsonFiles = files.filter(
    (f) =>
      f.name.toLowerCase().endsWith(".json") &&
      !f.name.endsWith("_DONE") &&
      // Skip the aggregate (no _in/_out dims → bogus "in=0").
      f.name !== "result.json" &&
      !f.name.endsWith("/result.json"),
  );
  if (jsonFiles.length === 0) return [];
  const parsed = await Promise.all(
    jsonFiles.map(async (f) => {
      try {
        const r = await fetch(gateway.benchmarkFileContentUrl(benchId, f.name));
        if (!r.ok) throw new Error(`fetch ${f.name}: ${r.status}`);
        const json = (await r.json()) as Record<string, unknown>;
        return rowFromJson(f.name, json);
      } catch {
        return null;
      }
    }),
  );
  return parsed.filter((x): x is Row => x !== null);
}
