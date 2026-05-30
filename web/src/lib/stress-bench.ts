// Isomorphic load generator shared by the stress tab. It runs identically in
// the browser (driving the same-origin /api/proxy) and on the Next.js server
// (driving the gateway directly with a bearer token) — only the target URL and
// headers differ. Server-side it escapes the browser's ~6-connection-per-host
// cap, so concurrency is real. Pure fetch/performance, no React or Node APIs.

import type { Stat, Summary } from "./stress-history";

export type StressConfig = {
  model: string;
  inputLen: number;
  outputLen: number;
  numPrompts: number;
  concurrency: number;
};

export type ReqResult = {
  ok: boolean;
  ttftMs: number; // time to first token
  e2eMs: number; // end to end
  tpotMs: number; // mean time per output token
  outTokens: number;
  promptTokens: number;
  error?: string;
};

export function perfNow(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function deltaContent(chunk: Record<string, unknown>): string {
  const choices = chunk.choices as Array<{ delta?: { content?: unknown } }> | undefined;
  const c = choices?.[0]?.delta?.content;
  return typeof c === "string" ? c : "";
}

function usageTokens(chunk: Record<string, unknown>): { out: number | null; prompt: number | null } {
  const u = (chunk.usage ?? null) as { completion_tokens?: unknown; prompt_tokens?: unknown } | null;
  const out = typeof u?.completion_tokens === "number" ? u.completion_tokens : null;
  const prompt = typeof u?.prompt_tokens === "number" ? u.prompt_tokens : null;
  return { out, prompt };
}

function pct(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.ceil((p / 100) * s.length) - 1));
  return s[i];
}
const mean = (arr: number[]) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
const statOf = (arr: number[]): Stat => ({ mean: mean(arr), median: pct(arr, 50), p99: pct(arr, 99) });

// ~1 token per "word ", good enough to size an input prompt for load testing.
function makePrompt(inputLen: number): string {
  const n = Math.max(1, inputLen);
  return "word ".repeat(n).trim();
}

/** One streaming chat-completion against `streamUrl`, timing TTFT / E2E / tokens. */
export async function oneRequest(
  streamUrl: string,
  headers: Record<string, string>,
  cfg: StressConfig,
  signal: AbortSignal,
): Promise<ReqResult> {
  const t0 = perfNow();
  let tFirst: number | null = null;
  let counted = 0;
  let usageOut: number | null = null;
  let promptTokens = 0;
  const body: Record<string, unknown> = {
    endpoint: "/v1/chat/completions",
    messages: [{ role: "user", content: makePrompt(cfg.inputLen) }],
    max_tokens: cfg.outputLen,
    stream_options: { include_usage: true },
  };
  if (cfg.model) body.model = cfg.model;
  try {
    const res = await fetch(streamUrl, {
      method: "POST",
      headers: { ...headers, "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok || !res.body) {
      const txt = await res.text().catch(() => "");
      throw new Error(txt || res.statusText);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done: rdone } = await reader.read();
      if (rdone) break;
      buf += decoder.decode(value, { stream: true });
      const frames = buf.split("\n\n");
      buf = frames.pop() ?? "";
      for (const frame of frames) {
        for (const lineRaw of frame.split("\n")) {
          const line = lineRaw.trimStart();
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (!data || data === "[DONE]") continue;
          let chunk: Record<string, unknown>;
          try {
            chunk = JSON.parse(data);
          } catch {
            continue;
          }
          if (chunk.error) throw new Error(String(chunk.error));
          const u = usageTokens(chunk);
          if (u.out != null) usageOut = u.out;
          if (u.prompt != null) promptTokens = u.prompt;
          if (deltaContent(chunk)) {
            if (tFirst === null) tFirst = perfNow();
            counted += 1;
          }
        }
      }
    }
    const tEnd = perfNow();
    const out = usageOut ?? counted;
    const ttft = (tFirst ?? tEnd) - t0;
    const e2e = tEnd - t0;
    const gen = tEnd - (tFirst ?? tEnd);
    return {
      ok: true,
      ttftMs: ttft,
      e2eMs: e2e,
      tpotMs: out > 1 ? gen / (out - 1) : 0,
      outTokens: out,
      promptTokens,
    };
  } catch (e) {
    return {
      ok: false,
      ttftMs: 0,
      e2eMs: 0,
      tpotMs: 0,
      outTokens: 0,
      promptTokens: 0,
      error: e instanceof Error ? e.message : String(e),
    };
  }
}

/** A right-aligned request-log line, identical in the browser and server runs. */
export function formatLine(seq: number, r: ReqResult): string {
  const n = String(seq).padStart(4);
  return r.ok
    ? `#${n}  ok    ttft=${r.ttftMs.toFixed(0).padStart(5)}ms  e2e=${r.e2eMs.toFixed(0).padStart(6)}ms  out=${r.outTokens}tok  tpot=${r.tpotMs.toFixed(1)}ms`
    : `#${n}  FAIL  ${r.error ?? "error"}`;
}

function summarize(results: ReqResult[], durationS: number): Summary {
  const ok = results.filter((r) => r.ok);
  const totalOut = ok.reduce((a, r) => a + r.outTokens, 0);
  const totalTok = ok.reduce((a, r) => a + r.outTokens + r.promptTokens, 0);
  return {
    successful: ok.length,
    failed: results.length - ok.length,
    durationS,
    reqThroughput: durationS > 0 ? ok.length / durationS : 0,
    outThroughput: durationS > 0 ? totalOut / durationS : 0,
    totalThroughput: durationS > 0 ? totalTok / durationS : 0,
    ttft: statOf(ok.map((r) => r.ttftMs)),
    tpot: statOf(ok.map((r) => r.tpotMs).filter((x) => x > 0)),
    e2e: statOf(ok.map((r) => r.e2eMs)),
  };
}

/** Run the worker-pool load test: fire `numPrompts` requests at `concurrency`,
 *  calling onResult after each. Returns the metric summary + the first error
 *  (handy for an "all requests failed" message). Honors `signal` for stop. */
export async function runStressBench(
  streamUrl: string,
  headers: Record<string, string>,
  cfg: StressConfig,
  opts: { signal: AbortSignal; onResult?: (r: ReqResult, done: number) => void },
): Promise<{ summary: Summary; firstError?: string }> {
  const total = Math.max(1, cfg.numPrompts);
  const conc = Math.max(1, Math.min(cfg.concurrency, total));
  const results: ReqResult[] = [];
  let launched = 0;
  let completed = 0;
  const start = perfNow();

  const worker = async () => {
    while (launched < total && !opts.signal.aborted) {
      launched += 1;
      const r = await oneRequest(streamUrl, headers, cfg, opts.signal);
      results.push(r);
      completed += 1;
      opts.onResult?.(r, completed);
    }
  };
  await Promise.all(Array.from({ length: conc }, () => worker()));

  const durationS = (perfNow() - start) / 1000;
  return { summary: summarize(results, durationS), firstError: results.find((r) => !r.ok)?.error };
}
