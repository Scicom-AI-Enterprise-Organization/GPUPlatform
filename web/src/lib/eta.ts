// ETA estimation for long-running dataset transformations (TTS pack, audio-zip
// transform). Both stream a log whose progress lines look like:
//   [AUTOTRAIN_PROGRESS] step=<name> processed=<N> total=<M> percent=<P>
// We parse the latest marker and estimate time-to-completion by sampling how
// fast `processed` grows across the card's poll interval. No backend timestamp
// plumbing needed — the rate is measured client-side and reset per step.
import { useEffect, useRef, useState } from "react";

export type ProgressMarker = {
  step: string | null;
  processed: number | null;
  total: number | null;
  percent: number | null;
};

const MARKER_RE = /\[AUTOTRAIN_PROGRESS\]([^\n]*)/g;

/** Parse the LAST `[AUTOTRAIN_PROGRESS]` marker from a log tail (markers
 * accumulate; the newest reflects current progress). Returns null when none. */
export function parseAutotrainProgress(log: string | null | undefined): ProgressMarker | null {
  if (!log) return null;
  let last: string | null = null;
  for (const m of log.matchAll(MARKER_RE)) last = m[1];
  if (last === null) return null;
  const kv: Record<string, string> = {};
  for (const tok of last.trim().split(/\s+/)) {
    const eq = tok.indexOf("=");
    if (eq > 0) kv[tok.slice(0, eq)] = tok.slice(eq + 1);
  }
  const num = (v: string | undefined): number | null => {
    if (v === undefined) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };
  const processed = num(kv.processed);
  const total = num(kv.total);
  let percent = num(kv.percent);
  if (percent === null && processed !== null && total) percent = (processed / total) * 100;
  if (processed === null && total === null && percent === null) return null;
  return { step: kv.step ?? null, processed, total, percent };
}

const WINDOW_MS = 90_000; // smoothing window for the rate estimate
const MAX_SAMPLES = 16;

/** Estimate seconds-to-completion from a stream of progress markers. Samples
 * the (time, processed) pairs the parent feeds on each poll; resets the window
 * when the step changes or the counter restarts. Returns null until there's
 * enough signal (≥2 samples with forward progress). */
export function useEta(marker: ProgressMarker | null, running: boolean): number | null {
  const samples = useRef<{ t: number; v: number }[]>([]);
  const keyRef = useRef<string>("");
  const [eta, setEta] = useState<number | null>(null);

  const processed = marker?.processed ?? null;
  const total = marker?.total ?? null;
  // Prefer absolute processed/total; fall back to percent toward 100.
  const usingProcessed = processed !== null && total !== null && total > 0;
  const value = usingProcessed ? processed : marker?.percent ?? null;
  const target = usingProcessed ? (total as number) : 100;
  const stepKey = `${marker?.step ?? ""}|${target}`;

  useEffect(() => {
    let next: number | null = null;
    if (running && value !== null) {
      if (keyRef.current !== stepKey) {
        // New step → restart the rate window (so a step boundary doesn't blend).
        keyRef.current = stepKey;
        samples.current = [];
      }
      const now = Date.now();
      const arr = samples.current;
      if (arr.length && value < arr[arr.length - 1].v) arr.length = 0; // counter restarted
      arr.push({ t: now, v: value });
      while (arr.length > 2 && now - arr[0].t > WINDOW_MS) arr.shift();
      if (arr.length > MAX_SAMPLES) arr.splice(0, arr.length - MAX_SAMPLES);
      if (arr.length >= 2) {
        const first = arr[0];
        const lastS = arr[arr.length - 1];
        const dv = lastS.v - first.v;
        const dt = (lastS.t - first.t) / 1000;
        if (dv > 0 && dt > 0) next = Math.max(0, target - value) / (dv / dt);
      }
    } else {
      samples.current = [];
      keyRef.current = "";
    }
    // The effect runs at most once per parent poll (~3-4s) and only when the
    // sampled value/step changes, so publishing the fresh estimate here can't loop.
    setEta(next);
  }, [value, stepKey, running, target]);

  return eta;
}

/** "~45s" / "~3m 20s" / "~1h 5m". null for missing/invalid input. */
export function formatEta(seconds: number | null): string | null {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return null;
  const s = Math.round(seconds);
  if (s < 60) return `~${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  if (m < 60) return rs ? `~${m}m ${rs}s` : `~${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm ? `~${h}h ${rm}m` : `~${h}h`;
}

const STEP_LABELS: Record<string, string> = {
  download: "downloading",
  convert_neucodec: "encoding",
  pack_stage1: "packing",
  tts_eval_gen: "evaluating",
  upload_s3: "uploading",
};

/** Human label for a progress step name (falls back to the raw name). */
export function prettyStep(step: string | null | undefined): string | null {
  if (!step) return null;
  return STEP_LABELS[step] ?? step;
}
