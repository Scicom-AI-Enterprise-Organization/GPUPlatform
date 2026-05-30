import { useCallback, useEffect, useState } from "react";

// Saved stress-test runs, persisted server-side (gateway: stress_runs table,
// /apps/{id}/stress-runs) so runs / models can be compared over time and the
// comparison shared by link — anyone who can access the endpoint sees the same
// saved runs. Reached through the same-origin /api/proxy, which forwards the
// session cookie as a bearer token.

export type Stat = { mean: number; median: number; p99: number };

export type Summary = {
  successful: number;
  failed: number;
  durationS: number;
  reqThroughput: number; // req/s
  outThroughput: number; // output tok/s
  totalThroughput: number; // (in+out) tok/s
  ttft: Stat;
  tpot: Stat;
  e2e: Stat;
};

// Mirrors gateway StressRunRecord (snake_case). `created_at` is an ISO string.
export type StressRun = {
  id: string;
  app_id: string;
  created_by: string;
  model: string;
  input_len: number;
  output_len: number;
  num_prompts: number;
  concurrency: number;
  summary: Summary;
  created_at: string;
};

// What the client sends to persist a finished run (gateway StressRunCreate).
export type StressRunInput = {
  model: string;
  input_len: number;
  output_len: number;
  num_prompts: number;
  concurrency: number;
  summary: Summary;
};

async function readErr(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  try {
    const o = JSON.parse(text) as { detail?: unknown };
    if (typeof o.detail === "string") return o.detail;
  } catch {
    /* not json */
  }
  return text || res.statusText;
}

/** Server-backed per-endpoint stress-run history (newest first). */
export function useStressHistory(appId: string) {
  const [runs, setRuns] = useState<StressRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const base = `/api/proxy/apps/${encodeURIComponent(appId)}/stress-runs`;

  const reload = useCallback(async () => {
    try {
      const res = await fetch(base, { cache: "no-store" });
      if (!res.ok) throw new Error(await readErr(res));
      setRuns((await res.json()) as StressRun[]);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [base]);

  // Load on mount / appId change. setState lands only in promise callbacks (not
  // synchronously in the effect body), mirroring the useApiToken pattern.
  useEffect(() => {
    let cancelled = false;
    fetch(base, { cache: "no-store" })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readErr(res));
        const data = (await res.json()) as StressRun[];
        if (!cancelled) {
          setRuns(data);
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [base]);

  const add = useCallback(
    async (input: StressRunInput) => {
      const res = await fetch(base, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
      });
      if (!res.ok) throw new Error(await readErr(res));
      const rec = (await res.json()) as StressRun;
      setRuns((prev) => [rec, ...prev]);
      return rec;
    },
    [base],
  );

  const remove = useCallback(
    async (id: string) => {
      setRuns((prev) => prev.filter((r) => r.id !== id)); // optimistic
      try {
        await fetch(`${base}/${encodeURIComponent(id)}`, { method: "DELETE" });
      } catch {
        void reload(); // resync on failure
      }
    },
    [base, reload],
  );

  const clear = useCallback(async () => {
    setRuns([]); // optimistic
    try {
      await fetch(base, { method: "DELETE" });
    } catch {
      void reload();
    }
  }, [base, reload]);

  return { runs, loading, error, add, remove, clear };
}
