"use client";

import { CompareView, benchDataFromPublic } from "../../../(app)/benchmark/compare/compare-view";

export type PublicComparePayload = {
  token: string;
  notes?: string;
  // Frozen accuracy→speed pairing captured when the link was minted.
  pairing?: Record<string, string>;
  benchmarks: Array<{
    id: string;
    name?: string;
    status?: string;
    config_yaml?: string;
    result_json?: Record<string, unknown> | null;
    result_rows?: Array<Record<string, unknown>>;
  }>;
};

/** Renders a shared comparison from the public (no-auth) payload — converts each
 * inlined record into BenchData and hands it to CompareView in public mode (no
 * fetching, no in-app chrome / share button). */
export function PublicCompareClient({ payload }: { payload: PublicComparePayload }) {
  const benches = (payload.benchmarks ?? []).map(benchDataFromPublic);
  const ids = (payload.benchmarks ?? []).map((b) => b.id);
  return (
    <CompareView
      ids={ids}
      initialBenches={benches}
      initialNotes={payload.notes ?? ""}
      initialPairing={payload.pairing}
      publicMode
    />
  );
}
