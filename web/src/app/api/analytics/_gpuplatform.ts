import { cookies } from "next/headers";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";

export const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";
export const PAGE = 1000;
export const MAX_PAGES = 5;

export type JobRecord = Record<string, unknown>;

export const GPUPLATFORM_ALL_KINDS: { kind: string; path: string }[] = [
  { kind: "benchmark", path: "benchmarks" },
  { kind: "training", path: "training" },
  { kind: "compute", path: "compute" },
  { kind: "inference", path: "inference" },
  { kind: "endpoint", path: "endpoints" },
];

export const GPUPLATFORM_OVERVIEW_KINDS = GPUPLATFORM_ALL_KINDS.filter(
  ({ kind }) => kind !== "inference",
);

export async function getGatewayToken() {
  const jar = await cookies();
  return jar.get(TOKEN_COOKIE)?.value ?? "";
}

export async function fetchKind(
  token: string,
  path: string,
  since: string,
  until: string,
): Promise<{ jobs: JobRecord[]; truncated: boolean }> {
  const jobs: JobRecord[] = [];
  for (let page = 0; page < MAX_PAGES; page++) {
    const qs = new URLSearchParams({
      since,
      until,
      order: "desc",
      limit: String(PAGE),
      offset: String(page * PAGE),
    });
    const r = await fetch(`${GATEWAY}/v1/history/${path}?${qs}`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!r.ok) return { jobs, truncated: false };
    const body = (await r.json()) as { jobs: JobRecord[]; has_more: boolean };
    jobs.push(...body.jobs);
    if (!body.has_more) return { jobs, truncated: false };
  }
  return { jobs, truncated: true };
}

export async function fetchKinds(
  token: string,
  kinds: { kind: string; path: string }[],
  since: string,
  until: string,
) {
  const results = await Promise.all(
    kinds.map(async ({ kind, path }) => ({ kind, ...(await fetchKind(token, path, since, until)) })),
  );
  const payload: Record<string, JobRecord[]> = {};
  const truncated: string[] = [];
  for (const result of results) {
    payload[result.kind] = result.jobs;
    if (result.truncated) truncated.push(result.kind);
  }
  return { kinds: payload, truncated };
}

export async function fetchInferenceSummary(
  token: string,
  since: string,
  until: string,
  tz: string,
) {
  return fetch(
    `${GATEWAY}/v1/history/summary?${new URLSearchParams({ since, until, tz })}`,
    { headers: { Authorization: `Bearer ${token}` }, cache: "no-store" },
  )
    .then(async (r) => (r.ok ? ((await r.json()) as { rows: unknown[] }).rows : null))
    .catch(() => null);
}
