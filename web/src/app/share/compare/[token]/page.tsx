import { PublicCompareClient, type PublicComparePayload } from "./public-compare-client";

// Public, no-auth shared comparison. Lives OUTSIDE the (app) route group, so it
// renders with the bare root layout — no sidebar, no topbar. Data comes from the
// gateway's unauthenticated /benchmarks/public-compare/{token} endpoint.
const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

export const dynamic = "force-dynamic";

export default async function PublicComparePage({
  params,
}: {
  params: Promise<{ token: string }>;
}) {
  const { token } = await params;
  let payload: PublicComparePayload | null = null;
  let error: string | null = null;
  try {
    const r = await fetch(
      `${GATEWAY}/benchmarks/public-compare/${encodeURIComponent(token)}`,
      { cache: "no-store" },
    );
    if (r.ok) {
      payload = (await r.json()) as PublicComparePayload;
    } else if (r.status === 404) {
      error = "This share link doesn't exist or was removed.";
    } else {
      error = `Failed to load this comparison (${r.status}).`;
    }
  } catch {
    error = "Couldn't reach the server.";
  }

  if (!payload) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-background px-4 text-center">
        <div className="max-w-md">
          <h1 className="text-lg font-semibold">Comparison unavailable</h1>
          <p className="mt-2 text-sm text-muted-foreground">{error ?? "Not found."}</p>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <PublicCompareClient payload={payload} />
      </div>
    </main>
  );
}
