import { redirect } from "next/navigation";
import { cookies } from "next/headers";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";
import { getMe } from "@/lib/me";
import { ConsoleTopbar } from "@/components/console/topbar";
import type { GlobalEnvRecord, TrackingCredentialRecord } from "@/lib/types";
import { SecretsManager } from "./secrets-manager";
import { TrackingCredentialsManager } from "./tracking-credentials-manager";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

async function loadJson<T>(path: string, token: string, fallback: T): Promise<T> {
  try {
    const r = await fetch(`${GATEWAY}${path}`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!r.ok) return fallback;
    return (await r.json()) as T;
  } catch {
    return fallback;
  }
}

export default async function SecretsPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");

  const jar = await cookies();
  const token = jar.get(TOKEN_COOKIE)?.value ?? "";
  const [entries, trackingCreds] = await Promise.all([
    loadJson<GlobalEnvRecord[]>("/v1/global-env", token, []),
    loadJson<TrackingCredentialRecord[]>("/v1/tracking-credentials", token, []),
  ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Secrets" }]} username={me.username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <header className="mb-8">
          <h1 className="text-2xl font-semibold tracking-tight">Global env &amp; secrets</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Org-wide environment variables (e.g. <span className="font-mono">HF_TOKEN</span>) injected into every
            benchmark run and serverless worker. A per-benchmark / per-endpoint variable of the same name overrides
            the global one. Secret values are encrypted at rest and never shown again after you save them.
          </p>
        </header>

        <SecretsManager initial={entries} />
        <TrackingCredentialsManager initial={trackingCreds} />
      </div>
    </div>
  );
}
