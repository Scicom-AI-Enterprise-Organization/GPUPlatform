"use client";

import { useEffect, useMemo, useState } from "react";
import yaml from "js-yaml";
import {
  AlertCircle,
  Check,
  ChevronRight,
  Copy,
  Cpu,
  FileCode2,
  Gauge,
  Server,
  Settings2,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { gateway } from "@/lib/gateway";
import type { BenchmarkRecord, ProviderRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

/** A loose shape for the parsed benchmaq runpod-mode YAML — every field is
 * optional because users can drop into YAML mode and remove or rename keys. */
type Parsed = {
  runpod?: {
    pod?: {
      name?: string;
      gpu_type?: string;
      gpu_count?: number;
      instance_type?: string;
      secure_cloud?: boolean;
    };
    container?: { image?: string; disk_size?: number };
    storage?: { volume_size?: number; mount_path?: string };
    ports?: Record<string, unknown>;
    env?: Record<string, unknown>;
  };
  remote?: {
    host?: string;
    port?: number;
    username?: string;
    key_filename?: string;
    uv?: { path?: string; python_version?: string };
    dependencies?: string[];
  };
  benchmark?: Array<{
    name?: string;
    engine?: string;
    model?: { repo_id?: string; local_dir?: string };
    serve?: Record<string, unknown>;
    bench?: Array<Record<string, unknown>>;
    results?: Record<string, unknown>;
  }>;
};

export function ParametersTab({ bench }: { bench: BenchmarkRecord }) {
  const [parseError, setParseError] = useState<string | null>(null);
  // For VM runs the submitted YAML doesn't contain host/port/user — those
  // are injected by the gateway at run time. Resolve them by looking up the
  // provider record on mount. Falls back gracefully if the provider has been
  // deleted since the bench ran.
  const [provider, setProvider] = useState<ProviderRecord | null>(null);
  useEffect(() => {
    if (!bench.provider_id) return;
    gateway
      .listProviders()
      .then((rows) => {
        const hit = rows.find((p) => p.id === bench.provider_id) ?? null;
        setProvider(hit);
      })
      .catch(() => setProvider(null));
  }, [bench.provider_id]);
  const parsed = useMemo<Parsed | null>(() => {
    try {
      const v = yaml.load(bench.config_yaml);
      setParseError(null);
      return (v && typeof v === "object" ? (v as Parsed) : null);
    } catch (e) {
      setParseError(e instanceof Error ? e.message : String(e));
      return null;
    }
  }, [bench.config_yaml]);

  if (parseError) {
    return (
      <div className="space-y-3">
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <AlertCircle className="mr-2 inline h-4 w-4" />
          Couldn&apos;t parse config: {parseError}
        </div>
        <RawYamlBlock yaml={bench.config_yaml} />
      </div>
    );
  }

  if (!parsed) {
    return <RawYamlBlock yaml={bench.config_yaml} />;
  }

  const pod = parsed.runpod?.pod ?? {};
  const container = parsed.runpod?.container ?? {};
  const storage = parsed.runpod?.storage ?? {};
  const env = parsed.runpod?.env ?? {};
  const benches = parsed.benchmark ?? [];
  // Almost always 1 benchmark[] item (we don't expose multi-config in the form).
  const first = benches[0] ?? {};
  const serve = (first.serve ?? {}) as Record<string, unknown>;
  const benchEntries = (first.bench ?? []) as Array<Record<string, unknown>>;
  const totalRuns = benchEntries.length;

  // Sweep dimensions — extract unique values across bench[] for input/output/concurrency.
  const inputLens = uniqueNums(benchEntries, "random_input_len");
  const outputLens = uniqueNums(benchEntries, "random_output_len");
  const concurrencies = uniqueNums(benchEntries, "max_concurrency");
  const isSweep = totalRuns > 1;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Parameters</h2>
        <p className="text-xs text-muted-foreground">
          Captured at submit time. The config below is the YAML benchmaq actually ran.
        </p>
      </div>

      {bench.provider_id && provider?.kind === "vm" ? (
        <ParamsCard
          icon={<Server className="h-4 w-4" />}
          title="Pod (bare metal)"
          description="benchmaq ran directly on a registered VM via SSH — no pod was spawned."
          action={
            <Badge variant="secondary" className="font-mono text-[10px]">
              VM
            </Badge>
          }
        >
          <KvGrid>
            <Kv
              label="Provider"
              value={provider?.name ?? bench.provider_id}
              mono
              wide
            />
            <Kv
              label="Host"
              value={provider?.host ?? parsed.remote?.host}
              mono
              wide
            />
            <Kv
              label="SSH user"
              value={provider?.user ?? parsed.remote?.username}
              mono
            />
            <Kv
              label="SSH port"
              value={provider?.port ?? parsed.remote?.port}
            />
            <Kv
              label="GPUs (last probed)"
              value={
                provider?.gpu_count != null && provider.gpu_count > 0
                  ? `${(provider.gpus ?? []).slice(0, 1).join("").replace(/^NVIDIA\s+/i, "") || "GPU"}${provider.gpu_count > 1 ? ` × ${provider.gpu_count}` : ""}`
                  : undefined
              }
              mono
              wide
            />
          </KvGrid>
        </ParamsCard>
      ) : (
        <ParamsCard
          icon={<Server className="h-4 w-4" />}
          title="Pod"
          description={
            bench.provider_id
              ? `What benchmaq spawned on RunPod (account: ${provider?.name ?? bench.provider_id}).`
              : "What benchmaq spawned on RunPod."
          }
          action={
            bench.provider_id && provider?.kind === "runpod" ? (
              <Badge variant="secondary" className="font-mono text-[10px]">
                RunPod
              </Badge>
            ) : undefined
          }
        >
          <KvGrid>
            {bench.provider_id && (
              <Kv
                label="Account (API key)"
                value={
                  provider
                    ? `${provider.name}${provider.api_key_last4 ? ` · ****${provider.api_key_last4}` : ""}`
                    : bench.provider_id
                }
                mono
                wide
              />
            )}
            <Kv label="GPU type" value={pod.gpu_type} mono />
            <Kv label="GPU count" value={pod.gpu_count} />
            <Kv
              label="Cloud"
              value={pod.secure_cloud ? "Secure" : "Community"}
            />
            <Kv label="Disk" value={container.disk_size ? `${container.disk_size} GB` : undefined} />
            <Kv label="Volume" value={storage.volume_size ? `${storage.volume_size} GB` : undefined} />
            <Kv label="Pod name" value={pod.name} mono />
          </KvGrid>
          <Detail label="Container image">
            <code className="font-mono text-xs">{container.image ?? "—"}</code>
          </Detail>
          {Object.keys(env).length > 0 && (
            <Detail label="Pod env">
              <div className="flex flex-wrap gap-1">
                {Object.entries(env).map(([k, v]) => (
                  <Badge key={k} variant="secondary" className="font-mono text-[10px]">
                    {k}={String(v)}
                  </Badge>
                ))}
              </div>
            </Detail>
          )}
        </ParamsCard>
      )}

      <ParamsCard
        icon={<Cpu className="h-4 w-4" />}
        title="Model"
        description="Model + vLLM serve config."
      >
        <KvGrid>
          <Kv label="Model" value={first.model?.repo_id} mono wide />
          <Kv label="Local dir" value={first.model?.local_dir} mono wide />
          <Kv label="Engine" value={first.engine} />
        </KvGrid>
        {Object.keys(serve).length > 0 && (
          <Detail label="vLLM engine args">
            <div className="rounded-md bg-muted/50 px-3 py-2">
              <pre className="overflow-x-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-foreground">
                {Object.entries(serve)
                  .map(([k, v]) => `${k}: ${formatValue(v)}`)
                  .join("\n")}
              </pre>
            </div>
          </Detail>
        )}
      </ParamsCard>

      <ParamsCard
        icon={<Gauge className="h-4 w-4" />}
        title="Workload"
        description={
          isSweep
            ? `Sweep — ${totalRuns} bench runs across ${inputLens.length} input length${
                inputLens.length === 1 ? "" : "s"
              } × ${concurrencies.length} concurrenc${
                concurrencies.length === 1 ? "y" : "ies"
              }.`
            : "Single bench run."
        }
        action={
          <Badge variant={isSweep ? "default" : "secondary"} className="font-mono text-[10px]">
            {totalRuns} run{totalRuns === 1 ? "" : "s"}
          </Badge>
        }
      >
        {isSweep ? (
          <>
            <KvGrid>
              <Kv
                label="Input lengths"
                value={inputLens.length ? inputLens.join(", ") : undefined}
                mono
                wide
              />
              <Kv
                label="Output lengths"
                value={outputLens.length ? outputLens.join(", ") : undefined}
                mono
              />
              <Kv
                label="Concurrencies"
                value={concurrencies.length ? concurrencies.join(", ") : undefined}
                mono
                wide
              />
            </KvGrid>
            <Detail label="All bench runs">
              <div className="overflow-hidden rounded-md border border-border">
                <table className="w-full text-sm">
                  <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
                    <tr>
                      <th className="px-3 py-1.5 text-left">#</th>
                      <th className="px-3 py-1.5 text-right">input</th>
                      <th className="px-3 py-1.5 text-right">output</th>
                      <th className="px-3 py-1.5 text-right">prompts</th>
                      <th className="px-3 py-1.5 text-right">concurrency</th>
                      <th className="px-3 py-1.5 text-right">rate</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {benchEntries.map((b, i) => (
                      <tr key={i}>
                        <td className="px-3 py-1.5 font-mono text-xs text-muted-foreground">{i}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-xs">
                          {(b.random_input_len as number) ?? "—"}
                        </td>
                        <td className="px-3 py-1.5 text-right font-mono text-xs">
                          {(b.random_output_len as number) ?? "—"}
                        </td>
                        <td className="px-3 py-1.5 text-right font-mono text-xs">
                          {(b.num_prompts as number) ?? "—"}
                        </td>
                        <td className="px-3 py-1.5 text-right font-mono text-xs">
                          {(b.max_concurrency as number) ?? "—"}
                        </td>
                        <td className="px-3 py-1.5 text-right font-mono text-xs">
                          {String(b.request_rate ?? "inf")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Detail>
          </>
        ) : (
          <KvGrid>
            <Kv label="Input length" value={(benchEntries[0]?.random_input_len as number) ?? undefined} />
            <Kv label="Output length" value={(benchEntries[0]?.random_output_len as number) ?? undefined} />
            <Kv label="Num prompts" value={(benchEntries[0]?.num_prompts as number) ?? undefined} />
            <Kv label="Max concurrency" value={(benchEntries[0]?.max_concurrency as number) ?? undefined} />
            <Kv
              label="Request rate"
              value={String((benchEntries[0]?.request_rate as unknown) ?? "inf")}
              mono
            />
            <Kv
              label="Endpoint"
              value={(benchEntries[0]?.endpoint as string) ?? undefined}
              mono
              wide
            />
          </KvGrid>
        )}
      </ParamsCard>

      <ParamsCard
        icon={<Settings2 className="h-4 w-4" />}
        title="Remote setup"
        description="Python env + dependencies installed on the pod by benchmaq."
      >
        <KvGrid>
          <Kv label="Python" value={parsed.remote?.uv?.python_version} />
          <Kv label="venv path" value={parsed.remote?.uv?.path} mono wide />
        </KvGrid>
        {parsed.remote?.dependencies && parsed.remote.dependencies.length > 0 && (
          <Detail label="Dependencies">
            <div className="flex flex-wrap gap-1">
              {parsed.remote.dependencies.map((d) => (
                <Badge key={d} variant="secondary" className="font-mono text-[10px]">
                  {d}
                </Badge>
              ))}
            </div>
          </Detail>
        )}
      </ParamsCard>

      <RawYamlBlock yaml={bench.config_yaml} />
    </div>
  );
}

function uniqueNums(rows: Array<Record<string, unknown>>, key: string): number[] {
  const set = new Set<number>();
  for (const r of rows) {
    const v = r[key];
    if (typeof v === "number" && Number.isFinite(v)) set.add(v);
  }
  return Array.from(set).sort((a, b) => a - b);
}

function formatValue(v: unknown): string {
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  if (v === null || v === undefined) return "null";
  return JSON.stringify(v);
}

function ParamsCard({
  icon,
  title,
  description,
  action,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted text-muted-foreground">
              {icon}
            </div>
            <div>
              <CardTitle className="text-sm">{title}</CardTitle>
              {description && (
                <CardDescription className="text-xs">{description}</CardDescription>
              )}
            </div>
          </div>
          {action}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">{children}</CardContent>
    </Card>
  );
}

function KvGrid({ children }: { children: React.ReactNode }) {
  return (
    <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2 lg:grid-cols-4">
      {children}
    </dl>
  );
}

function Kv({
  label,
  value,
  mono,
  wide,
}: {
  label: string;
  value: string | number | undefined;
  mono?: boolean;
  wide?: boolean;
}) {
  const display =
    value === undefined || value === null || value === "" ? "—" : String(value);
  return (
    <div className={cn(wide ? "sm:col-span-2" : "")}>
      <dt className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </dt>
      <dd
        className={cn(
          "mt-0.5 truncate text-sm",
          mono && "font-mono text-xs",
          display === "—" && "text-muted-foreground",
        )}
        title={display}
      >
        {display}
      </dd>
    </div>
  );
}

function Detail({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      {children}
    </div>
  );
}

function RawYamlBlock({ yaml: src }: { yaml: string }) {
  const [copied, setCopied] = useState(false);
  function copy() {
    navigator.clipboard.writeText(src).then(() => {
      setCopied(true);
      toast.success("YAML copied", { duration: 3000 });
      setTimeout(() => setCopied(false), 1500);
    });
  }
  return (
    <details className="group rounded-lg border border-border">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-4 py-3 text-sm font-medium hover:bg-muted/40 [&::-webkit-details-marker]:hidden">
        <div className="flex items-center gap-2">
          <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-90" />
          <FileCode2 className="h-4 w-4 text-muted-foreground" />
          Raw YAML
          <Badge variant="secondary" className="text-[10px]">
            as submitted
          </Badge>
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={(e) => {
            e.preventDefault();
            copy();
          }}
        >
          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          {copied ? "Copied" : "Copy"}
        </Button>
      </summary>
      <pre className="max-h-[60vh] overflow-auto rounded-b-lg border-t border-border bg-muted/40 px-4 py-3 font-mono text-xs leading-relaxed text-foreground">
        {src}
      </pre>
    </details>
  );
}
