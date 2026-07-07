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
  Pencil,
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
  // Ingress runs (bench an already-served endpoint — no pod spawned) put
  // base_url either top-level or on a benchmark[] item.
  base_url?: string;
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
    base_url?: string;
    model?: { repo_id?: string; local_dir?: string };
    serve?: Record<string, unknown>;
    bench?: Array<Record<string, unknown>>;
    accuracy?: {
      datasets?: unknown[];
      limit?: number;
      concurrency?: number;
      languages?: string[];
      max_tokens?: number;
    };
    results?: Record<string, unknown>;
  }>;
};

// Datalist suggestions for the manual GPU-type editor — dropdown + free text in
// one control. Full RunPod-style names so manually-tagged ingress runs group
// with pod-run benches in stats/aggregate and external consumers.
const GPU_TYPE_SUGGESTIONS = [
  "NVIDIA H20",
  "NVIDIA H100 80GB HBM3",
  "NVIDIA H200",
  "NVIDIA B200",
  "NVIDIA B300",
  "NVIDIA A100 80GB PCIe",
  "NVIDIA A100-SXM4-80GB",
  "NVIDIA L40S",
  "NVIDIA L4",
  "NVIDIA RTX A6000",
  "NVIDIA GeForce RTX 4090",
  "NVIDIA GeForce RTX 5090",
  "AMD Instinct MI300X",
  "Ascend 910B3",
];

export function ParametersTab({
  bench,
  canEdit = false,
  onBenchChange,
}: {
  bench: BenchmarkRecord;
  canEdit?: boolean;
  onBenchChange?: (b: BenchmarkRecord) => void;
}) {
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
  // Accuracy runs carry an `accuracy:` block instead of `bench:` rows.
  const accuracy = first.accuracy ?? null;
  const isAccuracy = !!accuracy;

  // Sweep dimensions — extract unique values across bench[] for input/output/concurrency.
  const inputLens = uniqueNums(benchEntries, "random_input_len");
  const outputLens = uniqueNums(benchEntries, "random_output_len");
  const concurrencies = uniqueNums(benchEntries, "max_concurrency");
  const isSweep = totalRuns > 1;

  // Ingress/external runs (Slurm-submitted via API, benching an existing
  // endpoint): no provider bound and no runpod block — nothing was spawned,
  // so hardware identity is whatever the user sets manually.
  const isIngress = !bench.provider_id && !parsed.runpod;
  const baseUrl = parsed.base_url ?? first.base_url;

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
      ) : isIngress ? (
        <ParamsCard
          icon={<Server className="h-4 w-4" />}
          title="Hardware"
          description="External endpoint run — nothing was spawned, so the platform can't detect the GPU behind it. Set it manually so this run groups by GPU in stats, comparisons, and the API."
          action={
            <Badge variant="secondary" className="font-mono text-[10px]">
              ingress
            </Badge>
          }
        >
          <KvGrid>
            <GpuIdentity
              bench={bench}
              fallbackType={pod.gpu_type}
              fallbackCount={pod.gpu_count}
              canEdit={canEdit}
              onBenchChange={onBenchChange}
            />
            <Kv label="Endpoint" value={baseUrl} mono wide />
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
            <GpuIdentity
              bench={bench}
              fallbackType={pod.gpu_type}
              fallbackCount={pod.gpu_count}
              canEdit={canEdit}
              onBenchChange={onBenchChange}
            />
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

      {(bench.visible_devices || (bench.env_vars && Object.keys(bench.env_vars).length > 0)) && (
        <ParamsCard
          icon={<Cpu className="h-4 w-4" />}
          title="Runtime environment"
          description={
            provider?.kind === "vm"
              ? "Exported on the VM before the run (absolute-path values are auto-created). Applied at run time — not part of the submitted YAML above."
              : "Passed to the pod at launch. Applied at run time — not part of the submitted YAML above."
          }
        >
          {bench.visible_devices && (
            <KvGrid>
              <Kv label="CUDA_VISIBLE_DEVICES" value={bench.visible_devices} mono wide />
            </KvGrid>
          )}
          {bench.env_vars && Object.keys(bench.env_vars).length > 0 && (
            <Detail label="Environment variables">
              <div className="flex flex-wrap gap-1">
                {Object.entries(bench.env_vars).map(([k, v]) => (
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

      {isAccuracy ? (
        <ParamsCard
          icon={<Gauge className="h-4 w-4" />}
          title="Accuracy eval"
          description="Quality eval — serves the model and scores datasets. No throughput rows."
          action={
            <Badge variant="default" className="font-mono text-[10px]">
              accuracy
            </Badge>
          }
        >
          <KvGrid>
            <Kv label="Samples / dataset" value={accuracy?.limit} />
            <Kv label="Concurrency" value={accuracy?.concurrency} />
            <Kv label="Max tokens" value={accuracy?.max_tokens} />
          </KvGrid>
          <Detail label="Datasets">
            <div className="flex flex-wrap gap-1">
              {(accuracy?.datasets ?? []).map((d, i) => (
                <Badge key={i} variant="secondary" className="font-mono text-[10px]">
                  {datasetLabel(d)}
                </Badge>
              ))}
            </div>
          </Detail>
          {accuracy?.languages && accuracy.languages.length > 0 && (
            <Detail label="MMLU languages">
              <div className="flex flex-wrap gap-1">
                {accuracy.languages.map((l) => (
                  <Badge key={l} variant="secondary" className="font-mono text-[10px]">
                    {l}
                  </Badge>
                ))}
              </div>
            </Detail>
          )}
        </ParamsCard>
      ) : (
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
      )}

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

/** GPU type + count cells for a KvGrid, with an inline editor (owner/admin).
 * The manual value lives on the benchmark row and wins over the config-derived
 * one everywhere — it's the only way ingress/Slurm runs get a GPU identity.
 * The input is a datalist: pick a known GPU or type any string. */
function GpuIdentity({
  bench,
  fallbackType,
  fallbackCount,
  canEdit,
  onBenchChange,
}: {
  bench: BenchmarkRecord;
  fallbackType?: string;
  fallbackCount?: number;
  canEdit?: boolean;
  onBenchChange?: (b: BenchmarkRecord) => void;
}) {
  // bench.gpu_type is already the server-resolved effective value (manual →
  // config); the fallbacks only cover older gateway payloads without the field.
  const effType = bench.gpu_type ?? fallbackType;
  const effCount = bench.gpu_count ?? fallbackCount;
  const [editing, setEditing] = useState(false);
  const [typeDraft, setTypeDraft] = useState("");
  const [countDraft, setCountDraft] = useState("");
  const [saving, setSaving] = useState(false);

  function start() {
    setTypeDraft(effType ?? "");
    setCountDraft(effCount != null ? String(effCount) : "");
    setEditing(true);
  }

  async function save() {
    setSaving(true);
    try {
      const next = await gateway.updateBenchmark(bench.id, {
        // "" / 0 clear the manual value (back to config-derived).
        gpu_type: typeDraft.trim(),
        gpu_count: countDraft.trim() === "" ? 0 : Math.max(0, parseInt(countDraft, 10) || 0),
      });
      onBenchChange?.(next);
      setEditing(false);
      toast.success("GPU type saved");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  if (editing) {
    return (
      <div className="sm:col-span-2">
        <dt className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          GPU type × count
        </dt>
        <dd className="mt-1 flex flex-wrap items-center gap-2">
          <input
            list="gpu-type-suggestions"
            value={typeDraft}
            onChange={(e) => setTypeDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") save();
              if (e.key === "Escape") setEditing(false);
            }}
            placeholder="NVIDIA H20"
            autoFocus
            className="h-7 w-48 rounded-md border border-border bg-background px-2 font-mono text-xs outline-none focus:ring-1 focus:ring-ring"
          />
          <datalist id="gpu-type-suggestions">
            {GPU_TYPE_SUGGESTIONS.map((g) => (
              <option key={g} value={g} />
            ))}
          </datalist>
          <input
            type="number"
            min={0}
            value={countDraft}
            onChange={(e) => setCountDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") save();
              if (e.key === "Escape") setEditing(false);
            }}
            placeholder="×"
            title="GPU count"
            className="h-7 w-14 rounded-md border border-border bg-background px-2 text-xs outline-none focus:ring-1 focus:ring-ring"
          />
          <Button type="button" size="sm" className="h-7" onClick={save} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </Button>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="h-7"
            onClick={() => setEditing(false)}
            disabled={saving}
          >
            Cancel
          </Button>
        </dd>
      </div>
    );
  }

  return (
    <>
      <div>
        <dt className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          GPU type
        </dt>
        <dd className="mt-0.5 flex items-center gap-1 text-sm">
          <span
            className={cn("truncate font-mono text-xs", !effType && "text-muted-foreground")}
            title={effType ?? "—"}
          >
            {effType ?? "—"}
          </span>
          {canEdit && (
            <button
              type="button"
              onClick={start}
              title="Set GPU type"
              className="shrink-0 rounded p-0.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <Pencil className="h-3 w-3" />
            </button>
          )}
        </dd>
      </div>
      <Kv label="GPU count" value={effCount ?? undefined} />
    </>
  );
}

function datasetLabel(d: unknown): string {
  if (typeof d === "string") return d;
  if (d && typeof d === "object" && "name" in d) {
    const o = d as { name?: unknown; config?: unknown };
    return o.config ? `${String(o.name)} (${String(o.config)})` : String(o.name);
  }
  return String(d);
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
