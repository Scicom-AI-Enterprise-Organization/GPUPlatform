"use client";

import { useEffect, useState, useTransition } from "react";
import { AlertTriangle, ArrowUpRight, Copy, Eye, EyeOff, Loader2, Pencil, Plus, RotateCw, Save, Trash2, X } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import type { AppRecord } from "@/lib/types";
import { gateway, type AppStatus } from "@/lib/gateway";
import { parseGpuIds, parsePhys, suggestPacking } from "@/lib/gpu-pin";
import { cleanVllmArgs } from "@/lib/vllm-args";
import { restartEndpoint, updateAutoscaler } from "../../actions";

export function OverviewTab({ app }: { app: AppRecord }) {
  // A multi-model VM endpoint is one fixed, always-on node that time-shares its
  // GPUs via vLLM sleep/wake — the scale/idle knobs don't apply (read-only card).
  // A multi-model *cloud* endpoint (RunPod, gpu != "vm") still scales the pod and
  // honours idle_timeout_s (idle → delete the pod), so it gets the editable scale
  // card. Each member model has its own vLLM args either way.
  const isMulti = app.mode === "multi";
  const isVm = app.gpu === "vm";
  return (
    <div className="space-y-4">
      <ProvisionErrorBanner appId={app.app_id} />
      <RequestPanel app={app} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <DetailCard app={app} />
        {isMulti && isVm ? <VmServingCard app={app} /> : <ScaleStrategyCard app={app} />}
      </div>

      {isMulti ? <MultiModelArgsCard app={app} /> : <EngineArgsCard app={app} />}

      <EnvVarsCard app={app} />
    </div>
  );
}

// The export env the endpoint was created with (HF_HOME, cache dirs, …),
// applied to every vLLM process on the worker. Read-only; secret-looking
// values are masked behind a reveal toggle.
function EnvVarsCard({ app }: { app: AppRecord }) {
  const [reveal, setReveal] = useState(false);
  const entries = Object.entries(app.env_vars ?? {});
  if (entries.length === 0) return null;

  const isSecret = (k: string) => /TOKEN|KEY|SECRET|PASSWORD|CRED/i.test(k);
  const line = (k: string, v: string, masked: boolean) =>
    `export ${k}=${JSON.stringify(masked && isSecret(k) ? "••••••••" : v)}`;
  const displayCode = entries.map(([k, v]) => line(k, v, !reveal)).join("\n");
  const copyCode = entries.map(([k, v]) => line(k, v, false)).join("\n");
  const hasSecret = entries.some(([k]) => isSecret(k));

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-2">
        <div className="flex flex-col gap-0.5">
          <CardTitle className="text-sm font-medium">Environment variables</CardTitle>
          <span className="text-xs text-muted-foreground">
            {entries.length} var{entries.length === 1 ? "" : "s"} exported into every vLLM process on the worker.
          </span>
        </div>
        {hasSecret && (
          <Button variant="outline" size="xs" onClick={() => setReveal((v) => !v)}>
            {reveal ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
            {reveal ? "Hide" : "Reveal"} values
          </Button>
        )}
      </CardHeader>
      <CardContent>
        <CodeBlock displayCode={displayCode} copyCode={copyCode} />
      </CardContent>
    </Card>
  );
}

// Multi-model VM serving: a single fixed node, always-on, GPU-pinned, with
// idle models evicted via vLLM sleep/wake. Read-only — there's nothing to
// autoscale.
function VmServingCard({ app }: { app: AppRecord }) {
  const models = app.models ?? [];
  const gpuIds = (app.visible_devices ?? "").trim();
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">VM serving</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <Row label="Mode" value="Multi-model (always-on)" />
        <Row label="Models" value={<code className="font-mono">{models.length}</code>} />
        <Row label="GPUs" value={<code className="font-mono">{gpuIds || `×${app.gpu_count ?? 0}`}</code>} />
        <Row label="Eviction" value={`Sleep/wake · level ${app.sleep_level ?? 1}`} />
        {app.vllm_version && (
          <Row label="vLLM" value={<code className="font-mono">{app.vllm_version}</code>} />
        )}
        {app.venv_path && (
          <Row label="venv" value={<code className="font-mono text-xs">{app.venv_path}</code>} />
        )}
        <p className="rounded-md border border-border bg-muted/40 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
          One fixed VM node, always on — no scale-to-zero. Models whose GPUs are
          contended are slept (level {app.sleep_level ?? 1}) and woken on demand; the first
          request to a sleeping model waits for the swap.
        </p>
      </CardContent>
    </Card>
  );
}

// Per-model vLLM args for a multi-model endpoint: each member has its own
// tensor-parallel size + extra args. Editable inline — click "Edit" to add /
// remove models, change TP, or rewrite args, then "Save" re-provisions the
// worker (in-flight requests drain first) via PATCH /apps/{id}/models.
type ModelRow = { model: string; tp: number; pp: number; extra_args: string; gpus: string; audio: boolean };

const TP_CHOICES = [1, 2, 4, 8];

function toRows(models: AppRecord["models"]): ModelRow[] {
  return (models ?? []).map((m) => ({
    model: m.model,
    tp: m.tp ?? 1,
    pp: m.pp ?? 1,
    extra_args: m.extra_args ?? "",
    gpus: (m.gpu_indices ?? []).join(","),
    audio: m.task === "transcription",
  }));
}

/** Pull a readable message out of the gateway's {detail:{error}} / {detail} shape. */
function errText(body: unknown, fallback: string): string {
  if (typeof body === "string") return body || fallback;
  if (body && typeof body === "object") {
    const o = body as Record<string, unknown>;
    const d = o.detail;
    if (typeof d === "string") return d;
    if (d && typeof d === "object" && typeof (d as Record<string, unknown>).error === "string") {
      return (d as Record<string, string>).error;
    }
    if (typeof o.error === "string") return o.error;
  }
  return fallback;
}

function MultiModelArgsCard({ app }: { app: AppRecord }) {
  const router = useRouter();
  const [editing, setEditing] = useState(false);
  const [rows, setRows] = useState<ModelRow[]>(() => toRows(app.models));
  const [visibleDevices, setVisibleDevices] = useState(app.visible_devices ?? "");
  const [sleepLevel, setSleepLevel] = useState<number>(app.sleep_level ?? 1);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const models = app.models ?? [];
  const physIds = parsePhys(visibleDevices);
  const gpuCount = physIds.length;
  // Each member occupies tp × pp consecutive GPUs.
  const suggestions = suggestPacking(rows.map((r) => r.tp * (r.pp || 1)), physIds);

  function startEdit() {
    setRows(toRows(app.models));
    setVisibleDevices(app.visible_devices ?? "");
    setSleepLevel(app.sleep_level ?? 1);
    setErr(null);
    setMsg(null);
    setEditing(true);
  }
  function cancel() {
    setEditing(false);
    setErr(null);
  }
  const update = (i: number, patch: Partial<ModelRow>) =>
    setRows((rs) => rs.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  const addRow = () => setRows((rs) => [...rs, { model: "", tp: 1, pp: 1, extra_args: "", gpus: "", audio: false }]);
  const removeRow = (i: number) => setRows((rs) => rs.filter((_, idx) => idx !== i));

  async function save() {
    setSaving(true);
    setErr(null);
    setMsg(null);
    let payload: Array<{ model: string; tp: number; pp: number; extra_args: string; gpu_indices?: number[]; task?: string }>;
    try {
      payload = rows
        .filter((r) => r.model.trim())
        .map((r) => {
          const pp = r.pp || 1;
          const gpu_indices = parseGpuIds(r.gpus, r.tp * pp, r.model.trim() || "model");
          return {
            model: r.model.trim(),
            tp: r.tp,
            pp,
            extra_args: r.extra_args.trim(),
            ...(gpu_indices ? { gpu_indices } : {}),
            ...(r.audio ? { task: "transcription" } : {}),
          };
        });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setSaving(false);
      return;
    }
    if (!payload.length) {
      setErr("Add at least one model (or delete the endpoint instead).");
      setSaving(false);
      return;
    }
    try {
      const r = await fetch(`/api/proxy/apps/${encodeURIComponent(app.app_id)}/models`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          models: payload,
          sleep_level: sleepLevel,
          ...(visibleDevices.trim() ? { visible_devices: visibleDevices.trim() } : {}),
        }),
      });
      const text = await r.text();
      let parsed: unknown = text;
      try {
        parsed = text ? JSON.parse(text) : null;
      } catch {
        /* keep raw */
      }
      if (!r.ok) {
        setErr(errText(parsed, r.statusText));
        return;
      }
      setEditing(false);
      setMsg(`Saved — ${payload.length} model(s). The fleet is re-provisioning; watch the Workers tab as it reloads.`);
      router.refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-2 space-y-0">
        <div className="flex flex-col gap-0.5">
          <CardTitle className="text-sm font-medium">vLLM engine args — per model</CardTitle>
          <span className="text-xs text-muted-foreground">
            {editing
              ? "Add or remove models, change tensor- / pipeline-parallel size (a model uses TP×PP GPUs), or edit args. Saving re-provisions the worker — in-flight requests drain first."
              : "Each model launches its own "}
            {!editing && <code className="font-mono">vllm serve</code>}
            {!editing && " with these args."}
          </span>
        </div>
        {!editing && (
          <Button variant="outline" size="xs" onClick={startEdit}>
            <Pencil className="h-3 w-3" /> Edit
          </Button>
        )}
      </CardHeader>

      <CardContent className="space-y-3 text-sm">
        {/* ---- read-only view ---- */}
        {!editing &&
          (models.length === 0 ? (
            <p className="text-xs text-muted-foreground">No models configured.</p>
          ) : (
            models.map((m) => (
              <div key={m.model} className="rounded-md border border-border">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border bg-muted/30 px-3 py-1.5">
                  <code className="font-mono text-xs text-foreground">{m.model}</code>
                  <div className="flex items-center gap-1.5">
                    {m.gpu_indices && m.gpu_indices.length > 0 && (
                      <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                        GPU {m.gpu_indices.join(",")}
                      </span>
                    )}
                    <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                      TP={m.tp}
                    </span>
                    {(m.pp ?? 1) > 1 && (
                      <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                        PP={m.pp}
                      </span>
                    )}
                  </div>
                </div>
                <pre className="overflow-x-auto px-3 py-2 font-mono text-[11px] leading-relaxed text-foreground scrollbar-thin">
                  {(m.extra_args ?? "").trim() || "(no extra args)"}
                </pre>
              </div>
            ))
          ))}

        {/* ---- edit view: one card per model ---- */}
        {editing && (
          <>
            {rows.map((r, i) => {
              const tpOpts = Array.from(new Set([...TP_CHOICES, r.tp])).sort((a, b) => a - b);
              // PP need not be a power of two (e.g. TP=2 × PP=3 = 6 GPUs).
              const ppMax = Math.max(gpuCount || 8, r.pp);
              const ppOpts = Array.from({ length: ppMax }, (_, k) => k + 1);
              return (
                <div key={i} className="rounded-md border border-border">
                  <div className="flex flex-wrap items-center gap-2 border-b border-border bg-muted/30 px-3 py-2">
                    <Input
                      value={r.model}
                      onChange={(e) => update(i, { model: e.target.value })}
                      placeholder="qwen/qwen3.6-27b"
                      disabled={saving}
                      className="h-8 min-w-[200px] flex-1 font-mono text-xs"
                    />
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">TP</span>
                      <Select
                        value={String(r.tp)}
                        onValueChange={(v) => update(i, { tp: Number(v) })}
                        disabled={saving}
                      >
                        <SelectTrigger className="h-8 w-[68px] font-mono text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {tpOpts.map((n) => (
                            <SelectItem
                              key={n}
                              value={String(n)}
                              disabled={gpuCount > 0 && n * (r.pp || 1) > gpuCount}
                            >
                              {n}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">PP</span>
                      <Select
                        value={String(r.pp || 1)}
                        onValueChange={(v) => update(i, { pp: Number(v) })}
                        disabled={saving}
                      >
                        <SelectTrigger className="h-8 w-[68px] font-mono text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {ppOpts.map((n) => (
                            <SelectItem
                              key={n}
                              value={String(n)}
                              disabled={gpuCount > 0 && r.tp * n > gpuCount}
                            >
                              {n}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">GPUs</span>
                      <Input
                        value={r.gpus}
                        onChange={(e) => update(i, { gpus: e.target.value })}
                        placeholder={suggestions[i] || "auto"}
                        disabled={saving}
                        aria-label="GPU ids"
                        className="h-8 w-28 font-mono text-xs"
                      />
                      {r.gpus.trim() === "" && suggestions[i] && (
                        <button
                          type="button"
                          onClick={() => update(i, { gpus: suggestions[i] })}
                          disabled={saving}
                          className="text-[10px] text-primary hover:underline"
                        >
                          use {suggestions[i]}
                        </button>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      onClick={() => removeRow(i)}
                      disabled={saving}
                      aria-label="Remove model"
                      className="text-muted-foreground hover:text-destructive"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                  <div className="space-y-2 px-3 py-2">
                    <Textarea
                      value={r.extra_args}
                      onChange={(e) => update(i, { extra_args: cleanVllmArgs(e.target.value) })}
                      placeholder="--gpu-memory-utilization 0.9 --reasoning-parser qwen3 --max-model-len 262144 ..."
                      disabled={saving}
                      rows={2}
                      className="font-mono text-[11px] leading-relaxed"
                    />
                    <label
                      className="flex w-fit cursor-pointer select-none items-center gap-1.5 text-[11px] text-muted-foreground"
                      title="Marks this as an audio/ASR (Whisper) model so the worker installs audio-decode deps. Set it for ASR finetunes whose name doesn't say 'whisper'."
                    >
                      <input
                        type="checkbox"
                        checked={r.audio}
                        onChange={(e) => update(i, { audio: e.target.checked })}
                        disabled={saving}
                        className="h-3.5 w-3.5 accent-primary"
                      />
                      Audio / ASR model (Whisper) — enables transcription
                    </label>
                  </div>
                </div>
              );
            })}

            <Button variant="outline" size="xs" onClick={addRow} disabled={saving}>
              <Plus className="h-3 w-3" /> Add model
            </Button>

            <div className="flex flex-wrap items-end gap-x-4 gap-y-2 border-t border-border pt-3">
              <div className="flex flex-col gap-1">
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                  visible_devices {gpuCount > 0 && <span className="normal-case">({gpuCount} GPU)</span>}
                </span>
                <Input
                  value={visibleDevices}
                  onChange={(e) => setVisibleDevices(e.target.value)}
                  placeholder="0,1,2,3"
                  disabled={saving}
                  className="h-8 w-32 font-mono text-xs"
                />
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground">sleep level</span>
                <Select
                  value={String(sleepLevel)}
                  onValueChange={(v) => setSleepLevel(Number(v))}
                  disabled={saving}
                >
                  <SelectTrigger className="h-8 w-16 font-mono text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">1</SelectItem>
                    <SelectItem value="2">2</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex-1" />
              <Button variant="ghost" size="sm" onClick={cancel} disabled={saving}>
                <X className="h-4 w-4" /> Cancel
              </Button>
              <Button size="sm" onClick={save} disabled={saving}>
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                Save &amp; re-provision
              </Button>
            </div>
          </>
        )}

        {err && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {err}
          </div>
        )}
        {msg && !editing && (
          <div className="flex items-center gap-2 rounded-md border border-status-active/40 bg-status-active/10 px-3 py-2 text-xs text-status-active">
            <RotateCw className="h-3 w-3" /> {msg}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ProvisionErrorBanner({ appId }: { appId: string }) {
  const [status, setStatus] = useState<AppStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const res = await fetch(
          `/api/proxy/apps/${encodeURIComponent(appId)}/status`,
          { cache: "no-store" },
        );
        if (!res.ok) return;
        const data = (await res.json()) as AppStatus;
        if (!cancelled) setStatus(data);
      } catch {
        // best-effort; banner stays hidden on failure
      }
    }
    poll();
    const id = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [appId]);

  if (!status?.last_provision_error) return null;

  const cooldown = status.provision_cooldown_remaining_s;
  const at = status.last_provision_error_at
    ? new Date(status.last_provision_error_at * 1000)
    : null;
  const ago = at ? formatAgo(at) : null;

  return (
    <div className="flex items-start gap-3 rounded-md border border-red-500/30 bg-red-500/5 px-4 py-3 text-sm text-red-700 dark:text-red-300">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="flex-1 space-y-1">
        <div className="font-medium">
          Couldn't start a worker
          {ago ? <span className="text-xs font-normal opacity-75"> · {ago}</span> : null}
        </div>
        <div className="font-mono text-xs leading-relaxed opacity-90 break-words">
          {status.last_provision_error}
        </div>
        {cooldown > 0 ? (
          <div className="text-xs opacity-75">
            Auto-retry in {cooldown}s. Pick a different GPU / count if this combo isn't in stock.
          </div>
        ) : (
          <div className="text-xs opacity-75">
            The autoscaler will retry on the next request — or change GPU / count above.
          </div>
        )}
      </div>
    </div>
  );
}

function formatAgo(d: Date): string {
  const s = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

function RequestPanel({ app }: { app: AppRecord }) {
  const [reveal, setReveal] = useState(false);
  const { token, loading: tokenLoading } = useApiToken();
  const base = process.env.NEXT_PUBLIC_GATEWAY_URL ?? gateway.baseUrl;
  // The OpenAI `model` field. A multi-model endpoint rejects the endpoint name
  // (you must name a member), so default to the first member; single-mode keeps
  // using the endpoint name, which resolves back-compat.
  const isMulti = app.mode === "multi";
  const exampleModel =
    isMulti && app.models && app.models.length > 0 ? app.models[0].model : app.app_id;

  // The visible / copyable forms of every snippet. Visible may be masked;
  // copy always pastes the real key so the user gets a working command.
  const visibleToken = reveal && token ? token : token ? maskToken(token) : "YOUR_API_KEY";
  const realToken = token ?? "YOUR_API_KEY";

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <CardTitle className="text-sm font-medium">Run a job</CardTitle>
          <span className="text-xs text-muted-foreground">
            OpenAI-compatible. Autoscales to meet demand.
          </span>
        </div>
        {token ? (
          <Button variant="outline" size="xs" onClick={() => setReveal((v) => !v)}>
            {reveal ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
            {reveal ? "Hide" : "Reveal"} key
          </Button>
        ) : !tokenLoading ? (
          <Link
            href="/login?next=/serverless"
            className="text-xs text-primary hover:underline"
          >
            Sign in to use your key
          </Link>
        ) : null}
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="curl">
          <TabsList variant="line" className="bg-transparent">
            <TabsTrigger value="curl">cURL</TabsTrigger>
            <TabsTrigger value="curl-stream">cURL (stream)</TabsTrigger>
            <TabsTrigger value="openai">OpenAI client</TabsTrigger>
          </TabsList>

          <TabsContent value="curl" className="mt-3 space-y-3">
            <p className="text-sm text-muted-foreground">
              OpenAI <code className="font-mono">/{app.app_id}/v1/chat/completions</code> — scoped to this endpoint; returns the full completion JSON in one call.
            </p>
            <CodeBlock
              displayCode={curlChatSnippet(base, visibleToken, exampleModel, app.app_id)}
              copyCode={curlChatSnippet(base, realToken, exampleModel, app.app_id)}
            />
            <DocsLink />
          </TabsContent>

          <TabsContent value="curl-stream" className="mt-3 space-y-3">
            <p className="text-sm text-muted-foreground">
              Same endpoint with <code className="font-mono">&quot;stream&quot;: true</code> — token-by-token Server-Sent Events.
            </p>
            <CodeBlock
              displayCode={curlChatStreamSnippet(base, visibleToken, exampleModel, app.app_id)}
              copyCode={curlChatStreamSnippet(base, realToken, exampleModel, app.app_id)}
            />
            <DocsLink />
          </TabsContent>

          <TabsContent value="openai" className="mt-3 space-y-3">
            <p className="text-sm text-muted-foreground">
              Point any OpenAI client at this endpoint&apos;s base URL.{" "}
              {isMulti
                ? "Set model to one of its member models."
                : "The model field is ignored — this endpoint serves one model."}
            </p>
            <CodeBlock
              displayCode={openaiSnippet(base, visibleToken, exampleModel, app.app_id)}
              copyCode={openaiSnippet(base, realToken, exampleModel, app.app_id)}
            />
            <DocsLink />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

function useApiToken() {
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let abort = false;
    fetch("/api/auth/token", { cache: "no-store" })
      .then(async (r) => {
        if (abort) return;
        if (!r.ok) {
          setToken(null);
          return;
        }
        const body = (await r.json()) as { token?: string };
        setToken(body.token ?? null);
      })
      .catch(() => !abort && setToken(null))
      .finally(() => !abort && setLoading(false));
    return () => {
      abort = true;
    };
  }, []);
  return { token, loading };
}

function maskToken(t: string) {
  if (t.length <= 8) return "•".repeat(t.length);
  return `${t.slice(0, 4)}${"•".repeat(Math.max(8, t.length - 8))}${t.slice(-4)}`;
}

function CopyButton({ text }: { text: string }) {
  return (
    <Button
      variant="outline"
      size="icon-sm"
      onClick={() => {
        navigator.clipboard.writeText(text);
        toast.success("Copied", { duration: 3000 });
      }}
    >
      <Copy className="h-3.5 w-3.5" />
    </Button>
  );
}

function CodeBlock({
  displayCode,
  copyCode,
}: {
  displayCode: string;
  copyCode?: string;
}) {
  return (
    <div className="relative">
      <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-foreground scrollbar-thin">
        {displayCode}
      </pre>
      <div className="absolute right-2 top-2">
        <CopyButton text={copyCode ?? displayCode} />
      </div>
    </div>
  );
}

function DocsLink() {
  return (
    <a
      className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
      href="/api-docs#inference"
      target="_blank"
      rel="noopener noreferrer"
    >
      API reference — chat/completions, models, metrics, health
      <ArrowUpRight className="h-3 w-3" />
    </a>
  );
}

function curlChatSnippet(base: string, token: string, model: string, appId: string) {
  // OpenAI-compatible chat completions, scoped to THIS endpoint by URL path
  // (so multiple endpoints never collide on the `model` field). The gateway
  // polls internally and returns the full completion JSON in one call.
  return `curl -X POST '${base}/${appId}/v1/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${token}' \\
  -d '{
    "model": "${model}",
    "messages": [{"role": "user", "content": "Hello, world"}],
    "max_tokens": 1024
  }'`;
}

function curlChatStreamSnippet(base: string, token: string, model: string, appId: string) {
  return `curl -N -X POST '${base}/${appId}/v1/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${token}' \\
  -d '{
    "model": "${model}",
    "messages": [{"role": "user", "content": "Hello, world"}],
    "max_tokens": 1024,
    "stream": true
  }'`;
}

function openaiSnippet(base: string, token: string, model: string, appId: string) {
  return `from openai import OpenAI

client = OpenAI(
    base_url="${base}/${appId}/v1",
    api_key="${token}",
)

resp = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)`;
}

function DetailCard({ app }: { app: AppRecord }) {
  return (
    <Card>
      <CardContent className="space-y-3 px-6 py-4 text-sm">
        <Row label="Endpoint ID" value={<code className="font-mono">{app.app_id}</code>} />
        <Row
          label="Created"
          value={new Date(app.created_at).toLocaleDateString("en-GB", {
            day: "2-digit",
            month: "short",
            year: "numeric",
          })}
        />
        <Row
          label="Framework"
          value={
            <span className="inline-flex items-center gap-1.5">
              <span className="flex h-5 w-5 items-center justify-center rounded border border-border bg-muted text-[10px] font-semibold text-muted-foreground">
                v
              </span>
              vLLM
            </span>
          }
        />
        <Row label="GPU count" value={`×${app.gpu_count ?? 1}`} />
        <Row label="GPU types" value={<span className="font-mono">{app.gpu}</span>} />
      </CardContent>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-border/40 pb-2 last:border-b-0 last:pb-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-foreground">{value}</span>
    </div>
  );
}

function ScaleStrategyCard({ app }: { app: AppRecord }) {
  const router = useRouter();
  const [editing, setEditing] = useState(false);
  const [maxInput, setMaxInput] = useState(String(app.autoscaler.max_containers));
  const [idleInput, setIdleInput] = useState(String(app.autoscaler.idle_timeout_s));
  const [pending, startTransition] = useTransition();

  useEffect(() => {
    setMaxInput(String(app.autoscaler.max_containers));
    setIdleInput(String(app.autoscaler.idle_timeout_s));
  }, [app.autoscaler.max_containers, app.autoscaler.idle_timeout_s]);

  const parsedMax = Number.parseInt(maxInput, 10);
  const parsedIdle = Number.parseInt(idleInput, 10);
  const maxInvalid =
    !/^\d+$/.test(maxInput.trim()) || !Number.isFinite(parsedMax) || parsedMax < 1 || parsedMax > 20;
  const idleInvalid =
    !/^\d+$/.test(idleInput.trim()) || !Number.isFinite(parsedIdle) || parsedIdle < 0 || parsedIdle > 86400;

  function save() {
    if (maxInvalid) {
      toast.error("Max workers must be an integer between 1 and 20.", { duration: 5000 });
      return;
    }
    if (idleInvalid) {
      toast.error("Idle timeout must be an integer 0–86400 seconds (0 = always-on, { duration: 5000 }).");
      return;
    }
    startTransition(async () => {
      const res = await updateAutoscaler(app.app_id, {
        max_containers: parsedMax,
        idle_timeout_s: parsedIdle,
      });
      if (!res.ok) {
        toast.error(res.error, { duration: 5000 });
        return;
      }
      toast.success("Scale strategy updated", { duration: 3000 });
      setEditing(false);
      router.refresh();
    });
  }

  function cancel() {
    setMaxInput(String(app.autoscaler.max_containers));
    setIdleInput(String(app.autoscaler.idle_timeout_s));
    setEditing(false);
  }

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-2">
        <CardTitle className="text-sm font-medium">Scale strategy</CardTitle>
        {!editing ? (
          <Button variant="outline" size="xs" onClick={() => setEditing(true)}>
            <Pencil className="h-3 w-3" />
            Edit
          </Button>
        ) : (
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="xs" onClick={cancel} disabled={pending}>
              Cancel
            </Button>
            <Button
              size="xs"
              onClick={save}
              disabled={pending || maxInvalid || idleInvalid}
            >
              {pending && <Loader2 className="h-3 w-3 animate-spin" />}
              Save
            </Button>
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <Row label="Active workers" value={<code className="font-mono">0</code>} />
        {editing ? (
          <>
            <EditRow label="Max workers">
              <Input
                type="text"
                inputMode="numeric"
                value={maxInput}
                onChange={(e) => setMaxInput(e.target.value)}
                placeholder="1–20"
                aria-invalid={maxInvalid}
                className="h-8 w-24 text-right font-mono"
                disabled={pending}
              />
            </EditRow>
            <EditRow label="Idle timeout (s)">
              <Input
                type="text"
                inputMode="numeric"
                value={idleInput}
                onChange={(e) => setIdleInput(e.target.value)}
                placeholder="0 = always-on"
                aria-invalid={idleInvalid}
                className="h-8 w-24 text-right font-mono"
                disabled={pending}
              />
            </EditRow>
          </>
        ) : (
          <>
            <Row
              label="Max workers"
              value={<code className="font-mono">{app.autoscaler.max_containers}</code>}
            />
            <Row
              label="Idle timeout"
              value={<code className="font-mono">{app.autoscaler.idle_timeout_s} s</code>}
            />
          </>
        )}
        <Row label="Auto scaling method" value="Queue delay" />
        <p className="rounded-md border border-border bg-muted/40 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
          Scale up after <strong className="text-foreground">4</strong> seconds of queue delay.
          With zero workers initially, the first request adds one worker. Subsequent requests
          add workers only after waiting in the queue for 4 seconds.
        </p>
        <p className="text-xs text-muted-foreground">
          Assuming <strong className="text-foreground">1</strong> req/sec with{" "}
          <strong className="text-foreground">0.5</strong> s processing time.
        </p>
      </CardContent>
    </Card>
  );
}

function EngineArgsCard({ app }: { app: AppRecord }) {
  const router = useRouter();
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(app.vllm_args ?? "");
  const [pending, startTransition] = useTransition();
  const [restarting, startRestart] = useTransition();
  const [confirmRestart, setConfirmRestart] = useState(false);

  useEffect(() => {
    setValue(app.vllm_args ?? "");
  }, [app.vllm_args]);

  const tooLong = value.length > 2048;

  function save() {
    if (tooLong) {
      toast.error("Engine args too long (max 2048 chars, { duration: 5000 }).");
      return;
    }
    startTransition(async () => {
      const res = await updateAutoscaler(app.app_id, { vllm_args: value.trim() });
      if (!res.ok) {
        toast.error(res.error, { duration: 5000 });
        return;
      }
      toast.success("Engine args saved. Click Restart to apply now.", { duration: 3000 });
      setEditing(false);
      router.refresh();
    });
  }

  function restart() {
    startRestart(async () => {
      const res = await restartEndpoint(app.app_id);
      if (!res.ok) {
        toast.error(res.error, { duration: 5000 });
        return;
      }
      if (res.drained === 0) {
        toast.success("No live workers to restart — next cold start will use the latest config.", { duration: 3000 });
      } else {
        toast.success(`Draining ${res.drained} worker${res.drained === 1 ? "" : "s"} — autoscaler will respawn.`, { duration: 3000 });
      }
      setConfirmRestart(false);
      router.refresh();
    });
  }

  function cancel() {
    setValue(app.vllm_args ?? "");
    setEditing(false);
  }

  const display = (app.vllm_args ?? "").trim();
  return (
    <>
    <Dialog open={confirmRestart} onOpenChange={(open) => !restarting && setConfirmRestart(open)}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Restart workers?</DialogTitle>
          <DialogDescription>
            All running workers for this endpoint will be drained. In-flight requests
            finish; new ones spawn with the latest config.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="ghost" onClick={() => setConfirmRestart(false)} disabled={restarting}>
            Cancel
          </Button>
          <Button onClick={restart} disabled={restarting}>
            {restarting && <Loader2 className="h-4 w-4 animate-spin" />}
            Restart workers
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-2">
        <div className="flex flex-col gap-0.5">
          <CardTitle className="text-sm font-medium">vLLM engine args</CardTitle>
          <span className="text-xs text-muted-foreground">
            Appended to the <code className="font-mono">vllm serve</code> command on each worker
            boot. Changes apply on the next cold start.
          </span>
        </div>
        {!editing ? (
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="xs"
              onClick={() => setConfirmRestart(true)}
              disabled={restarting}
              title="Drain workers so the next cold start picks up the latest config"
            >
              {restarting ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <RotateCw className="h-3 w-3" />
              )}
              Restart workers
            </Button>
            <Button variant="outline" size="xs" onClick={() => setEditing(true)}>
              <Pencil className="h-3 w-3" />
              Edit
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="xs" onClick={cancel} disabled={pending}>
              Cancel
            </Button>
            <Button size="xs" onClick={save} disabled={pending || tooLong}>
              {pending && <Loader2 className="h-3 w-3 animate-spin" />}
              Save
            </Button>
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {editing ? (
          <>
            <textarea
              value={value}
              onChange={(e) => setValue(cleanVllmArgs(e.target.value))}
              placeholder="--max-model-len 4096 --gpu-memory-utilization 0.9"
              rows={3}
              aria-invalid={tooLong}
              className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/30 aria-invalid:border-destructive"
              disabled={pending}
            />
            <p className="text-xs text-muted-foreground">
              See{" "}
              <a
                href="https://docs.vllm.ai/en/stable/configuration/engine_args/"
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-foreground"
              >
                vLLM engine args
              </a>
              . {value.length}/2048 chars.
            </p>
          </>
        ) : display ? (
          <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-foreground scrollbar-thin">
            {display}
          </pre>
        ) : (
          <p className="rounded-md border border-dashed border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
            No custom args — vLLM uses its built-in defaults.
          </p>
        )}
      </CardContent>
    </Card>
    </>
  );
}

function EditRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-border/40 pb-2 last:border-b-0 last:pb-0">
      <Label className="text-muted-foreground">{label}</Label>
      {children}
    </div>
  );
}

