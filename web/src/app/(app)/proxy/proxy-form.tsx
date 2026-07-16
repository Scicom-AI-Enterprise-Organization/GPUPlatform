"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Plus, Trash2, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { gateway } from "@/lib/gateway";
import type { ProxyEndpoint, ProxyUpstreamSpec } from "@/lib/types";
import { FormFooter, FormShell } from "@/components/form-shell";

type KeyMode = "secret" | "paste" | "keep";
type ModelPair = { alias: string; real: string };
type UpstreamDraft = {
  id?: string;
  name: string;
  base_url: string;
  keyMode: KeyMode;
  api_key_secret: string;
  api_key: string;
  models: ModelPair[];
  priority: number;
  enabled: boolean;
  hadKey: boolean;
  extraBody: string; // raw JSON text; parsed + validated on submit
  testMode: "chat" | "embedding";
  test: { status: "idle" | "running" | "ok" | "fail"; message?: string };
};

function blankUpstream(): UpstreamDraft {
  return {
    name: "", base_url: "", keyMode: "secret", api_key_secret: "", api_key: "",
    models: [{ alias: "", real: "" }], priority: 0, enabled: true, hadKey: false,
    extraBody: "", testMode: "chat", test: { status: "idle" },
  };
}

// Parse an upstream's raw extra_body text. "" → undefined (field omitted). Non-empty
// must be a JSON object; anything else is a form error surfaced on submit/test.
function parseExtraBody(text: string): { ok: true; value?: Record<string, unknown> } | { ok: false } {
  const t = text.trim();
  if (!t) return { ok: true, value: undefined };
  try {
    const v = JSON.parse(t);
    if (v && typeof v === "object" && !Array.isArray(v)) return { ok: true, value: v as Record<string, unknown> };
    return { ok: false };
  } catch {
    return { ok: false };
  }
}

// Seed the first upstream from a deep-link prefill (e.g. the serverless "Proxy"
// tab pre-pointing at an endpoint's serving URL + model). The admin still adds
// the API key. No prefill → a blank upstream.
function seededUpstream(prefill?: ProxyPrefill): UpstreamDraft {
  const u = blankUpstream();
  if (!prefill) return u;
  u.name = prefill.name ? `${prefill.name}-endpoint` : "";
  if (prefill.base) u.base_url = prefill.base;
  if (prefill.model) {
    const alias = prefill.name || prefill.model.split("/").pop() || prefill.model;
    u.models = [{ alias, real: prefill.model }];
  }
  return u;
}

export type ProxyPrefill = { name?: string; base?: string; model?: string };

function fromEndpoint(ep: ProxyEndpoint): UpstreamDraft[] {
  return ep.upstreams.map((u) => ({
    id: u.id,
    name: u.name,
    base_url: u.base_url,
    keyMode: u.has_inline_key ? "keep" : u.api_key_secret ? "secret" : "paste",
    api_key_secret: u.api_key_secret ?? "",
    api_key: "",
    models: Object.entries(u.models).map(([alias, real]) => ({ alias, real })),
    priority: u.priority,
    enabled: u.enabled,
    hadKey: u.has_inline_key || !!u.api_key_secret,
    extraBody: u.extra_body && Object.keys(u.extra_body).length ? JSON.stringify(u.extra_body, null, 2) : "",
    testMode: "chat",
    test: { status: "idle" },
  }));
}

export function ProxyForm({ initial, prefill }: { initial?: ProxyEndpoint; prefill?: ProxyPrefill }) {
  const router = useRouter();
  const editing = !!initial;
  const [name, setName] = useState(initial?.name ?? prefill?.name ?? "");
  const [enabled, setEnabled] = useState(initial?.enabled ?? true);
  const [isPublic, setIsPublic] = useState(initial?.public ?? false);
  const [maxConc, setMaxConc] = useState(String(initial?.max_concurrency ?? 0));
  const [timeoutS, setTimeoutS] = useState(String(initial?.timeout_s ?? 3600));
  const [ups, setUps] = useState<UpstreamDraft[]>(initial ? fromEndpoint(initial) : [seededUpstream(prefill)]);
  const [secretKeys, setSecretKeys] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/proxy/v1/global-env", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => { if (Array.isArray(rows)) setSecretKeys(rows.map((r: { key: string }) => r.key)); })
      .catch(() => {});
  }, []);

  const patch = (i: number, p: Partial<UpstreamDraft>) =>
    setUps((arr) => arr.map((u, j) => (j === i ? { ...u, ...p } : u)));

  const onTest = async (i: number) => {
    const u = ups[i];
    // End-to-end test the first real model the upstream serves, against the
    // endpoint matching the chosen mode (chat vs embeddings). Falls back to a
    // plain /models probe if no model is set yet.
    const model = u.models.map((m) => m.real.trim()).find((x) => x) || undefined;
    const eb = parseExtraBody(u.extraBody);
    if (!eb.ok) { patch(i, { test: { status: "fail", message: "Extra body must be valid JSON object" } }); return; }
    patch(i, { test: { status: "running" } });
    try {
      const r = await gateway.testProxyUpstream({
        base_url: u.base_url.trim(),
        api_key_secret: u.keyMode === "secret" ? u.api_key_secret.trim() || null : null,
        api_key: u.keyMode === "paste" ? u.api_key.trim() || null : null,
        model,
        mode: u.testMode,
        extra_body: eb.value ?? null,
      });
      patch(i, { test: { status: r.ok ? "ok" : "fail", message: r.ok ? `${r.message} · ${r.latency_ms ?? "?"}ms` : r.message } });
    } catch (e) {
      patch(i, { test: { status: "fail", message: e instanceof Error ? e.message : String(e) } });
    }
  };

  const build = (): { ok: true; upstreams: ProxyUpstreamSpec[] } | { ok: false; err: string } => {
    if (!name.trim()) return { ok: false, err: "Endpoint name is required." };
    if (ups.length === 0) return { ok: false, err: "Add at least one upstream." };
    const specs: ProxyUpstreamSpec[] = [];
    for (const u of ups) {
      if (!u.name.trim()) return { ok: false, err: "Each upstream needs a name." };
      if (!u.base_url.trim()) return { ok: false, err: `Upstream "${u.name}" needs a base URL.` };
      const models: Record<string, string> = {};
      for (const m of u.models) {
        if (m.alias.trim() && m.real.trim()) models[m.alias.trim()] = m.real.trim();
      }
      if (Object.keys(models).length === 0) return { ok: false, err: `Upstream "${u.name}" needs at least one model mapping (alias → upstream model).` };
      const eb = parseExtraBody(u.extraBody);
      if (!eb.ok) return { ok: false, err: `Upstream "${u.name}": Extra body must be a valid JSON object.` };
      specs.push({
        id: u.id,
        name: u.name.trim(),
        base_url: u.base_url.trim(),
        api_key_secret: u.keyMode === "secret" ? u.api_key_secret.trim() || null : null,
        api_key: u.keyMode === "paste" ? u.api_key.trim() || null : null,
        models,
        priority: Number(u.priority) || 0,
        enabled: u.enabled,
        extra_body: eb.value ?? null,
      });
    }
    return { ok: true, upstreams: specs };
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const b = build();
    if (!b.ok) { setError(b.err); return; }
    setSubmitting(true);
    try {
      const body = {
        name: name.trim(),
        max_concurrency: Number(maxConc) || 0,
        timeout_s: Number(timeoutS) || 3600,
        enabled,
        public: isPublic,
        upstreams: b.upstreams,
      };
      const ep = editing
        ? await gateway.updateProxy(initial!.id, body)
        : await gateway.createProxy(body);
      router.push(`/proxy/${ep.id}`);
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSubmitting(false);
    }
  };

  return (
    <FormShell>
    <form onSubmit={onSubmit} className="flex w-full flex-col gap-5">
      <section data-form-section="Endpoint" className="scroll-mt-6 rounded-lg border border-border bg-card p-5">
        <h2 className="mb-4 text-base font-medium">Endpoint</h2>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="md:col-span-1">
            <Label htmlFor="px-name" className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Name</Label>
            <Input id="px-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="myteam" disabled={editing} />
            <p className="mt-1 text-[11px] text-muted-foreground">URL segment: <span className="font-mono">/proxy/{name || "myteam"}/v1/…</span></p>
          </div>
          <div>
            <Label htmlFor="px-conc" className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Max concurrency</Label>
            <Input id="px-conc" type="number" min={0} value={maxConc} onChange={(e) => setMaxConc(e.target.value)} />
            <p className="mt-1 text-[11px] text-muted-foreground">0 = unlimited (no queue)</p>
          </div>
          <div>
            <Label htmlFor="px-timeout" className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Timeout (s)</Label>
            <Input id="px-timeout" type="number" min={1} value={timeoutS} onChange={(e) => setTimeoutS(e.target.value)} />
          </div>
        </div>
        <div className="mt-3 flex items-center justify-between border-t border-border pt-3">
          <Label className="text-xs uppercase tracking-wide text-muted-foreground">Enabled</Label>
          <Switch checked={enabled} onCheckedChange={setEnabled} />
        </div>
        <div className="mt-3 flex items-center justify-between border-t border-border pt-3">
          <div>
            <Label className="text-xs uppercase tracking-wide text-muted-foreground">Public</Label>
            <p className="mt-1 text-[11px] text-muted-foreground">
              Read-only visible to every logged-in user (name, serving URL, model aliases only) and usable via the data plane. Upstreams &amp; keys stay admin-only.
            </p>
          </div>
          <Switch checked={isPublic} onCheckedChange={setIsPublic} />
        </div>
      </section>

      <section data-form-section="Upstreams" className="scroll-mt-6 rounded-lg border border-border bg-card p-5">
        <div className="mb-1 flex items-center justify-between">
          <h2 className="text-base font-medium">Upstreams</h2>
          <Button type="button" variant="outline" size="sm" onClick={() => setUps((a) => [...a, blankUpstream()])}>
            <Plus className="h-4 w-4" /> Add upstream
          </Button>
        </div>
        <p className="mb-4 text-xs text-muted-foreground">
          OpenAI-compatible backends. Lower <span className="font-mono">priority</span> is preferred; requests fail over to the next alive one. Map your stable alias (e.g. <span className="font-mono">qwen</span>) to each backend&apos;s real model name.
        </p>
        <div className="space-y-4">
          {ups.map((u, i) => (
            <div key={i} className="rounded-md border border-border p-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">Upstream {i + 1}</span>
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-muted-foreground">enabled</span>
                  <Switch checked={u.enabled} onCheckedChange={(v) => patch(i, { enabled: v })} />
                  <Button type="button" variant="ghost" size="icon-sm" className="text-muted-foreground hover:text-destructive"
                          onClick={() => setUps((a) => a.filter((_, j) => j !== i))} aria-label="Remove">
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-12">
                <div className="md:col-span-4">
                  <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Name</Label>
                  <Input value={u.name} onChange={(e) => patch(i, { name: e.target.value })} placeholder="openai-1" />
                </div>
                <div className="md:col-span-6">
                  <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Base URL</Label>
                  <Input value={u.base_url} onChange={(e) => patch(i, { base_url: e.target.value })} placeholder="https://api.openai.com/v1" className="font-mono text-xs" />
                </div>
                <div className="md:col-span-2">
                  <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Priority</Label>
                  <Input type="number" value={u.priority} onChange={(e) => patch(i, { priority: Number(e.target.value) })} />
                </div>
              </div>

              {/* API key */}
              <div className="mt-3">
                <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">API key</Label>
                <div className="mt-1 flex flex-wrap items-center gap-2">
                  <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
                    {(["secret", "paste", ...(u.hadKey ? ["keep" as const] : [])] as KeyMode[]).map((m) => (
                      <button key={m} type="button" onClick={() => patch(i, { keyMode: m })}
                              className={"rounded px-2 py-1 " + (u.keyMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                        {m === "secret" ? "Secret ref" : m === "paste" ? "Paste" : "Keep existing"}
                      </button>
                    ))}
                  </div>
                  {u.keyMode === "secret" && (
                    <Input list="px-secret-keys" value={u.api_key_secret} onChange={(e) => patch(i, { api_key_secret: e.target.value })}
                           placeholder="SECRETS_KEY (e.g. OPENAI_KEY)" className="h-8 max-w-xs font-mono text-xs" />
                  )}
                  {u.keyMode === "paste" && (
                    <Input type="password" autoComplete="off" value={u.api_key} onChange={(e) => patch(i, { api_key: e.target.value })}
                           placeholder="sk-… (stored encrypted)" className="h-8 max-w-xs font-mono text-xs" />
                  )}
                  {u.keyMode === "keep" && <span className="text-xs text-muted-foreground">existing key kept</span>}
                </div>
              </div>

              {/* model alias map */}
              <div className="mt-3">
                <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Models (alias → upstream model)</Label>
                <div className="mt-1 space-y-1.5">
                  {u.models.map((m, k) => (
                    <div key={k} className="flex items-center gap-2">
                      <Input value={m.alias} onChange={(e) => patch(i, { models: u.models.map((x, j) => j === k ? { ...x, alias: e.target.value } : x) })} placeholder="qwen" className="h-8 max-w-[200px] font-mono text-xs" />
                      <span className="text-muted-foreground">→</span>
                      <Input value={m.real} onChange={(e) => patch(i, { models: u.models.map((x, j) => j === k ? { ...x, real: e.target.value } : x) })} placeholder="Qwen/Qwen2.5-72B-Instruct" className="h-8 font-mono text-xs" />
                      <Button type="button" variant="ghost" size="icon-sm" className="text-muted-foreground hover:text-destructive"
                              onClick={() => patch(i, { models: u.models.filter((_, j) => j !== k) })} aria-label="Remove mapping">
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  ))}
                  <Button type="button" variant="ghost" size="xs" onClick={() => patch(i, { models: [...u.models, { alias: "", real: "" }] })}>
                    <Plus className="h-3 w-3" /> Add model
                  </Button>
                </div>
              </div>

              {/* extra body — optional JSON merged into every forwarded request for this upstream */}
              <div className="mt-3">
                <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Extra body (JSON)</Label>
                <Textarea value={u.extraBody} onChange={(e) => patch(i, { extraBody: e.target.value })} rows={4} spellCheck={false}
                          placeholder={'{\n  "provider": { "order": ["ModelRun"], "allow_fallbacks": false }\n}'}
                          className="font-mono text-xs" />
                <p className="mt-1 text-[11px] text-muted-foreground">
                  Optional. Merged into every forwarded body — e.g. OpenRouter <span className="font-mono">provider</span> pinning. The upstream&apos;s keys win over the caller&apos;s; <span className="font-mono">model</span> always wins.
                </p>
              </div>

              {/* test — sends a real "hello" to the endpoint matching the chosen mode (chat or embeddings) */}
              <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border/60 pt-3">
                <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
                  {(["chat", "embedding"] as const).map((m) => (
                    <button key={m} type="button" onClick={() => patch(i, { testMode: m, test: { status: "idle" } })}
                            className={"rounded px-2 py-1 " + (u.testMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                      {m === "chat" ? "Chat" : "Embedding"}
                    </button>
                  ))}
                </div>
                <Button type="button" variant="outline" size="xs" onClick={() => onTest(i)} disabled={u.test.status === "running" || !u.base_url.trim()}>
                  {u.test.status === "running" ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />} Test
                </Button>
                <span className="text-[11px] text-muted-foreground">
                  {u.testMode === "embedding"
                    ? "sends a “hello” embedding using the first model"
                    : "sends a “hello” chat completion using the first model"}
                </span>
                {u.test.status !== "idle" && u.test.status !== "running" && (
                  <span className={"text-xs " + (u.test.status === "ok" ? "text-emerald-600 dark:text-emerald-400" : "text-destructive")}>{u.test.message}</span>
                )}
              </div>
            </div>
          ))}
        </div>
        <datalist id="px-secret-keys">{secretKeys.map((k) => <option key={k} value={k} />)}</datalist>
      </section>

      <FormFooter error={error}>
        <Button type="button" variant="ghost" onClick={() => router.push(editing ? `/proxy/${initial!.id}` : "/proxy")}>Cancel</Button>
        <Button type="submit" disabled={submitting}>
          {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
          {editing ? "Save changes" : "Create endpoint"}
        </Button>
      </FormFooter>
    </form>
    </FormShell>
  );
}
