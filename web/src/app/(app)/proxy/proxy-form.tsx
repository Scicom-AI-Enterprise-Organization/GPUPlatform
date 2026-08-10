"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Plus, Trash2, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { gateway } from "@/lib/gateway";
import { RED_TEAM_DEFAULT_TYPES, type DatasetRecord, type EvalProxyRedTeamResult, type GlobalEnvRecord, type ProxyEndpoint, type ProxyRedTeam, type ProxyUpstreamSpec, type StorageRecord, type TestProxyRedTeamProbe } from "@/lib/types";
import { FormFooter, FormShell } from "@/components/form-shell";
import { RoutingPanel } from "./routing-panel";
import { modelKind } from "./model-kind";

type KeyMode = "secret" | "paste" | "keep";
type ModelPair = { alias: string; real: string };
type UpstreamDraft = {
  // Stable React key. A new upstream is PREPENDED, so every other card's index
  // shifts — keying on the index would hand the new blank card the old one's DOM
  // node (stale focus, mis-animated switches). Saved upstreams reuse their real id.
  uid: string;
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
  testMode: "chat" | "embedding" | "rerank" | "transcription" | "tts";
  // true once the admin picks a mode by hand — stops guessTestMode from overriding it
  testModeTouched: boolean;
  test: { status: "idle" | "running" | "ok" | "fail"; message?: string };
};

type TestState = { status: "idle" | "running" | "ok" | "fail"; message?: string };

let uidSeq = 0;
const newUid = () => `draft-${++uidSeq}`;

function blankUpstream(): UpstreamDraft {
  return {
    uid: newUid(),
    name: "", base_url: "", keyMode: "secret", api_key_secret: "", api_key: "",
    models: [{ alias: "", real: "" }], priority: 0, enabled: true, hadKey: false,
    extraBody: "", testMode: "chat", testModeTouched: false, test: { status: "idle" },
  };
}

// Guess which endpoint an upstream serves from its model names, so the Test toggle
// opens on the right one instead of always "Chat" (a reranker upstream 404s the chat
// test, which reads as a broken upstream when it's really the wrong probe).
// Shares modelKind with the Routing panel so the two can't disagree about a model.
// Only a hint — the admin's own click always overrides it.
function guessTestMode(models: ModelPair[]): UpstreamDraft["testMode"] {
  return modelKind(...models.map((m) => `${m.alias} ${m.real}`));
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
    u.testMode = guessTestMode(u.models);
  }
  return u;
}

export type ProxyPrefill = { name?: string; base?: string; model?: string };

const NO_SECRET = "__none__"; // Select sentinel: clear the ref (= keyless upstream)

// The "Secret ref" key picker — a dropdown of the admin-managed global secrets,
// matching /datasets/new?source=generate. It replaced a free-text datalist input:
// a mistyped key name resolved to "" at call time, so the upstream silently went
// out unauthenticated instead of failing the form.
//
// `api_key_secret` resolves against ALL of global-env (proxy_api._resolve_key), not
// just the is_secret rows, so this lists every key — a plaintext row is flagged so a
// key parked in the wrong kind of row is visible rather than surprising.
function SecretRefSelect({
  value, onChange, rows, id,
}: {
  value: string;
  onChange: (v: string) => void;
  rows: GlobalEnvRecord[];
  id?: string;
}) {
  // A ref whose secret was deleted (or that predates this dropdown) is kept as its
  // own item — otherwise opening the edit form would blank a live, still-resolving
  // key the moment the admin saved anything else.
  const missing = !!value && !rows.some((r) => r.key === value);
  if (!rows.length && !value) {
    return (
      <span className="text-xs text-muted-foreground">
        No global secrets yet — add one under{" "}
        <a href="/admin/secrets" className="underline">Secrets</a>, or switch to{" "}
        <span className="font-medium">Paste</span>.
      </span>
    );
  }
  return (
    <Select value={value || NO_SECRET} onValueChange={(v) => onChange(v === NO_SECRET ? "" : v)}>
      <SelectTrigger id={id} className="h-8 max-w-xs font-mono text-xs">
        <SelectValue placeholder="Pick a secret" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={NO_SECRET} className="text-xs text-muted-foreground">— none —</SelectItem>
        {missing && (
          <SelectItem value={value} className="font-mono text-xs">
            {value} <span className="text-destructive">— missing</span>
          </SelectItem>
        )}
        {rows.map((r) => (
          <SelectItem key={r.key} value={r.key} className="font-mono text-xs">
            {r.key}
            {r.is_secret
              ? (r.value_preview ? ` — ${r.value_preview}` : "")
              : " — plaintext"}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function fromEndpoint(ep: ProxyEndpoint): UpstreamDraft[] {
  return ep.upstreams.map((u) => {
    const models = Object.entries(u.models).map(([alias, real]) => ({ alias, real }));
    return {
      uid: u.id,
      id: u.id,
      name: u.name,
      base_url: u.base_url,
      keyMode: u.has_inline_key ? "keep" : u.api_key_secret ? "secret" : "paste",
      api_key_secret: u.api_key_secret ?? "",
      api_key: "",
      models,
      priority: u.priority,
      enabled: u.enabled,
      hadKey: u.has_inline_key || !!u.api_key_secret,
      extraBody: u.extra_body && Object.keys(u.extra_body).length ? JSON.stringify(u.extra_body, null, 2) : "",
      testMode: guessTestMode(models),
      testModeTouched: false,
      test: { status: "idle" },
    };
  });
}

export function ProxyForm({ initial, prefill }: { initial?: ProxyEndpoint; prefill?: ProxyPrefill }) {
  const router = useRouter();
  const editing = !!initial;
  const [name, setName] = useState(initial?.name ?? prefill?.name ?? "");
  const [enabled, setEnabled] = useState(initial?.enabled ?? true);
  const [isPublic, setIsPublic] = useState(initial?.public ?? false);
  const [maxConc, setMaxConc] = useState(String(initial?.max_concurrency ?? 0));
  const [timeoutS, setTimeoutS] = useState(String(initial?.timeout_s ?? 3600));
  // Sub-500 statuses that fail over to the next upstream. Defaults to the three
  // OpenRouter documents as transient — 402 out of credits, 408 timed out, 429 throttled.
  // Blank = off (every 4xx goes straight back to the caller).
  const [failoverStatus, setFailoverStatus] = useState((initial?.failover_status ?? [402, 408, 429]).join(", "));
  const parsedFailover = failoverStatus.split(/[,\s]+/).map((x) => Number(x.trim()))
    .filter((n) => Number.isFinite(n) && n >= 400 && n < 500);
  const [ups, setUps] = useState<UpstreamDraft[]>(initial ? fromEndpoint(initial) : [seededUpstream(prefill)]);
  // Which upstream is open in the graph's inline editor (edit-through-node-workflow).
  // On the create page, auto-select the single seeded upstream so its editor is visible.
  const [selectedUid, setSelectedUid] = useState<string | null>(initial ? null : (ups[0]?.uid ?? null));
  // STT callback (CER/WER) — a whisper-compatible endpoint the TTS proxy transcribes
  // its generated audio through, async, to record CER/WER. Not a data-plane upstream.
  const stt0 = initial?.stt_callback ?? undefined;
  // Default OFF when nothing is stored (same as capture below). An unconfigured block
  // has NOWHERE to record an "off": _build_stt_callback drops the whole thing when the
  // URL/model are blank, so a default of `true` made the switch snap back on after every
  // save — the user turns it off, saves, reopens, and it's on again. Off IS the truth
  // when there's no callback configured.
  const [sttEnabled, setSttEnabled] = useState(stt0?.enabled ?? false);
  const [sttBase, setSttBase] = useState(stt0?.base_url ?? "");
  const [sttModel, setSttModel] = useState(stt0?.model ?? "");
  const sttHadKey = !!(stt0?.has_inline_key || stt0?.api_key_secret);
  const [sttKeyMode, setSttKeyMode] = useState<KeyMode>(
    stt0?.has_inline_key ? "keep" : stt0?.api_key_secret ? "secret" : "paste");
  const [sttKeySecret, setSttKeySecret] = useState(stt0?.api_key_secret ?? "");
  const [sttKey, setSttKey] = useState("");
  const [sttTest, setSttTest] = useState<TestState>({ status: "idle" });
  const [secrets, setSecrets] = useState<GlobalEnvRecord[]>([]);
  // Capture drift samples — save audio (+ sidecar) to storage when a threshold is crossed.
  const cap0 = initial?.capture ?? undefined;
  const [capEnabled, setCapEnabled] = useState(cap0?.enabled ?? false);
  const [capStorage, setCapStorage] = useState(cap0?.storage_id ?? "");
  const [capPrefix, setCapPrefix] = useState(cap0?.prefix ?? "drift/");
  const numStr = (n?: number | null) => (n === null || n === undefined ? "" : String(n));
  const [capLogprob, setCapLogprob] = useState(numStr(cap0?.logprob_threshold));
  const [capCer, setCapCer] = useState(numStr(cap0?.cer_threshold));
  const [capWer, setCapWer] = useState(numStr(cap0?.wer_threshold));
  const [storages, setStorages] = useState<StorageRecord[]>([]);
  // LLM red teaming — inline screening of chat requests by a classifier or LLM judge.
  // A positive verdict never reaches an upstream; the block carries its category in
  // the X-SGPU-Red-Team-Type header.
  const rt0 = initial?.red_team ?? undefined;
  const [rtEnabled, setRtEnabled] = useState(rt0?.enabled ?? false);  // see sttEnabled
  const [rtMode, setRtMode] = useState<"classifier" | "llm">(rt0?.mode ?? "classifier");
  const [rtBase, setRtBase] = useState(rt0?.base_url ?? "");
  const [rtModel, setRtModel] = useState(rt0?.model ?? "");
  const rtHadKey = !!(rt0?.has_inline_key || rt0?.api_key_secret);
  const [rtKeyMode, setRtKeyMode] = useState<KeyMode>(
    rt0?.has_inline_key ? "keep" : rt0?.api_key_secret ? "secret" : "paste");
  const [rtKeySecret, setRtKeySecret] = useState(rt0?.api_key_secret ?? "");
  const [rtKey, setRtKey] = useState("");
  const [rtTypes, setRtTypes] = useState((rt0?.types ?? []).join(", "));
  const [rtThreshold, setRtThreshold] = useState(String(rt0?.threshold ?? 0.5));
  const [rtFlagLabels, setRtFlagLabels] = useState((rt0?.flag_labels ?? []).join(", "));
  const [rtPrompt, setRtPrompt] = useState(rt0?.prompt ?? "");
  const [rtNoSystem, setRtNoSystem] = useState(rt0?.no_system ?? false);
  const [rtReasoning, setRtReasoning] = useState(rt0?.reasoning ?? "");
  const [rtScan, setRtScan] = useState<"last_user" | "user" | "full">(rt0?.scan ?? "last_user");
  // Both are rebuilt from the spec on every save, so the form has to carry them or a
  // value set elsewhere (API/automation) silently snaps back to the default.
  const [rtMaxChars, setRtMaxChars] = useState(String(rt0?.max_chars ?? 8000));
  const [rtTimeoutS, setRtTimeoutS] = useState(String(rt0?.timeout_s ?? 15));
  const [rtOnError, setRtOnError] = useState<"allow" | "block">(rt0?.on_error ?? "allow");
  const [rtAction, setRtAction] = useState<"respond" | "llm_respond" | "error">(rt0?.action ?? "respond");
  const [rtMessage, setRtMessage] = useState(rt0?.message ?? "");
  const [rtRespBase, setRtRespBase] = useState(rt0?.responder_base_url ?? "");
  const [rtRespModel, setRtRespModel] = useState(rt0?.responder_model ?? "");
  const [rtRespPrompt, setRtRespPrompt] = useState(rt0?.responder_prompt ?? "");
  const rtRespHadKey = !!(rt0?.responder_has_inline_key || rt0?.responder_api_key_secret);
  const [rtRespKeyMode, setRtRespKeyMode] = useState<KeyMode>(
    rt0?.responder_has_inline_key ? "keep" : rt0?.responder_api_key_secret ? "secret" : "paste");
  const [rtRespKeySecret, setRtRespKeySecret] = useState(rt0?.responder_api_key_secret ?? "");
  const [rtRespKey, setRtRespKey] = useState("");
  const [rtErrorStatus, setRtErrorStatus] = useState(String(rt0?.error_status ?? 403));
  const [rtTest, setRtTest] = useState<{
    status: "idle" | "running" | "ok" | "fail";
    message?: string;
    probes?: TestProxyRedTeamProbe[];
  }>({ status: "idle" });
  // Corpus evaluation — the two-probe test says the detector WORKS, this says how
  // often it is RIGHT. Datasets are fetched lazily: most edits never open this.
  const [rtEvalOpen, setRtEvalOpen] = useState(false);
  const [rtDatasets, setRtDatasets] = useState<DatasetRecord[]>([]);
  const [rtDatasetId, setRtDatasetId] = useState("");
  const [rtEvalLimit, setRtEvalLimit] = useState("100");
  const [rtEval, setRtEval] = useState<{
    status: "idle" | "running" | "ok" | "fail";
    message?: string;
    result?: EvalProxyRedTeamResult;
  }>({ status: "idle" });
  // What the Routing graph draws as the first stage of every chat route. Built from
  // live form state (not `initial`) so the canvas answers "what will this endpoint do
  // when I save?" — and matches the server's own rule for when screening is active:
  // enabled + a detector URL + a detector model (_resolve_red_team / _build_red_team).
  const rtPreview: ProxyRedTeam | null = rtEnabled && rtBase.trim() && rtModel.trim()
    ? {
        enabled: true, mode: rtMode, base_url: rtBase.trim(), model: rtModel.trim(),
        scan: rtScan, action: rtAction, on_error: rtOnError,
        error_status: Number(rtErrorStatus) || 403,
      }
    : null;
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/proxy/v1/global-env", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => { if (Array.isArray(rows)) setSecrets(rows as GlobalEnvRecord[]); })
      .catch(() => {});
    fetch("/api/proxy/v1/storage", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => { if (Array.isArray(rows)) setStorages(rows as StorageRecord[]); })
      .catch(() => {});
  }, []);

  const patch = (i: number, p: Partial<UpstreamDraft>) =>
    setUps((arr) => arr.map((u, j) => {
      if (j !== i) return u;
      const next = { ...u, ...p };
      // Re-guess on every model edit until the admin picks a mode by hand: typing
      // "Qwen3-Reranker-8B" should move the toggle off Chat without being told.
      if (p.models && !next.testModeTouched) next.testMode = guessTestMode(next.models);
      return next;
    }));

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

  // Dry-run the detector against a known-bad + a known-good probe. Uses the LIVE form
  // values, so it answers "will screening work if I save this?" — including the case a
  // single attack probe can't see: a detector that flags everything.
  const onTestRedTeam = async () => {
    if (!rtBase.trim() || !rtModel.trim()) {
      setRtTest({ status: "fail", message: "Set the detector base URL + model first" });
      return;
    }
    setRtTest({ status: "running" });
    try {
      const r = await gateway.testProxyRedTeam({
        mode: rtMode,
        base_url: rtBase.trim(),
        model: rtModel.trim(),
        api_key_secret: rtKeyMode === "secret" ? (rtKeySecret.trim() || null) : null,
        api_key: rtKeyMode === "paste" ? (rtKey.trim() || null) : null,
        // "Keep existing" sends no key at all — the server tests the stored one.
        proxy_id: rtKeyMode === "keep" ? (initial?.id ?? null) : null,
        types: rtTypes.split(/[,\n]+/).map((s) => s.trim()).filter(Boolean),
        threshold: rtThreshold.trim() === "" ? 0.5 : Number(rtThreshold),
        flag_labels: rtFlagLabels.split(/[,\n]+/).map((s) => s.trim()).filter(Boolean),
        prompt: rtPrompt.trim() || null,
        no_system: rtNoSystem,
        reasoning: rtReasoning,
        timeout_s: Number(rtTimeoutS) || 15,
      });
      setRtTest({
        status: r.ok ? "ok" : "fail",
        message: `${r.message}${r.latency_ms != null ? ` · ${r.latency_ms}ms` : ""}`,
        probes: r.probes ?? [],
      });
    } catch (e) {
      setRtTest({ status: "fail", message: e instanceof Error ? e.message : String(e) });
    }
  };

  const openRedTeamEval = () => {
    setRtEvalOpen(true);
    if (rtDatasets.length) return;
    gateway.listDatasets()
      .then((rows) => setRtDatasets(Array.isArray(rows) ? rows : []))
      .catch(() => {});
  };

  const onEvalRedTeam = async () => {
    if (!rtDatasetId) {
      setRtEval({ status: "fail", message: "Pick a dataset with expected.attack labels" });
      return;
    }
    setRtEval({ status: "running" });
    try {
      const r = await gateway.evaluateProxyRedTeam({
        mode: rtMode,
        base_url: rtBase.trim(),
        model: rtModel.trim(),
        api_key_secret: rtKeyMode === "secret" ? (rtKeySecret.trim() || null) : null,
        api_key: rtKeyMode === "paste" ? (rtKey.trim() || null) : null,
        proxy_id: rtKeyMode === "keep" ? (initial?.id ?? null) : null,
        types: rtTypes.split(/[,\n]+/).map((s) => s.trim()).filter(Boolean),
        threshold: rtThreshold.trim() === "" ? 0.5 : Number(rtThreshold),
        flag_labels: rtFlagLabels.split(/[,\n]+/).map((s) => s.trim()).filter(Boolean),
        prompt: rtPrompt.trim() || null,
        no_system: rtNoSystem,
        reasoning: rtReasoning,
        timeout_s: Number(rtTimeoutS) || 15,
        dataset_id: rtDatasetId,
        limit: Number(rtEvalLimit) || 100,
      });
      setRtEval({ status: r.ok ? "ok" : "fail", message: r.message, result: r });
    } catch (e) {
      setRtEval({ status: "fail", message: e instanceof Error ? e.message : String(e) });
    }
  };

  const onTestStt = async () => {
    if (!sttBase.trim() || !sttModel.trim()) {
      setSttTest({ status: "fail", message: "Set the STT base URL + model first" });
      return;
    }
    setSttTest({ status: "running" });
    try {
      const r = await gateway.testProxyUpstream({
        base_url: sttBase.trim(),
        api_key_secret: sttKeyMode === "secret" ? (sttKeySecret.trim() || null) : null,
        api_key: sttKeyMode === "paste" ? (sttKey.trim() || null) : null,
        model: sttModel.trim(),
        mode: "transcription",
      });
      setSttTest({ status: r.ok ? "ok" : "fail", message: r.ok ? `${r.message} · ${r.latency_ms ?? "?"}ms` : r.message });
    } catch (e) {
      setSttTest({ status: "fail", message: e instanceof Error ? e.message : String(e) });
    }
  };

  // The upstream editor, rendered inline inside the Routing node graph when a backend node
  // is selected. Same fields as the old standalone Upstreams list — name, URL, priority,
  // API key, model alias→name mappings, extra body, test — just reached by clicking a node.
  const renderUpstreamEditor = (uid: string) => {
    const i = ups.findIndex((x) => x.uid === uid);
    if (i < 0) return null;
    const u = ups[i];
    return (
      <div>
        <div className="mb-3 flex items-center justify-end gap-2">
          <span className="text-[11px] text-muted-foreground">enabled</span>
          <Switch checked={u.enabled} onCheckedChange={(v) => patch(i, { enabled: v })} />
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
              <SecretRefSelect value={u.api_key_secret} rows={secrets}
                               onChange={(v) => patch(i, { api_key_secret: v })} />
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
                <Input value={m.real} onChange={(e) => patch(i, { models: u.models.map((x, j) => j === k ? { ...x, real: e.target.value } : x) })} placeholder="Qwen/Qwen2.5-72B-Instruct" className="h-8 min-w-0 flex-1 font-mono text-xs" />
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

        {/* test — sends a real "hello" to the endpoint matching the chosen mode */}
        <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border/60 pt-3">
          <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
            {(["chat", "embedding", "rerank", "transcription", "tts"] as const).map((m) => (
              <button key={m} type="button" onClick={() => patch(i, { testMode: m, testModeTouched: true, test: { status: "idle" } })}
                      className={"rounded px-2 py-1 " + (u.testMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                {{ chat: "Chat", embedding: "Embedding", rerank: "Rerank", transcription: "Transcribe", tts: "TTS" }[m]}
              </button>
            ))}
          </div>
          <Button type="button" variant="outline" size="xs" onClick={() => onTest(i)} disabled={u.test.status === "running" || !u.base_url.trim()}>
            {u.test.status === "running" ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />} Test
          </Button>
          <span className="text-[11px] text-muted-foreground">
            {u.testMode === "embedding" ? "sends a “hello” embedding"
              : u.testMode === "rerank" ? "ranks 3 docs (incl. a hard negative) via /rerank"
              : u.testMode === "transcription" ? "sends a short tone to /audio/transcriptions"
              : u.testMode === "tts" ? "synthesizes “hello world” via /audio/speech"
              : "sends a “hello” chat completion"} using the first model
          </span>
          {u.test.status !== "idle" && u.test.status !== "running" && (
            <span className={"text-xs " + (u.test.status === "ok" ? "text-emerald-600 dark:text-emerald-400" : "text-destructive")}>{u.test.message}</span>
          )}
        </div>
      </div>
    );
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
    if (rtBase.trim() && rtAction === "llm_respond" && rtMode === "classifier"
        && !(rtRespBase.trim() && rtRespModel.trim())) {
      setError("Red teaming: “LLM writes the refusal” with a classifier detector needs a responder base URL + model.");
      return;
    }
    setSubmitting(true);
    try {
      const body = {
        name: name.trim(),
        max_concurrency: Number(maxConc) || 0,
        timeout_s: Number(timeoutS) || 3600,
        failover_status: parsedFailover,
        enabled,
        public: isPublic,
        upstreams: b.upstreams,
        // Always sent: blank base_url/model clears the callback server-side (edit) or
        // is ignored (create). keyMode "keep" leaves the stored key untouched.
        stt_callback: {
          enabled: sttEnabled,
          base_url: sttBase.trim(),
          model: sttModel.trim(),
          api_key_secret: sttKeyMode === "secret" ? (sttKeySecret.trim() || null) : null,
          api_key: sttKeyMode === "paste" ? (sttKey.trim() || null) : null,
        },
        // Always sent: blank storage clears it (edit) / is ignored (create). Empty
        // threshold → null (that dimension never triggers).
        capture: {
          enabled: capEnabled,
          storage_id: capStorage.trim(),
          prefix: capPrefix.trim(),
          logprob_threshold: capLogprob.trim() === "" ? null : Number(capLogprob),
          cer_threshold: capCer.trim() === "" ? null : Number(capCer),
          wer_threshold: capWer.trim() === "" ? null : Number(capWer),
        },
        // Always sent: blank base_url/model clears red teaming server-side (edit) or
        // is ignored (create). keyMode "keep" leaves stored keys untouched.
        red_team: {
          enabled: rtEnabled,
          mode: rtMode,
          base_url: rtBase.trim(),
          model: rtModel.trim(),
          api_key_secret: rtKeyMode === "secret" ? (rtKeySecret.trim() || null) : null,
          api_key: rtKeyMode === "paste" ? (rtKey.trim() || null) : null,
          types: rtTypes.split(/[,\n]+/).map((s) => s.trim()).filter(Boolean),
          threshold: rtThreshold.trim() === "" ? 0.5 : Number(rtThreshold),
          flag_labels: rtFlagLabels.split(/[,\n]+/).map((s) => s.trim()).filter(Boolean),
          prompt: rtPrompt.trim() || null,
          no_system: rtNoSystem,
          reasoning: rtReasoning as NonNullable<ProxyEndpoint["red_team"]>["reasoning"],
          scan: rtScan,
          max_chars: Number(rtMaxChars) || 8000,
          timeout_s: Number(rtTimeoutS) || 15,
          on_error: rtOnError,
          action: rtAction,
          message: rtMessage.trim(),
          responder_base_url: rtRespBase.trim(),
          responder_model: rtRespModel.trim(),
          responder_prompt: rtRespPrompt.trim() || null,
          responder_api_key_secret: rtRespKeyMode === "secret" ? (rtRespKeySecret.trim() || null) : null,
          responder_api_key: rtRespKeyMode === "paste" ? (rtRespKey.trim() || null) : null,
          error_status: Number(rtErrorStatus) || 403,
        },
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
          <div>
            <Label htmlFor="px-failover" className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Fail over on status</Label>
            <Input id="px-failover" value={failoverStatus} onChange={(e) => setFailoverStatus(e.target.value)} placeholder="402, 408, 429" />
            <p className="mt-1 text-[11px] text-muted-foreground">
              4xx codes that try the next upstream — a backend that&apos;s up but can&apos;t serve
              right now. 5xx and timeouts always do. Blank = off.
            </p>
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

      <section data-form-section="Routing" className="scroll-mt-6 rounded-lg border border-border bg-card p-5">
        <h2 className="text-base font-medium">Routing &amp; upstreams</h2>
        <p className="mb-4 mt-1 text-xs text-muted-foreground">
          Which backend serves each model, and where it goes when that backend fails. Add a backend,
          or click any node to edit its URL, priority and model mappings. Expand a model for the full chain.
        </p>
        <RoutingPanel
          upstreams={ups}
          maxConcurrency={Number(maxConc) || 0}
          timeoutS={Number(timeoutS) || 0}
          failoverStatus={parsedFailover}
          proxyId={initial?.id}
          // Live, unsaved guard state — edit the Red teaming card below and the chat
          // routes above redraw immediately, same as editing an upstream does.
          redTeam={rtPreview}
          defaultOpen={false}
          // Edit page: the graph IS the upstream editor. Clicking a node selects it; the
          // editor for that upstream renders inline below the graph.
          editable
          selectedUid={selectedUid}
          onSelect={setSelectedUid}
          renderEditor={renderUpstreamEditor}
          // PREPEND, not append: a new backend lands in view. Position is cosmetic — routing
          // sorts by priority + liveness, and the save path matches kept keys by id, not order.
          onAddUpstream={() => {
            const u = blankUpstream();
            setUps((a) => [u, ...a]);
            setSelectedUid(u.uid);
          }}
          onDeleteUpstream={(uid) => {
            setUps((a) => a.filter((u) => u.uid !== uid));
            setSelectedUid((cur) => (cur === uid ? null : cur));
          }}
          // Ties break on list position, so moving an upstream to the front is what makes
          // it the primary — the only way to change the winner without editing a priority.
          onPromote={(uid) => setUps((a) => {
            const i = a.findIndex((u) => u.uid === uid);
            if (i <= 0) return a;
            const next = [...a];
            next.unshift(next.splice(i, 1)[0]);
            return next;
          })}
        />
      </section>

      <section data-form-section="Red teaming" className="scroll-mt-6 rounded-lg border border-border bg-card p-5">
        <div className="mb-1 flex items-center justify-between">
          <h2 className="text-base font-medium">Red teaming — chat guardrail</h2>
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-muted-foreground">enabled</span>
            <Switch checked={rtEnabled} onCheckedChange={setRtEnabled} />
          </div>
        </div>
        <p className="mb-4 text-xs text-muted-foreground">
          Optional — screen every <span className="font-mono">/v1/chat/completions</span> request through a detector <span className="font-medium">before</span> it
          reaches an upstream. A hit is answered by this gateway (the model never sees the request) and carries its category in the
          <span className="font-mono"> X-SGPU-Red-Team-Type</span> response header. Leave the URL/model blank to disable. Detector latency is paid inline by each request.
        </p>
        {/* Collapsed while the switch is off — the fields keep their state and
            reappear on re-enable; the stored config is untouched until save. */}
        {rtEnabled && (<>

        {/* detector */}
        <div className="mb-3">
          <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Detector</Label>
          <div className="mt-1 inline-flex rounded-md border border-border p-0.5 text-xs">
            {(["classifier", "llm"] as const).map((m) => (
              <button key={m} type="button" onClick={() => setRtMode(m)}
                      className={"rounded px-2 py-1 " + (rtMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                {m === "classifier" ? "Classifier" : "LLM as judge"}
              </button>
            ))}
          </div>
          <p className="mt-1 text-[11px] text-muted-foreground">
            {rtMode === "classifier"
              ? <>a classification endpoint — vLLM <span className="font-mono">/classify</span> (server root, e.g. a prompt-injection classifier) or an OpenAI-style <span className="font-mono">/v1/moderations</span> URL pasted in full</>
              : <>any OpenAI-compatible chat-completions endpoint — server root, <span className="font-mono">/v1</span> base, or the full <span className="font-mono">/chat/completions</span> URL. The judge answers <span className="font-mono">UNSAFE &lt;type&gt;</span> or <span className="font-mono">SAFE</span> (Llama-Guard-style models work too)</>}
          </p>
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-12">
          <div className="md:col-span-7">
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Detector base URL</Label>
            <Input value={rtBase} onChange={(e) => setRtBase(e.target.value)}
                   placeholder={rtMode === "classifier" ? "http://guard-box:8000 (or …/v1/moderations)" : "https://…/proxy/guard/v1"}
                   className="font-mono text-xs" />
          </div>
          <div className="md:col-span-5">
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Detector model</Label>
            <Input value={rtModel} onChange={(e) => setRtModel(e.target.value)}
                   placeholder={rtMode === "classifier" ? "protectai/deberta-v3-base-prompt-injection-v2" : "meta-llama/Llama-Guard-4-12B"}
                   className="font-mono text-xs" />
          </div>
        </div>
        <div className="mt-3">
          <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Detector API key</Label>
          <div className="mt-1 flex flex-wrap items-center gap-2">
            <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
              {(["secret", "paste", ...(rtHadKey ? ["keep" as const] : [])] as KeyMode[]).map((m) => (
                <button key={m} type="button" onClick={() => setRtKeyMode(m)}
                        className={"rounded px-2 py-1 " + (rtKeyMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                  {m === "secret" ? "Secret ref" : m === "paste" ? "Paste" : "Keep existing"}
                </button>
              ))}
            </div>
            {rtKeyMode === "secret" && (
              <SecretRefSelect value={rtKeySecret} rows={secrets} onChange={setRtKeySecret} />
            )}
            {rtKeyMode === "paste" && (
              <Input type="password" autoComplete="off" value={rtKey} onChange={(e) => setRtKey(e.target.value)}
                     placeholder="sk-… (stored encrypted)" className="h-8 max-w-xs font-mono text-xs" />
            )}
            {rtKeyMode === "keep" && <span className="text-xs text-muted-foreground">existing key kept</span>}
            <span className="text-[11px] text-muted-foreground">optional — omit for a keyless detector</span>
          </div>
        </div>

        {/* taxonomy */}
        <div className="mt-3">
          <div className="mb-1.5 flex items-center justify-between">
            <Label className="text-xs uppercase tracking-wide text-muted-foreground">Attack types</Label>
            {/* Materialize the built-in taxonomy so it can be edited/extended instead of
                retyped — blank still means "server default", so this is purely a seed. */}
            <Button type="button" variant="ghost" size="xs"
                    onClick={() => setRtTypes(RED_TEAM_DEFAULT_TYPES.join(", "))}
                    disabled={rtTypes.trim() === RED_TEAM_DEFAULT_TYPES.join(", ")}>
              <Plus className="h-3 w-3" /> Use defaults
            </Button>
          </div>
          <Input value={rtTypes} onChange={(e) => setRtTypes(e.target.value)}
                 placeholder={RED_TEAM_DEFAULT_TYPES.join(", ")}
                 className="font-mono text-xs" />
          <p className="mt-1 text-[11px] text-muted-foreground">
            Comma-separated taxonomy reported in <span className="font-mono">X-SGPU-Red-Team-Type</span> — the judge picks from this list; a classifier
            label / moderation category is matched against it. Blank = the built-in default shown above; an unmatched hit reports <span className="font-mono">unclassified</span>.
          </p>
        </div>

        {rtMode === "classifier" ? (
          <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-12">
            <div className="md:col-span-3">
              <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Threshold</Label>
              <Input type="number" step="0.05" min={0} max={1} value={rtThreshold} onChange={(e) => setRtThreshold(e.target.value)} className="font-mono text-xs" />
              <p className="mt-1 text-[11px] text-muted-foreground">flag if probability ≥ this</p>
            </div>
            <div className="md:col-span-9">
              <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Flag labels</Label>
              <Input value={rtFlagLabels} onChange={(e) => setRtFlagLabels(e.target.value)}
                     placeholder="unsafe, injection, jailbreak, … (blank = built-in set)" className="font-mono text-xs" />
              <p className="mt-1 text-[11px] text-muted-foreground">classifier labels counted as a hit (substring, case-insensitive) — e.g. <span className="font-mono">INJECTION</span>, <span className="font-mono">LABEL_1</span></p>
            </div>
          </div>
        ) : (
          <>
            <div className="mt-3">
              <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Judge prompt</Label>
              <Textarea value={rtPrompt} onChange={(e) => setRtPrompt(e.target.value)} rows={3} spellCheck={false}
                        placeholder="Blank = built-in: answer 'UNSAFE <type>' (from the attack types above) or 'SAFE'."
                        className="font-mono text-xs" />
            </div>
            <div className="mt-3 flex items-center justify-between border-t border-border/60 pt-3">
              <div>
                <Label className="text-xs uppercase tracking-wide text-muted-foreground">No system prompt</Label>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  send only the scanned text — for guard models whose chat template bakes in the policy (e.g. Llama-Guard, which answers <span className="font-mono">unsafe + S-code</span>)
                </p>
              </div>
              <Switch checked={rtNoSystem} onCheckedChange={setRtNoSystem} />
            </div>
          </>
        )}

        {/* scope + failure policy + reasoning control */}
        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Scan</Label>
            <Select value={rtScan} onValueChange={(v) => setRtScan(v as typeof rtScan)}>
              <SelectTrigger className="h-9 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="last_user" className="text-xs">Last user message (default)</SelectItem>
                <SelectItem value="user" className="text-xs">All user messages</SelectItem>
                <SelectItem value="full" className="text-xs">Full conversation</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">If the detector fails</Label>
            <Select value={rtOnError} onValueChange={(v) => setRtOnError(v as typeof rtOnError)}>
              <SelectTrigger className="h-9 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="allow" className="text-xs">Allow the request (fail open)</SelectItem>
                <SelectItem value="block" className="text-xs">Block the request (fail closed)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Reasoning</Label>
            {/* Radix Select rejects value="" — "__default" stands in for model default. */}
            <Select value={rtReasoning || "__default"} onValueChange={(v) => setRtReasoning(v === "__default" ? "" : (v as typeof rtReasoning))}>
              <SelectTrigger className="h-9 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="__default" className="text-xs">Model default</SelectItem>
                <SelectItem value="disable" className="text-xs">Disable thinking (enable_thinking=false)</SelectItem>
                <SelectItem value="none" className="text-xs">reasoning_effort: none</SelectItem>
                <SelectItem value="minimal" className="text-xs">reasoning_effort: minimal</SelectItem>
                <SelectItem value="low" className="text-xs">reasoning_effort: low</SelectItem>
                <SelectItem value="medium" className="text-xs">reasoning_effort: medium</SelectItem>
                <SelectItem value="high" className="text-xs">reasoning_effort: high</SelectItem>
              </SelectContent>
            </Select>
            <p className="mt-1 text-[11px] text-muted-foreground">
              applies to the judge <span className="font-italic">and</span> responder calls — a reasoning model left thinking can spend its whole token budget and return an empty verdict
            </p>
          </div>
        </div>

        {/* how much text the detector sees, and how long the request waits for it */}
        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Max scanned chars</Label>
            <Input type="number" min={200} step={100} value={rtMaxChars}
                   onChange={(e) => setRtMaxChars(e.target.value)} className="font-mono text-xs" />
            <p className="mt-1 text-[11px] text-muted-foreground">
              the scanned text is truncated from the <span className="font-italic">head</span> to this length — injections ride at the END of long context. Default 8000, floor 200.
            </p>
          </div>
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Detector timeout (s)</Label>
            <Input type="number" min={1} step={1} value={rtTimeoutS}
                   onChange={(e) => setRtTimeoutS(e.target.value)} className="font-mono text-xs" />
            <p className="mt-1 text-[11px] text-muted-foreground">
              read timeout on the detector call, paid inline by every chat request; a timeout follows the failure policy above. Default 15.
            </p>
          </div>
        </div>

        {/* test — runs the settings above through the SAME detector path the live gate
            uses, on two probes. The benign control is the point: a detector that flags
            everything passes an attack-only test and then refuses all traffic. */}
        <div className="mt-3 border-t border-border/60 pt-3">
          <div className="flex flex-wrap items-center gap-2">
            <Button type="button" variant="outline" size="xs" onClick={onTestRedTeam}
                    disabled={rtTest.status === "running" || !rtBase.trim() || !rtModel.trim()}>
              {rtTest.status === "running" ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />} Test detector
            </Button>
            <span className="text-[11px] text-muted-foreground">
              screens one attack probe and one benign probe with the settings above — nothing is saved, and a
              detector that flags <span className="font-italic">everything</span> fails this too
              {rtKeyMode === "keep" && " (tests the key already stored on this endpoint)"}
            </span>
          </div>
          {rtTest.status !== "idle" && rtTest.status !== "running" && (
            <>
              <p className={"mt-2 text-xs " + (rtTest.status === "ok" ? "text-emerald-600 dark:text-emerald-400" : "text-destructive")}>
                {rtTest.message}
              </p>
              {!!rtTest.probes?.length && (
                <div className="mt-1.5 space-y-1 rounded-md border border-border bg-muted/20 p-2 font-mono text-[10px] leading-relaxed">
                  {rtTest.probes.map((p) => (
                    <div key={p.label} className="break-words">
                      <span className={p.ok ? "text-emerald-600 dark:text-emerald-400" : "text-destructive"}>
                        {p.ok ? "✓" : "✗"}
                      </span>{" "}
                      <span className="text-muted-foreground">{p.label} (expect {p.expected}):</span>{" "}
                      {p.error
                        ? <span className="text-destructive">{p.error}</span>
                        : <>{p.flagged ? `flagged${p.rt_type ? ` · ${p.rt_type}` : ""}` : "passed"}
                            {p.latency_ms != null && ` · ${p.latency_ms}ms`}
                            {p.reason && <span className="text-muted-foreground"> · {p.reason}</span>}</>}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {/* corpus evaluation — two probes prove it runs; a labelled corpus proves
              it's right. Attack-only corpora report recall and ABSTAIN on precision. */}
          {!rtEvalOpen ? (
            <button type="button" onClick={openRedTeamEval}
                    className="mt-2 text-[11px] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline">
              Evaluate against a labelled dataset →
            </button>
          ) : (
            <div className="mt-3 rounded-md border border-border bg-muted/10 p-2.5">
              <div className="flex flex-wrap items-end gap-2">
                <div className="min-w-[240px] flex-1">
                  <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Labelled dataset</Label>
                  <Select value={rtDatasetId} onValueChange={setRtDatasetId}>
                    <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="pick a red-team corpus…" /></SelectTrigger>
                    <SelectContent>
                      {rtDatasets.map((d) => (
                        <SelectItem key={d.id} value={d.id} className="text-xs">
                          {d.name}{d.num_rows ? ` · ${d.num_rows} rows` : ""}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="w-[110px]">
                  <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Rows</Label>
                  <Input type="number" min={1} max={500} value={rtEvalLimit}
                         onChange={(e) => setRtEvalLimit(e.target.value)} className="h-8 font-mono text-xs" />
                </div>
                <Button type="button" variant="outline" size="xs" onClick={onEvalRedTeam}
                        disabled={rtEval.status === "running" || !rtBase.trim() || !rtModel.trim()}>
                  {rtEval.status === "running" ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />} Evaluate
                </Button>
              </div>
              <p className="mt-1.5 text-[11px] text-muted-foreground">
                Needs rows carrying <span className="font-mono">expected.attack</span> — the red-team dataset
                generator writes them. One detector call per row (billed); rows without a label are skipped, not guessed.
              </p>
              {rtEval.status !== "idle" && rtEval.status !== "running" && (
                <>
                  <p className={"mt-2 text-xs " + (rtEval.status === "ok" ? "text-emerald-600 dark:text-emerald-400" : "text-destructive")}>
                    {rtEval.message}
                  </p>
                  {rtEval.result && rtEval.result.scored > 0 && (() => {
                    const r = rtEval.result;
                    const pct = (v?: number | null) => (v == null ? "n/a" : `${(v * 100).toFixed(0)}%`);
                    return (
                      <div className="mt-2 space-y-2">
                        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                          {([["Recall", pct(r.recall), "attacks caught"],
                             ["Precision", pct(r.precision), "of blocks that were attacks"],
                             ["F1", r.f1 == null ? "n/a" : r.f1.toFixed(2), "needs both halves"],
                             ["Category accuracy", pct(r.type_accuracy), "named the right type"]] as const).map(([l, v, h]) => (
                            <div key={l} className="rounded-md border border-border bg-card px-2 py-1.5">
                              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{l}</div>
                              <div className="font-mono text-sm">{v}</div>
                              <div className="text-[10px] text-muted-foreground">{h}</div>
                            </div>
                          ))}
                        </div>
                        <div className="font-mono text-[10px] text-muted-foreground">
                          TP {r.true_positives} · FN {r.false_negatives} · TN {r.true_negatives} · FP {r.false_positives}
                          {" · "}{r.attack_rows} attack / {r.benign_rows} benign rows
                          {r.skipped > 0 && ` · ${r.skipped} unlabelled skipped`}
                          {r.errors > 0 && ` · ${r.errors} detector errors`}
                          {r.latency_ms_p50 != null && ` · p50 ${r.latency_ms_p50}ms / p95 ${r.latency_ms_p95}ms`}
                        </div>
                        {Object.keys(r.recall_by_type).length > 0 && (
                          <div className="space-y-0.5 font-mono text-[10px]">
                            {Object.entries(r.recall_by_type).map(([t, v]) => (
                              <div key={t} className="flex items-center gap-2">
                                <span className="w-52 shrink-0 truncate text-muted-foreground">{t}</span>
                                <span className={v < 1 ? "text-destructive" : "text-emerald-600 dark:text-emerald-400"}>{pct(v)}</span>
                                <span className="text-muted-foreground">({r.rows_by_type[t]} rows)</span>
                              </div>
                            ))}
                          </div>
                        )}
                        {r.misses.length > 0 && (
                          <div className="space-y-1 rounded-md border border-border bg-muted/20 p-2 font-mono text-[10px] leading-relaxed">
                            <div className="text-muted-foreground">rows it got wrong — this is the tuning list:</div>
                            {r.misses.map((m, i) => (
                              <div key={i} className="break-words">
                                <span className="text-destructive">
                                  {m.kind === "false_negative" ? "MISSED" : m.kind === "false_positive" ? "OVER-BLOCKED" : "ERROR"}
                                </span>{" "}
                                {(m.attack_type || m.predicted_type) && (
                                  <span className="text-muted-foreground">[{m.attack_type || m.predicted_type}]</span>
                                )}{" "}
                                {m.text || m.reason}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </>
              )}
            </div>
          )}
        </div>

        {/* action on a hit */}
        <div className="mt-4 border-t border-border/60 pt-3">
          <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">On a hit</Label>
          <div className="mt-1 inline-flex rounded-md border border-border p-0.5 text-xs">
            {(["respond", "llm_respond", "error"] as const).map((m) => (
              <button key={m} type="button" onClick={() => setRtAction(m)}
                      className={"rounded px-2 py-1 " + (rtAction === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                {{ respond: "Canned reply", llm_respond: "LLM writes the refusal", error: "HTTP error" }[m]}
              </button>
            ))}
          </div>
          <p className="mt-1 text-[11px] text-muted-foreground">
            {rtAction === "respond" ? <>return a normal chat completion with the message below (<span className="font-mono">finish_reason: content_filter</span>) — SDKs keep working, streamed requests get an SSE-shaped reply</>
              : rtAction === "llm_respond" ? <>a responder LLM writes a contextual refusal (falls back to the canned message if it fails)</>
              : <>reject outright with the status code below and an <span className="font-mono">{"{error: …}"}</span> body</>}
          </p>
        </div>
        <div className="mt-3">
          <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">
            {rtAction === "error" ? "Error message"
              : rtAction === "llm_respond" ? "Fallback message"
              : "Canned message"}
          </Label>
          <Textarea value={rtMessage} onChange={(e) => setRtMessage(e.target.value)} rows={2}
                    placeholder="Blank = built-in: “I can't help with that request — it was flagged by this endpoint's safety screening.”"
                    className="text-xs" />
          {rtAction === "llm_respond" && (
            /* Not leftover from the canned-reply action: `_rt_llm_respond` returns None on
               ANY responder failure and the gate falls back to this text, so a dead
               responder can't turn a block into a 502. */
            <p className="mt-1 text-[11px] text-muted-foreground">
              Still used: this is the fallback whenever the responder LLM fails — non-2xx, timeout, or an
              empty reply — so a broken responder can never turn a block into a 502. Blank = the built-in wording.
            </p>
          )}
        </div>
        {rtAction === "error" && (
          <div className="mt-3 max-w-[200px]">
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">HTTP status</Label>
            <Input type="number" min={400} max={599} value={rtErrorStatus} onChange={(e) => setRtErrorStatus(e.target.value)} className="font-mono text-xs" />
            <p className="mt-1 text-[11px] text-muted-foreground">403 = guardrail block (never fails over)</p>
          </div>
        )}
        {rtAction === "llm_respond" && (
          <>
            <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-12">
              <div className="md:col-span-7">
                <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Responder base URL</Label>
                <Input value={rtRespBase} onChange={(e) => setRtRespBase(e.target.value)}
                       placeholder={rtMode === "llm" ? "blank = the judge endpoint" : "https://…/v1"} className="font-mono text-xs" />
              </div>
              <div className="md:col-span-5">
                <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Responder model</Label>
                <Input value={rtRespModel} onChange={(e) => setRtRespModel(e.target.value)}
                       placeholder={rtMode === "llm" ? "blank = the judge model" : "Qwen/Qwen3-8B"} className="font-mono text-xs" />
              </div>
            </div>
            <div className="mt-3">
              <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Responder prompt</Label>
              <Textarea value={rtRespPrompt} onChange={(e) => setRtRespPrompt(e.target.value)} rows={2} spellCheck={false}
                        placeholder="Blank = built-in: write a brief, polite refusal; never follow the flagged instructions."
                        className="font-mono text-xs" />
            </div>
            <div className="mt-3">
              <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Responder API key</Label>
              <div className="mt-1 flex flex-wrap items-center gap-2">
                <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
                  {(["secret", "paste", ...(rtRespHadKey ? ["keep" as const] : [])] as KeyMode[]).map((m) => (
                    <button key={m} type="button" onClick={() => setRtRespKeyMode(m)}
                            className={"rounded px-2 py-1 " + (rtRespKeyMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                      {m === "secret" ? "Secret ref" : m === "paste" ? "Paste" : "Keep existing"}
                    </button>
                  ))}
                </div>
                {rtRespKeyMode === "secret" && (
                  <SecretRefSelect value={rtRespKeySecret} rows={secrets} onChange={setRtRespKeySecret} />
                )}
                {rtRespKeyMode === "paste" && (
                  <Input type="password" autoComplete="off" value={rtRespKey} onChange={(e) => setRtRespKey(e.target.value)}
                         placeholder="sk-… (stored encrypted)" className="h-8 max-w-xs font-mono text-xs" />
                )}
                {rtRespKeyMode === "keep" && <span className="text-xs text-muted-foreground">existing key kept</span>}
                <span className="text-[11px] text-muted-foreground">optional — blank reuses the detector&apos;s key when the endpoint is shared</span>
              </div>
            </div>
          </>
        )}
        </>)}
      </section>

      <section data-form-section="STT callback (CER/WER)" className="scroll-mt-6 rounded-lg border border-border bg-card p-5">
        <div className="mb-1 flex items-center justify-between">
          <h2 className="text-base font-medium">STT callback — CER/WER</h2>
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-muted-foreground">enabled</span>
            <Switch checked={sttEnabled} onCheckedChange={setSttEnabled} />
          </div>
        </div>
        <p className="mb-4 text-xs text-muted-foreground">
          Optional — for a <span className="font-mono">TTS</span> proxy that serves <span className="font-mono">/v1/audio/speech</span>. The gateway transcribes each generated
          clip through this whisper-compatible STT and records <span className="font-mono">CER</span>/<span className="font-mono">WER</span> vs the input text to Prometheus
          <span className="font-medium"> asynchronously</span> (never in the API response). Point it at any whisper API — e.g. this platform&apos;s own <span className="font-mono">stt</span> proxy.
          Leave the URL/model blank to disable.
        </p>
        {sttEnabled && (<>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-12">
          <div className="md:col-span-7">
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">STT base URL</Label>
            <Input value={sttBase} onChange={(e) => setSttBase(e.target.value)}
                   placeholder="https://inferencegpu.aies.scicom.dev/proxy/stt/v1" className="font-mono text-xs" />
          </div>
          <div className="md:col-span-5">
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">STT model</Label>
            <Input value={sttModel} onChange={(e) => setSttModel(e.target.value)}
                   placeholder="scicom-ai-enterprise/whisper-large-v3-turbo-…" className="font-mono text-xs" />
          </div>
        </div>
        <div className="mt-3">
          <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">API key</Label>
          <div className="mt-1 flex flex-wrap items-center gap-2">
            <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
              {(["secret", "paste", ...(sttHadKey ? ["keep" as const] : [])] as KeyMode[]).map((m) => (
                <button key={m} type="button" onClick={() => setSttKeyMode(m)}
                        className={"rounded px-2 py-1 " + (sttKeyMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                  {m === "secret" ? "Secret ref" : m === "paste" ? "Paste" : "Keep existing"}
                </button>
              ))}
            </div>
            {sttKeyMode === "secret" && (
              <SecretRefSelect value={sttKeySecret} rows={secrets} onChange={setSttKeySecret} />
            )}
            {sttKeyMode === "paste" && (
              <Input type="password" autoComplete="off" value={sttKey} onChange={(e) => setSttKey(e.target.value)}
                     placeholder="sgpu-… (stored encrypted)" className="h-8 max-w-xs font-mono text-xs" />
            )}
            {sttKeyMode === "keep" && <span className="text-xs text-muted-foreground">existing key kept</span>}
            <span className="text-[11px] text-muted-foreground">optional — omit for a keyless / auth-disabled STT</span>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border/60 pt-3">
          <Button type="button" variant="outline" size="xs" onClick={onTestStt}
                  disabled={sttTest.status === "running" || !sttBase.trim() || !sttModel.trim()}>
            {sttTest.status === "running" ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />} Test transcription
          </Button>
          <span className="text-[11px] text-muted-foreground">sends a short tone to <span className="font-mono">/audio/transcriptions</span> (paste/secret key only — kept keys can&apos;t be tested)</span>
          {sttTest.status !== "idle" && sttTest.status !== "running" && (
            <span className={"text-xs " + (sttTest.status === "ok" ? "text-emerald-600 dark:text-emerald-400" : "text-destructive")}>{sttTest.message}</span>
          )}
        </div>
        </>)}
      </section>

      <section data-form-section="Capture drift samples" className="scroll-mt-6 rounded-lg border border-border bg-card p-5">
        <div className="mb-1 flex items-center justify-between">
          <h2 className="text-base font-medium">Capture drift samples</h2>
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-muted-foreground">enabled</span>
            <Switch checked={capEnabled} onCheckedChange={setCapEnabled} />
          </div>
        </div>
        <p className="mb-4 text-xs text-muted-foreground">
          Optional — when a request crosses a quality threshold, save the audio (+ a JSON sidecar with the text/scores)
          to a storage backend for inspection. <span className="font-mono">STT</span>: captures when the transcription&apos;s
          <span className="font-mono"> avg_logprob</span> falls below the floor. <span className="font-mono">TTS</span>: captures when the
          round-trip <span className="font-mono">CER</span> or <span className="font-mono">WER</span> exceeds the ceiling (needs the STT callback above).
          Leave a threshold blank to never trigger on it.
        </p>
        {capEnabled && (<>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-12">
          <div className="md:col-span-7">
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Storage backend</Label>
            <Select value={capStorage} onValueChange={setCapStorage}>
              <SelectTrigger className="h-9 text-xs"><SelectValue placeholder="Select a storage backend…" /></SelectTrigger>
              <SelectContent>
                {storages.map((s) => (
                  <SelectItem key={s.id} value={s.id} className="text-xs">
                    {s.name} <span className="text-muted-foreground">({s.kind}{s.bucket ? `: ${s.bucket}` : ""})</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="md:col-span-5">
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Key prefix</Label>
            <Input value={capPrefix} onChange={(e) => setCapPrefix(e.target.value)} placeholder="drift/" className="font-mono text-xs" />
          </div>
        </div>
        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">STT logprob floor</Label>
            <Input type="number" step="0.05" value={capLogprob} onChange={(e) => setCapLogprob(e.target.value)} placeholder="e.g. -0.5" className="font-mono text-xs" />
            <p className="mt-1 text-[11px] text-muted-foreground">save if avg_logprob &lt; this</p>
          </div>
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">TTS CER ceiling</Label>
            <Input type="number" step="0.05" min={0} value={capCer} onChange={(e) => setCapCer(e.target.value)} placeholder="e.g. 0.2" className="font-mono text-xs" />
            <p className="mt-1 text-[11px] text-muted-foreground">save if CER &gt; this</p>
          </div>
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">TTS WER ceiling</Label>
            <Input type="number" step="0.05" min={0} value={capWer} onChange={(e) => setCapWer(e.target.value)} placeholder="e.g. 0.3" className="font-mono text-xs" />
            <p className="mt-1 text-[11px] text-muted-foreground">save if WER &gt; this</p>
          </div>
        </div>
        </>)}
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
