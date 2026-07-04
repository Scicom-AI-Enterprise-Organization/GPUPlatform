"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Check, Loader2, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SearchableSelect, type SearchableOption } from "@/components/ui/searchable-select";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import {
  ComputeTargetPicker,
  computeVisibleDevicesError,
  defaultComputeTarget,
  type ComputeTarget,
} from "./compute-target-picker";
import type {
  CatalogRecord,
  DatasetRecord,
  GlobalEnvRecord,
  TrainingRunRecord,
} from "@/lib/types";


// Export-to-Label as a tab: synthesize N clips from the finished TTS model and seed
// a Label-platform recording + MOS project. "Run on" mirrors serverless/new — a
// fresh RunPod pod (pick GPU type/count/tier) or a registered bare-metal VM.
export function LabelExportTab({
  run,
  onStarted,
}: {
  run: TrainingRunRecord;
  onStarted?: () => void;
}) {
  const lcfg = (run.config_json ?? {}) as Record<string, unknown>;
  const str = (k: string, d = ""): string => (typeof lcfg[k] === "string" ? (lcfg[k] as string) : d);
  const num = (k: string, d: number): number => (typeof lcfg[k] === "number" ? (lcfg[k] as number) : d);
  const arr = (k: string): string => (Array.isArray(lcfg[k]) ? (lcfg[k] as unknown[]).map(String).join(", ") : "");

  // ---- Label-platform creds + project knobs (prefilled from the run's config) ----
  const [urlMode, setUrlMode] = useState<"paste" | "secret">(str("label_base_url_secret") ? "secret" : "paste");
  const [url, setUrl] = useState(str("label_base_url") || "http://localhost:3002");
  const [urlSecret, setUrlSecret] = useState(str("label_base_url_secret"));
  const [tokenMode, setTokenMode] = useState<"paste" | "secret">(str("label_token_secret") ? "secret" : "paste");
  const [token, setToken] = useState("");
  const [tokenSecret, setTokenSecret] = useState(str("label_token_secret"));
  const [secrets, setSecrets] = useState<GlobalEnvRecord[]>([]);
  const [project, setProject] = useState(str("label_project_name"));
  const [samples, setSamples] = useState(num("label_samples", 32));
  const [axes, setAxes] = useState(arr("label_mos_axes") || "Naturalness, Intelligibility, Noise");
  const [speakers, setSpeakers] = useState(arr("label_speakers"));
  const [speakerPrefix, setSpeakerPrefix] = useState(!!lcfg.label_speaker_prefix);
  const [rejectKeywords, setRejectKeywords] = useState(arr("label_reject_keywords"));
  const [perSpeaker, setPerSpeaker] = useState(!!lcfg.label_per_speaker);

  // ---- LLM label export (task_type=llm): generate responses from the trained model
  // and seed a Label-platform human_mos project instead of synthesizing audio clips. ----
  const isLlm = run.task_type === "llm";
  const [llmEvalDatasetId, setLlmEvalDatasetId] = useState(str("llm_label_eval_dataset_id"));
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [catalogDatasets, setCatalogDatasets] = useState<CatalogRecord[]>([]);
  const [llmSamples, setLlmSamples] = useState(num("llm_label_samples", 110));
  const [llmAxes, setLlmAxes] = useState(arr("llm_label_mos_axes") || "Relevance, Accuracy, Helpfulness, Tone");
  const [llmMaxNewTokens, setLlmMaxNewTokens] = useState(num("llm_label_max_new_tokens", 512));
  // vLLM version for the merge→serve venv (the export merges the LoRA + generates with
  // vLLM offline). Default 0.23.0, like serverless/new.
  const [vllmVersion, setVllmVersion] = useState(str("label_vllm_version") || "0.23.0");
  // HF token for the (gated) base-model download during the LoRA merge. Defaults to the
  // run's own token / the platform HF_TOKEN; override with a global secret or a pasted token.
  const [baseHfTokenMode, setBaseHfTokenMode] = useState<"reuse" | "secret" | "paste">(str("base_hf_token_secret") ? "secret" : "reuse");
  const [baseHfToken, setBaseHfToken] = useState("");
  const [baseHfTokenSecret, setBaseHfTokenSecret] = useState(str("base_hf_token_secret"));

  // ---- Run-on (pod card) — shared ComputeTargetPicker ----
  // LLM export runs from the run's LLM venv (/share/autotrain-llm-<arch>, filled
  // from the run config); leave the fallback empty so the gateway picks the arch
  // venv. TTS export uses the NeuCodec venv.
  const [compute, setCompute] = useState<ComputeTarget>(() => {
    const c = defaultComputeTarget(run);
    return { ...c, venvPath: c.venvPath || (run.task_type === "llm" ? "" : "/share/autotrain-tts") };
  });
  // NeuCodec decoder: upstream neuphonic/neucodec (24 kHz) or the Scicom 44k-d20 fork.
  const [codec, setCodec] = useState(str("tts_codec") || "neucodec");

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  const vdError = useMemo(
    () => computeVisibleDevicesError(compute.visibleDevices, 0),
    [compute.visibleDevices],
  );

  useEffect(() => {
    gateway.listGlobalEnv().then(setSecrets).catch(() => {});
    // Eval-dataset picker source (LLM export): registered kind=hf datasets +
    // standalone catalog repos pushed via the hf CLI. Mirrors autotrain/new.
    gateway.listDatasets().then(setDatasets).catch(() => {});
    gateway.listCatalog("mine", "dataset").then(setCatalogDatasets).catch(() => {});
  }, []);

  // Eval-dataset options for the LLM label export (search-select): registered
  // kind=hf datasets, uploaded chat datasets (kind=upload with a messages column),
  // and standalone catalog repos pushed via the hf CLI.
  const evalDatasetOptions = useMemo<SearchableOption[]>(() => {
    const hfDatasets = datasets.filter((d) => d.kind === "hf");
    const uploadChat = datasets.filter((d) => d.kind === "upload" && !!d.messages_field);
    const linkedRepoIds = new Set(datasets.map((d) => d.catalog_repo_id).filter(Boolean));
    const standaloneHosted = catalogDatasets.filter((r) => !linkedRepoIds.has(r.id));
    return [
      ...hfDatasets.map((d) => ({ value: d.id, label: d.name || d.hf_repo || d.id, hint: d.hf_repo || undefined })),
      ...uploadChat.map((d) => ({ value: d.id, label: d.name || d.id, hint: "uploaded chat" })),
      ...standaloneHosted.map((r) => ({ value: r.id, label: r.full_id })),
    ];
  }, [datasets, catalogDatasets]);

  const urlOk = urlMode === "paste" ? !!url.trim() : !!urlSecret;
  const tokenOk = tokenMode === "paste" ? !!token.trim() : !!tokenSecret;
  const running = run.result_json?.label_export?.status === "running";

  async function submit() {
    setErr(null);
    if (compute.runOn === "vm" && !compute.vmProviderId) return setErr("Pick a VM provider, or switch to cloud.");
    if (compute.runOn === "cloud" && !compute.runpodProviderId) return setErr("Select a RunPod provider — add one under GPU Providers.");
    if (vdError) return setErr(vdError);
    setBusy(true);
    try {
      await gateway.retryLabelExport(run.id, {
        base_url: urlMode === "paste" ? (url.trim() || undefined) : undefined,
        base_url_secret: urlMode === "secret" ? (urlSecret || null) : null,
        token: tokenMode === "paste" ? (token.trim() || undefined) : undefined,
        token_secret: tokenMode === "secret" ? (tokenSecret || null) : null,
        project_name: project.trim() || null,
        ...(isLlm
          ? {
              llm_eval_dataset_id: llmEvalDatasetId.trim() || null,
              llm_samples: llmSamples,
              llm_mos_axes: llmAxes.split(",").map((s) => s.trim()).filter(Boolean),
              llm_max_new_tokens: llmMaxNewTokens,
              vllm_version: vllmVersion.trim() || null,
              base_hf_token: baseHfTokenMode === "paste" ? (baseHfToken.trim() || undefined) : undefined,
              base_hf_token_secret: baseHfTokenMode === "secret" ? (baseHfTokenSecret || null) : null,
            }
          : {
              samples,
              mos_axes: axes.split(",").map((s) => s.trim()).filter(Boolean),
              speakers: speakers.split(",").map((s) => s.trim()).filter(Boolean),
              speaker_prefix: speakerPrefix,
              reject_keywords: rejectKeywords.split(/[,\n]/).map((s) => s.trim()).filter(Boolean),
              per_speaker: perSpeaker,
              tts_codec: codec,
            }),
        run_on: compute.runOn,
        provider_id: compute.runOn === "vm" ? compute.vmProviderId : compute.runpodProviderId,
        gpu_type: compute.gpuType,
        gpu_count: compute.gpuCount,
        secure_cloud: compute.secureCloud,
        data_center_id: compute.dataCenterId.trim() || null,
        disk_gb: compute.diskGb,
        volume_gb: compute.volumeGb,
        visible_devices: compute.visibleDevices.trim() || null,
        venv_path: compute.venvPath.trim() || null,
      });
      setDone(true);
      onStarted?.();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function cancel() {
    setErr(null);
    setCancelling(true);
    try {
      await gateway.cancelLabelExport(run.id);
      setDone(false);
      onStarted?.(); // parent refreshes → label_export.status flips off "running"
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setCancelling(false);
    }
  }

  if (done || running) {
    return (
      <div className="space-y-3">
        <p className="flex items-center gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2.5 text-sm text-emerald-700 dark:text-emerald-400">
          <Check className="h-4 w-4 shrink-0" />
          Export {running ? "is running" : "started"} — the run status shows “exporting to Label” and synthesis streams to
          the Logs tab; an “Open in Label” link appears on the Metrics tab when it finishes.
        </p>
        {running && (
          <div className="flex items-center gap-3">
            <Button variant="destructive" onClick={cancel} disabled={cancelling}>
              {cancelling ? <Loader2 className="h-4 w-4 animate-spin" /> : <X className="h-4 w-4" />}
              Cancel export
            </Button>
            <span className="text-xs text-muted-foreground">
              Stops the synthesis, tears down any pod it spawned, and clears the “exporting to Label” status.
            </span>
            {err && <p className="text-sm text-destructive">{err}</p>}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <p className="text-sm text-muted-foreground">
        {isLlm
          ? `Generate ${llmSamples} response${llmSamples === 1 ? "" : "s"} from this run's trained model and create a Label-platform human_mos project seeded with them. Runs in the background; watch the Logs tab.`
          : `Synthesize ${samples} clip${samples === 1 ? "" : "s"} from this run's trained model and create a Label-platform recording project (MOS rating), seeded with them. Runs in the background; watch the Logs tab.`}
      </p>

      {/* Run on + Pod — shared serverless-style compute picker */}
      <ComputeTargetPicker
        run={run}
        value={compute}
        onChange={setCompute}
        venvLabel={isLlm ? "uv venv path (LLM)" : "uv venv path (TTS)"}
        venvPlaceholder={isLlm ? "/share/autotrain-llm-<arch>" : "/share/autotrain-tts"}
        vramHint={isLlm
          ? "Pick a GPU with enough VRAM to load the trained model for text generation."
          : "Pick a GPU with enough VRAM to load the trained model and NeuCodec for synthesis."}
      />

      {/* NeuCodec decoder (TTS-only) — stays out of the shared picker. */}
      {!isLlm && (
        <Section title="Audio decoder" description="Which NeuCodec decodes the model's speech tokens back to audio.">
          <div className="space-y-1.5">
            <Label className="text-xs uppercase tracking-wide text-muted-foreground">NeuCodec (audio decoder)</Label>
            <Select value={codec} onValueChange={setCodec}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="neucodec">neuphonic/neucodec — 24 kHz (upstream)</SelectItem>
                <SelectItem value="neucodec-44k">Scicom neucodec-44k-d20 — 44.1 kHz</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              {codec === "neucodec-44k"
                ? "Scicom 44k-d20 fork — 44.1 kHz output (installs from git; slower first build)."
                : "Upstream neuphonic/neucodec — 24 kHz output. Same speech tokens, so either decodes the model fine."}
            </p>
          </div>
        </Section>
      )}

      {/* Label project — destination + project knobs */}
      <Section
        title="Label project"
        description="Where the synthesized clips land — the Label platform URL, an admin token, and the recording project's settings."
      >
        <div className="space-y-3">
        {/* Label platform URL */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-3">
            <label className="text-xs uppercase tracking-wide text-muted-foreground">Label platform URL</label>
            <div className="inline-flex overflow-hidden rounded-md border border-border text-xs">
              {(["paste", "secret"] as const).map((m) => (
                <button key={m} type="button" onClick={() => setUrlMode(m)}
                  className={cn("px-2.5 py-1 transition-colors",
                    urlMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                  {m === "paste" ? "Paste" : "From secret"}
                </button>
              ))}
            </div>
          </div>
          {urlMode === "paste" ? (
            <Input className="font-mono" value={url} placeholder="http://localhost:3002" onChange={(e) => setUrl(e.target.value)} />
          ) : (
            <Select value={urlSecret} onValueChange={setUrlSecret}>
              <SelectTrigger><SelectValue placeholder={secrets.length ? "Choose a secret" : "No secrets configured"} /></SelectTrigger>
              <SelectContent>
                {secrets.map((s) => (
                  <SelectItem key={s.key} value={s.key}>{s.key}{s.value_preview ? ` — ${s.value_preview}` : ""}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
        {/* API token */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-3">
            <label className="text-xs uppercase tracking-wide text-muted-foreground">API token</label>
            <div className="inline-flex overflow-hidden rounded-md border border-border text-xs">
              {(["paste", "secret"] as const).map((m) => (
                <button key={m} type="button" onClick={() => setTokenMode(m)}
                  className={cn("px-2.5 py-1 transition-colors",
                    tokenMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                  {m === "paste" ? "Paste" : "From secret"}
                </button>
              ))}
            </div>
          </div>
          {tokenMode === "paste" ? (
            <>
              <Input type="password" className="font-mono" value={token} placeholder="lpat_…" onChange={(e) => setToken(e.target.value)} />
              <p className="text-[11px] text-muted-foreground">Admin personal access token. Stored encrypted on the run.</p>
            </>
          ) : (
            <Select value={tokenSecret} onValueChange={setTokenSecret}>
              <SelectTrigger><SelectValue placeholder={secrets.some((s) => s.is_secret) ? "Choose a secret" : "No secrets configured"} /></SelectTrigger>
              <SelectContent>
                {secrets.filter((s) => s.is_secret).map((s) => (
                  <SelectItem key={s.key} value={s.key}>{s.key}{s.value_preview ? ` — ${s.value_preview}` : ""}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
        <div className="space-y-1.5">
          <label className="text-xs uppercase tracking-wide text-muted-foreground">Project name</label>
          <Input value={project} placeholder={`${run.name}-eval`} onChange={(e) => setProject(e.target.value)} />
        </div>
        {isLlm ? (
          <>
            <div className="space-y-1.5">
              <label className="text-xs uppercase tracking-wide text-muted-foreground">Eval dataset</label>
              <SearchableSelect
                value={llmEvalDatasetId}
                onChange={setLlmEvalDatasetId}
                options={evalDatasetOptions}
                placeholder={evalDatasetOptions.length ? "Pick an eval dataset…" : "No datasets yet — push via hf CLI first"}
                searchPlaceholder="Search datasets by name…"
              />
              <p className="text-xs text-muted-foreground">
                The chat dataset whose rows are the prompts — a kind=hf dataset (HF push) or an uploaded
                chat file (kind=upload with a messages column). Register / upload it on the Datasets page.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-xs uppercase tracking-wide text-muted-foreground">Responses</label>
                <Input type="number" min={1} value={llmSamples}
                  onChange={(e) => setLlmSamples(Math.max(1, Number.parseInt(e.target.value, 10) || 1))} />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs uppercase tracking-wide text-muted-foreground">Max new tokens</label>
                <Input type="number" min={64} step={64} value={llmMaxNewTokens}
                  onChange={(e) => setLlmMaxNewTokens(Math.max(64, Number.parseInt(e.target.value, 10) || 512))} />
              </div>
            </div>
            <div className="space-y-1.5">
              <label className="text-xs uppercase tracking-wide text-muted-foreground">MOS axes</label>
              <Input value={llmAxes} placeholder="Relevance, Accuracy, Helpfulness, Tone" onChange={(e) => setLlmAxes(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs uppercase tracking-wide text-muted-foreground">vLLM version</label>
              <Input className="font-mono sm:max-w-xs" value={vllmVersion} placeholder="0.23.0" onChange={(e) => setVllmVersion(e.target.value)} />
              <p className="text-xs text-muted-foreground">
                The LoRA is merged (FP8→fp16 for MiniMax/Mistral) and served with vLLM offline. Version installed in the serve venv.
              </p>
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center gap-3">
                <label className="text-xs uppercase tracking-wide text-muted-foreground">HF token — base model</label>
                <div className="inline-flex overflow-hidden rounded-md border border-border text-xs">
                  {(["reuse", "secret", "paste"] as const).map((m) => (
                    <button key={m} type="button" onClick={() => setBaseHfTokenMode(m)}
                      className={cn("px-2.5 py-1 transition-colors",
                        baseHfTokenMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                      {m === "reuse" ? "Run / platform" : m === "secret" ? "From secret" : "Paste"}
                    </button>
                  ))}
                </div>
              </div>
              {baseHfTokenMode === "secret" ? (
                <Select value={baseHfTokenSecret} onValueChange={setBaseHfTokenSecret}>
                  <SelectTrigger><SelectValue placeholder={secrets.some((s) => s.is_secret) ? "Choose a secret" : "No secrets configured"} /></SelectTrigger>
                  <SelectContent>
                    {secrets.filter((s) => s.is_secret).map((s) => (
                      <SelectItem key={s.key} value={s.key}>{s.key}{s.value_preview ? ` — ${s.value_preview}` : ""}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : baseHfTokenMode === "paste" ? (
                <Input type="password" className="font-mono" value={baseHfToken} placeholder="hf_…" onChange={(e) => setBaseHfToken(e.target.value)} />
              ) : null}
              <p className="text-xs text-muted-foreground">
                Downloads the (usually gated) base model to merge the LoRA. Defaults to the run&apos;s HF token (its secret) or the platform <span className="font-mono">HF_TOKEN</span>; override if a different account owns the base model.
              </p>
            </div>
          </>
        ) : (
          <>
            <div className="space-y-1.5">
              <label className="text-xs uppercase tracking-wide text-muted-foreground">Samples</label>
              <Input type="number" min={1} value={samples}
                onChange={(e) => setSamples(Math.max(1, Number.parseInt(e.target.value, 10) || 1))} />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs uppercase tracking-wide text-muted-foreground">MOS axes</label>
              <Input value={axes} placeholder="Naturalness, Intelligibility, Noise" onChange={(e) => setAxes(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs uppercase tracking-wide text-muted-foreground">Reject keywords (optional)</label>
              <Input value={rejectKeywords} placeholder="EMGS, E M G S, Husein" onChange={(e) => setRejectKeywords(e.target.value)} />
              <p className="text-xs text-muted-foreground">
                Comma- or newline-separated. Text samples containing any phrase are dropped (case-insensitive, spacing-agnostic).
              </p>
            </div>
            <div className="space-y-1.5">
              <label className="text-xs uppercase tracking-wide text-muted-foreground">Speaker names (optional)</label>
              <Input value={speakers} placeholder="speakerA, speakerB" onChange={(e) => setSpeakers(e.target.value)} />
              <p className="text-xs text-muted-foreground">
                {perSpeaker
                  ? `Comma-separated. One project per speaker, each from that speaker's own clips (${samples} per speaker). Names must match the dataset's speaker labels.`
                  : `Comma-separated. Balances the clips evenly across these voices (e.g. 2 speakers + ${samples} samples → ${Math.floor(samples / 2)} each). Blank → the dataset's original voices.`}
              </p>
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-sm">
              <input type="checkbox" checked={perSpeaker} onChange={(e) => setPerSpeaker(e.target.checked)} className="h-4 w-4 accent-primary" />
              <span>Separate project per speaker <span className="text-muted-foreground">(each from that speaker&apos;s own clips)</span></span>
            </label>
            <label className="flex cursor-pointer items-center gap-2 text-sm">
              <input type="checkbox" checked={speakerPrefix} onChange={(e) => setSpeakerPrefix(e.target.checked)} className="h-4 w-4 accent-primary" />
              <span>Prefix transcription with speaker name <span className="text-muted-foreground">(e.g. “TM_Mandarin: …”)</span></span>
            </label>
          </>
        )}
        </div>
      </Section>

      <div className="flex items-center justify-end gap-3">
        {err && <p className="mr-auto text-sm text-destructive">{err}</p>}
        <Button onClick={submit} disabled={busy || !urlOk || !tokenOk}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
          Start export
        </Button>
      </div>
    </div>
  );
}

// Card section matching serverless/new's "Run on" / "Pod" cards.
function Section({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-lg border border-border bg-card p-5">
      <div className="mb-4">
        <h2 className="text-sm font-semibold">{title}</h2>
        {description && <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>}
      </div>
      {children}
    </section>
  );
}

