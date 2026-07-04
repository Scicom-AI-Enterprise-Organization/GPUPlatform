"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { ExternalLink, Loader2, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { gateway } from "@/lib/gateway";
import {
  ComputeTargetPicker,
  computeVisibleDevicesError,
  defaultComputeTarget,
  type ComputeTarget,
} from "./compute-target-picker";
import type { StorageRecord, TrainingRunRecord } from "@/lib/types";

// Export-to-HF as a tab: push the finished run's best/final model to a Hugging Face
// repo. For LLM the artifact is a raw LoRA checkpoint, so merging it into the base
// (GPU work) is required — that adds a "Run on" compute picker. ASR/TTS artifacts are
// already merged models, so they export straight from the gateway/VM (no compute pick).
export function HfExportTab({
  run,
  onStarted,
}: {
  run: TrainingRunRecord;
  onStarted?: () => void;
}) {
  const lcfg = (run.config_json ?? {}) as Record<string, unknown>;
  const isLlm = run.task_type === "llm";

  const [storages, setStorages] = useState<StorageRecord[]>([]);
  const [storageId, setStorageId] = useState("");
  const [repo, setRepo] = useState(typeof lcfg.hf_push_repo === "string" ? (lcfg.hf_push_repo as string) : "");
  const [isPrivate, setIsPrivate] = useState(true); // default private
  // HF token for the GATED BASE MODEL download during merge (serverless/new-style) — a
  // separate account may own the base model vs. the push target. Reuse the push token,
  // or override with a global secret / a pasted token. LLM-merge only.
  const [tokenSource, setTokenSource] = useState<"reuse" | "secret" | "paste">("reuse");
  const [hfToken, setHfToken] = useState("");
  const [hfTokenSecret, setHfTokenSecret] = useState("");
  const [secretKeys, setSecretKeys] = useState<string[]>([]);
  // Merge only applies to LLM (raw LoRA → loadable model); default on + required there.
  const [merge, setMerge] = useState(isLlm);

  // Compute target for the LLM merge job (leave venv blank → gateway picks the arch venv).
  const [compute, setCompute] = useState<ComputeTarget>(() => defaultComputeTarget(run));

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);

  const vdError = useMemo(
    () => computeVisibleDevicesError(compute.visibleDevices, 0),
    [compute.visibleDevices],
  );

  useEffect(() => {
    gateway
      .listStorage()
      .then((rows) => setStorages(rows.filter((s) => s.kind === "huggingface")))
      .catch(() => {});
  }, []);

  // Global-secret keys the HF token can reference (keys only; resolved server-side).
  useEffect(() => {
    fetch("/api/proxy/v1/global-env", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => {
        if (Array.isArray(rows)) setSecretKeys(rows.map((x: { key: string }) => x.key));
      })
      .catch(() => {});
  }, []);

  const hf = run.result_json?.hf_export;
  const running = hf?.status === "running";

  async function submit() {
    setErr(null);
    if (!repo.trim()) return setErr("Enter a repo name (org/model-name).");
    if (isLlm) {
      if (compute.runOn === "vm" && !compute.vmProviderId) return setErr("Pick a VM provider, or switch to cloud.");
      if (compute.runOn === "cloud" && !compute.runpodProviderId) return setErr("Select a RunPod provider — add one under GPU Providers.");
      if (vdError) return setErr(vdError);
    }
    setBusy(true);
    try {
      await gateway.exportToHuggingFace(run.id, {
        repo: repo.trim(),
        storage_id: storageId || null,
        private: isPrivate,
        // Merge + compute target are LLM-only; ASR/TTS export straight from the gateway/VM.
        ...(isLlm
          ? {
              merge,
              // Base-model (gated) download token — separate from the push token above.
              ...(tokenSource === "paste" && hfToken.trim() ? { base_hf_token: hfToken.trim() } : {}),
              ...(tokenSource === "secret" && hfTokenSecret ? { base_hf_token_secret: hfTokenSecret } : {}),
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
            }
          : {}),
      });
      onStarted?.();  // parent refreshes → hf_export.status flips to "running" → Cancel appears
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    setStopping(true);
    try {
      await gateway.cancelHuggingFaceExport(run.id);
      onStarted?.(); // parent refreshes → hf_export.status flips off "running"
    } catch {
      // best-effort; the next poll reflects the real state
    } finally {
      setStopping(false);
    }
  }

  return (
    <div className="space-y-5">
      <p className="text-sm text-muted-foreground">
        Push this run&apos;s <span className="font-medium">best (final) checkpoint</span> to a Hugging Face model repo.
        Runs in the background — watch the Logs tab; an “Open on Hugging Face” link appears below when it finishes.
      </p>

      <Section title="Destination" description="The Hugging Face repo and the token used to push to it.">
        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label className="text-xs uppercase tracking-wide text-muted-foreground">Repo name</Label>
            <Input className="font-mono" value={repo} placeholder="org/model-name"
              onChange={(e) => setRepo(e.target.value)} />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs uppercase tracking-wide text-muted-foreground">Hugging Face storage</Label>
            <Select value={storageId} onValueChange={setStorageId}>
              <SelectTrigger>
                <SelectValue placeholder={storages.length ? "Choose a HuggingFace storage" : "None configured — platform HF_TOKEN used"} />
              </SelectTrigger>
              <SelectContent>
                {storages.map((s) => (
                  <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-[11px] text-muted-foreground">
              Provides the push token + any custom endpoint (self-hosted mirror).{storages.length === 0 ? " None configured — the platform HF_TOKEN secret is used, if set." : ""}
            </p>
          </div>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input type="checkbox" checked={isPrivate} onChange={(e) => setIsPrivate(e.target.checked)}
              className="h-4 w-4 accent-primary" />
            <span>Private repo</span>
          </label>
        </div>
      </Section>

      {/* Merge + compute target — LLM only (raw LoRA needs a GPU merge to become loadable). */}
      {isLlm && (
        <>
          <Section title="Merge" description="The LLM artifact is a raw LoRA checkpoint. Merging it into the base produces a loadable model.">
            <label className="flex cursor-pointer items-center gap-2 text-sm">
              <input type="checkbox" checked={merge} onChange={(e) => setMerge(e.target.checked)}
                className="h-4 w-4 accent-primary" />
              <span>Merge LoRA into base model</span>
            </label>
            <p className="mt-1.5 text-xs text-muted-foreground">
              Required — the LLM artifact is a raw LoRA checkpoint; merging produces a loadable model (runs on GPU).
            </p>

            <div className="mt-4 space-y-1.5 border-t border-border pt-4">
              <Label className="text-xs uppercase tracking-wide text-muted-foreground">HF token — base model</Label>
              <p className="text-[11px] text-muted-foreground">
                Downloads the base model{typeof lcfg.base_model === "string" ? <> (<span className="font-mono">{lcfg.base_model as string}</span>)</> : ""}, which is usually gated — needs an HF account with read access to it. Separate from the push token above.
              </p>
              <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
                {(["reuse", "secret", "paste"] as const).map((src) => (
                  <button
                    key={src}
                    type="button"
                    onClick={() => setTokenSource(src)}
                    className={`rounded px-2.5 py-1 transition-colors ${
                      tokenSource === src ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {src === "reuse" ? "Same as push token" : src === "secret" ? "Global secret" : "Paste a token"}
                  </button>
                ))}
              </div>
              {tokenSource === "secret" ? (
                secretKeys.length > 0 ? (
                  <Select value={hfTokenSecret} onValueChange={setHfTokenSecret}>
                    <SelectTrigger><SelectValue placeholder="Select a secret (e.g. HF_TOKEN)" /></SelectTrigger>
                    <SelectContent>
                      {secretKeys.map((k) => <SelectItem key={k} value={k}>{k}</SelectItem>)}
                    </SelectContent>
                  </Select>
                ) : (
                  <p className="text-[11px] text-muted-foreground">
                    No global secrets yet — add one under{" "}
                    <a href="/admin/secrets" className="underline underline-offset-2 hover:text-foreground">Secrets</a>{" "}
                    (e.g. <span className="font-mono">HF_TOKEN</span>), or switch to Paste.
                  </p>
                )
              ) : tokenSource === "paste" ? (
                <Input type="password" value={hfToken} onChange={(e) => setHfToken(e.target.value)}
                  placeholder="hf_…" autoComplete="off" className="font-mono text-xs" />
              ) : (
                <p className="text-[11px] text-muted-foreground">Reuses the destination push token to download the base model.</p>
              )}
            </div>
          </Section>

          <ComputeTargetPicker
            run={run}
            value={compute}
            onChange={setCompute}
            venvLabel="uv venv path (LLM)"
            venvPlaceholder="/share/autotrain-llm-<arch>"
            vramHint="Pick a GPU with enough VRAM to load the base model for the merge (dequant FP8→fp16 for MiniMax/Mistral needs the full training footprint)."
          />
        </>
      )}

      <div className="flex items-center justify-end gap-3">
        {err && <p className="mr-auto text-sm text-destructive">{err}</p>}
        {running && (
          <Button variant="destructive" onClick={stop} disabled={stopping}>
            {stopping ? <Loader2 className="h-4 w-4 animate-spin" /> : <X className="h-4 w-4" />}
            Cancel export
          </Button>
        )}
        <Button onClick={submit} disabled={busy || running || !repo.trim()}>
          {(busy || running) ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
          {running ? "Exporting…" : (isLlm && merge ? "Merge & push to HF" : "Push to HF")}
        </Button>
      </div>

      {/* Export status — pushing (while running), a link when done, or a failed/cancelled note. */}
      {hf && <HfExportStatus run={run} />}
    </div>
  );
}

// The HF-export status card: pushing (while running), a link when done, or a
// failed/cancelled note. Cancelling is the "Cancel export" button beside Push above.
function HfExportStatus({ run }: { run: TrainingRunRecord }) {
  const hf = run.result_json?.hf_export;
  if (!hf) return null;
  return (
    <section className="rounded-lg border border-border bg-card p-5">
      <h2 className="mb-3 text-sm font-semibold">Hugging Face export</h2>
      <div className="flex flex-wrap items-center gap-x-8 gap-y-2 text-sm">
        {hf.status === "running" && (
          <span className="flex items-center gap-1.5 text-amber-600 dark:text-amber-400">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> pushing {hf.repo} …
          </span>
        )}
        {hf.status === "cancelled" && (
          <span className="flex items-center gap-x-3 text-muted-foreground">
            push stopped{hf.error ? ` — ${hf.error}` : ""}
          </span>
        )}
        {hf.status === "done" && hf.url && (
          <a
            href={hf.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 font-medium text-primary hover:underline"
          >
            Open on Hugging Face — {hf.repo}
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        )}
        {hf.status === "failed" && (
          <span className="text-destructive">push failed: {hf.error}</span>
        )}
      </div>
    </section>
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
