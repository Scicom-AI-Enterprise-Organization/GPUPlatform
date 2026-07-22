"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight, ChevronDown, Loader2, Package, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ProgressEta } from "@/components/progress-eta";
import { gateway, GatewayError } from "@/lib/gateway";
import type { StorageRecord } from "@/lib/types";

type SplitInfo = { split: string; columns: string[]; num_rows?: number | null };

const DEFAULT_TOKENIZER = "google/gemma-4-31B-it";
const DEFAULT_SEQ_LEN = 32768;

function errText(body: unknown, fallback: string): string {
  if (typeof body === "string") return body || fallback;
  if (body && typeof body === "object") {
    const d = (body as Record<string, unknown>).detail;
    if (typeof d === "string") return d;
    if (d && typeof d === "object" && typeof (d as Record<string, unknown>).error === "string") {
      return (d as Record<string, string>).error;
    }
  }
  return fallback;
}

// Chat → multipack: tokenize a kind=llm dataset's messages column (+ optional
// tools) via the chosen tokenizer's chat template and bin-pack into a ChiniDataset
// (kind=llm_packed) for LLM finetuning. Runs in-process on the gateway (CPU
// tokenization — no GPU box), so it's just "pick a subset + tokenizer + storage".
export function LlmPackCard({
  datasetId,
  messagesField,
  rejectedField,
  s3Storages,
  initialStatus,
  initialLog,
  initialSplit,
}: {
  datasetId: string;
  messagesField: string;
  rejectedField?: string | null;
  s3Storages: StorageRecord[];
  initialStatus: string | null;
  initialLog: string | null;
  initialSplit?: string | null;
}) {
  const router = useRouter();
  const [splits, setSplits] = useState<SplitInfo[] | null>(null);
  const [loadingSplits, setLoadingSplits] = useState(false);
  // Multiselect: pack one or several subsets/splits together (rows concatenated).
  const [subsets, setSubsets] = useState<string[]>([]);
  const [tokenizer, setTokenizer] = useState(DEFAULT_TOKENIZER);
  const [seqLen, setSeqLen] = useState(DEFAULT_SEQ_LEN);
  const [storageId, setStorageId] = useState(s3Storages[0]?.id ?? "");
  const [toolsField, setToolsField] = useState("functions");
  const [allReasoning, setAllReasoning] = useState(true);
  const [fullSeqLabels, setFullSeqLabels] = useState(false);
  // Objective: sft = the messages column → kind=llm_packed; dpo = chosen/rejected
  // preference pairs → kind=llm_dpo_packed (whole pairs per bin, for DPO runs).
  // Default to the dataset's configured mode: a mapped rejected column ⇒ DPO, with
  // chosen = the messages column and rejected = the mapped rejected column.
  const isPref = !!(rejectedField ?? "").trim();
  const [objective, setObjective] = useState<"sft" | "dpo">(isPref ? "dpo" : "sft");
  const [chosenField, setChosenField] = useState(isPref ? messagesField : "chosen");
  const [rejectedFieldName, setRejectedFieldName] = useState((rejectedField ?? "").trim() || "rejected");
  const [promptField, setPromptField] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [status, setStatus] = useState<string | null>(initialStatus);
  const [log, setLog] = useState<string | null>(initialLog);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);
  const logRef = useRef<HTMLPreElement | null>(null);

  const running = status === "running";
  const newDatasetId =
    status === "done" ? log?.match(/created dataset (ds-[0-9a-f]+)/i)?.[1] : undefined;

  // Load the source's subsets/splits (the same labels the row preview shows) so
  // the user can pick which one to pack.
  const loadSplits = useCallback(async () => {
    setLoadingSplits(true);
    try {
      const r = await fetch(`/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}/splits`, {
        cache: "no-store",
      });
      const data = (await r.json()) as { splits?: SplitInfo[] };
      const info = data.splits ?? [];
      setSplits(info);
      // Seed the selection: prefer the split(s) carried in the URL (the row browser
      // uses a comma-joined `split` param), else fall back to the first split. Keep
      // the picks in the dataset's split order for a stable concat order.
      setSubsets((cur) => {
        if (cur.length) return cur;
        const avail = info.map((s) => s.split);
        const want = (initialSplit ?? "")
          .split(",")
          .map((s) => s.trim())
          .filter((s) => s && avail.includes(s));
        const seeded = want.length ? avail.filter((s) => want.includes(s)) : avail.slice(0, 1);
        return seeded;
      });
    } catch {
      setSplits([]);
    } finally {
      setLoadingSplits(false);
    }
  }, [datasetId, initialSplit]);

  useEffect(() => {
    void loadSplits();
  }, [loadSplits]);

  // Poll while a pack is running; refresh the page when it ends.
  useEffect(() => {
    if (!running) {
      if (poll.current) {
        clearInterval(poll.current);
        poll.current = null;
      }
      return;
    }
    const id = setInterval(async () => {
      try {
        const d = await gateway.getDataset(datasetId);
        setStatus(d.transform_status ?? null);
        setLog(d.transform_log ?? null);
        if (d.transform_status !== "running") router.refresh();
      } catch {
        /* transient; keep polling */
      }
    }, 3000);
    poll.current = id;
    return () => clearInterval(id);
  }, [running, datasetId, router]);

  useEffect(() => {
    if (running && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log, running]);

  async function run() {
    setErr(null);
    if (!tokenizer.trim()) {
      setErr("Enter a tokenizer (the chat template comes from it).");
      return;
    }
    if (!storageId) {
      setErr("Pick an S3 storage for the packed shards.");
      return;
    }
    if (!Number.isFinite(seqLen) || seqLen < 1) {
      setErr("Sequence length must be a positive integer.");
      return;
    }
    if (splits && splits.length && !subsets.length) {
      setErr("Pick at least one subset / split to pack.");
      return;
    }
    if (objective === "dpo" && (!chosenField.trim() || !rejectedFieldName.trim())) {
      setErr("DPO packing needs the chosen and rejected column names.");
      return;
    }
    setStarting(true);
    try {
      const d = await gateway.packLlmDataset(datasetId, {
        storage_id: storageId,
        tokenizer: tokenizer.trim(),
        subsets: subsets.length ? subsets : null,
        sequence_length: seqLen,
        tools_field: toolsField.trim() || null,
        all_reasoning: allReasoning,
        full_seq_labels: fullSeqLabels,
        objective,
        ...(objective === "dpo"
          ? {
              chosen_field: chosenField.trim(),
              rejected_field: rejectedFieldName.trim(),
              prompt_field: promptField.trim() || null,
            }
          : {}),
      });
      setStatus(d.transform_status ?? "running");
      setLog(d.transform_log ?? null);
    } catch (e) {
      setErr(
        e instanceof GatewayError
          ? errText(e.parsed, e.message)
          : e instanceof Error ? e.message : String(e),
      );
    } finally {
      setStarting(false);
    }
  }

  async function cancel() {
    setErr(null);
    setCancelling(true);
    try {
      const d = await gateway.cancelDatasetTransform(datasetId);
      setStatus(d.transform_status ?? null);
      setLog(d.transform_log ?? null);
      router.refresh();
    } catch (e) {
      setErr(
        e instanceof GatewayError
          ? errText(e.parsed, e.message)
          : e instanceof Error ? e.message : String(e),
      );
    } finally {
      setCancelling(false);
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-col gap-0.5">
        <CardTitle className="text-base">Pack for LLM — tokenize + multipack</CardTitle>
        <span className="text-xs text-muted-foreground">
          {objective === "dpo" ? (
            <>
              Tokenize the <span className="font-mono">{chosenField || "chosen"}</span> /{" "}
              <span className="font-mono">{rejectedFieldName || "rejected"}</span> preference pairs
              (rendered via the tokenizer&apos;s chat template) and bin-pack them — whole pairs per
              bin — into a DPO ChiniDataset. Pairs longer than the sequence length are dropped
              (never split). Runs on the gateway (CPU tokenization); watch progress below.
            </>
          ) : (
            <>
              Tokenize the <span className="font-mono">{messagesField}</span> column (rendered via the
              tokenizer&apos;s chat template, with any tool/function declarations) and greedily bin-pack
              conversations into a ChiniDataset for LLM finetuning — the chat-text analogue of TTS packing.
              Conversations longer than the sequence length are dropped (never split). Runs on the gateway
              (CPU tokenization); watch progress below.
            </>
          )}
        </span>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="space-y-1">
            <Label className="text-xs">Objective</Label>
            <Select
              value={objective}
              onValueChange={(v) => setObjective(v as "sft" | "dpo")}
              disabled={running}
            >
              <SelectTrigger className="text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="sft">SFT — supervised finetune (messages column)</SelectItem>
                <SelectItem value="dpo">DPO — preference pairs (chosen / rejected columns)</SelectItem>
              </SelectContent>
            </Select>
            {objective === "dpo" && (
              <p className="text-[11px] text-muted-foreground">
                Produces a <span className="font-mono">kind=llm_dpo_packed</span> dataset for Autotrain&apos;s
                DPO training type (Qwen3.5/3.6 and Gemma-4 base models). Chosen/rejected are full message
                lists sharing the prompt turns (ultrafeedback style), or plain response strings + a prompt column.
              </p>
            )}
          </div>

          {objective === "dpo" && (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="space-y-1">
                <Label htmlFor="lp-chosen" className="text-xs">Chosen column</Label>
                <Input
                  id="lp-chosen"
                  value={chosenField}
                  onChange={(e) => setChosenField(e.target.value)}
                  placeholder="chosen"
                  disabled={running}
                  className="font-mono text-xs"
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="lp-rejected" className="text-xs">Rejected column</Label>
                <Input
                  id="lp-rejected"
                  value={rejectedFieldName}
                  onChange={(e) => setRejectedFieldName(e.target.value)}
                  placeholder="rejected"
                  disabled={running}
                  className="font-mono text-xs"
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="lp-prompt" className="text-xs">Prompt column (optional)</Label>
                <Input
                  id="lp-prompt"
                  value={promptField}
                  onChange={(e) => setPromptField(e.target.value)}
                  placeholder="prompt"
                  disabled={running}
                  className="font-mono text-xs"
                />
                <p className="text-[11px] text-muted-foreground">
                  Only when chosen/rejected are plain strings.
                </p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="space-y-1">
              <Label className="text-xs">Subset / split</Label>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    disabled={running || loadingSplits || !(splits && splits.length)}
                    className="h-9 w-full justify-between px-3 text-xs font-normal"
                  >
                    <span className="truncate">
                      {loadingSplits
                        ? "Loading subsets…"
                        : !(splits && splits.length)
                          ? "No subsets found"
                          : subsets.length === 0
                            ? "Pick subset(s)"
                            : subsets.length === 1
                              ? subsets[0]
                              : `${subsets.length} subsets`}
                    </span>
                    <ChevronDown className="ml-1 h-3.5 w-3.5 shrink-0 opacity-60" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="max-h-72 overflow-y-auto">
                  {(splits ?? []).map((s) => (
                    <DropdownMenuCheckboxItem
                      key={s.split}
                      checked={subsets.includes(s.split)}
                      // keep the menu open so several subsets can be ticked in one go
                      onSelect={(e) => e.preventDefault()}
                      onCheckedChange={(c) => {
                        setSubsets((prev) => {
                          const next = c ? [...prev, s.split] : prev.filter((x) => x !== s.split);
                          // keep the dataset's split order (stable concat order)
                          const order = (splits ?? []).map((x) => x.split);
                          return order.filter((x) => next.includes(x));
                        });
                      }}
                      className="text-xs"
                    >
                      {s.split}
                      {typeof s.num_rows === "number" ? ` · ${s.num_rows} rows` : ""}
                    </DropdownMenuCheckboxItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
              {subsets.length > 1 && (
                <p className="text-[11px] text-muted-foreground">
                  {subsets.length} subsets → concatenated into one packed dataset.
                </p>
              )}
            </div>
            <div className="space-y-1">
              <Label htmlFor="lp-seq" className="text-xs">Sequence length (tokens)</Label>
              <Input
                id="lp-seq"
                type="number"
                min={1}
                value={seqLen}
                onChange={(e) => setSeqLen(parseInt(e.target.value, 10))}
                disabled={running}
                className="font-mono text-xs"
              />
            </div>
          </div>

          <div className="space-y-1">
            <Label htmlFor="lp-tok" className="text-xs">Tokenizer (chat template)</Label>
            <Input
              id="lp-tok"
              value={tokenizer}
              onChange={(e) => setTokenizer(e.target.value)}
              placeholder="google/gemma-4-31B-it"
              disabled={running}
              className="font-mono text-xs"
            />
            <p className="text-xs text-muted-foreground">
              The HF tokenizer whose chat template renders each conversation. Tokenization happens once,
              here — the trainer reads the packed ids as-is. Gated repos resolve with the gateway HF token.
            </p>
          </div>

          <div className="space-y-1">
            <Label className="text-xs">S3 storage (packed shards)</Label>
            <Select value={storageId} onValueChange={setStorageId} disabled={running}>
              <SelectTrigger className="text-xs">
                <SelectValue placeholder={s3Storages.length ? "Choose an S3 storage" : "No S3 storage configured"} />
              </SelectTrigger>
              <SelectContent>
                {s3Storages.map((s) => (
                  <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            {showAdvanced ? "▾" : "▸"} Advanced
          </button>
          {showAdvanced && (
            <div className="space-y-3 rounded-md border border-border p-3">
              {objective === "sft" && (
              <div className="space-y-1">
                <Label htmlFor="lp-tools" className="text-xs">Tools / functions column</Label>
                <Input
                  id="lp-tools"
                  value={toolsField}
                  onChange={(e) => setToolsField(e.target.value)}
                  placeholder="functions"
                  disabled={running}
                  className="font-mono text-xs"
                />
                <p className="text-xs text-muted-foreground">
                  Source column of OpenAI-style tool declarations, rendered as <span className="font-mono">tools=</span>{" "}
                  into the chat template. Leave blank to pack without tools.
                </p>
              </div>
              )}
              <label className="flex items-start gap-2 text-xs">
                <Checkbox
                  checked={allReasoning}
                  onCheckedChange={(v) => setAllReasoning(v === true)}
                  disabled={running}
                  className="mt-0.5"
                />
                <span>
                  <span className="font-medium">Render all assistant reasoning</span> — for templates that
                  gate reasoning to tool-call turns after the last user message (e.g. gemma-4, MiniMax-M2):
                  train on every assistant turn&apos;s reasoning. No-op on templates without that guard.
                </span>
              </label>
              {objective === "sft" && (
              <label className="flex items-start gap-2 text-xs">
                <Checkbox
                  checked={fullSeqLabels}
                  onCheckedChange={(v) => setFullSeqLabels(v === true)}
                  disabled={running}
                  className="mt-0.5"
                />
                <span>
                  <span className="font-medium">Full-sequence labels</span> — train on EVERY token
                  (system tool declarations, user turns, tool responses) instead of assistant-only
                  masking. More robust stop-token behaviour at the cost of also learning to imitate
                  environment/user text.
                </span>
              </label>
              )}
            </div>
          )}

          {err && <p className="text-sm text-destructive">{err}</p>}

          <div className="flex items-center gap-3">
            <Button onClick={run} disabled={running || starting}>
              {running || starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Package className="h-4 w-4" />}
              {running ? "Packing…" : objective === "dpo" ? "Pack for DPO" : "Pack for LLM"}
            </Button>
            {running && (
              <Button variant="outline" onClick={cancel} disabled={cancelling} className="text-destructive">
                {cancelling ? <Loader2 className="h-4 w-4 animate-spin" /> : <X className="h-4 w-4" />}
                {cancelling ? "Cancelling…" : "Cancel"}
              </Button>
            )}
            {status && status !== "running" && (
              <span className={status === "done" ? "text-sm text-emerald-600 dark:text-emerald-400" : "text-sm text-destructive"}>
                {status === "done" ? "✓ done" : `✕ ${status}`}
              </span>
            )}
            {newDatasetId && (
              <Link
                href={`/datasets/${newDatasetId}`}
                className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
              >
                Open packed dataset <span className="font-mono text-xs">{newDatasetId}</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            )}
          </div>

          {log && (
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                {running && <Loader2 className="h-3 w-3 animate-spin" />}
                <span>{running ? "Live log" : "Log"}</span>
                <ProgressEta log={log} running={running} />
              </div>
              <pre ref={logRef} className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin">
                {log}
              </pre>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
