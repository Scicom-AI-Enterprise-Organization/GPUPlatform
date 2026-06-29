"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Check, Copy, Database, FileText, Loader2, Lock, Package, RefreshCw, Search, Trash2,
} from "lucide-react";
import { gateway } from "@/lib/gateway";
import type { CatalogDataPreview, CatalogFile, CatalogRecord, CatalogRef } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Pagination } from "@/components/ui/pagination";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { fmtBytes } from "./catalog-list";

function CopyBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="relative">
      <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 pr-10 text-xs leading-relaxed scrollbar-thin">
        <code className="font-mono">{code}</code>
      </pre>
      <button
        onClick={async () => {
          await navigator.clipboard.writeText(code);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        }}
        className="absolute right-2 top-2 rounded-md bg-background/80 p-1.5 text-muted-foreground hover:text-foreground"
        title="Copy"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-0.5 truncate text-lg font-semibold tabular-nums" title={typeof value === "string" ? value : undefined}>
        {value}
      </div>
    </div>
  );
}

export function CatalogDetail({
  repo,
  gatewayUrl,
  backHref,
  initialView,
}: {
  repo: CatalogRecord;
  gatewayUrl: string;
  backHref: string;
  initialView?: string;
}) {
  const router = useRouter();
  const [reindexing, setReindexing] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [wipe, setWipe] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const endpoint = `${gatewayUrl.replace(/\/$/, "")}/hf`;
  const isDataset = repo.repo_type === "dataset";

  // Revision selector (versioned repos only): the page server-fetches the head;
  // switching a branch client-fetches that revision's files. Flat repos have just
  // `main` and no selector.
  const defaultBranch = repo.default_branch || "main";
  const [revision, setRevision] = useState(repo.revision || defaultBranch);
  const [files, setFiles] = useState<CatalogFile[]>(repo.files ?? []);
  const [sha, setSha] = useState<string | null | undefined>(repo.sha);
  // Seed with the default branch so a fresh versioned repo shows "main" in the
  // picker immediately (the full list loads from /refs just after).
  const [branches, setBranches] = useState<CatalogRef[]>(
    repo.versioned ? [{ name: defaultBranch, sha: repo.sha }] : [],
  );
  const [revBusy, setRevBusy] = useState(false);

  useEffect(() => {
    if (!repo.versioned) return;
    gateway.listCatalogRefs(repo.id).then((r) => setBranches(r.branches)).catch(() => {});
  }, [repo.id, repo.versioned]);

  const tabs = [
    { value: "overview", label: "Overview" },
    ...(isDataset ? [{ value: "data", label: "Data" }] : []),
    { value: "files", label: "Files" },
  ];
  const valid = tabs.map((t) => t.value);
  // Top-level tab lives in `?view=`. The initial value comes from the server
  // (page.tsx reads `?view=`) as a prop, NOT useSearchParams() — reading the URL
  // during the first client render shifts the React tree boundary vs the server,
  // drifting Radix's useId seed → a hydration mismatch on the tabs. Updates use
  // history.replaceState so switching tabs doesn't re-run the page's server fetch.
  const [tab, setTabState] = useState(() =>
    initialView && valid.includes(initialView) ? initialView : "overview",
  );
  const setTab = (v: string) => {
    setTabState(v);
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    params.set("view", v);
    window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
  };
  const viewHref = (v: string) => `?view=${v}`;

  // Repos can have thousands of files — search + paginate the list.
  const [fileQ, setFileQ] = useState("");
  const [filePage, setFilePage] = useState(1);
  const [filePageSize, setFilePageSize] = useState(50);

  async function selectRevision(rev: string) {
    setRevision(rev);
    setRevBusy(true);
    try {
      const r = await gateway.getCatalogRepo(repo.id, rev);
      setFiles(r.files ?? []);
      setSha(r.sha);
      setFilePage(1);
    } catch {
      /* keep the current view on error */
    } finally {
      setRevBusy(false);
    }
  }
  const filteredFiles = useMemo(() => {
    const q = fileQ.trim().toLowerCase();
    return q ? files.filter((f) => f.path.toLowerCase().includes(q)) : files;
  }, [files, fileQ]);
  const filePageCount = Math.max(1, Math.ceil(filteredFiles.length / filePageSize));
  const curFilePage = Math.min(filePage, filePageCount);
  const pageFiles = filteredFiles.slice((curFilePage - 1) * filePageSize, curFilePage * filePageSize);

  const envSnippet = `export HF_ENDPOINT=${endpoint}\nexport HF_TOKEN=sgpu_…   # your platform API key\n# or:  hf auth login --token sgpu_…\nexport HF_HUB_DISABLE_XET=1   # this mirror uses LFS, not Xet — required for hf upload (else 404)`;

  const pullSnippet = isDataset
    ? `# Python\nfrom huggingface_hub import snapshot_download\nsnapshot_download("${repo.full_id}", repo_type="dataset")\n\n# CLI\nhf download ${repo.full_id} --repo-type dataset`
    : `# Python\nfrom transformers import AutoModel\nAutoModel.from_pretrained("${repo.full_id}")\n\n# CLI\nhf download ${repo.full_id}`;

  const pushSnippet = isDataset
    ? `# CLI\nhf upload ${repo.full_id} ./my-dataset --repo-type dataset\n\n# Python\nfrom huggingface_hub import HfApi\nHfApi().upload_folder(folder_path="./my-dataset", repo_id="${repo.full_id}", repo_type="dataset")`
    : `# CLI\nhf upload ${repo.full_id} ./my-model\n\n# Python\nmodel.push_to_hub("${repo.full_id}")`;

  async function reindex() {
    setReindexing(true);
    setErr(null);
    try {
      await gateway.reindexCatalogRepo(repo.id);
      router.refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setReindexing(false);
    }
  }

  async function doDelete() {
    setBusy(true);
    setErr(null);
    try {
      await gateway.deleteCatalogRepo(repo.id, wipe);
      router.push(backHref);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="border-b border-border bg-sidebar/40 px-6 pt-4 lg:px-10">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex min-w-0 items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
              {isDataset ? <Database className="h-5 w-5" /> : <Package className="h-5 w-5" />}
            </div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="truncate font-mono text-xl font-semibold tracking-tight">{repo.full_id}</h1>
                <Badge variant="outline">{repo.repo_type}</Badge>
                {repo.private && (
                  <Badge variant="secondary">
                    <Lock className="h-3 w-3" /> private
                  </Badge>
                )}
              </div>
              {repo.description && <p className="mt-1 text-sm text-muted-foreground">{repo.description}</p>}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={reindex} disabled={reindexing}>
              <RefreshCw className={`h-4 w-4 ${reindexing ? "animate-spin" : ""}`} />
              Reindex
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="text-destructive hover:text-destructive"
              onClick={() => {
                setDeleteOpen(true);
                setWipe(false);
                setErr(null);
              }}
            >
              <Trash2 className="h-4 w-4" />
              Delete
            </Button>
          </div>
        </div>

        {err && (
          <div className="mt-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {err}
          </div>
        )}

        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Kpi label="Files" value={String(files.length)} />
          <Kpi label="Size" value={fmtBytes(repo.size_bytes)} />
          <Kpi label="Storage" value={repo.storage_name ?? repo.storage_id ?? "—"} />
          {repo.versioned && branches.length > 0 ? (
            <Kpi label="Revision" value={
              <Select value={revision} onValueChange={selectRevision} disabled={revBusy}>
                <SelectTrigger className="h-7 w-full font-mono text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {branches.map((b) => (
                    <SelectItem key={b.name} value={b.name} className="font-mono text-xs">
                      {b.name}{b.name === defaultBranch ? " (default)" : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            } />
          ) : (
            <Kpi label={repo.versioned ? "Branch · commit" : "Revision"} value={
              <span className="font-mono text-base">
                {(repo.versioned ? `${revision} · ` : "") + ((sha ?? "").slice(0, 12) || "—")}
              </span>
            } />
          )}
        </div>

        <Tabs value={tab} onValueChange={setTab} className="mt-4">
          <TabsList variant="line" className="bg-transparent">
            {tabs.map((t) => (
              <TabsTrigger key={t.value} value={t.value} asChild>
                <a
                  href={viewHref(t.value)}
                  onClick={(e) => {
                    // Modifier/middle clicks → let the browser open a new tab;
                    // plain left-click switches in place (no server re-fetch).
                    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
                    e.preventDefault();
                    setTab(t.value);
                  }}
                >
                  {t.label}
                </a>
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <Tabs value={tab} onValueChange={setTab} className="!block">
          <TabsContent value="overview" className="!flex-none">
            <section className="space-y-3">
              <h2 className="text-base font-medium">Use this repo</h2>
              <div className="rounded-lg border border-border bg-card p-4 space-y-4">
                <div>
                  <p className="mb-1.5 text-xs font-medium text-muted-foreground">1 · Point HF tooling at this gateway</p>
                  <CopyBlock code={envSnippet} />
                </div>
                <div>
                  <p className="mb-1.5 text-xs font-medium text-muted-foreground">Pull (download)</p>
                  <CopyBlock code={pullSnippet} />
                </div>
                <div>
                  <p className="mb-1.5 text-xs font-medium text-muted-foreground">Push (upload)</p>
                  <CopyBlock code={pushSnippet} />
                </div>
              </div>
            </section>
          </TabsContent>

          {isDataset && (
            <TabsContent value="data" className="!flex-none">
              <DataPreview repoId={repo.id} />
            </TabsContent>
          )}

          <TabsContent value="files" className="!flex-none space-y-3">
            <div className="flex items-baseline gap-3 border-b border-border pb-2">
              <h2 className="text-base font-medium">Files</h2>
              <span className="text-xs text-muted-foreground">
                {files.length} {files.length === 1 ? "file" : "files"} on <span className="font-mono">{repo.prefix}</span>
              </span>
            </div>
            {files.length === 0 ? (
              <p className="px-2 py-8 text-center text-sm text-muted-foreground">
                No files yet — <span className="font-mono">hf upload</span> to this repo, then it&apos;ll appear here.
              </p>
            ) : (
              <>
                {files.length > filePageSize && (
                  <div className="relative max-w-sm">
                    <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                    <input
                      value={fileQ}
                      onChange={(e) => {
                        setFileQ(e.target.value);
                        setFilePage(1);
                      }}
                      placeholder="Filter files by path…"
                      className="h-9 w-full rounded-md border border-input bg-transparent pl-8 pr-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
                    />
                  </div>
                )}
                <ul className="divide-y divide-border rounded-lg border border-border">
                  {pageFiles.map((f) => (
                    <li key={f.path} className="flex items-center gap-3 px-4 py-2.5 text-sm">
                      <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                      <span className="min-w-0 flex-1 truncate font-mono text-xs" title={f.path}>{f.path}</span>
                      {f.lfs && <Badge variant="outline" className="shrink-0">LFS</Badge>}
                      <span className="shrink-0 tabular-nums text-xs text-muted-foreground">{fmtBytes(f.size)}</span>
                    </li>
                  ))}
                  {pageFiles.length === 0 && (
                    <li className="px-4 py-8 text-center text-sm text-muted-foreground">No files match.</li>
                  )}
                </ul>
                <Pagination
                  page={curFilePage}
                  pageCount={filePageCount}
                  total={filteredFiles.length}
                  pageSize={filePageSize}
                  onPageChange={setFilePage}
                  onPageSizeChange={(n) => {
                    setFilePageSize(n);
                    setFilePage(1);
                  }}
                  itemLabel="files"
                />
              </>
            )}
          </TabsContent>
        </Tabs>
      </div>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete repo</DialogTitle>
            <DialogDescription>
              Remove <span className="font-mono">{repo.full_id}</span> from the catalog.
            </DialogDescription>
          </DialogHeader>
          <label className="flex items-start gap-2 rounded-md border border-border px-3 py-2 text-sm">
            <Checkbox checked={wipe} onCheckedChange={(v) => setWipe(v === true)} className="mt-0.5" />
            <span>
              Also delete all files from storage (
              <span className="font-mono text-xs">{repo.prefix}</span>). Permanent.
            </span>
          </label>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteOpen(false)} disabled={busy}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={doDelete} disabled={busy}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function cellText(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/** Parquet row preview for a dataset repo, with subset (config) + split pickers. */
function DataPreview({ repoId }: { repoId: string }) {
  const LIMIT = 20;
  // undefined = "let the server pick the default"; a value = user-selected.
  const [config, setConfig] = useState<string | undefined>(undefined);
  const [split, setSplit] = useState<string | undefined>(undefined);
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<CatalogDataPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancel = false;
    setLoading(true);
    setError(null);
    gateway
      .getCatalogData(repoId, { config, split, offset, limit: LIMIT })
      .then((d) => {
        if (cancel) return;
        setData(d);
        if (d.error) setError(d.error);
      })
      .catch((e) => !cancel && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => !cancel && setLoading(false));
    return () => {
      cancel = true;
    };
  }, [repoId, config, split, offset]);

  const curConfig = config ?? data?.config ?? "";
  const curSplit = split ?? data?.split ?? "";
  const columns = data?.columns ?? [];
  const rows = data?.rows ?? [];
  const total = data?.num_rows ?? 0;
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + LIMIT, total);

  return (
    <section className="space-y-3">
      {loading ? (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> loading rows…
        </div>
      ) : error ? (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2">
            {(data?.configs?.length ?? 0) > 1 && (
              <Select
                value={curConfig}
                onValueChange={(v) => {
                  setConfig(v);
                  setSplit(undefined);
                  setOffset(0);
                }}
              >
                <SelectTrigger className="h-9 w-[180px]" title="Subset (config)">
                  <SelectValue placeholder="Subset" />
                </SelectTrigger>
                <SelectContent>
                  {data!.configs.map((c) => (
                    <SelectItem key={c} value={c}>{c}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            {(data?.splits?.length ?? 0) > 1 && (
              <Select
                value={curSplit}
                onValueChange={(v) => {
                  setSplit(v);
                  setOffset(0);
                }}
              >
                <SelectTrigger className="h-9 w-[150px]" title="Split">
                  <SelectValue placeholder="Split" />
                </SelectTrigger>
                <SelectContent>
                  {data!.splits.map((s) => (
                    <SelectItem key={s} value={s}>{s}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            <span className="text-xs text-muted-foreground">
              <span className="font-mono">{curConfig}</span> / <span className="font-mono">{curSplit}</span>
              {(data?.shards ?? 0) > 1 && <> · {data!.shards} shards (previewing the first)</>}
            </span>
          </div>
          {columns.length === 0 ? (
            <p className="py-10 text-center text-sm text-muted-foreground">No rows.</p>
          ) : (
            <>
          <div className="overflow-x-auto rounded-lg border border-border scrollbar-thin">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/40 text-left">
                  <th className="px-3 py-2 text-xs font-medium text-muted-foreground">#</th>
                  {columns.map((c) => (
                    <th key={c} className="px-3 py-2 font-mono text-xs font-medium">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={offset + i} className="border-b border-border/60 last:border-0 align-top">
                    <td className="px-3 py-2 text-xs tabular-nums text-muted-foreground">{offset + i}</td>
                    {columns.map((c) => (
                      <td key={c} className="max-w-[28rem] truncate px-3 py-2 font-mono text-xs" title={cellText(row[c])}>
                        {cellText(row[c])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              Showing <span className="font-medium text-foreground">{start}</span>–
              <span className="font-medium text-foreground">{end}</span> of{" "}
              <span className="font-medium text-foreground">{total.toLocaleString()}</span> rows
            </span>
            <div className="flex gap-1">
              <Button variant="outline" size="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - LIMIT))}>
                Prev
              </Button>
              <Button variant="outline" size="sm" disabled={end >= total} onClick={() => setOffset(offset + LIMIT)}>
                Next
              </Button>
            </div>
          </div>
            </>
          )}
        </>
      )}
    </section>
  );
}
