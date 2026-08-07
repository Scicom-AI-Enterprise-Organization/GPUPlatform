"use client";

import { useCallback, useEffect, useState, type ComponentProps } from "react";
import { useRouter } from "next/navigation";
import {
  Check,
  ChevronRight,
  Copy,
  Download,
  File as FileIcon,
  FileAudio,
  FileImage,
  FileText,
  Folder,
  Inbox,
  Loader2,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";
import type { StorageBrowseResponse, StorageEntry, StorageRecord } from "@/lib/types";

// Mirrors the gateway's caps (storage_api.MAX_INLINE_BYTES / DEFAULT_PREVIEW_BYTES).
// Media is all-or-nothing — half a WAV is not a smaller WAV — so anything over
// the inline cap is download-only; text is head-read and marked truncated.
const MEDIA_MAX_BYTES = 25 * 1024 * 1024;
const TEXT_PREVIEW_BYTES = 512 * 1024;

const IMAGE_EXTS = ["png", "jpg", "jpeg", "gif", "webp", "svg", "bmp", "ico"];
const AUDIO_EXTS = ["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac"];
const VIDEO_EXTS = ["mp4", "webm", "mov"];
const TEXT_EXTS = [
  "txt", "log", "json", "jsonl", "ndjson", "csv", "tsv", "md", "yaml", "yml",
  "py", "sh", "toml", "cfg", "ini", "xml", "html", "js", "ts", "sql", "env",
];

type PreviewKind = "text" | "image" | "audio" | "video" | "binary";

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i < 0 ? "" : name.slice(i + 1).toLowerCase();
}

function previewKind(name: string): PreviewKind {
  const ext = extOf(name);
  if (IMAGE_EXTS.includes(ext)) return "image";
  if (AUDIO_EXTS.includes(ext)) return "audio";
  if (VIDEO_EXTS.includes(ext)) return "video";
  if (TEXT_EXTS.includes(ext)) return "text";
  return "binary";
}

function EntryIcon({ entry }: { entry: StorageEntry }) {
  if (entry.kind === "folder") return <Folder className="h-4 w-4 text-sky-500" />;
  const kind = previewKind(entry.name);
  if (kind === "image") return <FileImage className="h-4 w-4 text-muted-foreground" />;
  if (kind === "audio" || kind === "video") return <FileAudio className="h-4 w-4 text-muted-foreground" />;
  if (kind === "text") return <FileText className="h-4 w-4 text-muted-foreground" />;
  return <FileIcon className="h-4 w-4 text-muted-foreground" />;
}

function formatBytes(b?: number | null): string {
  if (b == null) return "—";
  let n = b;
  for (const u of ["B", "KB", "MB", "GB", "TB", "PB"]) {
    if (n < 1024) return `${n < 10 && u !== "B" ? n.toFixed(1) : Math.round(n)} ${u}`;
    n /= 1024;
  }
  return `${n.toFixed(1)} EB`;
}

/** The absolute location, for pasting into a config: `s3://bucket/root/path`
 * for a bucket, `/abs/root/path` for a local directory. */
function absoluteUri(res: StorageBrowseResponse | null, path: string): string {
  if (!res) return path;
  if (res.kind === "local") {
    return [res.root.replace(/\/+$/, ""), path].filter(Boolean).join("/");
  }
  const parts = [res.root, path].filter(Boolean).join("/");
  return `s3://${res.bucket}${parts ? `/${parts}` : ""}`;
}

/** `?path=…&file=…` — the browsed directory plus (optionally) the previewed file. */
function browseQuery(dir: string, file?: string | null): string {
  const p = new URLSearchParams();
  if (dir) p.set("path", dir);
  if (file) p.set("file", file);
  const qs = p.toString();
  return qs ? `?${qs}` : "?";
}

/** Copy-to-clipboard button that confirms inline — the icon flips to a tick for
 * a moment instead of toasting. Failures still toast (a silent tick would lie). */
function CopyButton({
  value,
  iconClass = "h-4 w-4",
  children,
  ...props
}: { value: string; iconClass?: string } & Omit<ComponentProps<typeof Button>, "onClick" | "value">) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(t);
  }, [copied]);
  return (
    <Button
      {...props}
      onClick={async (ev) => {
        ev.stopPropagation();
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
        } catch {
          toast.error("Couldn't copy to the clipboard");
        }
      }}
    >
      {copied ? (
        <Check className={cn(iconClass, "text-emerald-500 animate-in zoom-in-50 duration-200")} />
      ) : (
        <Copy className={iconClass} />
      )}
      {children}
    </Button>
  );
}

async function openDownload(storage: StorageRecord, path: string) {
  // Local files have nothing to presign — they stream off the gateway's disk.
  if (storage.kind !== "s3") {
    window.open(gateway.storageDownloadUrl(storage.id, path), "_blank", "noopener,noreferrer");
    return;
  }
  try {
    // Presigned + direct from S3 — a multi-GB checkpoint never streams through
    // the gateway.
    const { url } = await gateway.storageObjectUrl(storage.id, path);
    window.open(url, "_blank", "noopener,noreferrer");
  } catch (e) {
    toast.error(e instanceof Error ? e.message : String(e));
  }
}

/**
 * Read-only file viewer over a storage backend (s3 bucket or local directory):
 * one delimited LIST / readdir per directory, click a file to preview it,
 * download presigned (s3) or streamed off disk (local). The browsed path lives
 * in `?path=` and an open preview in `?file=` so both a directory and a file
 * are linkable and survive a refresh.
 *
 * ⚠ Every listing is server-paged and the name filter is a SERVER-side prefix
 * query — a directory here can hold a million objects, so filtering whatever
 * happens to be loaded would quietly answer the wrong question.
 */
export function FileBrowser({
  storage,
  initialPath,
  initialFile,
}: {
  storage: StorageRecord;
  initialPath: string;
  initialFile?: string;
}) {
  const router = useRouter();
  const [path, setPath] = useState(initialPath.replace(/^\/+|\/+$/g, ""));
  const [res, setRes] = useState<StorageBrowseResponse | null>(null);
  const [entries, setEntries] = useState<StorageEntry[]>([]);
  const [nextToken, setNextToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [query, setQuery] = useState(""); // debounced — what the server sees
  const [preview, setPreview] = useState<StorageEntry | null>(null);
  // A `?file=` deep link, consumed once the first listing is in (below).
  const [pendingFile, setPendingFile] = useState<string | null>(
    () => initialFile?.replace(/^\/+|\/+$/g, "") || null,
  );

  const load = useCallback(
    async (dir: string, prefix: string, token?: string | null) => {
      if (token) setLoadingMore(true);
      else setLoading(true);
      setError(null);
      try {
        const r = await gateway.storageBrowse(storage.id, dir, token, 300, prefix);
        setRes(r);
        setEntries((prev) => (token ? [...prev, ...r.entries] : r.entries));
        setNextToken(r.next_token ?? null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        if (!token) setEntries([]);
      } finally {
        setLoading(false);
        setLoadingMore(false);
      }
    },
    [storage.id],
  );

  useEffect(() => {
    const t = setTimeout(() => setQuery(q.trim()), 300);
    return () => clearTimeout(t);
  }, [q]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load(path, query);
  }, [load, path, query]);

  const goto = (dir: string) => {
    setQ("");
    setQuery("");
    setNextToken(null);
    setPath(dir);
    setPendingFile(null);
    // Keep the URL in sync so a directory is shareable (replace, not push, so
    // Back leaves the viewer instead of walking every directory visited).
    router.replace(browseQuery(dir), { scroll: false });
  };

  // Preview open/close rides the URL too (`?file=`), so a single object is
  // linkable — the copied link reopens the dialog on top of its directory.
  const openPreview = (e: StorageEntry) => {
    setPreview(e);
    router.replace(browseQuery(path, e.name), { scroll: false });
  };
  const closePreview = () => {
    setPreview(null);
    router.replace(browseQuery(path), { scroll: false });
  };

  // Consume the deep link: prefer the listed entry (real size/modified); a file
  // the first page doesn't carry still opens via a minimal entry — the preview
  // fetches by path, and the gateway is the one that knows whether it exists.
  useEffect(() => {
    if (!pendingFile || loading) return;
    const found = entries.find((e) => e.kind === "file" && e.name === pendingFile);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPreview(
      found ?? { kind: "file", name: pendingFile, path: [path, pendingFile].filter(Boolean).join("/") },
    );
    setPendingFile(null);
  }, [pendingFile, loading, entries, path]);

  const segments = path ? path.split("/") : [];
  const folderCount = entries.filter((e) => e.kind === "folder").length;
  const fileCount = entries.length - folderCount;
  const rootLabel =
    storage.kind === "local" ? res?.root ?? storage.path ?? "" : `${storage.bucket}${res?.root ? `/${res.root}` : ""}`;
  const uriLabel = storage.kind === "local" ? "Path" : "S3 URI";

  return (
    <div>
      {/* breadcrumbs + toolbar */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1 text-sm">
          <button
            type="button"
            onClick={() => goto("")}
            className={cn(
              "max-w-[22rem] truncate rounded px-1.5 py-0.5 font-mono text-xs hover:bg-muted",
              path ? "text-muted-foreground hover:text-foreground" : "text-foreground",
            )}
            title={absoluteUri(res, "")}
          >
            {rootLabel}
          </button>
          {segments.map((seg, i) => {
            const dir = segments.slice(0, i + 1).join("/");
            const last = i === segments.length - 1;
            return (
              <span key={dir} className="flex items-center gap-1">
                <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                <button
                  type="button"
                  onClick={() => goto(dir)}
                  disabled={last}
                  className={cn(
                    "rounded px-1.5 py-0.5 font-mono text-xs",
                    last ? "text-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                  )}
                >
                  {seg}
                </button>
              </span>
            );
          })}
        </div>
        <div className="relative w-56">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            placeholder="Name starts with…"
            title="Server-side prefix search within this folder — matches names that START with what you type."
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="h-9 w-full rounded-md border border-input bg-background pl-8 pr-8 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
          />
          {q && (
            <button
              type="button"
              onClick={() => setQ("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              title="Clear"
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
        <Button variant="outline" size="sm" onClick={() => load(path, query)} disabled={loading}>
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} /> Refresh
        </Button>
      </div>

      {error && (
        <div className="mb-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="overflow-hidden rounded-xl border border-border">
        <table className="w-full text-sm">
          <thead className="bg-muted/40 text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Name</th>
              <th className="w-28 px-3 py-2 text-right font-medium">Size</th>
              <th className="w-52 px-3 py-2 text-left font-medium">Modified</th>
              <th className="w-24 px-3 py-2"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {path && (
              <tr
                className="cursor-pointer hover:bg-muted/40"
                onClick={() => goto(segments.slice(0, -1).join("/"))}
              >
                <td className="px-3 py-2 font-mono text-xs text-muted-foreground" colSpan={4}>
                  ../
                </td>
              </tr>
            )}
            {loading ? (
              <tr>
                <td colSpan={4} className="px-3 py-12 text-center text-sm text-muted-foreground">
                  <Loader2 className="mx-auto h-5 w-5 animate-spin" />
                </td>
              </tr>
            ) : entries.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-3 py-12 text-center">
                  <Inbox className="mx-auto h-6 w-6 text-muted-foreground/60" />
                  <p className="mt-2 text-sm text-muted-foreground">
                    {query
                      ? `No entries in this folder start with “${query}”.`
                      : "This folder is empty."}
                  </p>
                </td>
              </tr>
            ) : (
              entries.map((e) => (
                <tr
                  key={`${e.kind}:${e.path}`}
                  className="cursor-pointer hover:bg-muted/40"
                  onClick={() => (e.kind === "folder" ? goto(e.path) : openPreview(e))}
                >
                  <td className="px-3 py-2">
                    <span className="flex min-w-0 items-center gap-2">
                      <EntryIcon entry={e} />
                      <span className="truncate font-mono text-xs" title={e.path}>
                        {e.name}
                        {e.kind === "folder" && "/"}
                      </span>
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">
                    {e.kind === "folder" ? "—" : formatBytes(e.size)}
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">
                    {e.modified ? new Date(e.modified).toLocaleString() : "—"}
                  </td>
                  <td className="px-3 py-2">
                    {e.kind === "file" && (
                      <span className="flex items-center justify-end gap-1">
                        <CopyButton
                          variant="ghost"
                          size="icon-sm"
                          aria-label={`Copy ${uriLabel.toLowerCase()}`}
                          title={`Copy ${uriLabel.toLowerCase()}`}
                          value={absoluteUri(res, e.path)}
                          iconClass="h-3.5 w-3.5"
                        />
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          aria-label="Download"
                          title={storage.kind === "s3" ? "Download (presigned)" : "Download"}
                          onClick={(ev) => {
                            ev.stopPropagation();
                            void openDownload(storage, e.path);
                          }}
                        >
                          <Download className="h-3.5 w-3.5" />
                        </Button>
                      </span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
        <span>
          {folderCount} folder{folderCount === 1 ? "" : "s"} · {fileCount} file
          {fileCount === 1 ? "" : "s"} loaded
          {query && ` · prefix “${query}”`}
          {nextToken && " · more available"}
          {res?.note && ` · ${res.note}`}
        </span>
        {nextToken && (
          <Button variant="outline" size="sm" onClick={() => load(path, query, nextToken)} disabled={loadingMore}>
            {loadingMore ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            Load more
          </Button>
        )}
      </div>

      {preview && (
        <PreviewDialog
          storage={storage}
          entry={preview}
          uri={absoluteUri(res, preview.path)}
          uriLabel={uriLabel}
          onClose={closePreview}
        />
      )}
    </div>
  );
}

/** One object's contents. Text is fetched through the gateway (same-origin, so
 * no bucket CORS policy is needed) and may be head-read; media is rendered from
 * the same endpoint; anything else is download-only. */
function PreviewDialog({
  storage,
  entry,
  uri,
  uriLabel,
  onClose,
}: {
  storage: StorageRecord;
  entry: StorageEntry;
  uri: string;
  uriLabel: string;
  onClose: () => void;
}) {
  const storageId = storage.id;
  const kind = previewKind(entry.name);
  const size = entry.size ?? 0;
  const tooBigForMedia = kind !== "text" && kind !== "binary" && size > MEDIA_MAX_BYTES;
  const src = gateway.storageObjectContentUrl(storageId, entry.path, MEDIA_MAX_BYTES);
  // Unknown/extensionless files get a text attempt too — the gateway sniffs the
  // head and serves a textual type when it really is text (a 413 means it isn't
  // text and is over the inline cap).
  const tryText = kind === "text" || kind === "binary";

  const [text, setText] = useState<string | null>(null);
  const [truncated, setTruncated] = useState(false);
  const [loading, setLoading] = useState(tryText);
  const [error, setError] = useState<string | null>(null);
  const [downloadOnly, setDownloadOnly] = useState(false);

  useEffect(() => {
    if (!tryText) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(
          gateway.storageObjectContentUrl(storageId, entry.path, TEXT_PREVIEW_BYTES),
        );
        if (cancelled) return;
        if (r.status === 413) {
          setDownloadOnly(true);
          return;
        }
        const ct = r.headers.get("content-type") ?? "";
        const body = await r.text();
        if (cancelled) return;
        if (!r.ok) {
          setError(body || `gateway error ${r.status}`);
          return;
        }
        // The gateway resolved it as binary after all — don't dump bytes into a <pre>.
        if (!/^(text\/|application\/(json|x-ndjson|xml|javascript|x-yaml))/.test(ct)) {
          setDownloadOnly(true);
          return;
        }
        setTruncated(r.headers.get("x-object-truncated") === "1");
        // Pretty-print JSON when it parses; a .json that doesn't is more useful raw.
        let out = body;
        if (extOf(entry.name) === "json") {
          try {
            out = JSON.stringify(JSON.parse(body), null, 2);
          } catch {
            /* leave it raw */
          }
        }
        setText(out);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [storageId, entry.path, entry.name, tryText]);

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-4xl">
        <DialogHeader className="min-w-0">
          <DialogTitle className="truncate font-mono text-sm" title={entry.path}>
            {entry.name}
          </DialogTitle>
          <p className="text-xs text-muted-foreground">
            {formatBytes(entry.size)}
            {entry.modified ? ` · ${new Date(entry.modified).toLocaleString()}` : ""}
            {truncated && " · preview truncated"}
          </p>
        </DialogHeader>

        <div className="relative">
          {/* Floats over the pane (not inside the scroller, or it would scroll
              away); copies the text as displayed — pretty-printed for JSON. */}
          {text !== null && (
            <CopyButton
              variant="outline"
              size="icon-sm"
              className="absolute right-2 top-2 z-10 bg-background/80 backdrop-blur-sm"
              aria-label="Copy contents"
              title={truncated ? "Copy contents (preview is truncated)" : "Copy contents"}
              value={text}
              iconClass="h-3.5 w-3.5"
            />
          )}
          <div className="max-h-[60vh] overflow-auto rounded-md border border-border bg-muted/20 p-3 scrollbar-thin">
          {error ? (
            <p className="py-8 text-center text-sm text-destructive">{error}</p>
          ) : tooBigForMedia ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              {formatBytes(size)} is over the {formatBytes(MEDIA_MAX_BYTES)} inline
              limit — download it instead.
            </p>
          ) : loading ? (
            <p className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading…
            </p>
          ) : downloadOnly ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No inline preview for this file — download it instead.
            </p>
          ) : text !== null ? (
            <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed">
              {text}
            </pre>
          ) : kind === "image" ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={src} alt={entry.name} className="mx-auto max-h-[55vh] object-contain" />
          ) : kind === "audio" ? (
            <audio src={src} controls className="w-full" />
          ) : kind === "video" ? (
            <video src={src} controls className="mx-auto max-h-[55vh]" />
          ) : (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No inline preview for this file type — download it instead.
            </p>
          )}
          </div>
        </div>

        {/* min-w-0: DialogContent is a grid, and a grid item's min-width:auto
            would size this row to the URI's full nowrap width — past the card. */}
        <DialogFooter className="min-w-0 sm:items-center">
          <span className="mr-auto min-w-0 truncate font-mono text-[11px] text-muted-foreground" title={uri}>
            {uri}
          </span>
          <CopyButton variant="outline" className="shrink-0" value={uri}>
            Copy {uriLabel.toLowerCase()}
          </CopyButton>
          <Button className="shrink-0" onClick={() => openDownload(storage, entry.path)}>
            <Download className="h-4 w-4" /> Download
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
