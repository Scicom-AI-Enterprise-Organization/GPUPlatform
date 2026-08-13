"use client";

// Multimodal attachments for the chat playground: images, audio clips and PDFs sent
// alongside the prompt as OpenAI content parts.
//
// Everything happens in the browser — the file never touches a server other than the
// endpoint under test, and the request is the same one a customer's app would send:
//   image → {type:"image_url", image_url:{url:"data:…;base64,…"}}
//   audio → {type:"input_audio", input_audio:{data:"<base64>", format:"wav"|"mp3"|…}}
//
// ⚠ A PDF is NOT a content part any inference server understands. vLLM (which is what
// serves gemma-4 here) accepts images and audio only, so a PDF is RASTERIZED page by
// page into images at pick time — that is also the only representation that works for a
// SCANNED document, where text extraction returns nothing. The user sees the pages it
// became, so a 40-page upload can't silently turn into a 40-image request.

import { useCallback, useRef, useState } from "react";
import { FileAudio, FileText, ImageIcon, Loader2, Paperclip, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  attachmentsBytes, audioFormat, fmtBytes, type Attachment, type AudioPartKind,
} from "@/components/playground/attachment-parts";

// Re-exported so callers keep importing attachments from one place.
export {
  attachmentParts, attachmentsBytes, audioFormat, chatContent, fmtBytes, partSummary,
} from "@/components/playground/attachment-parts";
export type { Attachment, AttachmentKind, AudioPartKind } from "@/components/playground/attachment-parts";

// Caps. These exist because the whole thing is one JSON body: a 12-megapixel phone photo
// is ~8 MB, and base64 adds a third on top. Images are downscaled rather than rejected —
// a 1600px long edge is past the point where a vision encoder gains accuracy, and it
// keeps a multi-page PDF inside a sane request.
const PDF_WORKER_URL = "/pdfjs/pdf.worker.min.mjs";
const MAX_EDGE_PX = 1600;
const JPEG_QUALITY = 0.9;
const MAX_PDF_PAGES = 12;
const MAX_AUDIO_BYTES = 25 * 1024 * 1024;   // matches the usual /v1/audio upload ceiling
const WARN_TOTAL_BYTES = 8 * 1024 * 1024;   // body size at which we warn, not block

const readAsDataUrl = (file: Blob) => new Promise<string>((resolve, reject) => {
  const fr = new FileReader();
  fr.onload = () => resolve(String(fr.result));
  fr.onerror = () => reject(fr.error ?? new Error("could not read the file"));
  fr.readAsDataURL(file);
});

/** Downscale to MAX_EDGE_PX. Below that, the original bytes are kept as-is: re-encoding
 *  a small PNG screenshot as JPEG only adds artefacts to the text in it. */
async function imageAttachment(file: File): Promise<Attachment> {
  const dataUrl = await readAsDataUrl(file);
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error(`${file.name} is not a readable image`));
    img.src = dataUrl;
  });
  const edge = Math.max(img.naturalWidth, img.naturalHeight);
  if (edge <= MAX_EDGE_PX) {
    return { id: crypto.randomUUID(), kind: "image", name: file.name, mime: file.type || "image/png", dataUrl, bytes: dataUrl.length };
  }
  const scale = MAX_EDGE_PX / edge;
  const cv = document.createElement("canvas");
  cv.width = Math.round(img.naturalWidth * scale);
  cv.height = Math.round(img.naturalHeight * scale);
  cv.getContext("2d")!.drawImage(img, 0, 0, cv.width, cv.height);
  const out = cv.toDataURL("image/jpeg", JPEG_QUALITY);
  return { id: crypto.randomUUID(), kind: "image", name: file.name, mime: "image/jpeg", dataUrl: out, bytes: out.length };
}

async function audioAttachment(file: File): Promise<Attachment> {
  if (file.size > MAX_AUDIO_BYTES) throw new Error(`${file.name} is ${fmtBytes(file.size)} — over the ${fmtBytes(MAX_AUDIO_BYTES)} limit`);
  const dataUrl = await readAsDataUrl(file);
  return {
    id: crypto.randomUUID(), kind: "audio", name: file.name,
    mime: file.type || `audio/${audioFormat("", file.name)}`, dataUrl, bytes: dataUrl.length,
  };
}

/** Rasterize a PDF into one image attachment per page (capped). Loaded lazily so the
 *  ~1 MB pdf.js bundle is only fetched by someone who actually picks a PDF. */
async function pdfAttachments(file: File): Promise<Attachment[]> {
  // ⚠ The LEGACY build, deliberately — and the worker below must match it.
  // pdfjs-dist 6's modern build calls `Map.prototype.getOrInsertComputed`, a TC39
  // proposal current browsers don't ship, and carries no polyfill: dropping a PDF fails
  // with "this[#methodPromises].getOrInsertComputed is not a function" from deep inside
  // its message handler, which reads like a bundler problem and isn't one. The legacy
  // build ships the polyfill. Same reason the copy script takes legacy/build/.
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
  // Served out of public/ by scripts/copy-pdf-worker.mjs (predev/prebuild) rather than
  // resolved through the bundler — see that script for why. pdf.js loads an .mjs worker
  // as a module worker automatically.
  pdfjs.GlobalWorkerOptions.workerSrc = PDF_WORKER_URL;
  const probe = await fetch(PDF_WORKER_URL, { method: "HEAD" }).catch(() => null);
  if (!probe?.ok) {
    throw new Error(`${file.name}: the PDF worker is missing at ${PDF_WORKER_URL} — `
      + "restart the web app (npm run dev) to copy it, then try again");
  }
  const doc = await pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;
  const pages = Math.min(doc.numPages, MAX_PDF_PAGES);
  const out: Attachment[] = [];
  for (let i = 1; i <= pages; i++) {
    const page = await doc.getPage(i);
    const base = page.getViewport({ scale: 1 });
    // Render at whatever scale lands the long edge on MAX_EDGE_PX — a 72-dpi viewport
    // is unreadable to a vision model, and 300 dpi is megabytes per page for no gain.
    const viewport = page.getViewport({ scale: MAX_EDGE_PX / Math.max(base.width, base.height) });
    const cv = document.createElement("canvas");
    cv.width = Math.ceil(viewport.width);
    cv.height = Math.ceil(viewport.height);
    const ctx = cv.getContext("2d")!;
    ctx.fillStyle = "#fff";                       // PDFs are transparent; JPEG needs a ground
    ctx.fillRect(0, 0, cv.width, cv.height);
    await page.render({ canvas: cv, canvasContext: ctx, viewport }).promise;
    const dataUrl = cv.toDataURL("image/jpeg", JPEG_QUALITY);
    out.push({
      id: crypto.randomUUID(), kind: "image", mime: "image/jpeg", dataUrl, bytes: dataUrl.length,
      name: `${file.name} p${i}/${doc.numPages}`, fromPdf: true,
    });
  }
  return out;
}

async function toAttachments(file: File): Promise<Attachment[]> {
  const mime = (file.type || "").toLowerCase();
  const name = file.name.toLowerCase();
  if (mime === "application/pdf" || name.endsWith(".pdf")) return pdfAttachments(file);
  if (mime.startsWith("image/")) return [await imageAttachment(file)];
  if (mime.startsWith("audio/") || /\.(wav|mp3|m4a|aac|ogg|opus|flac|webm)$/.test(name)) return [await audioAttachment(file)];
  throw new Error(`${file.name}: only images, audio and PDF are supported`);
}

/**
 * Convert a picked/dropped/pasted set of files. One unreadable file never discards the
 * ones that did convert — a 40-page PDF plus a corrupt WAV should still give you the
 * pages, with the WAV's reason reported.
 */
export async function readFiles(files: FileList | File[] | null): Promise<{ added: Attachment[]; errors: string[] }> {
  const added: Attachment[] = [];
  const errors: string[] = [];
  for (const f of Array.from(files ?? [])) {
    try {
      added.push(...(await toAttachments(f)));
    } catch (e) {
      errors.push(e instanceof Error ? e.message : String(e));
    }
  }
  return { added, errors };
}

const ACCEPT = "image/*,audio/*,application/pdf,.pdf,.wav,.mp3,.m4a,.flac,.ogg,.opus";

export function AttachmentBar({
  attachments, onChange, disabled, audioAs = "input_audio", onAudioAsChange,
}: {
  attachments: Attachment[];
  onChange: (next: Attachment[]) => void;
  disabled?: boolean;
  audioAs?: AudioPartKind;
  onAudioAsChange?: (k: AudioPartKind) => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const add = useCallback(async (files: FileList | File[] | null) => {
    const list = Array.from(files ?? []);
    if (!list.length) return;
    setBusy(true);
    setErr(null);
    const { added, errors } = await readFiles(list);
    if (added.length) onChange([...attachments, ...added]);
    if (errors.length) setErr(errors.join(" · "));
    setBusy(false);
  }, [attachments, onChange]);

  const total = attachmentsBytes(attachments);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); void add(e.dataTransfer?.files ?? null); }}
      className={"rounded-md border border-dashed p-2 transition-colors " + (dragOver ? "border-primary bg-primary/5" : "border-border")}
    >
      <div className="flex flex-wrap items-center gap-2">
        <input ref={inputRef} type="file" multiple accept={ACCEPT} className="hidden"
               onChange={(e) => { void add(e.target.files); e.target.value = ""; }} />
        <Button type="button" variant="outline" size="xs" disabled={disabled || busy}
                onClick={() => inputRef.current?.click()}>
          {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Paperclip className="h-3 w-3" />} Attach
        </Button>
        <span className="text-[11px] text-muted-foreground">
          image · audio · PDF — drop files here or paste an image into the prompt.
          {" "}PDFs are converted to page images (first {MAX_PDF_PAGES}).
        </span>
        {attachments.length > 0 && (
          <>
            <span className="text-[11px] text-muted-foreground">
              {attachments.length} part{attachments.length === 1 ? "" : "s"} · {fmtBytes(total)}
            </span>
            <Button type="button" variant="ghost" size="xs" onClick={() => onChange([])} disabled={disabled}>Clear</Button>
          </>
        )}
      </div>

      {attachments.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2">
          {attachments.map((a) => (
            <div key={a.id} className="flex max-w-[260px] items-center gap-2 rounded border border-border bg-muted/30 p-1 pr-1.5">
              {a.kind === "image" ? (
                // eslint-disable-next-line @next/next/no-img-element -- a data: URL, nothing for next/image to optimize
                <img src={a.dataUrl} alt={a.name} className="h-9 w-9 rounded object-cover" />
              ) : (
                <span className="flex h-9 w-9 items-center justify-center rounded bg-muted">
                  <FileAudio className="h-4 w-4 text-muted-foreground" />
                </span>
              )}
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-1 truncate font-mono text-[10px]" title={a.name}>
                  {a.fromPdf ? <FileText className="h-2.5 w-2.5 shrink-0 text-muted-foreground" />
                    : a.kind === "image" ? <ImageIcon className="h-2.5 w-2.5 shrink-0 text-muted-foreground" /> : null}
                  {a.name}
                </span>
                <span className="block text-[10px] text-muted-foreground">
                  {a.kind === "image" ? "image_url" : `input_audio · ${audioFormat(a.mime, a.name)}`} · {fmtBytes(a.bytes)}
                </span>
              </span>
              <button type="button" title="Remove" disabled={disabled}
                      onClick={() => onChange(attachments.filter((x) => x.id !== a.id))}
                      className="rounded p-0.5 text-muted-foreground hover:text-destructive">
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Audio is the one kind whose wire format is not settled across backends, and the
          failure is silent: a 200 whose answer says "I do not hear any audio". Surface the
          choice and the failure mode instead of leaving someone to debug their microphone. */}
      {attachments.some((a) => a.kind === "audio") && (
        <div className="mt-2 space-y-1 border-t border-border/60 pt-2 text-[11px] text-muted-foreground">
          <div className="flex flex-wrap items-center gap-2">
            <span>audio part</span>
            <div className="inline-flex rounded-md border border-border p-0.5">
              {(["input_audio", "audio_url"] as const).map((k) => (
                <button key={k} type="button" disabled={disabled || !onAudioAsChange}
                        onClick={() => onAudioAsChange?.(k)}
                        className={"rounded px-1.5 py-0.5 font-mono " + (audioAs === k
                          ? "bg-primary text-primary-foreground"
                          : "text-muted-foreground hover:text-foreground")}>
                  {k}
                </button>
              ))}
            </div>
            <span>{audioAs === "input_audio" ? "OpenAI / OpenRouter spelling" : "vLLM / Dynamo extension"}</span>
          </div>
          <div>
            &#9888; Not every chat backend consumes audio. Some accept the part and drop it — you get a
            200 whose answer is &ldquo;I do not hear any audio&rdquo; — and some reject it outright. If that
            happens the upload is fine and the backend is the limit; use the{" "}
            <span className="font-medium">Transcribe</span> mode for the whisper path instead.
          </div>
        </div>
      )}

      {err && <p className="mt-1 text-[11px] text-destructive">{err}</p>}
      {total > WARN_TOTAL_BYTES && (
        <p className="mt-1 text-[11px] text-amber-600 dark:text-amber-400">
          {fmtBytes(total)} of attachments — one large JSON body. Expect a slow upload, and check the
          endpoint&apos;s own body-size limit if it 413s.
        </p>
      )}
    </div>
  );
}

/** Paste handler for the prompt box: image on the clipboard → an attachment. */
export function useImagePaste(add: (files: File[]) => void) {
  return useCallback((e: React.ClipboardEvent) => {
    const files = Array.from(e.clipboardData?.items ?? [])
      .filter((it) => it.kind === "file" && it.type.startsWith("image/"))
      .map((it) => it.getAsFile())
      .filter((f): f is File => f !== null);
    if (files.length) {
      e.preventDefault();   // don't also paste the filename as text
      add(files);
    }
  }, [add]);
}
