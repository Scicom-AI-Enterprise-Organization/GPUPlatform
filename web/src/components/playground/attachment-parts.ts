// Pure half of playground attachments: the OpenAI content parts an attachment list
// becomes. Split from attachments.tsx (which owns the picker UI and the browser-only
// FileReader/canvas/pdf.js work) so the request SHAPE — the thing that has to match what
// vLLM accepts — is unit-testable in a node environment.
//
//   image → {type:"image_url",   image_url:  {url:"data:image/jpeg;base64,…"}}
//   audio → {type:"input_audio", input_audio:{data:"<base64>", format:"wav"}}
//
// Media comes BEFORE the text part, matching vLLM's own multi-image example.

export type AttachmentKind = "image" | "audio";

export type Attachment = {
  id: string;
  kind: AttachmentKind;
  name: string;          // display name (a PDF page carries "report.pdf p2/7")
  mime: string;
  dataUrl: string;       // data:<mime>;base64,<…> — what the request carries
  bytes: number;         // size of the encoded payload, for the budget warning
  fromPdf?: boolean;
};

const AUDIO_FORMATS: Record<string, string> = {
  "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav",
  "audio/mpeg": "mp3", "audio/mp3": "mp3",
  "audio/mp4": "m4a", "audio/x-m4a": "m4a", "audio/aac": "aac",
  "audio/ogg": "ogg", "audio/opus": "opus", "audio/webm": "webm", "audio/flac": "flac",
  "audio/x-flac": "flac",
};

/** OpenAI's `format` for an input_audio part: the mime first, then the file extension.
 *  Browsers leave `type` empty for plenty of audio files, so the extension is not a
 *  cosmetic fallback — it is the only signal left when it happens. */
export function audioFormat(mime: string, name: string): string {
  const byMime = AUDIO_FORMATS[(mime || "").toLowerCase()];
  if (byMime) return byMime;
  const ext = (name.split(".").pop() || "").toLowerCase();
  return ext && ext !== name.toLowerCase() && ext.length <= 5 ? ext : "wav";
}

export const fmtBytes = (n: number) =>
  n >= 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;

/** An input_audio part carries BARE base64 — the `data:…;base64,` prefix an image_url
 *  needs would be part of the payload here, and the server would fail to decode it. */
const b64Of = (dataUrl: string) => dataUrl.slice(dataUrl.indexOf(",") + 1);

/**
 * How an audio clip is spelled. There is no single answer across backends:
 *   input_audio — the OpenAI spelling (bare base64 + a `format`). OpenAI, OpenRouter.
 *   audio_url   — the vLLM/Dynamo extension, shaped like image_url ({url: data-URL}).
 * Measured on this platform's gemma-4-31B (tm-h20-llm-3, 2026-08-13): `input_audio` is
 * accepted with HTTP 200 and then SILENTLY DROPPED — the model answers "I do not hear
 * any audio" — while `audio_url` 500s. So the choice is exposed rather than guessed, and
 * a backend that ignores audio can't be mistaken for a broken upload.
 */
export type AudioPartKind = "input_audio" | "audio_url";

export function attachmentParts(atts: Attachment[], audioAs: AudioPartKind = "input_audio"): unknown[] {
  return atts.map((a) => {
    if (a.kind === "image") return { type: "image_url", image_url: { url: a.dataUrl } };
    return audioAs === "audio_url"
      ? { type: "audio_url", audio_url: { url: a.dataUrl } }
      : { type: "input_audio", input_audio: { data: b64Of(a.dataUrl), format: audioFormat(a.mime, a.name) } };
  });
}

/** Total encoded size — what decides the "this is a big request" warning. */
export const attachmentsBytes = (atts: Attachment[]) => atts.reduce((n, a) => n + a.bytes, 0);

/**
 * The `content` of the user message. No attachments → a plain STRING, byte-identical to
 * what this playground always sent, so no text-only backend sees a new shape. With
 * attachments → media parts then the text part (omitted when the prompt is empty: an
 * empty text part makes some templates emit a stray blank turn).
 */
export function chatContent(
  prompt: string, atts: Attachment[] | undefined, audioAs: AudioPartKind = "input_audio",
): unknown {
  const list = atts ?? [];
  if (!list.length) return prompt;
  return [
    ...attachmentParts(list, audioAs),
    ...(prompt.trim() ? [{ type: "text", text: prompt }] : []),
  ];
}

/**
 * "2x image_url, 1x input_audio" — what the curl comment says the body contains, since
 * the body itself is a downloaded file rather than inline base64. Naming the part TYPES
 * is the part worth reading anyway: it is what a backend accepts or ignores.
 */
export function partSummary(atts: Attachment[], audioAs: AudioPartKind = "input_audio"): string {
  const counts = new Map<string, number>();
  for (const a of atts) {
    const t = a.kind === "image" ? "image_url" : audioAs;
    counts.set(t, (counts.get(t) ?? 0) + 1);
  }
  return [...counts.entries()].map(([t, n]) => `${n}x ${t}`).join(", ");
}
