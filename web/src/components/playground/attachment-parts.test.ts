import { describe, expect, it } from "vitest";
import {
  attachmentParts, attachmentsBytes, audioFormat, chatContent, fmtBytes, partSummary,
  type Attachment,
} from "./attachment-parts";

// The playground's attachment picker is browser-only (FileReader, canvas, pdf.js), but the
// REQUEST SHAPE is what has to match what vLLM accepts — so that half lives in a pure
// module and is pinned here. A wrong part name or a base64 payload that still carries its
// `data:` prefix produces an HTTP 400 from the model server, or worse, an image the model
// silently ignores.

const img = (over: Partial<Attachment> = {}): Attachment => ({
  id: "a1", kind: "image", name: "shot.png", mime: "image/png",
  dataUrl: "data:image/png;base64,AAAA", bytes: 4096, ...over,
});
const aud = (over: Partial<Attachment> = {}): Attachment => ({
  id: "a2", kind: "audio", name: "clip.wav", mime: "audio/wav",
  dataUrl: "data:audio/wav;base64,BBBB", bytes: 8192, ...over,
});

describe("chatContent", () => {
  it("stays a plain STRING with no attachments", () => {
    // Every text-only backend has always received a string here; attachments must not
    // change the shape of a request that has none.
    expect(chatContent("hello", undefined)).toBe("hello");
    expect(chatContent("hello", [])).toBe("hello");
  });

  it("puts media BEFORE the text part, in pick order", () => {
    // vLLM's own multi-image example orders it this way, and the gemma template reads
    // the question as being about the images above it.
    const parts = chatContent("compare these", [img(), aud()]) as Array<{ type: string }>;
    expect(parts.map((p) => p.type)).toEqual(["image_url", "input_audio", "text"]);
  });

  it("omits the text part when the prompt is empty", () => {
    // "what is in this image?" with no words is a real request; an empty text part makes
    // some chat templates emit a stray blank turn.
    const parts = chatContent("   ", [img()]) as Array<{ type: string }>;
    expect(parts.map((p) => p.type)).toEqual(["image_url"]);
  });
});

describe("attachmentParts", () => {
  it("sends an image as image_url carrying the whole data URL", () => {
    expect(attachmentParts([img()])).toEqual([
      { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
    ]);
  });

  it("sends audio as input_audio with BARE base64 — no data: prefix", () => {
    // The prefix is part of the payload for image_url and poison for input_audio: the
    // server base64-decodes the string as-is.
    expect(attachmentParts([aud()])).toEqual([
      { type: "input_audio", input_audio: { data: "BBBB", format: "wav" } },
    ]);
  });
});

describe("audioFormat", () => {
  it("maps the common mimes", () => {
    expect(audioFormat("audio/mpeg", "x.mp3")).toBe("mp3");
    expect(audioFormat("audio/x-wav", "x.wav")).toBe("wav");
    expect(audioFormat("audio/x-m4a", "x.m4a")).toBe("m4a");
    expect(audioFormat("audio/flac", "x.flac")).toBe("flac");
  });

  it("falls back to the extension — browsers leave `type` empty often enough", () => {
    expect(audioFormat("", "recording.ogg")).toBe("ogg");
    expect(audioFormat("application/octet-stream", "voice.opus")).toBe("opus");
  });

  it("defaults to wav rather than guessing from a name with no extension", () => {
    expect(audioFormat("", "recording")).toBe("wav");
  });
});

describe("partSummary", () => {
  // The curl preview names the part TYPES instead of inlining base64: an earlier version
  // printed a body with the payload replaced by "<272 KB … elided>", which copies cleanly
  // and then fails at the model with a garbage image. A placeholder that looks like a
  // runnable command is worse than no command.
  it("counts parts by type", () => {
    expect(partSummary([img(), img(), aud()])).toBe("2x image_url, 1x input_audio");
  });

  it("follows the audio spelling", () => {
    expect(partSummary([aud()], "audio_url")).toBe("1x audio_url");
  });

  it("is empty for no attachments", () => {
    expect(partSummary([])).toBe("");
  });
});

describe("size helpers", () => {
  it("totals the encoded payloads", () => {
    expect(attachmentsBytes([img(), aud()])).toBe(4096 + 8192);
    expect(attachmentsBytes([])).toBe(0);
  });

  it("formats KB below a megabyte and MB above it", () => {
    expect(fmtBytes(4096)).toBe("4 KB");
    expect(fmtBytes(2 * 1024 * 1024)).toBe("2.0 MB");
    expect(fmtBytes(10)).toBe("1 KB");   // never "0 KB" for a non-empty file
  });
});

describe("audio spelling", () => {
  // Measured on this platform's gemma-4-31B: input_audio → 200 with the audio silently
  // dropped, audio_url → 500. Neither is a client bug, so the spelling is a choice and
  // both have to be built correctly.
  it("audio_url is shaped like image_url — a data URL under `url`", () => {
    expect(attachmentParts([aud()], "audio_url")).toEqual([
      { type: "audio_url", audio_url: { url: "data:audio/wav;base64,BBBB" } },
    ]);
  });

  it("defaults to the OpenAI spelling", () => {
    expect(attachmentParts([aud()])).toEqual(attachmentParts([aud()], "input_audio"));
  });

  it("images are unaffected by the audio spelling", () => {
    expect(attachmentParts([img()], "audio_url")).toEqual(attachmentParts([img()], "input_audio"));
  });

  it("chatContent threads the choice through", () => {
    const parts = chatContent("what is this?", [aud()], "audio_url") as Array<{ type: string }>;
    expect(parts.map((p) => p.type)).toEqual(["audio_url", "text"]);
  });

});
