"use client";

// Real-time factor readout for the audio playgrounds (transcribe + speech).
//
// The proxy's audio data plane reports how long the upstream took per second of audio
// on the response itself: `X-Audio-Seconds` (the measured duration), `X-RTF` (processing
// seconds per audio second — lower is faster, < 1 = faster than real time) and `X-RTFX`
// (its reciprocal, the "× real time" figure people quote). Absent when the duration
// couldn't be measured without a decoder (mp3/opus/flac) or on a streamed request (the
// headers flush before the audio does) — so every consumer treats it as optional.
export type Rtf = { audioSeconds: number; rtf: number; rtfx: number };

export function readRtf(h: Headers): Rtf | null {
  const audioSeconds = Number(h.get("x-audio-seconds"));
  const rtf = Number(h.get("x-rtf"));
  if (!Number.isFinite(audioSeconds) || audioSeconds <= 0 || !Number.isFinite(rtf) || rtf <= 0) return null;
  const rtfx = Number(h.get("x-rtfx"));
  return { audioSeconds, rtf, rtfx: Number.isFinite(rtfx) && rtfx > 0 ? rtfx : 1 / rtf };
}

/** `12.5s audio · RTF 0.031 · 32× real time` — the compact one-line form. */
export function rtfLabel(r: Rtf): string {
  return `${fmtSeconds(r.audioSeconds)} audio · RTF ${r.rtf.toFixed(3)} · ${fmtX(r.rtfx)}× real time`;
}

function fmtSeconds(s: number): string {
  return s >= 10 ? `${s.toFixed(1)}s` : `${s.toFixed(2)}s`;
}

function fmtX(x: number): string {
  return x >= 10 ? x.toFixed(0) : x.toFixed(1);
}

/** `compact` drops the duration (for a history row, where space is tight); the full
 *  reading stays available as the tooltip. */
export function RtfNote({ rtf, className = "", compact = false }:
                        { rtf: Rtf; className?: string; compact?: boolean }) {
  return (
    <span className={`shrink-0 font-mono text-[11px] text-muted-foreground ${className}`} title={rtfLabel(rtf)}>
      {compact ? `RTF ${rtf.rtf.toFixed(3)} · ${fmtX(rtf.rtfx)}×` : rtfLabel(rtf)}
    </span>
  );
}
