"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Gauge, Pause, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { computeWaveformPeaks, decodeContext } from "@/lib/wav";

const SPEED_PRESETS = [0.5, 0.75, 1, 1.25, 1.5, 2];
const BUCKETS = 800; // server peak resolution (aggregated down to the bar count)
// Match the Label platform's waveform exactly.
const WAVE = "oklch(0.6231 0.1880 259.8145 / 0.25)"; // unplayed
const WAVE_PAST = "oklch(0.6231 0.1880 259.8145)"; // played
const CURSOR = "oklch(0.5461 0.2152 262.8809)"; // playhead
const BAR_STEP = 3; // px per bar slot
const BAR_W = 2; // px bar width (1px gap)
const BAR_RADIUS = 1;
const BAR_MARGIN = 0.9; // leave a little headroom after normalising

function fmt(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

/** Speed presets + freeform input, mirroring the Label player. */
function SpeedControl({ rate, onChange }: { rate: number; onChange: (r: number) => void }) {
  const [text, setText] = useState(String(rate));
  const apply = (n: number) => {
    const safe = Math.max(0.1, Math.min(4, n));
    setText(String(safe));
    onChange(safe);
  };
  return (
    <div className="flex items-center gap-1">
      <Gauge className="h-3 w-3 text-muted-foreground" />
      {SPEED_PRESETS.map((p) => (
        <Button
          key={p}
          variant={Math.abs(rate - p) < 1e-6 ? "secondary" : "ghost"}
          size="xs"
          className="h-6 px-1.5 font-mono text-[10px]"
          onClick={() => apply(p)}
          title={`${p}x`}
        >
          {p}x
        </Button>
      ))}
      <Input
        type="number"
        step="0.05"
        min="0.1"
        max="4"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => {
          const n = parseFloat(text);
          if (Number.isFinite(n)) apply(n);
          else setText(String(rate));
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            const n = parseFloat(text);
            if (Number.isFinite(n)) apply(n);
            else setText(String(rate));
            (e.target as HTMLInputElement).blur();
          }
        }}
        className="h-6 w-14 px-1.5 text-[10px]"
      />
    </div>
  );
}

/**
 * Audio player with a canvas waveform — visually matched to the Label platform
 * (normalised, thin rounded mirrored bars, same colours + controls). Playback
 * runs through a native <audio> element (lenient decoder + Range seeking via the
 * gateway proxy); the waveform comes from server-side peaks (libsndfile), with a
 * client-side decodeAudioData fallback. Parent keys this by `src`.
 */
export function WaveformPlayer({ src }: { src: string }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number>(0);

  const [peaks, setPeaks] = useState<[number, number][]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRate] = useState(1);

  // Parent keys this by `src`, so a new clip remounts with fresh initial state.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      // 1) Server-side peaks (libsndfile) — reliable for codecs the browser
      // rejects; only the small peaks JSON is fetched here.
      const peaksUrl = src.includes("/audio?src=")
        ? `${src.replace("/audio?src=", "/audio-peaks?src=")}&buckets=${BUCKETS}`
        : null;
      if (peaksUrl) {
        try {
          const res = await fetch(peaksUrl);
          if (res.ok) {
            const data = (await res.json()) as { peaks?: [number, number][]; duration?: number };
            if (!cancelled && Array.isArray(data.peaks)) {
              setPeaks(data.peaks);
              if (typeof data.duration === "number" && data.duration > 0) setDuration(data.duration);
              setLoaded(true);
              return;
            }
          }
        } catch {
          /* fall through to client-side decode */
        }
      }
      // 2) Fallback: client-side decodeAudioData (absolute/non-proxied URLs).
      try {
        const res = await fetch(src);
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.error || `Audio failed: ${res.status} ${res.statusText}`);
        }
        const buf = await res.arrayBuffer();
        const ctx = decodeContext();
        if (ctx) {
          const audioBuf = await ctx.decodeAudioData(buf);
          if (!cancelled) setPeaks(computeWaveformPeaks(audioBuf, BUCKETS));
        }
      } catch (err) {
        if (!cancelled && err instanceof Error && err.message.startsWith("Audio failed")) {
          setError(err.message);
        }
      } finally {
        if (!cancelled) setLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [src]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0) return;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    const mid = h / 2;
    ctx.clearRect(0, 0, w, h);
    const progress = duration > 0 ? time / duration : 0;

    if (peaks.length) {
      // Normalise so the loudest bar nearly fills the height (like the Label app).
      let maxAbs = 1e-6;
      for (const [mn, mx] of peaks) {
        const a = Math.max(Math.abs(mn), Math.abs(mx));
        if (a > maxAbs) maxAbs = a;
      }
      const scale = (1 / maxAbs) * BAR_MARGIN;
      const nbars = Math.max(1, Math.floor(w / BAR_STEP));
      for (let b = 0; b < nbars; b++) {
        const s = Math.floor((b / nbars) * peaks.length);
        const e = Math.max(s + 1, Math.floor(((b + 1) / nbars) * peaks.length));
        let mn = 0;
        let mx = 0;
        for (let k = s; k < e && k < peaks.length; k++) {
          if (peaks[k][0] < mn) mn = peaks[k][0];
          if (peaks[k][1] > mx) mx = peaks[k][1];
        }
        const yTop = mid - mx * scale * mid;
        const barH = Math.max(1, (mx - mn) * scale * mid);
        const x = b * BAR_STEP;
        ctx.fillStyle = x / w <= progress ? WAVE_PAST : WAVE;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x, yTop, BAR_W, barH, BAR_RADIUS);
        else ctx.rect(x, yTop, BAR_W, barH);
        ctx.fill();
      }
    } else {
      // Undecodable codec → flat baseline so it's not blank; playback still works.
      ctx.fillStyle = WAVE;
      ctx.fillRect(0, mid - 1, w, 2);
      ctx.fillStyle = WAVE_PAST;
      ctx.fillRect(0, mid - 1, w * progress, 2);
    }

    const px = progress * w;
    ctx.strokeStyle = CURSOR;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
    ctx.stroke();
  }, [peaks, time, duration]);

  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    if (!playing) return;
    const tick = () => {
      const a = audioRef.current;
      if (a) setTime(a.currentTime);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing]);

  const togglePlay = useCallback(() => {
    const a = audioRef.current;
    if (!a) return;
    if (a.paused) a.play().catch(() => setError("Playback failed"));
    else a.pause();
  }, []);

  const applyRate = useCallback((r: number) => {
    setRate(r);
    if (audioRef.current) audioRef.current.playbackRate = r;
  }, []);

  function seek(e: React.MouseEvent<HTMLCanvasElement>) {
    const a = audioRef.current;
    const canvas = canvasRef.current;
    if (!a || !canvas || !duration) return;
    const rect = canvas.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    a.currentTime = Math.max(0, Math.min(duration, pct * duration));
    setTime(a.currentTime);
  }

  return (
    <div className="rounded-lg border bg-card p-3 space-y-2">
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onLoadedMetadata={() => audioRef.current && setDuration(audioRef.current.duration)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onError={() => setError("Failed to load audio")}
      />
      <canvas
        ref={canvasRef}
        onClick={seek}
        className={cn("h-24 w-full cursor-pointer rounded", !loaded && !error && "animate-pulse bg-muted")}
      />
      {error && <p className="text-xs text-destructive">{error}</p>}
      {!error && (
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="outline" size="icon" onClick={togglePlay} title="Play/Pause">
            {playing ? <Pause className="size-4" /> : <Play className="size-4 ml-0.5" />}
          </Button>
          <span className="font-mono text-xs tabular-nums text-muted-foreground">
            {fmt(time)} / {fmt(duration)}
          </span>
          <SpeedControl rate={rate} onChange={applyRate} />
        </div>
      )}
    </div>
  );
}
