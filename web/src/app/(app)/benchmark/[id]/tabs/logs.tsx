"use client";

import { useEffect, useRef, useState } from "react";
import { Pause, Play, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { gateway } from "@/lib/gateway";
import type { BenchmarkRecord } from "@/lib/types";

/** Terminal-style live log viewer.
 *
 * Subscribes to /benchmarks/<id>/logs/stream (SSE). Replays everything in the
 * redis list, then live-tails. Auto-scrolls to bottom unless the user has
 * scrolled up (paused state). `event: end` from the server closes the stream.
 */
export function LogsTab({ bench }: { bench: BenchmarkRecord }) {
  const [lines, setLines] = useState<string[]>([]);
  const [streaming, setStreaming] = useState(true);
  const [autoscroll, setAutoscroll] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!streaming) return;
    const es = new EventSource(gateway.benchmarkLogsStreamUrl(bench.id));
    // The server replays the whole Redis log list on every (re)connect. onopen
    // fires on the initial connect AND every transparent reconnect, so clear
    // here — otherwise a network blip doubles every line already shown.
    es.onopen = () => setLines([]);
    es.onmessage = (ev) => {
      setLines((prev) => [...prev, ev.data]);
    };
    es.addEventListener("end", () => {
      es.close();
      setStreaming(false);
    });
    es.onerror = () => {
      // Browser auto-retries unless we close. Once the bench is terminal,
      // an `end` event already closed us — anything else means transient,
      // let it retry.
    };
    return () => es.close();
  }, [bench.id, streaming]);

  // Auto-scroll to bottom on new lines, unless user has paused.
  useEffect(() => {
    if (!autoscroll) return;
    endRef.current?.scrollIntoView({ block: "end" });
  }, [lines, autoscroll]);

  function onScroll() {
    const el = containerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    setAutoscroll(atBottom);
  }

  const terminal =
    bench.status === "done" || bench.status === "failed" || bench.status === "cancelled";

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Logs</h2>
          <p className="text-xs text-muted-foreground">
            {streaming
              ? "Live — tailing benchmaq stdout"
              : terminal
              ? `Stream closed (status: ${bench.status})`
              : "Stream paused"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {streaming ? (
            <Button variant="outline" size="sm" onClick={() => setStreaming(false)}>
              <Pause className="h-4 w-4" /> Pause
            </Button>
          ) : !terminal ? (
            <Button variant="outline" size="sm" onClick={() => setStreaming(true)}>
              <Play className="h-4 w-4" /> Resume
            </Button>
          ) : null}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setLines([])}
            title="Clear local view (server retains)"
          >
            <Trash2 className="h-4 w-4" /> Clear
          </Button>
        </div>
      </div>

      <div
        ref={containerRef}
        onScroll={onScroll}
        className="terminal-block h-[60vh] overflow-y-auto rounded-md border border-border bg-zinc-950 p-3 font-mono text-xs leading-relaxed text-zinc-200"
      >
        {lines.length === 0 ? (
          <div className="text-zinc-500">
            {bench.status === "queued"
              ? "Queued — waiting for runner to start…"
              : "Waiting for output…"}
          </div>
        ) : (
          lines.map((l, i) => (
            <div
              key={i}
              className={
                l.startsWith("[stderr] ")
                  ? "text-rose-300"
                  : l.startsWith("[gateway] ")
                  ? "text-emerald-300"
                  : "text-zinc-200"
              }
            >
              {l}
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>

      {!autoscroll && streaming && (
        <div className="mt-2 text-right">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setAutoscroll(true);
              endRef.current?.scrollIntoView({ block: "end" });
            }}
          >
            Jump to bottom
          </Button>
        </div>
      )}

      {bench.error_text && terminal && bench.status === "failed" && (
        <details className="mt-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          <summary className="cursor-pointer">Error excerpt (last lines of stderr)</summary>
          <pre className="mt-2 whitespace-pre-wrap break-words">{bench.error_text}</pre>
        </details>
      )}
    </div>
  );
}
