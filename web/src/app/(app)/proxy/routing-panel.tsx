"use client";

// Routing canvas — a node graph, per model, of which backend serves a request and
// where it goes when that backend fails. Grouped type → endpoint → model, each group
// collapsible.
//
// The topology is a CHAIN, not a fan-out, because that is what the gateway does.
// `proxy_api._select_candidates` sorts the upstreams serving a model by (known-dead,
// priority) with a STABLE sort; the forwarder walks that list and stops at the first
// backend that answers. There is no round-robin: two upstreams at the same priority
// are not sharing traffic, the first in the list takes all of it and the rest are
// standbys. So exactly ONE edge leaves the request node, and every edge below it is
// failure-only. Fanning parallel edges out of the request would draw load balancing
// this proxy has never done.
//
// Models are listed by their REAL upstream name. The alias is what a client actually
// puts in `model`, so it stays on the request node — leaving it out entirely would
// tell you to send a name the proxy doesn't answer to.
import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProxyUpstreamHealth } from "@/lib/types";
import { KIND_LABEL, KIND_ORDER, KIND_PATH, modelKind, type ModelKind } from "./model-kind";

// Structural subset of the form's UpstreamDraft — also satisfied by a saved endpoint.
export type RoutingUpstream = {
  uid: string;
  name: string;
  base_url: string;
  priority: number;
  enabled: boolean;
  models: { alias: string; real: string }[];
};

type Route = { kind: ModelKind; real: string; alias: string; chain: RoutingUpstream[]; tie: number };

// What actually moves a request to the next backend. Identical in all three forwarders
// (_do_unary, _do_unary_multipart, _stream): status >= 500, or a transport failure —
// ConnectError / ConnectTimeout / ReadTimeout / ReadError / RemoteProtocolError. Anything
// under 500 is handed back to the client untouched and marks the backend alive, so a 429
// does NOT roll to the next backend even though people expect it to.
const HOP = "5xx · timeout · dropped";

// canvas geometry
const PAD = 16;
const NODE_H = 94;
const TERM_H = 34;
const TWO_COL_MIN = 660;

// Theme-aware palette. Set as CSS variables on the canvas so the SVG strokes and the
// card borders read from the same source in both light and dark.
const PALETTE = [
  "[--live:#059669] [--fall:#b45309] [--down:#e11d48] [--wire:#cbd5e1]",
  "[--dot:rgba(15,23,42,0.10)] [--surface:#ffffff] [--canvas:#f8fafc]",
  "dark:[--live:#34d399] dark:[--fall:#f59e0b] dark:[--down:#fb7185] dark:[--wire:#2a303c]",
  "dark:[--dot:rgba(255,255,255,0.07)] dark:[--surface:#161a22] dark:[--canvas:#0d0f14]",
].join(" ");

// Mirrors _select_candidates. The final `a.idx - b.idx` is the load-bearing line: it is
// Python's stable sort, i.e. a same-priority tie falls back to the order the upstreams
// appear in the config — the order of the Upstreams list below.
function selectCandidates(ups: RoutingUpstream[], alias: string, down: Set<string>) {
  return ups
    .map((u, idx) => ({ u, idx }))
    .filter(({ u }) => u.enabled && u.models.some((m) => m.alias.trim() === alias))
    .sort((a, b) => {
      const da = down.has(a.u.uid) ? 1 : 0;
      const db = down.has(b.u.uid) ? 1 : 0;
      if (da !== db) return da - db;
      if (a.u.priority !== b.u.priority) return a.u.priority - b.u.priority;
      return a.idx - b.idx;
    })
    .map(({ u }) => u);
}

const curveLR = (x1: number, y1: number, x2: number, y2: number) => {
  const dx = Math.max(36, (x2 - x1) * 0.55);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
};
const curveDown = (x: number, y1: number, y2: number, bow: number) =>
  `M ${x} ${y1} C ${x - bow} ${y1}, ${x - bow} ${y2}, ${x} ${y2}`;

export function RoutingPanel({ upstreams, maxConcurrency, timeoutS, proxyId, onPromote,
                               defaultOpen = true, healthRows }: {
  upstreams: RoutingUpstream[];
  maxConcurrency: number;
  timeoutS: number;
  proxyId?: string;   // saved endpoint → pull live upstream health
  // Move an upstream to the front of the list. Only meaningful for a tie: that is the
  // one case where list position, not the priority number, picks the winner.
  onPromote?: (uid: string) => void;
  // Open on the overview, where reading the routes is the point; folded in the edit form
  // so it doesn't push the fields you came to change off screen. Only the INITIAL state —
  // <details open> is uncontrolled after mount, so a section you expand stays expanded
  // through the re-renders that every keystroke in the form causes.
  defaultOpen?: boolean;
  // Health already polled by the host page — pass it instead of proxyId so the two
  // don't poll the same endpoint on separate timers.
  healthRows?: ProxyUpstreamHealth[];
}) {
  const [simDown, setSimDown] = useState<Set<string>>(new Set());
  const toggleDown = (uid: string) =>
    setSimDown((prev) => {
      const next = new Set(prev);
      if (next.has(uid)) next.delete(uid); else next.add(uid);
      return next;
    });

  // Live health for a saved endpoint. A probe older than HEALTH_TTL_S (120s) is "stale",
  // and the gateway treats stale as unknown — which sorts as alive, not dead.
  const [fetched, setFetched] = useState<Record<string, ProxyUpstreamHealth>>({});
  const health = useMemo(
    () => (healthRows ? Object.fromEntries(healthRows.map((h) => [h.upstream_id, h])) : fetched),
    [healthRows, fetched],
  );
  useEffect(() => {
    if (!proxyId || healthRows) return;
    let stop = false;
    const load = async () => {
      try {
        const r = await fetch(`/api/proxy/v1/proxy/${encodeURIComponent(proxyId)}/health`, { cache: "no-store" });
        if (!r.ok) return;
        const rows: ProxyUpstreamHealth[] = await r.json();
        if (!stop) setFetched(Object.fromEntries(rows.map((h) => [h.upstream_id, h])));
      } catch { /* degrades to "not probed" */ }
    };
    load();
    const t = setInterval(load, 15000);
    return () => { stop = true; clearInterval(t); };
  }, [proxyId, healthRows]);

  const down = useMemo(() => {
    const s = new Set<string>();
    for (const u of upstreams) {
      const h = health[u.uid];
      if (simDown.has(u.uid) || (h && h.alive === false && !h.stale)) s.add(u.uid);
    }
    return s;
  }, [upstreams, simDown, health]);

  // One route per ALIAS — the alias is the only thing that decides which upstreams match.
  // Keying on the real model name instead would split a single route in two whenever the
  // upstreams spell the model differently (a local vLLM serving "google/gemma-4-31B-it"
  // next to OpenRouter's "google/gemma-4-31b-it" is one route, not two). The route is
  // labelled with the real name of whichever backend is actually serving it.
  const routes = useMemo(() => {
    const aliases = new Set<string>();
    for (const u of upstreams) for (const m of u.models) if (m.alias.trim()) aliases.add(m.alias.trim());
    const out: Route[] = [];
    for (const alias of aliases) {
      const chain = selectCandidates(upstreams, alias, down);
      const realOf = (u?: RoutingUpstream) => u?.models.find((m) => m.alias.trim() === alias)?.real.trim() ?? "";
      const real = realOf(chain[0])
        || realOf(upstreams.find((u) => u.models.some((m) => m.alias.trim() === alias)))
        || alias;
      const top = chain.find((c) => !down.has(c.uid));
      const tie = top ? chain.filter((c) => !down.has(c.uid) && c.priority === top.priority).length : 0;
      out.push({ kind: modelKind(real, alias), real, alias, chain, tie });
    }
    return out.sort((a, b) => a.real.localeCompare(b.real));
  }, [upstreams, down]);

  const byKind = KIND_ORDER
    .map((kind) => ({ kind, items: routes.filter((r) => r.kind === kind) }))
    .filter((g) => g.items.length > 0);

  if (routes.length === 0) {
    return (
      <p className="rounded-md border border-dashed border-border px-3 py-6 text-center text-xs text-muted-foreground">
        Map a model below and its route appears here.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {byKind.map(({ kind, items }) => (
        <details key={kind} open={defaultOpen} className="group/kind rounded-md border border-border">
          <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2">
            <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform group-open/kind:rotate-90" />
            <span className="text-sm font-medium">{KIND_LABEL[kind]}</span>
            <code className="truncate text-[11px] text-muted-foreground">POST {KIND_PATH[kind]}</code>
            <span className="ml-auto shrink-0 text-[11px] text-muted-foreground">
              {items.length} model{items.length === 1 ? "" : "s"}
            </span>
          </summary>
          <div className="space-y-2 border-t border-border p-2">
            {items.map((r) => (
              <ModelFlow key={`${r.real}/${r.alias}`} route={r} down={down} health={health}
                         simDown={simDown} onToggle={toggleDown} onPromote={onPromote}
                         listOrder={upstreams} maxConcurrency={maxConcurrency} timeoutS={timeoutS}
                         defaultOpen={defaultOpen} />
            ))}
          </div>
        </details>
      ))}

      <details className="group/help rounded-md border border-border">
        <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-[11px] text-muted-foreground">
          <ChevronRight className="h-3.5 w-3.5 shrink-0 transition-transform group-open/help:rotate-90" />
          How failover works
        </summary>
        <ul className="space-y-1 border-t border-border px-3 py-2 text-[11px] text-muted-foreground">
          <li><b className="text-foreground">Moves to the next backend</b> on 5xx, a connection refused/timeout, or a dropped socket.</li>
          <li><b className="text-foreground">Does not move</b> on any 4xx — 400, 401, 404, 422 and <b className="text-foreground">429</b> all return to the client as-is, and the backend stays marked alive.</li>
          <li>Streaming can only fail over before the first byte; after that a dropped upstream just ends the stream.</li>
          <li>Lower priority number wins. Same number is not load balancing — the tie breaks on list order, so the winner moves if you reorder or add an upstream. Distinct numbers pin it.</li>
          <li>A dead backend sinks to last, not out. A probe older than 120s counts as alive again.</li>
          <li><code>X-SGPU-Upstream: name</code> pins one backend and skips failover.</li>
          <li>{maxConcurrency > 0 ? `Up to ${maxConcurrency} run at once, the rest queue.` : "No concurrency cap."} Each attempt gets {timeoutS}s.</li>
        </ul>
      </details>
    </div>
  );
}

function ModelFlow({ route, down, health, simDown, onToggle, onPromote, listOrder,
                     maxConcurrency, timeoutS, defaultOpen }: {
  route: Route;
  down: Set<string>;
  health: Record<string, ProxyUpstreamHealth>;
  simDown: Set<string>;
  onToggle: (uid: string) => void;
  onPromote?: (uid: string) => void;
  listOrder: RoutingUpstream[];
  maxConcurrency: number;
  timeoutS: number;
  defaultOpen: boolean;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [avail, setAvail] = useState(0);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setAvail(e.contentRect.width));
    ro.observe(el);
    setAvail(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const { chain, tie } = route;
  const W = avail || 640;
  const twoCol = W >= TWO_COL_MIN;
  const colGap = 84;
  const nodeW = twoCol
    ? Math.min(360, Math.floor((W - PAD * 2 - colGap) / 2))
    : Math.min(520, W - PAD * 2 - 24);
  const chainX = twoCol ? PAD + nodeW + colGap : PAD + 24;
  const rowGap = 40;
  const row = NODE_H + rowGap;
  const rows = chain.length + 1;
  const chainH = rows * row - rowGap - (NODE_H - TERM_H);
  const reqY = twoCol ? PAD + Math.max(0, (chainH - NODE_H) / 2) : PAD;
  const firstRowY = twoCol ? PAD : reqY + row;
  const rowY = (i: number) => firstRowY + i * row;
  const termY = rowY(chain.length);
  const termMid = termY + TERM_H / 2;
  const height = Math.max(reqY + NODE_H, termY + TERM_H) + PAD;
  const bow = twoCol ? 48 : 26;
  const fromPt = twoCol
    ? { x: PAD + nodeW, y: reqY + NODE_H / 2 }
    : { x: chainX, y: reqY + NODE_H + 6 };
  const entry = chain.length === 0
    ? (twoCol ? curveLR(fromPt.x, fromPt.y, chainX, termMid) : curveDown(chainX, fromPt.y, termMid, bow))
    : (twoCol ? curveLR(fromPt.x, fromPt.y, chainX, rowY(0) + NODE_H / 2)
              : curveDown(chainX, fromPt.y, rowY(0) + NODE_H / 2, bow));

  return (
    <details open={defaultOpen} className="group/model rounded-md border border-border">
      <summary className="flex cursor-pointer select-none flex-wrap items-center gap-x-2 gap-y-1 px-3 py-2">
        <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform group-open/model:rotate-90" />
        <code className="text-xs font-medium">{route.real}</code>
        {chain.length === 0 ? (
          <span className="text-[11px] text-rose-600 dark:text-rose-400">no backend — returns 404</span>
        ) : (
          <span className="truncate text-[11px] text-muted-foreground">
            served by <span className="font-medium text-emerald-600 dark:text-emerald-400">{chain[0].name || "unnamed"}</span>
            {chain.length > 1 && <> · {chain.length - 1} fallback{chain.length > 2 ? "s" : ""}</>}
          </span>
        )}
        {tie > 1 && (
          <span className="shrink-0 rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-amber-600 dark:text-amber-400">
            {tie}-way tie
          </span>
        )}
      </summary>

      <div className="border-t border-border p-2">
        <div
          ref={wrapRef}
          className={cn("relative overflow-hidden rounded-md", PALETTE)}
          style={{
            background: "var(--canvas)",
            backgroundImage: "radial-gradient(var(--dot) 1px, transparent 1px)",
            backgroundSize: "16px 16px",
          }}
        >
          <style>{`@media (prefers-reduced-motion: reduce){.pxr-crawl{animation:none!important}}
            @keyframes pxr-dash{to{stroke-dashoffset:-26}}
            .pxr-crawl{animation:pxr-dash 1.2s linear infinite}`}</style>

          <div className="relative" style={{ height }}>
            <svg className="absolute inset-0 h-full w-full" aria-hidden>
              {/* Tied nodes get a bracket: inside it the priority number is NOT what ranks
                  them — their position in the Upstreams list is. */}
              {tie > 1 && (
                <rect x={chainX - 8} y={rowY(0) - 8} rx={12}
                      width={nodeW + 16} height={(tie - 1) * row + NODE_H + 16}
                      fill="none" stroke="var(--fall)" strokeOpacity={0.45} strokeDasharray="4 5" />
              )}
              <path d={entry} fill="none" strokeWidth={1.75} strokeDasharray="7 5" strokeLinecap="round"
                    stroke={chain.length ? "var(--live)" : "var(--down)"}
                    className={chain.length ? "pxr-crawl" : undefined} />
              {chain.map((_, i) => {
                const last = i === chain.length - 1;
                return (
                  <path key={i} d={curveDown(chainX, rowY(i) + NODE_H / 2, last ? termMid : rowY(i + 1) + NODE_H / 2, bow)}
                        fill="none" stroke={last ? "var(--wire)" : "var(--fall)"} strokeWidth={1.5}
                        strokeDasharray="2 5" strokeLinecap="round" />
                );
              })}
              <circle cx={fromPt.x} cy={fromPt.y} r={3.5} fill={chain.length ? "var(--live)" : "var(--down)"} />
              {chain.map((u, i) => (
                <circle key={u.uid} cx={chainX} cy={rowY(i) + NODE_H / 2} r={3.5}
                        fill={i === 0 && !down.has(u.uid) ? "var(--live)" : "var(--canvas)"}
                        stroke={down.has(u.uid) ? "var(--down)" : i === 0 ? "var(--live)" : "var(--fall)"}
                        strokeWidth={1.5} />
              ))}
              <circle cx={chainX} cy={termMid} r={3.5} fill="var(--canvas)"
                      stroke={chain.length ? "var(--wire)" : "var(--down)"} strokeWidth={1.5} />
            </svg>

            {/* request */}
            <Node x={PAD} y={reqY} w={nodeW} tone={chain.length ? "live" : "down"} glow={chain.length > 0}>
              <div className="truncate font-mono text-[11px] text-emerald-600 dark:text-emerald-400">
                POST {KIND_PATH[route.kind]}
              </div>
              <div className="mt-1 truncate font-mono text-[11px] text-muted-foreground">
                &quot;model&quot;: &quot;{route.alias}&quot;
              </div>
              <div className="mt-1 truncate font-mono text-[11px] text-muted-foreground">
                {maxConcurrency > 0 ? `${maxConcurrency} at once` : "no cap"} · {timeoutS}s
              </div>
            </Node>

            {/* failure-hop labels sit in the vertical gaps, never over a card */}
            {chain.map((_, i) => (
              <span key={i}
                    className="absolute z-10 whitespace-nowrap font-mono text-[10px]"
                    style={{
                      left: chainX + 8,
                      top: (i === chain.length - 1 ? termY : rowY(i + 1)) - rowGap / 2,
                      transform: "translateY(-50%)",
                      color: i === chain.length - 1 ? "var(--wire)" : "var(--fall)",
                    }}>
                {i === chain.length - 1 ? "all failed · 502" : HOP}
              </span>
            ))}

            {/* candidates */}
            {chain.map((u, i) => {
              const dead = down.has(u.uid);
              const live = i === 0 && !dead;
              const h = health[u.uid];
              const state = simDown.has(u.uid) ? "down (simulated)"
                : h?.alive === false && !h.stale ? `down — ${h.error ?? "probe failed"}`
                : h?.alive ? `alive · ${h.latency_ms ?? "?"}ms`
                : "not probed";
              // Inside a tie the rank comes from list position, so show that position and
              // offer the one action that changes the winner without touching a number.
              const tied = tie > 1 && i < tie;
              const listPos = listOrder.findIndex((x) => x.uid === u.uid) + 1;
              const ownReal = u.models.find((m) => m.alias.trim() === route.alias)?.real.trim() ?? "";
              return (
                <Node key={u.uid} x={chainX} y={rowY(i)} w={nodeW}
                      tone={dead ? "down" : live ? "live" : "wait"} glow={live}>
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono text-[11px] tabular-nums text-muted-foreground">#{i + 1}</span>
                    <span className="truncate text-xs font-medium">{u.name || "unnamed"}</span>
                    {live && <Tag tone="live">serving</Tag>}
                    {!live && !dead && <Tag tone="wait">standby</Tag>}
                    {dead && <Tag tone="down">tried last</Tag>}
                    <button type="button" onClick={(e) => { e.preventDefault(); onToggle(u.uid); }}
                            className="ml-auto shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] text-muted-foreground hover:text-foreground"
                            title={simDown.has(u.uid) ? "Bring this backend back" : "Pretend this backend is down"}>
                      {simDown.has(u.uid) ? "bring back" : "take down"}
                    </button>
                  </div>
                  <div className="mt-1 truncate font-mono text-[11px] text-muted-foreground" title={u.base_url}>
                    {u.base_url || "no base URL"}
                  </div>
                  {/* Each backend's own name for the model. Usually identical down the
                      chain; when it isn't (a local vLLM vs a hosted provider spelling it
                      differently) that is worth seeing, not hiding behind the alias. */}
                  <div className={cn("truncate font-mono text-[11px]",
                                     ownReal && ownReal !== route.real
                                       ? "text-amber-600 dark:text-amber-400" : "text-muted-foreground")}
                       title={ownReal !== route.real ? `This backend calls it "${ownReal}"` : undefined}>
                    → {ownReal || "—"}
                  </div>
                  <div className="mt-1 flex items-baseline gap-2 font-mono text-[11px] text-muted-foreground">
                    <span>prio {u.priority}</span>
                    {tied && (
                      <span className="shrink-0 text-amber-600 dark:text-amber-400"
                            title="Tied on priority — the winner is whichever of these is listed first in Upstreams">
                        listed #{listPos}{live ? " ← wins" : ""}
                      </span>
                    )}
                    {tied && !live && onPromote && (
                      <button type="button" onClick={(e) => { e.preventDefault(); onPromote(u.uid); }}
                              className="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] hover:text-foreground"
                              title="Move this upstream to the top of the list — with a tie, that alone makes it the primary">
                        make primary
                      </button>
                    )}
                    <span className={cn("ml-auto truncate", dead && "text-rose-600 dark:text-rose-400")}>{state}</span>
                  </div>
                </Node>
              );
            })}

            {/* end-stop: where the chain would run out, not a failure that happened */}
            <Node x={chainX} y={termY} w={nodeW} h={TERM_H} tone={chain.length ? "wire" : "down"}>
              <div className="truncate font-mono text-[11px] text-muted-foreground">
                {chain.length === 0 ? `404 · nothing serves "${route.alias}"` : "if every one fails · 502 to client"}
              </div>
            </Node>
          </div>
        </div>

        {tie > 1 && chain[0] && (
          <p className="mt-2 text-[11px] text-amber-600 dark:text-amber-400">
            {tie} backends sit at priority {chain[0].priority}, so the tie breaks on list order —
            <span className="font-medium"> {chain[0].name || "unnamed"}</span> wins only because it is
            listed first, and takes every request (no round-robin). Reordering or adding an upstream
            silently moves the traffic. Give them distinct priorities to pin it.
          </p>
        )}
      </div>
    </details>
  );
}

function Node({ x, y, w, h = NODE_H, tone, glow, children }: {
  x: number; y: number; w: number; h?: number;
  tone: "live" | "wait" | "down" | "wire"; glow?: boolean; children: React.ReactNode;
}) {
  const color = `var(--${tone === "wait" ? "fall" : tone})`;
  return (
    <div
      className="absolute overflow-hidden rounded-lg px-2.5 py-2"
      style={{
        left: x, top: y, width: w, height: h, background: "var(--surface)",
        boxShadow: glow ? `inset 0 0 0 1px ${color}, 0 0 18px -8px ${color}` : `inset 0 0 0 1px ${color}`,
      }}
    >
      {children}
    </div>
  );
}

function Tag({ tone, children }: { tone: "live" | "wait" | "down"; children: React.ReactNode }) {
  return (
    <span className={cn(
      "shrink-0 rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide",
      tone === "live" ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
        : tone === "wait" ? "bg-amber-500/10 text-amber-600 dark:text-amber-400"
        : "bg-rose-500/10 text-rose-600 dark:text-rose-400",
    )}>{children}</span>
  );
}
