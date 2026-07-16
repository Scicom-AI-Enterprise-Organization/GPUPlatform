"use client";

// Shared shell for the long "create X" forms (/autotrain/new, /serverless/new,
// /benchmark/new). Three problems it solves, uniformly:
//
//   1. Width — the forms used to sprawl the full content area; the shell caps the
//      form at a readable column and centers it.
//   2. Orientation — these forms are 8–12 cards tall. A right-hand scrollspy rail
//      (xl screens) lists every visible section and jumps to it. Sections are
//      DISCOVERED from the DOM (`data-form-section="Title"`), so per-task/target
//      conditional cards appear and disappear from the rail automatically —
//      no per-form section registry to keep in sync.
//   3. The submit bar — it used to sit at the very bottom of the scroll (with the
//      error message next to it, far from its cause). FormFooter is sticky to the
//      bottom of the scrollport, so the primary action, its disabled state, and
//      any submit error stay visible from anywhere in the form.
//
// Usage:
//   <FormShell>
//     <form …>
//       …sections, each with data-form-section="Title" (+ scroll-mt-6)…
//       <FormFooter error={err}>…cancel/submit buttons…</FormFooter>
//     </form>
//   </FormShell>
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

/** Walk up to the ancestor that actually scrolls (the app shell scrolls an inner
 * div, not the window), so the IntersectionObserver root + scrollspy math track
 * the right viewport. */
function scrollParent(el: HTMLElement | null): HTMLElement | null {
  for (let p = el?.parentElement; p; p = p.parentElement) {
    const oy = getComputedStyle(p).overflowY;
    if (oy === "auto" || oy === "scroll") return p;
  }
  return null;
}

type NavSection = { id: string; title: string; el: HTMLElement };

export function FormShell({ children, className }: { children: React.ReactNode; className?: string }) {
  const bodyRef = useRef<HTMLDivElement>(null);
  const [sections, setSections] = useState<NavSection[]>([]);
  const [active, setActive] = useState<string>("");

  // Discover `[data-form-section]` descendants; re-scan on DOM mutations so
  // conditional sections (per task type / target) keep the rail in sync.
  useEffect(() => {
    const root = bodyRef.current;
    if (!root) return;
    let raf = 0;
    const scan = () => {
      const els = Array.from(root.querySelectorAll<HTMLElement>("[data-form-section]"));
      const next = els.map((el, i) => {
        const title = el.dataset.formSection || `Section ${i + 1}`;
        if (!el.id) el.id = `sec-${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
        return { id: el.id, title, el };
      });
      setSections((prev) =>
        prev.length === next.length && prev.every((s, i) => s.id === next[i].id && s.title === next[i].title)
          ? prev
          : next,
      );
    };
    scan();
    const mo = new MutationObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(scan);
    });
    mo.observe(root, { childList: true, subtree: true });
    return () => {
      mo.disconnect();
      cancelAnimationFrame(raf);
    };
  }, []);

  // Scrollspy: the active section is the last one whose top has passed the upper
  // third of the scrollport. Plain scroll math (not IntersectionObserver) so it
  // stays correct when sections are taller than the viewport.
  useEffect(() => {
    if (sections.length === 0) return;
    const sc = scrollParent(bodyRef.current);
    if (!sc) return;
    const onScroll = () => {
      const cut = sc.getBoundingClientRect().top + sc.clientHeight / 3;
      let cur = sections[0].id;
      for (const s of sections) {
        if (s.el.getBoundingClientRect().top <= cut) cur = s.id;
      }
      setActive(cur);
    };
    onScroll();
    sc.addEventListener("scroll", onScroll, { passive: true });
    return () => sc.removeEventListener("scroll", onScroll);
  }, [sections]);

  return (
    <div className={cn("mx-auto flex w-full items-start gap-4", className)}>
      <div ref={bodyRef} className="min-w-0 flex-1">
        {children}
      </div>
      {sections.length > 1 && (
        <nav className="sticky top-2 hidden w-44 shrink-0 xl:block" aria-label="Form sections">
          <p className="mb-2 px-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground/70">
            On this page
          </p>
          <ul className="space-y-0.5 border-l border-border">
            {sections.map((s) => (
              <li key={s.id}>
                <button
                  type="button"
                  onClick={() => s.el.scrollIntoView({ behavior: "smooth", block: "start" })}
                  className={cn(
                    "-ml-px block w-full truncate border-l-2 px-2 py-1 text-left text-xs transition-colors",
                    active === s.id
                      ? "border-primary font-medium text-foreground"
                      : "border-transparent text-muted-foreground hover:border-border hover:text-foreground",
                  )}
                >
                  {s.title}
                </button>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </div>
  );
}

/** Sticky action bar pinned to the bottom of the scrollport: error + actions stay
 * visible however tall the form is. Place as the LAST child inside the <form>. */
export function FormFooter({
  error,
  hint,
  children,
  className,
}: {
  /** Submit error — always visible here (the old bars buried it at page bottom). */
  error?: string | null;
  /** Muted helper line (e.g. why submit is disabled). */
  hint?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("sticky bottom-0 z-10 -mx-1 mt-6 px-1", className)}>
      <div className="rounded-t-lg border border-b-0 border-border bg-background/95 px-4 py-3 shadow-[0_-4px_16px_-8px_rgba(0,0,0,0.25)] backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <div className="flex flex-wrap items-center justify-end gap-3">
          {(error || hint) && (
            <div className="mr-auto min-w-0 flex-1 basis-64">
              {error ? (
                <p className="truncate text-sm text-destructive" title={error}>{error}</p>
              ) : (
                <div className="text-xs text-muted-foreground">{hint}</div>
              )}
            </div>
          )}
          {children}
        </div>
      </div>
    </div>
  );
}
