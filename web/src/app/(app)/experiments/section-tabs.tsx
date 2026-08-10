"use client";

// Section-level tabs for Experiments. The three things here are peer resources,
// not a hierarchy: a dataset and an evaluator are both reusable across runs, so
// neither belongs behind a button on the runs page.

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Boxes, FlaskConical, Microscope, Wand2 } from "lucide-react";
import { cn } from "@/lib/utils";

// Datasets deliberately absent: experiments read from the platform's own
// /datasets section rather than keeping a parallel corpus of their own.
// Optimize is a peer, not a sub-page of Runs: a GEPA search *uses* the same
// datasets and evaluators, and its output is a prompt you then run an
// experiment with — the two are siblings in the same loop.
// Sandboxes is a peer for the same reason: it's a reusable resource (the thing
// that ANSWERS a model's tool calls during a replay), not a per-run setting —
// and like an evaluator, a run snapshots it rather than referencing it live.
const TABS = [
  { label: "Runs", href: "/experiments", icon: Microscope },
  { label: "Optimize", href: "/experiments/optimize", icon: Wand2 },
  { label: "Evaluators", href: "/experiments/evaluators", icon: FlaskConical },
  { label: "Sandboxes", href: "/experiments/sandboxes", icon: Boxes },
];

export function SectionTabs() {
  const pathname = usePathname();
  // "Runs" owns /experiments and /experiments/<id>, but not the sibling tabs or
  // /experiments/new (which is a form reached from Runs).
  const isActive = (href: string) => {
    if (href === "/experiments") {
      return (
        pathname === "/experiments" ||
        (pathname.startsWith("/experiments/") &&
          !TABS.some((t) => t.href !== "/experiments" && pathname.startsWith(t.href)))
      );
    }
    return pathname === href || pathname.startsWith(`${href}/`);
  };

  return (
    <nav className="mb-6 flex items-center gap-1 border-b border-border" aria-label="Experiments">
      {TABS.map((t) => {
        const active = isActive(t.href);
        return (
          <Link
            key={t.href}
            href={t.href}
            aria-current={active ? "page" : undefined}
            className={cn(
              "-mb-px inline-flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm transition-colors",
              active
                ? "border-primary font-medium text-foreground"
                : "border-transparent text-muted-foreground hover:border-border hover:text-foreground",
            )}
          >
            <t.icon className="h-4 w-4" />
            {t.label}
          </Link>
        );
      })}
    </nav>
  );
}
