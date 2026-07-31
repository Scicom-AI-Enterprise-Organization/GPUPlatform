"use client";

// Section-level tabs for Experiments. The three things here are peer resources,
// not a hierarchy: a dataset and an evaluator are both reusable across runs, so
// neither belongs behind a button on the runs page.

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FlaskConical, Microscope } from "lucide-react";
import { cn } from "@/lib/utils";

// Datasets deliberately absent: experiments read from the platform's own
// /datasets section rather than keeping a parallel corpus of their own.
const TABS = [
  { label: "Runs", href: "/experiments", icon: Microscope },
  { label: "Evaluators", href: "/experiments/evaluators", icon: FlaskConical },
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
