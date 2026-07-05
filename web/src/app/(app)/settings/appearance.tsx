"use client";

import { useEffect, useState } from "react";
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  TERMINAL_THEMES,
  type TerminalTheme,
  readTerminalTheme,
  writeTerminalTheme,
} from "@/lib/terminal-theme";

const META: Record<TerminalTheme, { label: string; description: string }> = {
  default: {
    label: "Default",
    description: "Neutral grey block, matches the rest of the UI.",
  },
  classic: {
    label: "Classic",
    description: "Black background, green phosphor text. Hacker vibes.",
  },
  rainbow: {
    label: "Rainbow",
    description: "Animated gradient sweep. Loud on purpose.",
  },
  sparkle: {
    label: "Sparkle",
    description: "Purple haze with twinkling stars — hover to summon more.",
  },
  christmas: {
    label: "Christmas",
    description: "Snowy night with falling flakes and a little snowman. Ho ho ho.",
  },
};

export function AppearanceSettings() {
  // Read from localStorage on mount (after hydration) so we don't render a
  // mismatched value during SSR.
  const [theme, setTheme] = useState<TerminalTheme>("rainbow");
  useEffect(() => {
    setTheme(readTerminalTheme());
  }, []);

  function pick(next: TerminalTheme) {
    setTheme(next);
    writeTerminalTheme(next);
  }

  return (
    <section data-form-section="Terminal appearance" className="scroll-mt-6 rounded-lg border border-border bg-card p-5">
      <div className="mb-4">
        <h2 className="text-sm font-semibold">Terminal appearance</h2>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Affects how SSH commands and other terminal-like blocks look on the
          Compute pages.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {TERMINAL_THEMES.map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => pick(t)}
            className={cn(
              "relative overflow-hidden rounded-lg border p-3 text-left transition-colors hover:border-foreground/40",
              theme === t
                ? "border-foreground/60 ring-1 ring-foreground/20"
                : "border-border",
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium">{META[t].label}</span>
              {theme === t && <Check className="h-4 w-4 text-foreground" />}
            </div>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {META[t].description}
            </p>
            <Preview theme={t} />
          </button>
        ))}
      </div>
    </section>
  );
}

// Preview swatch — uses dedicated `.terminal-preview-*` classes (defined in
// globals.css) so each option always renders its own theme regardless of
// what's currently active globally on <html>.
function Preview({ theme }: { theme: TerminalTheme }) {
  return (
    <div className="mt-3">
      <code className={cn("terminal-preview", `terminal-preview-${theme}`)}>
        ssh -i ~/.ssh/sgpu-runpod -p 39342 root@1.2.3.4
      </code>
    </div>
  );
}
