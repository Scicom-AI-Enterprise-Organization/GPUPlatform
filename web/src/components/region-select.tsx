"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronDown } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { RegionOption } from "@/lib/types";

// Static fallback mirroring the gateway's RUNPOD_REGIONS — used until the live
// list (GET /compute/runpod/regions) loads, or if the gateway is older.
const FALLBACK: RegionOption[] = [
  { id: "US-KS-2", label: "US · Kansas", country: "US" },
  { id: "US-CA-2", label: "US · California", country: "US" },
  { id: "US-GA-1", label: "US · Georgia", country: "US" },
  { id: "US-TX-3", label: "US · Texas", country: "US" },
  { id: "US-NC-1", label: "US · North Carolina", country: "US" },
  { id: "US-WA-1", label: "US · Washington", country: "US" },
  { id: "CA-MTL-1", label: "Canada · Montreal", country: "CA" },
  { id: "EU-RO-1", label: "EU · Romania", country: "EU" },
  { id: "EU-CZ-1", label: "EU · Czechia", country: "EU" },
  { id: "EU-NL-1", label: "EU · Netherlands", country: "EU" },
  { id: "EU-SE-1", label: "EU · Sweden", country: "EU" },
  { id: "EU-FR-1", label: "EU · France", country: "EU" },
  { id: "AP-JP-1", label: "Asia · Japan", country: "AP" },
  { id: "OC-AU-1", label: "Oceania · Australia", country: "AP" },
];

const parse = (v: string) => new Set((v || "").split(",").map((s) => s.trim()).filter(Boolean));

/**
 * RunPod data-center picker (multi-select). `value` is a comma-separated allowlist
 * of RunPod `dataCenterIds`; empty = Auto (RunPod picks any region with capacity —
 * the default everywhere). RunPod deploys the pod into whichever selected DC has
 * capacity, so picking several broadens capacity while keeping a geo constraint.
 * The live list comes from the gateway; a curated fallback fills in until it loads.
 * Shared by every RunPod-provisioning form (serverless, compute, autotrain, …).
 */
export function RegionSelect({
  value,
  onChange,
  disabled,
  className = "h-8 text-xs",
}: {
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
  className?: string;
}) {
  const [regions, setRegions] = useState<RegionOption[]>(FALLBACK);
  useEffect(() => {
    gateway.listRunpodRegions().then((r) => { if (r?.length) setRegions(r); }).catch(() => {});
  }, []);

  const selected = useMemo(() => parse(value), [value]);
  const setSelected = (next: Set<string>) => onChange([...next].join(","));
  const toggle = (id: string, on: boolean) => {
    const next = new Set(selected);
    if (on) next.add(id); else next.delete(id);
    setSelected(next);
  };

  const label =
    selected.size === 0
      ? "Auto (any region)"
      : selected.size === 1
        ? (regions.find((r) => r.id === [...selected][0])?.label ?? [...selected][0])
        : `${selected.size} regions`;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className={cn(
            "flex w-full items-center justify-between gap-2 rounded-md border border-input bg-transparent px-3 py-1 shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50",
            className,
          )}
        >
          <span className="truncate">{label}</span>
          <ChevronDown className="h-4 w-4 shrink-0 opacity-50" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="max-h-72 w-64 overflow-y-auto">
        {/* Auto = clear the allowlist. Selecting any DC turns it off automatically. */}
        <DropdownMenuCheckboxItem
          checked={selected.size === 0}
          onCheckedChange={() => setSelected(new Set())}
          onSelect={(e) => e.preventDefault()}
          className="text-xs"
        >
          Auto (any region)
        </DropdownMenuCheckboxItem>
        {regions.map((r) => (
          <DropdownMenuCheckboxItem
            key={r.id}
            checked={selected.has(r.id)}
            onCheckedChange={(on) => toggle(r.id, !!on)}
            onSelect={(e) => e.preventDefault()}
            className="text-xs"
          >
            {r.label} <span className="ml-1 text-muted-foreground">· {r.id}</span>
          </DropdownMenuCheckboxItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
