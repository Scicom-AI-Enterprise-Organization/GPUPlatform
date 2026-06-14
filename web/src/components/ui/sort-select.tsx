"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export type SortDir = "newest" | "oldest";

/** Sort a list by its `created_at` timestamp. Returns a new array. */
export function sortByCreated<T extends { created_at: string }>(items: T[], dir: SortDir): T[] {
  return [...items].sort((a, b) => {
    const ta = new Date(a.created_at).getTime();
    const tb = new Date(b.created_at).getTime();
    return dir === "newest" ? tb - ta : ta - tb;
  });
}

/** Shared "Newest / Oldest first" sort control used across the list pages. */
export function SortSelect({
  value,
  onValueChange,
  className = "h-10! w-[150px]",
}: {
  value: SortDir;
  onValueChange: (v: SortDir) => void;
  className?: string;
}) {
  return (
    <Select value={value} onValueChange={(v) => onValueChange(v as SortDir)}>
      <SelectTrigger className={className} title="Sort by created time">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="newest">Newest first</SelectItem>
        <SelectItem value="oldest">Oldest first</SelectItem>
      </SelectContent>
    </Select>
  );
}
