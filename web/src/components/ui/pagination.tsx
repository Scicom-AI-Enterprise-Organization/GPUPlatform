"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

/** First page, last page, and current ±1, collapsing the gaps to an ellipsis. */
function pageWindow(page: number, pageCount: number): (number | "ellipsis")[] {
  const wanted = new Set<number>([1, pageCount, page, page - 1, page + 1]);
  const sorted = [...wanted].filter((p) => p >= 1 && p <= pageCount).sort((a, b) => a - b);
  const out: (number | "ellipsis")[] = [];
  let prev = 0;
  for (const p of sorted) {
    if (prev && p - prev > 1) out.push("ellipsis");
    out.push(p);
    prev = p;
  }
  return out;
}

const PAGE_SIZE_OPTIONS = [12, 24, 48, 96];

/**
 * Client-side pagination bar — drives slicing in the parent list. Shows the
 * visible range, numbered page buttons (collapsed with ellipses), prev/next,
 * and an optional page-size selector. Renders nothing when there's a single
 * page and no size selector to offer.
 */
export function Pagination({
  page,
  pageCount,
  total,
  pageSize,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = PAGE_SIZE_OPTIONS,
  itemLabel = "items",
}: {
  page: number;
  pageCount: number;
  total: number;
  pageSize: number;
  onPageChange: (p: number) => void;
  onPageSizeChange?: (n: number) => void;
  pageSizeOptions?: number[];
  itemLabel?: string;
}) {
  if (pageCount <= 1 && !onPageSizeChange) return null;

  const start = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const end = Math.min(total, page * pageSize);
  const btn =
    "inline-flex h-8 min-w-8 items-center justify-center rounded-md border border-input bg-background px-2 text-sm shadow-xs transition-colors disabled:cursor-not-allowed disabled:opacity-40";

  return (
    <div className="mt-4 flex flex-col items-center justify-between gap-3 border-t border-border pt-3 sm:flex-row">
      <div className="flex items-center gap-3 text-xs text-muted-foreground">
        <span>
          Showing <span className="font-medium text-foreground">{start}</span>–
          <span className="font-medium text-foreground">{end}</span> of{" "}
          <span className="font-medium text-foreground">{total}</span> {itemLabel}
        </span>
        {onPageSizeChange && (
          <Select
            value={String(pageSize)}
            onValueChange={(v) => onPageSizeChange(Number(v))}
          >
            <SelectTrigger size="sm" className="w-[110px]" aria-label="Items per page">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {pageSizeOptions.map((n) => (
                <SelectItem key={n} value={String(n)}>
                  {n} / page
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>

      {pageCount > 1 && (
        <div className="flex items-center gap-1">
          <button
            type="button"
            className={cn(btn, "hover:bg-muted")}
            onClick={() => onPageChange(page - 1)}
            disabled={page <= 1}
            aria-label="Previous page"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          {pageWindow(page, pageCount).map((p, i) =>
            p === "ellipsis" ? (
              <span key={`e${i}`} className="px-1 text-sm text-muted-foreground">
                …
              </span>
            ) : (
              <button
                key={p}
                type="button"
                onClick={() => onPageChange(p)}
                aria-current={p === page ? "page" : undefined}
                className={cn(
                  btn,
                  p === page
                    ? "border-primary bg-primary text-primary-foreground hover:bg-primary/90"
                    : "hover:bg-muted",
                )}
              >
                {p}
              </button>
            ),
          )}
          <button
            type="button"
            className={cn(btn, "hover:bg-muted")}
            onClick={() => onPageChange(page + 1)}
            disabled={page >= pageCount}
            aria-label="Next page"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );
}
