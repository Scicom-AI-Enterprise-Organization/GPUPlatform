"use client";

import Link from "next/link";
import {
  CloudUpload,
  Database,
  FileAudio,
  HardDrive,
  MessagesSquare,
  MoreHorizontal,
  Pencil,
  Rows3,
  Tags,
  Trash2,
  User,
} from "lucide-react";
import type { DatasetKind, DatasetRecord } from "@/lib/types";
import { avatarFor } from "@/lib/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

export const KIND_LABEL: Record<DatasetKind, string> = {
  upload: "upload",
  s3: "s3",
  hf: "huggingface",
  label: "labeling",
  tts_packed: "tts-packed",
  omnivoice_packed: "omnivoice-packed",
  hosted: "hf repo",
  llm: "llm",
  llm_packed: "llm-packed",
};

export function KindIcon({ kind, className }: { kind: DatasetKind; className?: string }) {
  if (kind === "hf" || kind === "hosted") return <Database className={className} />;
  if (kind === "s3") return <CloudUpload className={className} />;
  if (kind === "label") return <Tags className={className} />;
  if (kind === "llm" || kind === "llm_packed") return <MessagesSquare className={className} />;
  return <FileAudio className={className} />;
}

/** The dataset's source reference, shown as the first chip. */
function sourceDetail(d: DatasetRecord): string | null {
  if (d.kind === "hf") return d.hf_repo || null;
  if (d.kind === "s3") return d.s3_metadata_uri || null;
  if (d.kind === "label") return d.label_project_id ? `project ${d.label_project_id.slice(0, 8)}…` : null;
  return d.metadata_filename || null;
}

export function DatasetCard({
  dataset: d,
  onRename,
  onDelete,
  selectMode = false,
  selected = false,
  onToggle,
}: {
  dataset: DatasetRecord;
  onRename?: (d: DatasetRecord) => void;
  onDelete?: (d: DatasetRecord) => void;
  selectMode?: boolean;
  selected?: boolean;
  onToggle?: (id: string) => void;
}) {
  const avatar = avatarFor(d.name);
  const detail = sourceDetail(d);
  // A hosted HF-mirror repo (name = "ns/name") links to its name-based detail.
  const hosted = d.kind === "hosted";
  const href = hosted ? `/datasets/hosted/${d.name}` : `/datasets/${encodeURIComponent(d.id)}`;

  const inner = (
    <>
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          {selectMode && (
            <input
              type="checkbox"
              checked={selected}
              onChange={() => onToggle?.(d.id)}
              onClick={(e) => e.stopPropagation()}
              className="h-4 w-4 shrink-0 cursor-pointer accent-primary"
              aria-label={`Select ${d.name}`}
            />
          )}
          <div className={cn("flex h-11 w-11 shrink-0 items-center justify-center rounded-lg text-base font-semibold", avatar.bg, avatar.text)}>
            {avatar.letter}
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="truncate font-medium text-foreground">{d.name}</span>
              <span className="inline-flex shrink-0 items-center gap-1 rounded-full border border-border bg-muted px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                <KindIcon kind={d.kind} className="h-3 w-3" />
                {KIND_LABEL[d.kind]}
              </span>
            </div>
            <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="truncate font-mono" title={d.id}>{d.id}</span>
              <span>·</span>
              <User className="h-3 w-3" />
              <span className="truncate">{d.created_by}</span>
            </div>
          </div>
        </div>

        {!selectMode && (onRename || onDelete) && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                className="-mr-1 text-muted-foreground hover:text-foreground"
                aria-label="Actions"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                }}
              >
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
              {onRename && !hosted && (
                <DropdownMenuItem onSelect={(e) => { e.preventDefault(); onRename(d); }}>
                  <Pencil className="h-4 w-4" />
                  Rename
                </DropdownMenuItem>
              )}
              {onDelete && (
                <DropdownMenuItem variant="destructive" onSelect={(e) => { e.preventDefault(); onDelete(d); }}>
                  <Trash2 className="h-4 w-4" />
                  {hosted ? "Delete repo" : "Delete dataset"}
                </DropdownMenuItem>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        {detail && (
          <span className="inline-flex max-w-full items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs">
            <KindIcon kind={d.kind} className="h-3 w-3 shrink-0 text-muted-foreground" />
            <span className="truncate font-mono">{detail}</span>
          </span>
        )}
        {d.storage_name && (
          <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs">
            <HardDrive className="h-3 w-3 text-muted-foreground" />
            {d.storage_name}
          </span>
        )}
        {typeof d.num_rows === "number" && (
          <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
            <Rows3 className="h-3 w-3 text-muted-foreground" />
            {d.num_rows.toLocaleString()} rows
          </span>
        )}
      </div>

      <div className="mt-3 flex items-center justify-between gap-3 border-t border-border/60 pt-2 text-xs text-muted-foreground">
        <span className="truncate">{d.description || ""}</span>
        <span className="shrink-0" title={new Date(d.created_at).toISOString()}>
          {new Date(d.created_at).toLocaleDateString()}
        </span>
      </div>
    </>
  );

  const baseCard = "group block rounded-xl border border-border bg-card p-4 transition-all";

  if (selectMode) {
    return (
      <div
        role="button"
        tabIndex={0}
        onClick={() => onToggle?.(d.id)}
        onKeyDown={(e) => {
          if (e.key === " " || e.key === "Enter") {
            e.preventDefault();
            onToggle?.(d.id);
          }
        }}
        className={cn(
          baseCard,
          "cursor-pointer",
          selected ? "border-primary/60 bg-primary/5" : "hover:border-primary/40 hover:bg-card/80",
        )}
      >
        {inner}
      </div>
    );
  }

  return (
    <Link href={href} className={cn(baseCard, "hover:border-primary/40 hover:bg-card/80 hover:shadow-md")}>
      {inner}
    </Link>
  );
}
