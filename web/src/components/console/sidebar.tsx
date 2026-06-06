"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { BookOpen, Box, Boxes, CheckSquare, Cloud, Database, FlaskConical, GitBranch, KeyRound, Library, Lock, ScrollText, Server, Settings, Shield, Sparkles, Users } from "lucide-react";
import { cn } from "@/lib/utils";
import { useSidebarState } from "./sidebar-state";

type Item = {
  label: string;
  href: string;
  icon: React.ElementType;
  locked?: boolean;
  // If set, this item is dropped from the nav when the section is turned off
  // platform-wide via DISABLED_SECTIONS.
  section?: "inference" | "benchmark" | "compute" | "datasets";
};

const RESOURCES: Item[] = [
  { label: "Serverless Inference", href: "/serverless", icon: Boxes, section: "inference" },
  { label: "Benchmark", href: "/benchmark", icon: FlaskConical, section: "benchmark" },
  { label: "Storage", href: "/storage", icon: Database },
  { label: "Datasets", href: "/datasets", icon: Library, section: "datasets" },
  { label: "Autotrain", href: "/autotrain", icon: Sparkles },
  { label: "Compute", href: "/compute", icon: Box, section: "compute" },
  { label: "GPU Providers", href: "/providers", icon: Cloud },
];
const ACCOUNT: Item[] = [
  { label: "API tokens", href: "/api-keys", icon: KeyRound },
  { label: "API docs", href: "/api-docs", icon: BookOpen },
  { label: "Settings", href: "/settings", icon: Settings },
];
const ADMIN: Item[] = [
  { label: "GitOps", href: "/gitops", icon: GitBranch },
  { label: "Organization", href: "/organization", icon: Users },
  { label: "Roles", href: "/admin/roles", icon: Shield },
  { label: "Secrets", href: "/admin/secrets", icon: Lock },
  { label: "Audit log", href: "/admin/audit", icon: ScrollText },
];
const MANAGE: Item[] = [
  { label: "Compute approvals", href: "/admin/compute-approvals", icon: CheckSquare },
  { label: "Provisioned", href: "/admin/provisioned", icon: Server },
];

type Counts = { pendingApprovals: number; provisioned: number };

export function ConsoleSidebar({
  isAdmin = false,
  disabled = [],
  counts = { pendingApprovals: 0, provisioned: 0 },
}: {
  isAdmin?: boolean;
  // Surfaces turned off platform-wide (DISABLED_SECTIONS) — dropped from the nav.
  disabled?: string[];
  counts?: Counts;
} = {}) {
  const disabledSet = new Set(disabled);
  const resources = RESOURCES.filter((item) => !(item.section && disabledSet.has(item.section)));
  // Map of href → numeric badge to show next to the item label. Always
  // present (default 0) so admins know the rail is wired up even when
  // nothing's pending.
  const BADGES: Record<string, number> = {
    "/admin/compute-approvals": counts.pendingApprovals,
    "/admin/provisioned": counts.provisioned,
  };
  const pathname = usePathname();
  const { collapsed, mobileOpen, closeMobile } = useSidebarState();

  const isActive = (href: string) => {
    if (href === "/serverless") {
      return pathname === "/serverless" || pathname.startsWith("/serverless/");
    }
    if (href === "/benchmark") {
      return pathname === "/benchmark" || pathname.startsWith("/benchmark/");
    }
    if (href === "/compute") {
      return pathname === "/compute" || pathname.startsWith("/compute/");
    }
    if (href === "/providers") {
      return pathname === "/providers" || pathname.startsWith("/providers/");
    }
    if (href === "/storage") {
      return pathname === "/storage" || pathname.startsWith("/storage/");
    }
    if (href === "/datasets") {
      return pathname === "/datasets" || pathname.startsWith("/datasets/");
    }
    if (href === "/autotrain") {
      return pathname === "/autotrain" || pathname.startsWith("/autotrain/");
    }
    if (href === "/gitops") {
      return pathname === "/gitops" || pathname.startsWith("/gitops/");
    }
    return pathname === href;
  };

  // Show every resource so users see the full surface of the platform; the
  // page itself renders a "no permission" alert if they can't actually use it.
  const groups = (
    <>
      <SidebarGroup label="Resources" collapsed={collapsed}>
        {resources.map((item) => (
          <SidebarItem
            key={item.label}
            item={item}
            active={isActive(item.href)}
            collapsed={collapsed}
            onNavigate={closeMobile}
          />
        ))}
      </SidebarGroup>

      {isAdmin && (
        <SidebarGroup label="Manage" collapsed={collapsed}>
          {MANAGE.map((item) => (
            <SidebarItem
              key={item.label}
              item={item}
              active={isActive(item.href)}
              collapsed={collapsed}
              onNavigate={closeMobile}
              badge={BADGES[item.href]}
            />
          ))}
        </SidebarGroup>
      )}

      <SidebarGroup label="Account" collapsed={collapsed}>
        {ACCOUNT.map((item) => (
          <SidebarItem
            key={item.label}
            item={item}
            active={isActive(item.href)}
            collapsed={collapsed}
            onNavigate={closeMobile}
          />
        ))}
      </SidebarGroup>

      {isAdmin && (
        <SidebarGroup label="Admin" collapsed={collapsed}>
          {ADMIN.map((item) => (
            <SidebarItem
              key={item.label}
              item={item}
              active={isActive(item.href)}
              collapsed={collapsed}
              onNavigate={closeMobile}
            />
          ))}
        </SidebarGroup>
      )}
    </>
  );

  return (
    <>
      {/* Mobile drawer overlay */}
      {mobileOpen && (
        <button
          aria-label="Close sidebar"
          onClick={closeMobile}
          className="fixed inset-0 z-30 bg-background/70 backdrop-blur-sm md:hidden"
        />
      )}

      <aside
        className={cn(
          "h-full shrink-0 flex-col border-r border-sidebar-border bg-sidebar transition-[width,transform] duration-200 ease-out",
          // Desktop: visible, width depends on collapsed
          "hidden md:flex",
          collapsed ? "md:w-16" : "md:w-60",
          // Mobile: render as fixed drawer when mobileOpen
          mobileOpen
            ? "fixed inset-y-0 left-0 z-40 flex w-64 translate-x-0"
            : "max-md:-translate-x-full max-md:fixed max-md:inset-y-0 max-md:left-0 max-md:z-40 max-md:w-64",
        )}
      >
        <Link
          href="/"
          onClick={closeMobile}
          className={cn(
            "flex h-14 shrink-0 items-center gap-2 border-b border-sidebar-border hover:bg-sidebar-accent/40",
            collapsed ? "justify-center px-2" : "px-4",
          )}
        >
          <Image
            src="/logos/scicom-logo.png"
            alt="Scicom"
            width={96}
            height={24}
            priority
            className={cn("h-6 select-none", collapsed ? "w-6 object-contain" : "w-auto")}
          />
          {!collapsed && (
            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              GPU Platform
            </span>
          )}
        </Link>

        <nav className="flex-1 overflow-y-auto py-3 scrollbar-thin">{groups}</nav>

        {/* Build version — baked in at `next build` from APP_VERSION (git
            short-sha in CI). Defaults to "dev" for local/unversioned builds so
            the footer is always populated. */}
        <div
          className={cn(
            "shrink-0 border-t border-sidebar-border py-2",
            collapsed ? "px-2 text-center" : "px-4",
          )}
          title={`Build ${process.env.NEXT_PUBLIC_APP_VERSION || "dev"}`}
        >
          <p className="truncate font-mono text-[10px] text-muted-foreground">
            {collapsed
              ? (process.env.NEXT_PUBLIC_APP_VERSION || "dev").slice(0, 4)
              : `v${process.env.NEXT_PUBLIC_APP_VERSION || "dev"}`}
          </p>
        </div>
      </aside>
    </>
  );
}

function SidebarGroup({
  label,
  collapsed,
  children,
}: {
  label: string;
  collapsed?: boolean;
  children: React.ReactNode;
}) {
  return (
    <>
      {!collapsed && (
        <div className="mt-3 flex w-full items-center px-4 py-1.5 text-xs font-medium text-muted-foreground">
          {label}
        </div>
      )}
      <ul className={cn("space-y-px", collapsed ? "px-2 pt-2" : "px-2")}>{children}</ul>
    </>
  );
}

function SidebarItem({
  item,
  active,
  collapsed,
  onNavigate,
  badge,
}: {
  item: Item;
  active?: boolean;
  collapsed?: boolean;
  onNavigate?: () => void;
  /** Optional numeric badge (e.g. count of pending items). Defaults
   * undefined = no badge; pass 0 to show a neutral "0" pill. */
  badge?: number;
}) {
  if (item.locked) {
    return (
      <li>
        <div
          aria-disabled
          title={collapsed ? `${item.label} — coming soon` : "Coming soon"}
          className={cn(
            "group flex cursor-not-allowed items-center rounded-md px-2 py-1.5 text-sm text-muted-foreground/70",
            collapsed ? "justify-center" : "gap-2",
          )}
        >
          <item.icon className="h-4 w-4 shrink-0" />
          {!collapsed && (
            <>
              <span className="flex-1 truncate">{item.label}</span>
              <Lock className="h-3 w-3 shrink-0 opacity-70" />
            </>
          )}
        </div>
      </li>
    );
  }
  return (
    <li className="relative">
      <Link
        href={item.href}
        onClick={onNavigate}
        title={collapsed ? item.label : undefined}
        className={cn(
          "group flex w-full items-center rounded-md px-2 py-1.5 text-sm transition-colors",
          collapsed ? "justify-center" : "gap-2",
          active
            ? "bg-sidebar-accent text-sidebar-accent-foreground"
            : "text-sidebar-foreground hover:bg-sidebar-accent/60 hover:text-foreground",
        )}
      >
        <item.icon className="h-4 w-4 shrink-0" />
        {!collapsed && <span className="flex-1 truncate">{item.label}</span>}
        {!collapsed && badge !== undefined && (
          <span className="ml-auto inline-flex min-w-[1.25rem] items-center justify-center rounded-md border border-border bg-muted/60 px-1.5 text-[10px] font-medium tabular-nums text-muted-foreground">
            {badge}
          </span>
        )}
      </Link>
    </li>
  );
}
