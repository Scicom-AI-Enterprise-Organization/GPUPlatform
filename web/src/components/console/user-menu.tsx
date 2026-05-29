"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useTransition } from "react";
import { LogOut, Settings, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export function UserMenu({ username }: { username: string }) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();

  function logout() {
    startTransition(async () => {
      await fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
      router.replace("/login");
      router.refresh();
    });
  }

  const initial = username.charAt(0).toUpperCase() || "?";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="rounded-full p-0"
          aria-label="Account menu"
          // Radix derives this trigger's id/aria-controls from useId. Under the
          // async App Router shell (React 19 + Next 16), the server emits this
          // client component at a different useId tree-offset than hydration
          // reproduces, so the generated id mismatches. It's an aria-only id
          // (menu wiring is via context/refs, not the DOM id) — harmless, so we
          // suppress the dev-only hydration warning here rather than mask it app-wide.
          suppressHydrationWarning
        >
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-emerald-600 text-xs font-semibold text-emerald-50">
            {initial}
          </span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-44">
        <DropdownMenuLabel className="flex items-center gap-2">
          <User className="h-4 w-4" />
          <span className="truncate">{username}</span>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem asChild>
          <Link href="/settings">
            <Settings className="h-4 w-4" />
            Settings
          </Link>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem variant="destructive" disabled={pending} onSelect={logout}>
          <LogOut className="h-4 w-4" />
          Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
