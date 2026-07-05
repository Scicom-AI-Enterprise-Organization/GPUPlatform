import Link from "next/link";
import { BookOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { ApiKeyPanel } from "./api-key-panel";
import { FormShell } from "@/components/form-shell";

export default async function ApiKeysPage() {
  const username = await currentUsername();
  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "API tokens" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">API tokens</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Personal tokens for the gateway API — submit and list jobs, deploy endpoints,
              run benchmarks, and manage compute from scripts or CI. Pass a token as a{" "}
              <code className="font-mono">Bearer</code> credential. Treat tokens like passwords.
            </p>
          </div>
          <Button asChild variant="outline" size="sm" className="shrink-0">
            <Link href="/api-docs">
              <BookOpen className="h-4 w-4" /> API docs
            </Link>
          </Button>
        </div>
        <FormShell>
          <ApiKeyPanel />
        </FormShell>
      </div>
    </div>
  );
}
