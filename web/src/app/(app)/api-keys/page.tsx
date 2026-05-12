import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { ApiKeyPanel } from "./api-key-panel";

export default async function ApiKeysPage() {
  const username = await currentUsername();
  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "API keys" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="">
          <div className="mb-6">
            <h1 className="text-2xl font-semibold tracking-tight">API keys</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Use this token in the <code className="font-mono">Authorization</code> header to call the
              gateway from your own scripts and SDKs. Treat it like a password.
            </p>
          </div>
          <ApiKeyPanel />
        </div>
      </div>
    </div>
  );
}
