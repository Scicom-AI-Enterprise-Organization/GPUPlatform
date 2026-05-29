import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { ApiDocs } from "./api-docs";

export default async function ApiDocsPage() {
  const username = await currentUsername();
  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "API docs" }]} username={username} />
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <ApiDocs />
      </div>
    </div>
  );
}
