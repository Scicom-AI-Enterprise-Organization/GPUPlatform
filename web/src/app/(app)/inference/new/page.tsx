import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { InferenceForm } from "./inference-form";

export default async function NewEndpointPage() {
  const [me, username] = await Promise.all([getMe(), currentUsername()]);
  const noAccess = me?.role === "user";
  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[
          { label: "Inference", href: "/inference" },
          { label: "New endpoint" },
        ]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        {noAccess ? <NoAccessAlert /> : <InferenceForm />}
      </div>
    </div>
  );
}
