import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { TrainingForm } from "./training-form";

export default async function NewTrainingRunPage() {
  const me = await getMe();
  const noAccess = me?.role === "user";
  const username = await currentUsername();

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[
          { label: "Autotrain", href: "/autotrain" },
          { label: "New run" },
        ]}
        username={username}
      />
      <div className="relative flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        {noAccess ? <NoAccessAlert /> : <TrainingForm />}
      </div>
    </div>
  );
}
