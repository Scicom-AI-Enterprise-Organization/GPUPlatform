import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { QuantizationForm } from "./quantization-form";

export default async function NewQuantizationJobPage() {
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.quantization) : false;
  const username = await currentUsername();

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Quantization", href: "/quantization" }, { label: "New job" }]}
        username={username}
      />
      <div className="relative flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        {noAccess ? <NoAccessAlert /> : <QuantizationForm />}
      </div>
    </div>
  );
}
