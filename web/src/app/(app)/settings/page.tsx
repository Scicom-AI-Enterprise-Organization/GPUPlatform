import { redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { getMe } from "@/lib/me";
import { ProfileForm } from "./profile-form";
import { AppearanceSettings } from "./appearance";

export default async function SettingsPage() {
  const me = await getMe();
  if (!me) redirect("/login");

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Settings" }]} username={me.username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
      <div className="space-y-8">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Your profile. Username and email are tied to your account and can&apos;t be changed here.
          </p>
        </header>

        <ProfileForm
          username={me.username}
          email={me.email ?? ""}
          role={me.role}
        />

        <AppearanceSettings />
      </div>
      </div>
    </div>
  );
}
