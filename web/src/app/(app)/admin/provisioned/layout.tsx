import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// Part of the admin "Manage" group — switched off by DISABLED_SECTIONS=manage,
// which also hides the group from the sidebar. Guarded here too so the pages are
// gone, not merely unlinked.
export default function ProvisionedLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("manage")) notFound();
  return <>{children}</>;
}
