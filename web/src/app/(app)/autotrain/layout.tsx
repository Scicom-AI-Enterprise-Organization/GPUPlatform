import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When Autotrain is turned off via DISABLED_SECTIONS, 404 every /autotrain/* route —
// the pages are gone, not just hidden from the sidebar nav.
export default function AutotrainLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("autotrain")) notFound();
  return <>{children}</>;
}
