import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When GPU Providers is turned off via DISABLED_SECTIONS, 404 every /providers/* route —
// the pages are gone, not just hidden from the sidebar nav.
export default function ProvidersLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("providers")) notFound();
  return <>{children}</>;
}
