import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When Storage is turned off via DISABLED_SECTIONS, 404 every /storage/* route —
// the pages are gone, not just hidden from the sidebar nav.
export default function StorageLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("storage")) notFound();
  return <>{children}</>;
}
