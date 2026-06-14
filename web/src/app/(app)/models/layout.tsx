import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When the catalog surface is turned off via DISABLED_SECTIONS, 404 every
// /models/* route — the page is gone, not just hidden from the sidebar nav.
export default function ModelsLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("catalog")) notFound();
  return <>{children}</>;
}
