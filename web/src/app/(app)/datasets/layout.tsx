import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When the Datasets surface is turned off via DISABLED_SECTIONS, 404 every
// /datasets/* route — the page is gone, not just hidden from the sidebar nav.
export default function DatasetsLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("datasets")) notFound();
  return <>{children}</>;
}
