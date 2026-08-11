import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When Experiments is turned off via DISABLED_SECTIONS, 404 every /experiments/* route —
// the pages are gone, not just hidden from the sidebar nav.
export default function ExperimentsLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("experiments")) notFound();
  return <>{children}</>;
}
