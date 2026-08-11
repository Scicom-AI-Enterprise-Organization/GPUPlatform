import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When Quantization is turned off via DISABLED_SECTIONS, 404 every /quantization/* route —
// the pages are gone, not just hidden from the sidebar nav.
export default function QuantizationLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("quantization")) notFound();
  return <>{children}</>;
}
