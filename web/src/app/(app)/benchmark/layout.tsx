import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When the Benchmark surface is turned off via DISABLED_SECTIONS, 404 every
// /benchmark/* route — the page is gone, not just hidden from the sidebar nav.
export default function BenchmarkLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("benchmark")) notFound();
  return <>{children}</>;
}
