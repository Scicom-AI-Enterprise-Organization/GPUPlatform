import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When API Proxy is turned off via DISABLED_SECTIONS, 404 every /proxy/* route —
// the pages are gone, not just hidden from the sidebar nav.
export default function ProxyLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("proxy")) notFound();
  return <>{children}</>;
}
