import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When the Inference surface is turned off via DISABLED_SECTIONS, 404 every
// /serverless/* route — the page is gone, not just hidden from the sidebar nav.
export default function ServerlessLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("inference")) notFound();
  return <>{children}</>;
}
