import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When GitOps is turned off via DISABLED_SECTIONS, 404 every /gitops/* route —
// the page is gone, not just hidden from the sidebar nav. Mirrors the guards on
// /inference and /benchmark.
export default function GitOpsLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("gitops")) notFound();
  return <>{children}</>;
}
