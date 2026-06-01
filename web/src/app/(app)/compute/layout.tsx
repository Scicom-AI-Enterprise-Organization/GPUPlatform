import { notFound } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When the Compute surface is turned off via DISABLED_SECTIONS, 404 every
// /compute/* route (list, new, [id]) — the page is gone, not just hidden from the
// sidebar nav. (The gateway also 403s its compute routes for defense in depth.)
export default function ComputeLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("compute")) notFound();
  return <>{children}</>;
}
