import { redirect } from "next/navigation";
import { disabledSections } from "@/lib/sections";

// When the Compute surface is turned off via DISABLED_SECTIONS, bounce every
// /compute/* route (list, new, [id]) — the sidebar nav is already hidden, this
// blocks direct navigation too.
export default function ComputeLayout({ children }: { children: React.ReactNode }) {
  if (disabledSections().has("compute")) redirect("/serverless");
  return <>{children}</>;
}
