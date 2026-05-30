// Product surfaces the *web* turns off, independent of the gateway, via the
// server-side DISABLED_SECTIONS env (comma-separated: inference,benchmark,compute).
// Mirrors the gateway's DISABLED_SECTIONS so a section can be killed from either
// side. Read in server components only (plain env, not NEXT_PUBLIC).
export function disabledSections(): Set<string> {
  return new Set(
    (process.env.DISABLED_SECTIONS ?? "")
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean),
  );
}
