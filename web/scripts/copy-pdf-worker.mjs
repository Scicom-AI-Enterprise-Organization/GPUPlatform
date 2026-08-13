// Copy pdf.js's worker out of node_modules into public/ so the playground can load it
// from a URL we control (/pdfjs/pdf.worker.min.mjs).
//
// Why not `new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url)`: that form
// relies on the bundler rewriting a BARE package specifier inside new URL(), which is
// not standard (the spec resolves it relative to the module's own URL) and differs
// between webpack/turbopack versions. When it silently resolves wrong, the failure only
// shows up the first time a user picks a PDF, in the browser, as a 404 for a worker —
// the worst place to find it. A copied asset either exists or 404s at a path we can curl.
//
// Runs from predev/prebuild, so it is in place for `npm run dev` and `next build` alike.
// The copy is gitignored — node_modules is the source of truth for the version.

import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
// ⚠ legacy/, matching the legacy main build the playground imports: pdf.js 6's modern
// build assumes Map.prototype.getOrInsertComputed (a TC39 proposal browsers don't ship
// yet) with no polyfill, and fails with
//   "this[#methodPromises].getOrInsertComputed is not a function"
// the first time a PDF is dropped. The two halves must come from the SAME build — a
// modern worker under a legacy main build reintroduces the identical error.
const src = join(here, "..", "node_modules", "pdfjs-dist", "legacy", "build", "pdf.worker.min.mjs");
const destDir = join(here, "..", "public", "pdfjs");
const dest = join(destDir, "pdf.worker.min.mjs");

try {
  const s = statSync(src);
  mkdirSync(destDir, { recursive: true });
  // Skip an identical copy so a warm `npm run dev` doesn't churn the file.
  let same = false;
  try { same = statSync(dest).size === s.size; } catch { /* not there yet */ }
  if (!same) {
    copyFileSync(src, dest);
    console.log(`[pdf-worker] copied ${(s.size / 1024).toFixed(0)} KB → public/pdfjs/pdf.worker.min.mjs`);
  }
} catch (e) {
  // Not fatal: everything except PDF attachments works without it, and the playground
  // reports the missing worker inline if someone picks a PDF.
  console.warn(`[pdf-worker] skipped (${e.code ?? e.message}) — PDF attachments will report a missing worker`);
}
