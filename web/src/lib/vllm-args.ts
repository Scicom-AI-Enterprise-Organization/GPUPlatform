/** Clean a pasted vLLM-args string into one line. Copying a multi-line
 *  `vllm serve` block — where each flag sits on its own line ending in a shell
 *  line-continuation `\` — would otherwise paste backslashes + newlines that the
 *  gateway rejects. This turns that block into a single space-separated line:
 *  drops newlines and backslashes, then collapses runs of whitespace.
 *
 *  It's a no-op for text the user is normally typing (no newlines, backslashes,
 *  or double-spaces), so it's safe to run on every onChange without fighting the
 *  caret. Trailing spaces are kept so you can type "--foo " then the next token;
 *  callers still `.trim()` before submit. */
export function cleanVllmArgs(s: string): string {
  return s
    .replace(/\r?\n/g, " ") // newlines → space
    .replace(/\s*\\\s*/g, " ") // line-continuation / stray backslash → space
    .replace(/[ \t]{2,}/g, " ") // collapse runs of spaces/tabs
    .replace(/^[ \t]+/, ""); // drop leading whitespace (never meaningful here)
}
