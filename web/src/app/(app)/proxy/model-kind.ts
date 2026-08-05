// What kind of model a name refers to, guessed from the name itself.
//
// Used in two places that must agree: the upstream Test toggle (so a reranker
// upstream doesn't get chat-tested and 404) and the Routing panel (which groups
// models by kind). One source of truth so they can't drift.
//
// Order matters: "bge-reranker-v2" is a cross-encoder, not an embedder, so rerank
// has to be checked before embed. Only ever a hint — the admin's own choice wins.
export type ModelKind = "chat" | "embedding" | "rerank" | "transcription" | "tts";

export function modelKind(...names: string[]): ModelKind {
  const hay = names.join(" ").toLowerCase();
  if (/rerank|cross-encoder/.test(hay)) return "rerank";
  if (/embed|bge|gte-|e5-|nomic/.test(hay)) return "embedding";
  if (/whisper|\bstt\b|transcri/.test(hay)) return "transcription";
  // deliberately NOT "voice" — this fleet has voice-suffixed CHAT models (gemma-…-it-voice)
  if (/tts|speech|kokoro|orpheus/.test(hay)) return "tts";
  return "chat";
}

// The OpenAI-compatible path a request for this kind of model goes to.
export const KIND_PATH: Record<ModelKind, string> = {
  chat: "/v1/chat/completions",
  embedding: "/v1/embeddings",
  rerank: "/v1/rerank",
  transcription: "/v1/audio/transcriptions",
  tts: "/v1/audio/speech",
};

export const KIND_LABEL: Record<ModelKind, string> = {
  chat: "Chat",
  embedding: "Embeddings",
  rerank: "Rerank",
  transcription: "Transcription",
  tts: "Speech",
};

export const KIND_ORDER: ModelKind[] = ["chat", "embedding", "rerank", "transcription", "tts"];
