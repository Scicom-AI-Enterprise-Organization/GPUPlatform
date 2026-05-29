import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

// Unit tests for the JS API layer (creating serverless inference endpoints via
// the gateway client). Node env — these exercise the server-side request path.
export default defineConfig({
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
