import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  devIndicators: false,
  // The section moved /serverless → /inference (2026-08-10). Existing bookmarks,
  // Slack links and anything that deep-linked an endpoint would 404 otherwise, so
  // keep the old path redirecting. Permanent (308) — the old route is gone, not
  // temporarily elsewhere. Safe to drop once nothing points at /serverless.
  async redirects() {
    return [
      { source: "/serverless", destination: "/inference", permanent: true },
      { source: "/serverless/:path*", destination: "/inference/:path*", permanent: true },
    ];
  },
};

export default nextConfig;
