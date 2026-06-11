# syntax=docker/dockerfile:1.6
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# NEXT_PUBLIC_* vars are inlined at build time. Pass the public gateway URL
# here so the curl examples in the UI point at the live endpoint, not the
# `http://localhost:8080` fallback.
ARG NEXT_PUBLIC_GATEWAY_URL=https://serverlessgpu.aies.scicom.dev
ENV NEXT_PUBLIC_GATEWAY_URL=${NEXT_PUBLIC_GATEWAY_URL}
# In-cluster gateway Service URL. Baked at build time (like the public URL above)
# so the "Run a job" cards can offer a Public/Internal snippet toggle. Empty =
# toggle hidden. A runtime pod env can't drive this — NEXT_PUBLIC_* is inlined here.
ARG NEXT_PUBLIC_GATEWAY_INTERNAL_URL=http://serverlessgpu-gateway.serverlessgpu.svc.cluster.local
ENV NEXT_PUBLIC_GATEWAY_INTERNAL_URL=${NEXT_PUBLIC_GATEWAY_INTERNAL_URL}
# Release version (git short-sha) shown in the sidebar footer. CI passes
# `--build-arg APP_VERSION=<sha>`; defaults to "dev" for local builds.
ARG APP_VERSION=dev
ENV NEXT_PUBLIC_APP_VERSION=${APP_VERSION}
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0
RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
