"use client";

import { useEffect, useRef, useState } from "react";
import type { GpuAvailability } from "./gateway";

type State =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ok"; data: GpuAvailability }
  | { status: "error"; message: string };

export function useGpuAvailability(
  gpu: string,
  count: number,
  enabled = true,
  cloudType?: "COMMUNITY" | "SECURE",
): State {
  const [state, setState] = useState<State>({ status: "idle" });
  const reqId = useRef(0);

  useEffect(() => {
    if (!enabled || !gpu || count < 1) {
      setState({ status: "idle" });
      return;
    }
    const id = ++reqId.current;
    setState({ status: "loading" });
    const t = setTimeout(async () => {
      try {
        const params = new URLSearchParams({ gpu, count: String(count) });
        if (cloudType) params.set("cloud_type", cloudType);
        const res = await fetch(
          `/api/proxy/v1/availability?${params.toString()}`,
          { cache: "no-store" },
        );
        if (id !== reqId.current) return;
        if (!res.ok) {
          const body = await res.text().catch(() => "");
          setState({
            status: "error",
            message: `gateway ${res.status}: ${body || res.statusText}`,
          });
          return;
        }
        const data = (await res.json()) as GpuAvailability;
        if (id !== reqId.current) return;
        setState({ status: "ok", data });
      } catch (e) {
        if (id !== reqId.current) return;
        setState({
          status: "error",
          message: e instanceof Error ? e.message : String(e),
        });
      }
    }, 300);
    return () => clearTimeout(t);
  }, [gpu, count, enabled, cloudType]);

  return state;
}
