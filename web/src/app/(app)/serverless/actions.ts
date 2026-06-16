"use server";

import { revalidatePath } from "next/cache";
import { gateway, GatewayError } from "@/lib/gateway";
import type { CreateAppRequest } from "@/lib/types";

export type DeployResult =
  | { ok: true; app_id: string }
  | {
      ok: false;
      error: string;
      // Populated when the gateway rejected with a structured "GPU not
      // available" error from the create-time provision pre-flight (503).
      // The form uses these to render a clear modal instead of a toast.
      unavailable?: {
        gpu: string;
        gpu_count: number;
        reason: string;
      };
    };

export async function deployEndpoint(input: CreateAppRequest): Promise<DeployResult> {
  try {
    const res = await gateway.createApp(input);
    revalidatePath("/serverless");
    return { ok: true, app_id: res.app_id };
  } catch (e) {
    if (e instanceof GatewayError && e.status === 503 && e.parsed) {
      const detail = (e.parsed as { detail?: Record<string, unknown> }).detail;
      if (
        detail &&
        typeof detail === "object" &&
        "reason" in detail &&
        "gpu" in detail
      ) {
        return {
          ok: false,
          error: String(detail.error ?? "GPU not available"),
          unavailable: {
            gpu: String(detail.gpu),
            gpu_count: Number(detail.gpu_count ?? 1),
            reason: String(detail.reason),
          },
        };
      }
    }
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

export async function updateAutoscaler(
  appId: string,
  patch: Partial<{ max_containers: number; tasks_per_container: number; idle_timeout_s: number; worker_ttl_s: number; vllm_args: string }>,
): Promise<DeployResult> {
  try {
    await gateway.updateAutoscaler(appId, patch);
    revalidatePath(`/serverless/${appId}`);
    revalidatePath("/serverless");
    return { ok: true, app_id: appId };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

export async function restartEndpoint(
  appId: string,
): Promise<{ ok: true; drained: number } | { ok: false; error: string }> {
  try {
    const res = await gateway.restartApp(appId);
    revalidatePath(`/serverless/${appId}`);
    return { ok: true, drained: res.drained_workers };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

export async function purgeWorkers(
  appId: string,
): Promise<{ ok: true; terminated: number; purged: number } | { ok: false; error: string }> {
  try {
    const res = await gateway.purgeApp(appId);
    revalidatePath(`/serverless/${appId}`);
    return { ok: true, terminated: res.terminated, purged: res.purged };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

export async function deleteEndpoint(appId: string): Promise<DeployResult> {
  try {
    await gateway.deleteApp(appId);
    revalidatePath("/serverless");
    return { ok: true, app_id: appId };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}
