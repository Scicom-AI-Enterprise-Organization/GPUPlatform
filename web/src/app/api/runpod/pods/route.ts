// Server-side proxy to RunPod for the WorkersTab. We use GraphQL
// (api.runpod.io/graphql) instead of REST (rest.runpod.io/v1/pods) because
// the REST response leaves `machine: {}` empty — no gpuTypeId, no location.
// GraphQL `myself.pods` returns machine.gpuTypeId, machine.location, and
// secureCloud in a single call.
//
// RUNPOD_API_KEY stays server-side. Browser hits /api/runpod/pods?app=<id>.

import { NextRequest, NextResponse } from "next/server";
import { requireSession } from "@/lib/require-session";

const RUNPOD_GQL =
  process.env.RUNPOD_GRAPHQL_URL ?? "https://api.runpod.io/graphql";
const NAME_PREFIX = process.env.RUNPOD_NAME_PREFIX ?? "serverlessgpu";

type GqlMachine = {
  gpuTypeId?: string;
  dataCenterId?: string;
  location?: string;
  podHostId?: string;
  secureCloud?: boolean;
};

type GqlPod = {
  id: string;
  name: string;
  desiredStatus?: string;
  lastStatusChange?: string;
  gpuCount?: number;
  vcpuCount?: number;
  memoryInGb?: number;
  containerDiskInGb?: number;
  costPerHr?: number;
  createdAt?: string;
  machine?: GqlMachine;
};

export type WorkerRowResponse = {
  machine_id: string;
  pod_id: string;
  status: "running" | "initializing" | "terminating" | "terminated" | "unknown";
  raw_status: string;
  region: string;
  region_code: string;
  gpu: string;
  gpu_count: number;
  vcpus: number;
  ram_gb: number;
  disk_gb: number;
  created_at: string | null;
  cost_per_hr: number | null;
};

const apiKey = () => process.env.RUNPOD_API_KEY ?? "";

const QUERY = `
query MyPods {
  myself {
    pods {
      id
      name
      desiredStatus
      lastStatusChange
      gpuCount
      vcpuCount
      memoryInGb
      containerDiskInGb
      costPerHr
      createdAt
      machine {
        gpuTypeId
        dataCenterId
        location
        podHostId
        secureCloud
      }
    }
  }
}
`;

export async function GET(req: NextRequest) {
  // Enumerating the org's fleet uses the server's RUNPOD_API_KEY, so verify the
  // caller has a valid session before touching it (middleware skips /api/*).
  const session = await requireSession(req);
  if (!session.ok) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const key = apiKey();
  if (!key) {
    return NextResponse.json(
      { error: "RUNPOD_API_KEY is not set on the UI server" },
      { status: 503 },
    );
  }

  const appId = req.nextUrl.searchParams.get("app");
  if (!appId) {
    return NextResponse.json({ error: "missing ?app=<appId>" }, { status: 400 });
  }

  try {
    const r = await fetch(RUNPOD_GQL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: QUERY }),
      cache: "no-store",
    });
    if (!r.ok) {
      const text = await r.text();
      return NextResponse.json(
        { error: `runpod ${r.status}: ${text || r.statusText}` },
        { status: 502 },
      );
    }
    const payload = (await r.json()) as {
      data?: { myself?: { pods?: GqlPod[] } };
      errors?: { message: string }[];
    };
    if (payload.errors?.length) {
      return NextResponse.json(
        { error: `runpod graphql: ${payload.errors[0].message}` },
        { status: 502 },
      );
    }
    const pods = payload.data?.myself?.pods ?? [];
    const namePrefix = `${NAME_PREFIX}-${appId}-`;
    const rows: WorkerRowResponse[] = pods
      .filter((p) => (p.name ?? "").startsWith(namePrefix))
      .map(toRow);
    return NextResponse.json({ workers: rows, prefix: namePrefix });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : String(e) },
      { status: 502 },
    );
  }
}

function toRow(pod: GqlPod): WorkerRowResponse {
  const idx = pod.name.indexOf("m-rp-");
  const machineId = idx >= 0 ? pod.name.slice(idx) : pod.name;

  // Datacenter ids are empty for community-cloud pods (RunPod hides them).
  // Fall back to country code from machine.location ("CA", "US", "DE"…).
  const dc = pod.machine?.dataCenterId ?? "";
  const country = (pod.machine?.location ?? "").trim();
  const region = dc || country || (pod.machine?.secureCloud === false ? "community" : "");
  const regionCode = (dc.split("-")[0] || country || region).slice(0, 6).toUpperCase();

  const gpuId = pod.machine?.gpuTypeId ?? "—";

  return {
    machine_id: machineId,
    pod_id: pod.id,
    status: mapStatus(pod.desiredStatus, undefined),
    raw_status: (pod.desiredStatus || "").toLowerCase(),
    region,
    region_code: regionCode,
    gpu: prettyGpu(gpuId),
    gpu_count: pod.gpuCount ?? 1,
    vcpus: pod.vcpuCount ?? 0,
    ram_gb: pod.memoryInGb ?? 0,
    disk_gb: pod.containerDiskInGb ?? 0,
    created_at: pod.createdAt ?? null,
    cost_per_hr: pod.costPerHr ?? null,
  };
}

function mapStatus(
  desired?: string,
  current?: string,
): WorkerRowResponse["status"] {
  const d = (desired || "").toUpperCase();
  const c = (current || "").toUpperCase();
  if (c === "RUNNING" || d === "RUNNING") return "running";
  if (c === "EXITED" || c === "TERMINATED" || d === "TERMINATED" || d === "EXITED")
    return "terminated";
  if (d === "STOPPED" || c === "STOPPING") return "terminating";
  if (c === "STARTING" || c === "CREATED" || d === "CREATED") return "initializing";
  return "unknown";
}

function prettyGpu(id: string): string {
  // "NVIDIA H100 80GB HBM3" → "H100 80GB"; lowercase passthroughs keep their form.
  if (!id || id === "—") return "—";
  const stripped = id.replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "");
  return stripped;
}
