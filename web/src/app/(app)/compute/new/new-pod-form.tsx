"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SearchableSelect } from "@/components/ui/searchable-select";
import { AvailabilityBadge } from "@/components/availability-badge";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { gateway } from "@/lib/gateway";
import type {
  ComputeTemplate,
  GpuTypeOption,
  PiImageOption,
  ProviderRecord,
  RunpodTemplateSearchResult,
} from "@/lib/types";
import { cn } from "@/lib/utils";

// Same option list as the benchmark form so users see one consistent picker.
// Fallback list rendered if /compute/pi/images is unreachable (e.g. gateway
// pre-dates the route). Keeps the picker usable instead of stuck on "Loading…".
const PI_IMAGES_FALLBACK: PiImageOption[] = [
  {
    id: "cuda_12_6_pytorch_2_7",
    name: "PyTorch 2.7 + CUDA 12.6",
    description: "Newest CUDA/PyTorch combo PI offers. Broadest sub-provider support.",
  },
  {
    id: "cuda_12_4_pytorch_2_6",
    name: "PyTorch 2.6 + CUDA 12.4",
    description: "Slightly older — pick if you need PyTorch ≤ 2.6.",
  },
  {
    id: "cuda_12_4_pytorch_2_5",
    name: "PyTorch 2.5 + CUDA 12.4",
    description: "Same CUDA as our RunPod default. Good fallback when 12.6 is short of stock.",
  },
  {
    id: "ubuntu_22_cuda_12",
    name: "Ubuntu 22.04 + CUDA 12",
    description: "Minimal Ubuntu + CUDA 12. Bring your own framework.",
  },
  {
    id: "vllm_llama_70b",
    name: "vLLM + Llama-3 70B",
    description: "vLLM pre-loaded with a Llama-3 70B endpoint.",
  },
  {
    id: "stable_diffusion",
    name: "Stable Diffusion",
    description: "Stable Diffusion pre-configured.",
  },
];

// Provider-native GPU catalogs (matches gateway constants). Used as a fallback
// when /compute/{kind}/gpu-types is unreachable (older gateway). Otherwise the
// form pulls the live list so newly added SKUs show up without a frontend deploy.
const RUNPOD_GPU_FALLBACK: GpuTypeOption[] = [
  { id: "NVIDIA RTX A4000", label: "RTX A4000", vram_gb: 16, hint: "cheap baseline" },
  { id: "NVIDIA RTX A5000", label: "RTX A5000", vram_gb: 24, hint: "" },
  { id: "NVIDIA RTX A6000", label: "RTX A6000", vram_gb: 48, hint: "" },
  { id: "NVIDIA GeForce RTX 4090", label: "RTX 4090", vram_gb: 24, hint: "consumer" },
  { id: "NVIDIA L40", label: "L40", vram_gb: 48, hint: "" },
  { id: "NVIDIA L40S", label: "L40S", vram_gb: 48, hint: "faster L40" },
  { id: "NVIDIA A100 80GB PCIe", label: "A100 80GB PCIe", vram_gb: 80, hint: "datacenter" },
  { id: "NVIDIA H100 80GB HBM3", label: "H100 80GB SXM", vram_gb: 80, hint: "fastest H100" },
];
const PI_GPU_FALLBACK: GpuTypeOption[] = [
  { id: "RTX3090_24GB", label: "RTX 3090", vram_gb: 24, hint: "consumer" },
  { id: "RTX4090_24GB", label: "RTX 4090", vram_gb: 24, hint: "consumer" },
  { id: "RTX5090_32GB", label: "RTX 5090", vram_gb: 32, hint: "consumer · Blackwell" },
  { id: "A4000_16GB", label: "RTX A4000", vram_gb: 16, hint: "cheap baseline" },
  { id: "A5000_24GB", label: "RTX A5000", vram_gb: 24, hint: "" },
  { id: "A6000_48GB", label: "RTX A6000", vram_gb: 48, hint: "" },
  { id: "A10_24GB", label: "A10", vram_gb: 24, hint: "" },
  { id: "L4_24GB", label: "L4", vram_gb: 24, hint: "" },
  { id: "L40_48GB", label: "L40", vram_gb: 48, hint: "" },
  { id: "L40S_48GB", label: "L40S", vram_gb: 48, hint: "faster L40" },
  { id: "A100_40GB", label: "A100 40GB", vram_gb: 40, hint: "" },
  { id: "A100_80GB", label: "A100 80GB", vram_gb: 80, hint: "datacenter" },
  { id: "H100_80GB", label: "H100 80GB", vram_gb: 80, hint: "fastest H100" },
  { id: "H200_141GB", label: "H200", vram_gb: 141, hint: "newest" },
  { id: "B200_180GB", label: "B200", vram_gb: 180, hint: "Blackwell datacenter" },
  { id: "MI300X_192GB", label: "MI300X", vram_gb: 192, hint: "AMD" },
];

// Same count picker as the serverless Pod card.
const GPU_COUNT_CHOICES = [1, 2, 4, 8] as const;

// Inline GPU hint — total VRAM across the requested count.
function capacityHint(vramPerGpu: number, count: number): string {
  const total = vramPerGpu * count;
  const totalStr = total >= 100 ? `${Math.round(total)} GB` : `${total} GB`;
  return count === 1 ? `${totalStr} VRAM` : `${totalStr} VRAM · ${count} × ${vramPerGpu} GB`;
}

export function NewPodForm({ templates }: { templates: ComputeTemplate[] }) {
  const router = useRouter();
  const [name, setName] = useState("dev-pod");
  const [gpuType, setGpuType] = useState<string>(RUNPOD_GPU_FALLBACK[0].id);
  const [runpodGpus, setRunpodGpus] = useState<GpuTypeOption[]>(RUNPOD_GPU_FALLBACK);
  const [piGpus, setPiGpus] = useState<GpuTypeOption[]>(PI_GPU_FALLBACK);
  const [gpuCount, setGpuCount] = useState(1);
  const [containerDisk, setContainerDisk] = useState("40");
  const [volumeGb, setVolumeGb] = useState("0");
  const [templateId, setTemplateId] = useState(
    templates[0]?.id ?? "pytorch-2.4-cuda12.4",
  );
  // For non-curated RunPod templates picked via search we also carry the
  // resolved imageName so the gateway doesn't have to round-trip back to
  // RunPod's templates API at create time.
  const [imageOverride, setImageOverride] = useState<string | null>(null);
  const [secureCloud, setSecureCloud] = useState(false);
  const [providerId, setProviderId] = useState<string>("");
  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  const [piImages, setPiImages] = useState<PiImageOption[]>([]);
  const [piImagesError, setPiImagesError] = useState<string | null>(null);
  const [piImagesFiltered, setPiImagesFiltered] = useState<boolean>(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    gateway.listProviders().then(setProviders).catch(() => {});
    gateway
      .listPiImages()
      .then((rows) => {
        setPiImages(rows.length > 0 ? rows : PI_IMAGES_FALLBACK);
      })
      .catch((e) => {
        setPiImages(PI_IMAGES_FALLBACK);
        setPiImagesError(e instanceof Error ? e.message : String(e));
      });
    gateway
      .listRunpodGpuTypes()
      .then((rows) => {
        if (rows.length > 0) setRunpodGpus(rows);
      })
      .catch(() => {});
    gateway
      .listPiGpuTypes()
      .then((rows) => {
        if (rows.length > 0) setPiGpus(rows);
      })
      .catch(() => {});
  }, []);

  const selectedProvider = useMemo(
    () => providers.find((p) => p.id === providerId) ?? null,
    [providers, providerId],
  );
  const providerKind: "runpod" | "pi" = selectedProvider?.kind === "pi" ? "pi" : "runpod";
  const gpuCatalog: GpuTypeOption[] = providerKind === "pi" ? piGpus : runpodGpus;

  // Snap gpuType to the active catalog when provider kind changes (or when
  // catalogs first load) so we never send a RunPod-form name to PI's API.
  useEffect(() => {
    if (gpuCatalog.length === 0) return;
    if (!gpuCatalog.some((g) => g.id === gpuType)) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setGpuType(gpuCatalog[0].id);
    }
  }, [providerKind, gpuCatalog, gpuType]);

  // For PI: narrow the image list to ones at least one in-stock sub-provider
  // supports for the current (gpu, count, tier). Re-runs when those change.
  useEffect(() => {
    if (providerKind !== "pi") {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setPiImagesFiltered(false);
      return;
    }
    let cancelled = false;
    const tid = setTimeout(() => {
      gateway
        .listPiCompatibleImages({
          gpu: gpuType,
          count: gpuCount,
          cloud_type: secureCloud ? "SECURE" : "COMMUNITY",
          provider_id: providerId || null,
        })
        .then((rows) => {
          if (cancelled) return;
          if (rows.length === 0) {
            setPiImagesFiltered(false);
            return;
          }
          setPiImages(rows);
          setPiImagesFiltered(true);
        })
        .catch(() => {
          if (!cancelled) setPiImagesFiltered(false);
        });
    }, 300);
    return () => {
      cancelled = true;
      clearTimeout(tid);
    };
  }, [providerKind, gpuType, gpuCount, secureCloud, providerId]);

  // When switching provider kind, snap template_id back to the kind's default
  // so we don't ship a RunPod image id to PI (or vice versa).
  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect */
    if (providerKind === "pi") {
      setTemplateId((cur) =>
        piImages.some((i) => i.id === cur)
          ? cur
          : piImages[0]?.id ?? "cuda_12_6_pytorch_2_7",
      );
      setImageOverride(null);
    } else {
      setTemplateId((cur) =>
        templates.some((t) => t.id === cur) ? cur : templates[0]?.id ?? "pytorch-2.4-cuda12.4",
      );
      setImageOverride(null);
    }
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [providerKind, piImages, templates]);

  const availability = useGpuAvailability(
    gpuType,
    gpuCount,
    true,
    secureCloud ? "SECURE" : "COMMUNITY",
    { kind: providerKind, id: providerId || null },
  );

  const parsedDisk = Number.parseInt(containerDisk, 10);
  const diskInvalid = !Number.isFinite(parsedDisk) || parsedDisk < 10 || parsedDisk > 2000;
  const parsedVolume = Number.parseInt(volumeGb, 10);
  const volumeInvalid = !Number.isFinite(parsedVolume) || parsedVolume < 0 || parsedVolume > 2000;

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    if (!name.trim()) {
      setSubmitError("Name is required.");
      return;
    }
    if (diskInvalid) {
      setSubmitError("Container disk must be between 10 and 2000 GB.");
      return;
    }
    if (volumeInvalid) {
      setSubmitError("Volume must be between 0 and 2000 GB.");
      return;
    }
    setSubmitting(true);
    try {
      const pod = await gateway.createCompute({
        name: name.trim(),
        gpu_type: gpuType,
        gpu_count: gpuCount,
        container_disk_gb: parsedDisk,
        volume_gb: parsedVolume,
        template_id: templateId,
        image: imageOverride,
        cloud_type: secureCloud ? "SECURE" : "COMMUNITY",
        provider_id: providerId || null,
      });
      toast.success(
        pod.status === "pending_approval"
          ? "Request submitted — an admin will review and approve."
          : "Pod creating — provisioning takes a few minutes",
        { duration: 4000 },
      );
      router.push(`/compute/${pod.id}`);
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="space-y-6">
      {/* Section: identity */}
      <Section title="Pod" description="A short name to remember this pod by.">
        <div className="space-y-1.5">
          <Label htmlFor="cmp-name" className="text-xs uppercase tracking-wide text-muted-foreground">Name</Label>
          <Input
            id="cmp-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="dev-pod"
            maxLength={128}
            required
          />
        </div>
      </Section>

      {/* Section: cloud account */}
      <Section
        title="Cloud account"
        description="Which API key to bill against. Default = gateway RunPod env key."
      >
        <div className="space-y-1.5">
          <Label className="text-xs uppercase tracking-wide text-muted-foreground">API key</Label>
          <Select
            value={providerId || "__default__"}
            onValueChange={(v) => setProviderId(v === "__default__" ? "" : v)}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__default__">Gateway default (RunPod)</SelectItem>
              {providers
                .filter((p) => p.kind === "runpod" || p.kind === "pi")
                .map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.name}
                    {" · "}
                    {p.kind === "pi" ? "Prime Intellect" : "RunPod"}
                    {p.api_key_last4 ? ` · ****${p.api_key_last4}` : ""}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
          {providers.filter((p) => p.kind === "runpod" || p.kind === "pi").length === 0 && (
            <p className="text-xs text-muted-foreground">
              None registered. <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">Add a cloud account →</a>
            </p>
          )}
        </div>
      </Section>

      {/* Section: hardware */}
      <Section title="Hardware" description="GPU type, count, and storage.">
        <div className="space-y-5">
          <Field
            label="GPU"
            hint={(() => {
              const g = gpuCatalog.find((c) => c.id === gpuType);
              return g ? capacityHint(g.vram_gb, gpuCount) : undefined;
            })()}
            extra={<AvailabilityBadge state={availability} count={gpuCount} />}
          >
            <div className="flex gap-2">
              <SearchableSelect
                className="flex-1"
                value={gpuType}
                onChange={setGpuType}
                options={gpuCatalog.map((g) => ({
                  value: g.id,
                  label: g.label,
                  hint: `${g.vram_gb} GB${g.hint ? ` · ${g.hint}` : ""}`,
                }))}
                placeholder="Choose a GPU"
                searchPlaceholder="Search GPUs (e.g. h100, 24gb, ada)…"
              />
              <Select
                value={String(gpuCount)}
                onValueChange={(v) => setGpuCount(Number.parseInt(v, 10))}
              >
                <SelectTrigger className="w-24 shrink-0">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {GPU_COUNT_CHOICES.map((n) => (
                    <SelectItem key={n} value={String(n)}>
                      ×{n}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </Field>

          <Field
            label="Cloud tier"
            hint="Community is cheaper with variable hosts; Secure uses vetted hosts with more capacity."
          >
            <div className="grid grid-cols-2 gap-2">
              <TierButton
                active={!secureCloud}
                onClick={() => setSecureCloud(false)}
                label="Community"
                hint="cheaper, variable hosts"
              />
              <TierButton
                active={secureCloud}
                onClick={() => setSecureCloud(true)}
                label="Secure"
                hint="vetted hosts, more capacity"
              />
            </div>
          </Field>

          <div className="grid grid-cols-2 gap-3">
            <Field
              label="Container disk (GB)"
              hint="Working space for the container. Resets on pod stop."
            >
              <Input
                type="text"
                inputMode="numeric"
                value={containerDisk}
                onChange={(e) => setContainerDisk(e.target.value)}
                placeholder="40"
                aria-invalid={diskInvalid}
              />
            </Field>
            <Field
              label="Volume (GB)"
              hint="0 = no persistent volume. Volume keeps data across stop/start."
            >
              <Input
                type="text"
                inputMode="numeric"
                value={volumeGb}
                onChange={(e) => setVolumeGb(e.target.value)}
                placeholder="0"
                aria-invalid={volumeInvalid}
              />
            </Field>
          </div>
        </div>
      </Section>

      {/* Section: template / image */}
      {providerKind === "pi" ? (
        <Section
          title="Image"
          description="Prime Intellect provides a fixed set of pre-baked images."
        >
          <PiImagePicker
            images={piImages}
            filtered={piImagesFiltered}
            error={piImagesError}
            value={templateId}
            onChange={(id) => setTemplateId(id)}
          />
        </Section>
      ) : (
        <Section
          title="Template"
          description="JupyterLab is always enabled — every template here ships sshd + jupyter."
        >
          <RunpodTemplatePicker
            curated={templates}
            providerId={providerId || null}
            value={templateId}
            imageOverride={imageOverride}
            onChange={(id, image) => {
              setTemplateId(id);
              setImageOverride(image);
            }}
          />
        </Section>
      )}

      <div className="flex items-center justify-end gap-3 border-t border-border pt-4">
        {submitError && (
          <p className="flex-1 text-sm text-destructive">{submitError}</p>
        )}
        <Button
          type="button"
          variant="ghost"
          onClick={() => router.push("/compute")}
          disabled={submitting}
        >
          Cancel
        </Button>
        <Button type="submit" disabled={submitting}>
          {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
          {submitting ? "Creating…" : "Create pod"}
        </Button>
      </div>
    </form>
  );
}

function Section({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-border bg-card p-5">
      <div className="mb-4">
        <h2 className="text-sm font-semibold">{title}</h2>
        {description && (
          <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
        )}
      </div>
      {children}
    </section>
  );
}

function TierButton({
  active,
  onClick,
  label,
  hint,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  hint: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-md border bg-card p-3 text-left transition-colors",
        active ? "border-foreground/60 ring-1 ring-foreground/20" : "border-border hover:border-foreground/40",
      )}
    >
      <div className="text-sm font-medium">{label}</div>
      <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div>
    </button>
  );
}

// Label + optional inline `extra` (e.g. availability badge) + hint below —
// mirrors the serverless Pod card's field layout.
function Field({
  label,
  hint,
  children,
  extra,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
  extra?: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <Label className="text-xs uppercase tracking-wide text-muted-foreground">{label}</Label>
        {extra}
      </div>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

function PiImagePicker({
  images,
  filtered,
  error,
  value,
  onChange,
}: {
  images: PiImageOption[];
  filtered: boolean;
  error: string | null;
  value: string;
  onChange: (id: string) => void;
}) {
  // If the current `value` isn't in the (possibly newly-filtered) list, snap
  // to the first available so the dropdown can't show a label-less selection.
  useEffect(() => {
    if (images.length === 0) return;
    if (!images.some((i) => i.id === value)) {
      onChange(images[0].id);
    }
  }, [images, value, onChange]);

  if (images.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">Loading Prime Intellect images…</p>
    );
  }
  return (
    <div className="space-y-1.5">
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {images.map((i) => (
            <SelectItem key={i.id} value={i.id}>
              <span className="font-medium">{i.name}</span>
              <span className="ml-2 text-xs text-muted-foreground">{i.id}</span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        {images.find((i) => i.id === value)?.description ?? ""}
      </p>
      {filtered && (
        <p className="text-[11px] text-muted-foreground">
          Only images currently in stock on Prime Intellect for this GPU + tier are shown.
        </p>
      )}
      {error && (
        <p className="text-[11px] text-muted-foreground">
          Using built-in fallback list (couldn&apos;t reach gateway: {error}). Restart the gateway to load the live list.
        </p>
      )}
    </div>
  );
}

function RunpodTemplatePicker({
  curated,
  providerId,
  value,
  imageOverride,
  onChange,
}: {
  curated: ComputeTemplate[];
  providerId: string | null;
  value: string;
  imageOverride: string | null;
  onChange: (templateId: string, image: string | null) => void;
}) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<RunpodTemplateSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isCuratedPick = curated.some((t) => t.id === value);
  const selectedCustom = !isCuratedPick && imageOverride
    ? { id: value, name: value, image: imageOverride }
    : null;

  useEffect(() => {
    if (!searchOpen) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setSearching(true);
      setSearchError(null);
      try {
        const rows = await gateway.searchRunpodTemplates({
          q: query.trim(),
          limit: 50,
          provider_id: providerId,
        });
        setResults(rows);
      } catch (e) {
        setSearchError(e instanceof Error ? e.message : String(e));
        setResults([]);
      } finally {
        setSearching(false);
      }
    }, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query, searchOpen, providerId]);

  return (
    <div className="space-y-3">
      <div className="grid gap-2 sm:grid-cols-2">
        {curated.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => onChange(t.id, null)}
            className={cn(
              "rounded-lg border bg-card p-3 text-left transition-colors hover:border-foreground/40",
              value === t.id
                ? "border-foreground/60 ring-1 ring-foreground/20"
                : "border-border",
            )}
          >
            <div className="text-sm font-medium">{t.name}</div>
            <div className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
              {t.image}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{t.description}</p>
          </button>
        ))}
      </div>

      {selectedCustom && (
        <div className="rounded-lg border border-foreground/60 bg-card p-3 ring-1 ring-foreground/20">
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            Custom RunPod template
          </div>
          <div className="mt-1 text-sm font-medium">{selectedCustom.name}</div>
          <div className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
            {selectedCustom.image}
          </div>
        </div>
      )}

      <div>
        <button
          type="button"
          onClick={() => setSearchOpen((v) => !v)}
          className="text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
        >
          {searchOpen ? "Hide RunPod template search" : "Search all RunPod templates →"}
        </button>
      </div>

      {searchOpen && (
        <div className="space-y-2 rounded-lg border border-border bg-card p-3">
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="search by name, image, category…"
          />
          {searchError && (
            <p className="text-xs text-destructive">{searchError}</p>
          )}
          {searching && (
            <p className="text-xs text-muted-foreground">Searching…</p>
          )}
          {!searching && results.length > 0 && (
            <div className="max-h-64 space-y-1 overflow-y-auto">
              {results.map((r) => (
                <button
                  key={r.id}
                  type="button"
                  onClick={() => onChange(r.id, r.image)}
                  className={cn(
                    "block w-full rounded-md border bg-card p-2 text-left transition-colors hover:border-foreground/40",
                    value === r.id
                      ? "border-foreground/60 ring-1 ring-foreground/20"
                      : "border-border",
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{r.name}</span>
                    {r.is_runpod && (
                      <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                        official
                      </span>
                    )}
                  </div>
                  <div className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
                    {r.image}
                  </div>
                </button>
              ))}
            </div>
          )}
          {!searching && results.length === 0 && query && !searchError && (
            <p className="text-xs text-muted-foreground">No matches.</p>
          )}
        </div>
      )}
    </div>
  );
}
