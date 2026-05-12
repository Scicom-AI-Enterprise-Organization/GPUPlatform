"use client";

import { useState } from "react";
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
import { AvailabilityBadge } from "@/components/availability-badge";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { gateway } from "@/lib/gateway";
import type { ComputeTemplate } from "@/lib/types";
import { cn } from "@/lib/utils";

// Same option list as the benchmark form so users see one consistent picker.
const GPU_OPTIONS = [
  { id: "NVIDIA RTX A4000", label: "RTX A4000", hint: "16 GB · cheap baseline" },
  { id: "NVIDIA RTX A5000", label: "RTX A5000", hint: "24 GB" },
  { id: "NVIDIA RTX A6000", label: "RTX A6000", hint: "48 GB" },
  { id: "NVIDIA GeForce RTX 4090", label: "RTX 4090", hint: "24 GB · consumer" },
  { id: "NVIDIA L40", label: "L40", hint: "48 GB" },
  { id: "NVIDIA L40S", label: "L40S", hint: "48 GB · faster L40" },
  { id: "NVIDIA A100 80GB PCIe", label: "A100 80GB", hint: "datacenter" },
  { id: "NVIDIA H100 80GB HBM3", label: "H100 80GB", hint: "fastest" },
];

export function NewPodForm({ templates }: { templates: ComputeTemplate[] }) {
  const router = useRouter();
  const [name, setName] = useState("dev-pod");
  const [gpuType, setGpuType] = useState("NVIDIA RTX A4000");
  const [gpuCount, setGpuCount] = useState(1);
  const [diskGb, setDiskGb] = useState(40);
  const [volumeGb, setVolumeGb] = useState(0);
  const [templateId, setTemplateId] = useState(
    templates[0]?.id ?? "pytorch-2.4-cuda12.4",
  );
  const [secureCloud, setSecureCloud] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const availability = useGpuAvailability(
    gpuType,
    gpuCount,
    true,
    secureCloud ? "SECURE" : "COMMUNITY",
  );

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    if (!name.trim()) {
      setSubmitError("Name is required.");
      return;
    }
    setSubmitting(true);
    try {
      const pod = await gateway.createCompute({
        name: name.trim(),
        gpu_type: gpuType,
        gpu_count: gpuCount,
        container_disk_gb: diskGb,
        volume_gb: volumeGb,
        template_id: templateId,
        cloud_type: secureCloud ? "SECURE" : "COMMUNITY",
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
          <Label htmlFor="cmp-name">Name</Label>
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

      {/* Section: hardware */}
      <Section title="Hardware" description="GPU type, count, and storage.">
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label>GPU</Label>
            <Select value={gpuType} onValueChange={setGpuType}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {GPU_OPTIONS.map((g) => (
                  <SelectItem key={g.id} value={g.id}>
                    <span className="font-medium">{g.label}</span>
                    <span className="ml-2 text-xs text-muted-foreground">
                      {g.hint}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <AvailabilityBadge state={availability} count={gpuCount} className="mt-1" />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="cmp-count">GPU count</Label>
            <Input
              id="cmp-count"
              type="number"
              min={1}
              max={8}
              value={gpuCount}
              onChange={(e) => setGpuCount(Math.max(1, Math.min(8, Number(e.target.value) || 1)))}
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="cmp-disk">Container disk (GB)</Label>
            <Input
              id="cmp-disk"
              type="number"
              min={10}
              max={2000}
              value={diskGb}
              onChange={(e) => setDiskGb(Math.max(10, Math.min(2000, Number(e.target.value) || 10)))}
            />
            <p className="text-xs text-muted-foreground">
              Working space for the container. Resets on pod stop.
            </p>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="cmp-volume">Volume (GB)</Label>
            <Input
              id="cmp-volume"
              type="number"
              min={0}
              max={2000}
              value={volumeGb}
              onChange={(e) => setVolumeGb(Math.max(0, Math.min(2000, Number(e.target.value) || 0)))}
            />
            <p className="text-xs text-muted-foreground">
              0 = no persistent volume. Volume keeps data across stop/start.
            </p>
          </div>

          <div className="space-y-1.5 sm:col-span-2">
            <Label>Cloud tier</Label>
            <div className="flex gap-2">
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
          </div>
        </div>
      </Section>

      {/* Section: template */}
      <Section
        title="Template"
        description="JupyterLab is always enabled — every template here ships sshd + jupyter."
      >
        <div className="grid gap-2 sm:grid-cols-2">
          {templates.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTemplateId(t.id)}
              className={cn(
                "rounded-lg border bg-card p-3 text-left transition-colors hover:border-foreground/40",
                templateId === t.id
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
      </Section>

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
        "flex-1 rounded-md border bg-card px-3 py-2 text-left transition-colors hover:border-foreground/40",
        active ? "border-foreground/60 ring-1 ring-foreground/20" : "border-border",
      )}
    >
      <div className="text-sm font-medium">{label}</div>
      <div className="text-[11px] text-muted-foreground">{hint}</div>
    </button>
  );
}
