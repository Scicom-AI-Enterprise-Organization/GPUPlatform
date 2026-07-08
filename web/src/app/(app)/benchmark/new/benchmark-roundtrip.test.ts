// Round-trip tests for the Benchmark "New" form's Form<->YAML sync (DE-53).
// Two regressions are pinned here:
//   1. Editing volume_size in YAML must survive a YAML -> Form -> YAML round-trip
//      (it used to reset to the form default because parseYamlToForm never read
//      runpod.storage.volume_size back).
//   2. A storage named on a benchmark item (`benchmark[].storage`) must surface
//      so the Form's Storage dropdown can be selected without a manual pick.
import { describe, expect, it } from "vitest";
import { renderYaml, parseYamlToForm, DEFAULTS } from "./benchmark-form";

describe("benchmark Form<->YAML round-trip", () => {
  it("keeps an edited volume_size across YAML -> Form -> YAML (no revert to default)", () => {
    expect(DEFAULTS.volume_size).not.toBe(800); // guard: 800 must be a real edit

    // Start from a rendered config, then simulate the user editing the volume.
    const edited = renderYaml(DEFAULTS).replace(
      /volume_size: \d+/,
      "volume_size: 800",
    );
    expect(edited).toContain("volume_size: 800");

    // YAML -> Form: the parsed state must carry the edited volume, not the default.
    const parsed = parseYamlToForm(edited, DEFAULTS);
    expect(parsed.parseError).toBeNull();
    expect(parsed.state.volume_size).toBe(800);

    // Form -> YAML: re-rendering from that state must still say 800, not the
    // default. (Boundary-matched so "800" isn't read as the default "80".)
    const reRendered = renderYaml(parsed.state);
    expect(reRendered).toMatch(/volume_size: 800\b/);
    expect(reRendered).not.toMatch(/volume_size: 80(\D|$)/);
  });

  it("surfaces a storage named on the benchmark item so the form can select it", () => {
    // Hand-authored: storage key inside the benchmark item.
    const withStorage = renderYaml(DEFAULTS).replace(
      /(  - name: .*\n)/,
      `$1    storage: "prod-s3-logs"\n`,
    );
    const parsed = parseYamlToForm(withStorage, DEFAULTS);
    expect(parsed.parseError).toBeNull();
    expect(parsed.storageRef).toBe("prod-s3-logs");
  });

  it("renders the selected storage name inside the benchmark item (Form -> YAML)", () => {
    const out = renderYaml(DEFAULTS, "cloud", "prod-s3-logs");
    // Inside the benchmark item (4-space indent), not a top-level key.
    expect(out).toMatch(/^ {4}storage: "prod-s3-logs"$/m);
    expect(out).not.toMatch(/^storage:/m);
    // round-trips back out
    expect(parseYamlToForm(out, DEFAULTS).storageRef).toBe("prod-s3-logs");
  });

  it("omits the benchmark-item storage key when none is selected", () => {
    const out = renderYaml(DEFAULTS);
    expect(out).not.toMatch(/^ {4}storage:/m); // no bench-item storage line
    expect(parseYamlToForm(out, DEFAULTS).storageRef).toBeNull();
  });
});

// VM (bare-metal) Form<->YAML sync: provider, working dir, clean-up flag, GPU
// pin and runtime env must all round-trip through the `remote:` block so the
// form fills from YAML and flipping Form<->YAML<->Form doesn't drift/reset.
describe("benchmark VM Form<->YAML round-trip (DE-53)", () => {
  const vmExtras = {
    providerName: "TM-H20",
    cleanupModel: true,
    visibleDevices: "0,1",
    envText: "export HF_HOME=/share/huggingface\nexport TRITON_CACHE_DIR=/share/triton",
  };

  it("emits provider / workdir / cleanup_model / env in the VM remote block", () => {
    const out = renderYaml(DEFAULTS, "vm", "s3", vmExtras);
    expect(out).toMatch(/^ {2}provider: "TM-H20"$/m);
    expect(out).toMatch(/^ {2}workdir: "~"$/m);
    expect(out).toMatch(/^ {2}cleanup_model: true$/m);
    expect(out).toMatch(/^ {4}CUDA_VISIBLE_DEVICES: "0,1"$/m);
    expect(out).toMatch(/^ {4}HF_HOME: "\/share\/huggingface"$/m);
    expect(out).not.toMatch(/^runpod:/m); // VM template carries no runpod block
  });

  it("parses every VM remote.* section back out (YAML -> Form)", () => {
    const parsed = parseYamlToForm(renderYaml(DEFAULTS, "vm", "s3", vmExtras), DEFAULTS);
    expect(parsed.parseError).toBeNull();
    expect(parsed.providerRef).toBe("TM-H20");
    expect(parsed.cleanupModel).toBe(true);
    expect(parsed.visibleDevices).toBe("0,1");
    expect(parsed.state.vm_base_dir).toBe("~");
    expect(parsed.envText).toContain("HF_HOME=/share/huggingface");
    expect(parsed.envText).toContain("TRITON_CACHE_DIR=/share/triton");
    expect(parsed.envText).not.toContain("CUDA_VISIBLE_DEVICES"); // its own field
    expect(parsed.storageRef).toBe("s3");
  });

  it("is byte-stable across VM render -> parse -> render (no drift / no refresh)", () => {
    const first = renderYaml(DEFAULTS, "vm", "s3", vmExtras);
    const parsed = parseYamlToForm(first, DEFAULTS);
    const second = renderYaml(parsed.state, "vm", "s3", {
      providerName: parsed.providerRef ?? undefined,
      cleanupModel: parsed.cleanupModel ?? undefined,
      visibleDevices: parsed.visibleDevices ?? undefined,
      envText: parsed.envText ?? undefined,
    });
    expect(second).toBe(first);
  });

  it("round-trips a non-default working directory via remote.workdir", () => {
    const out = renderYaml({ ...DEFAULTS, vm_base_dir: "/mnt/scratch" }, "vm", undefined, {
      cleanupModel: true,
    });
    expect(out).toMatch(/^ {2}workdir: "\/mnt\/scratch"$/m);
    expect(parseYamlToForm(out, DEFAULTS).state.vm_base_dir).toBe("/mnt/scratch");
  });

  it("omits the env block when no GPU pin / env vars are set", () => {
    const out = renderYaml(DEFAULTS, "vm", undefined, { cleanupModel: true });
    expect(out).not.toMatch(/^ {2}env:$/m);
    const parsed = parseYamlToForm(out, DEFAULTS);
    expect(parsed.visibleDevices).toBeNull();
    expect(parsed.envText).toBeNull();
  });

  it("leaves VM-only fields null on the cloud template (no cross-contamination)", () => {
    const parsed = parseYamlToForm(renderYaml(DEFAULTS, "cloud", "s3"), DEFAULTS);
    expect(parsed.providerRef).toBeNull();
    expect(parsed.cleanupModel).toBeNull();
    expect(parsed.visibleDevices).toBeNull();
    expect(parsed.envText).toBeNull();
  });

  it("emits + round-trips the storage backend on the VM template", () => {
    const out = renderYaml(DEFAULTS, "vm", "s3", { cleanupModel: true });
    // storage sits on the benchmark item (4-space indent), same as cloud.
    expect(out).toMatch(/^ {4}storage: "s3"$/m);
    const parsed = parseYamlToForm(out, DEFAULTS);
    expect(parsed.parseError).toBeNull();
    expect(parsed.storageRef).toBe("s3"); // resolvable to the Storage dropdown
    // render -> parse -> render keeps the storage line stable.
    expect(renderYaml(parsed.state, "vm", "s3", { cleanupModel: true })).toBe(out);
  });

  it("ships a fillable provider placeholder + guidance comment when none is picked", () => {
    const out = renderYaml(DEFAULTS, "vm", "s3");
    expect(out).toMatch(/^ {2}provider: ""  # STATE THE NAME OF GPU PROVIDER$/m);
    // The empty placeholder must parse back to "no provider" (not the literal).
    const parsed = parseYamlToForm(out, DEFAULTS);
    expect(parsed.parseError).toBeNull();
    expect(parsed.providerRef).toBeNull();
    // And the comment survives a render -> parse -> render (no drift).
    expect(renderYaml(parsed.state, "vm", "s3")).toBe(out);
  });
});

// Ingress GPU identity (DE-59): the author-stated GPU behind an external
// endpoint must be emitted into the YAML template and round-trip, so the run's
// Parameters -> Hardware shows it without a manual post-run edit.
describe("benchmark ingress GPU Form<->YAML round-trip (DE-59)", () => {
  const ingress = {
    ...DEFAULTS,
    ingress_base_url: "https://tm-h20-llm-1.aies.scicom.dev",
    ingress_gpu_type: "NVIDIA H20",
    ingress_gpu_count: 4,
  };

  it("emits gpu_type / gpu_count on the ingress benchmark item", () => {
    const out = renderYaml(ingress, "ingress", "s3");
    expect(out).toMatch(/^ {4}gpu_type: "NVIDIA H20"/m);
    expect(out).toMatch(/^ {4}gpu_count: 4$/m);
    expect(out).not.toMatch(/^runpod:/m); // ingress spawns nothing
  });

  it("parses gpu_type / gpu_count back out (YAML -> Form)", () => {
    const parsed = parseYamlToForm(renderYaml(ingress, "ingress", "s3"), DEFAULTS);
    expect(parsed.parseError).toBeNull();
    expect(parsed.state.ingress_gpu_type).toBe("NVIDIA H20");
    expect(parsed.state.ingress_gpu_count).toBe(4);
  });

  it("is byte-stable across render -> parse -> render (no drift)", () => {
    const first = renderYaml(ingress, "ingress", "s3");
    const parsed = parseYamlToForm(first, DEFAULTS);
    expect(renderYaml(parsed.state, "ingress", "s3")).toBe(first);
  });

  it("keeps the empty-GPU template stable (blank stays blank)", () => {
    const out = renderYaml(DEFAULTS, "ingress", "s3");
    expect(out).toMatch(/^ {4}gpu_type: ""/m); // template always shows the key
    const parsed = parseYamlToForm(out, DEFAULTS);
    expect(parsed.state.ingress_gpu_type).toBe(""); // empty parses to no GPU
    expect(renderYaml(parsed.state, "ingress", "s3")).toBe(out);
  });
});

// TP/DP are now surfaced in the RunPod/VM serve template (default 1) and, for
// ingress runs, recorded as a descriptive serve block (nothing is launched, so
// they only annotate how the external endpoint is served).
describe("benchmark TP/DP template + ingress round-trip", () => {
  it("shows tensor_parallel_size / data_parallel_size in the cloud template (default 1)", () => {
    const out = renderYaml(DEFAULTS, "cloud", "s3");
    expect(out).toMatch(/^ {6}tensor_parallel_size: 1$/m);
    expect(out).toMatch(/^ {6}data_parallel_size: 1$/m);
  });

  it("shows tensor_parallel_size / data_parallel_size in the VM template too", () => {
    const out = renderYaml(DEFAULTS, "vm", "s3", { cleanupModel: true });
    expect(out).toMatch(/^ {6}tensor_parallel_size: 1$/m);
    expect(out).toMatch(/^ {6}data_parallel_size: 1$/m);
  });

  it("round-trips an edited TP/DP through cloud YAML -> Form -> YAML", () => {
    const edited = renderYaml(
      { ...DEFAULTS, tensor_parallel_size: "4", data_parallel_size: "2" },
      "cloud",
      "s3",
    );
    expect(edited).toMatch(/^ {6}tensor_parallel_size: 4$/m);
    expect(edited).toMatch(/^ {6}data_parallel_size: 2$/m);
    const parsed = parseYamlToForm(edited, DEFAULTS);
    expect(parsed.parseError).toBeNull();
    expect(parsed.state.tensor_parallel_size).toBe("4");
    expect(parsed.state.data_parallel_size).toBe("2");
  });

  it("records TP/DP on the ingress item as a descriptive serve block (no pod)", () => {
    const out = renderYaml(
      { ...DEFAULTS, ingress_base_url: "https://x", tensor_parallel_size: "8", data_parallel_size: "1" },
      "ingress",
      "s3",
    );
    expect(out).toMatch(/^ {4}serve:$/m);
    expect(out).toMatch(/^ {6}tensor_parallel_size: 8$/m);
    expect(out).toMatch(/^ {6}data_parallel_size: 1$/m);
    expect(out).not.toMatch(/^runpod:/m); // ingress still spawns nothing
  });

  it("parses ingress TP/DP back out and is byte-stable", () => {
    const first = renderYaml(
      { ...DEFAULTS, ingress_base_url: "https://x", tensor_parallel_size: "8", data_parallel_size: "2" },
      "ingress",
      "s3",
    );
    const parsed = parseYamlToForm(first, DEFAULTS);
    expect(parsed.parseError).toBeNull();
    expect(parsed.state.tensor_parallel_size).toBe("8");
    expect(parsed.state.data_parallel_size).toBe("2");
    expect(renderYaml(parsed.state, "ingress", "s3")).toBe(first);
  });
});
