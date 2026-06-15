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
