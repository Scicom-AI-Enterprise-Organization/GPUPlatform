import { describe, expect, it } from "vitest";
import { captureStateFromParam } from "./capture-dialog";

// The capture dialog's open/tab state lives in `?capture=` so it's shareable and
// survives a refresh. Radix renders the dialog through a portal, so this mapping
// is the only part that can be asserted without a browser — hence the unit test.
describe("captureStateFromParam", () => {
  it("opens on a known source and selects that tab", () => {
    expect(captureStateFromParam("platform")).toEqual({ open: true, source: "platform" });
    expect(captureStateFromParam("langfuse")).toEqual({ open: true, source: "langfuse" });
    expect(captureStateFromParam("synthetic")).toEqual({ open: true, source: "synthetic" });
  });

  it("stays closed with no param", () => {
    expect(captureStateFromParam(null)).toEqual({ open: false, source: "platform" });
    expect(captureStateFromParam("")).toEqual({ open: false, source: "platform" });
  });

  it("stays closed on an unknown value rather than guessing a tab", () => {
    // A stale or hand-edited link must not open the dialog on an arbitrary tab.
    expect(captureStateFromParam("bogus")).toEqual({ open: false, source: "platform" });
    expect(captureStateFromParam("Platform")).toEqual({ open: false, source: "platform" });
  });
});
