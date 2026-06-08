// DOCX export for the Usage Report. Ported from SlurmUI's reports exporter:
// `docx` is dynamically imported (lazy bundle), and charts are rasterised to PNG
// via a hand-written Canvas 2D renderer (no chart lib) and embedded as ImageRun.

import type { UsageReport, UsageSpend } from "@/lib/types";

const MODEL_COLORS = ["#2563eb", "#f97316", "#16a34a", "#9333ea", "#db2777", "#0891b2", "#ca8a04", "#dc2626", "#0d9488", "#7c3aed"];

const fmtInt = (n: number) => n.toLocaleString();
const fmtTok = (n: number) => (n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));
const fmtLat = (s: number | null) => (s == null ? "—" : s < 1 ? `${Math.round(s * 1000)}ms` : s < 60 ? `${s.toFixed(1)}s` : `${(s / 60).toFixed(1)}m`);

// Render a multi-line chart to a PNG (Uint8Array) using Canvas 2D.
async function chartPng(
  rows: Record<string, number | string>[],
  ids: string[],
): Promise<Uint8Array | null> {
  if (!rows.length || !ids.length) return null;
  const W = 720, H = 220;
  const PAD = { top: 16, right: 16, bottom: 40, left: 52 };
  const cW = W - PAD.left - PAD.right;
  const cH = H - PAD.top - PAD.bottom;

  const canvas = document.createElement("canvas");
  canvas.width = W * 2; canvas.height = H * 2;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.scale(2, 2);
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, W, H);

  const vals: number[] = [];
  for (const id of ids) for (const r of rows) if (typeof r[id] === "number") vals.push(r[id] as number);
  if (!vals.length) return null;
  const yMax = Math.max(...vals) * 1.1 || 1;
  const n = rows.length;
  const px = (i: number) => PAD.left + (i / Math.max(n - 1, 1)) * cW;
  const py = (v: number) => PAD.top + cH - (v / yMax) * cH;

  ctx.font = "9px Arial";
  for (let g = 0; g <= 4; g++) {
    const v = yMax - (g / 4) * yMax;
    const y = PAD.top + (g / 4) * cH;
    ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + cW, y); ctx.stroke();
    ctx.fillStyle = "#9ca3af"; ctx.textAlign = "right";
    ctx.fillText(`${v < 10 ? v.toFixed(1) : Math.round(v)}`, PAD.left - 4, y + 3);
  }

  const xStep = Math.max(1, Math.floor(n / 8));
  ctx.fillStyle = "#9ca3af"; ctx.font = "8px Arial"; ctx.textAlign = "center";
  for (let i = 0; i < n; i += xStep) ctx.fillText(String(rows[i].label ?? ""), px(i), PAD.top + cH + 12);

  ids.forEach((id, si) => {
    ctx.strokeStyle = MODEL_COLORS[si % MODEL_COLORS.length];
    ctx.lineWidth = 1.6; ctx.beginPath();
    let first = true;
    for (let i = 0; i < n; i++) {
      const v = rows[i][id];
      if (typeof v !== "number") continue;
      first ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v));
      first = false;
    }
    ctx.stroke();
  });

  ctx.font = "9px Arial"; ctx.textAlign = "left";
  let lx = PAD.left; const ly = H - 8;
  for (let si = 0; si < ids.length; si++) {
    if (lx + 100 > W) break;
    ctx.fillStyle = MODEL_COLORS[si % MODEL_COLORS.length];
    ctx.fillRect(lx, ly - 4, 14, 3);
    ctx.fillStyle = "#374151";
    ctx.fillText(ids[si], lx + 17, ly);
    lx += 17 + ctx.measureText(ids[si]).width + 12;
  }

  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      if (!blob) { resolve(null); return; }
      blob.arrayBuffer().then((buf) => resolve(new Uint8Array(buf)));
    }, "image/png");
  });
}

export async function exportUsageDocx(
  data: UsageReport,
  spend: UsageSpend | null,
  opts: { periodLabel: string; scopeLabel: string; tz: string },
) {
  const {
    Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
    HeadingLevel, WidthType, AlignmentType, BorderStyle, ImageRun,
  } = await import("docx");

  const border = (c = "BBBBBB") => ({
    top: { style: BorderStyle.SINGLE, size: 1, color: c },
    bottom: { style: BorderStyle.SINGLE, size: 1, color: c },
    left: { style: BorderStyle.SINGLE, size: 1, color: c },
    right: { style: BorderStyle.SINGLE, size: 1, color: c },
  });
  const cell = (text: string, bold = false, shade = false) =>
    new TableCell({
      shading: shade ? { fill: "F0F0F0" } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text, bold, size: 18 })] })],
    });
  const dataTable = (headers: string[], rows: string[][]) =>
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      borders: border(),
      rows: [
        new TableRow({ tableHeader: true, children: headers.map((h) => cell(h, true, true)) }),
        ...rows.map((r) => new TableRow({ children: r.map((c) => cell(c)) })),
      ],
    });
  const h = (text: string, level: (typeof HeadingLevel)[keyof typeof HeadingLevel]) =>
    new Paragraph({ heading: level, children: [new TextRun({ text, bold: true })] });
  const gap = () => new Paragraph({ children: [] });

  const children: object[] = [];
  children.push(h("GPU Platform — Usage Report", HeadingLevel.HEADING_1), gap());

  // Meta + summary
  const metaRow = (l: string, v: string) => new TableRow({ children: [cell(l, true, true), cell(v)] });
  children.push(
    new Table({
      width: { size: 70, type: WidthType.PERCENTAGE }, borders: border(),
      rows: [
        metaRow("Period", opts.periodLabel),
        metaRow("Scope", opts.scopeLabel),
        metaRow("Timezone", opts.tz),
        metaRow("Total requests", `${fmtInt(data.summary.total_requests)} · ${data.summary.distinct_models} models · ${data.summary.distinct_apps} endpoints`),
        metaRow("Outcomes", `${fmtInt(data.summary.completed)} ok · ${fmtInt(data.summary.client_cancelled)} ~4xx · ${fmtInt(data.summary.server_error)} ~5xx${data.summary.success_rate != null ? ` · ${data.summary.success_rate}% success` : ""}`),
        metaRow("Tokens", `${fmtInt(data.summary.tokens_total)} total (${data.summary.token_coverage_pct ?? 0}% coverage)`),
        metaRow("Latency", `p95 ${fmtLat(data.summary.p95_latency_s)} · avg ${fmtLat(data.summary.avg_latency_s)} (end-to-end)`),
      ],
    }),
    gap(),
  );

  // Chart: requests over time, by top model
  const totals: Record<string, number> = {};
  data.time_series.forEach((p) => Object.entries(p.by_model).forEach(([m, c]) => { totals[m] = (totals[m] || 0) + c; }));
  const topModels = Object.entries(totals).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([m]) => m);
  const chartRows = data.time_series.map((p) => {
    const row: Record<string, number | string> = { label: p.label };
    topModels.forEach((m) => { row[m] = p.by_model[m] || 0; });
    return row;
  });
  children.push(h("Requests over time, by model", HeadingLevel.HEADING_2));
  const png = await chartPng(chartRows, topModels);
  children.push(new Paragraph({
    children: png
      ? [new ImageRun({ type: "png", data: png, transformation: { width: 600, height: 183 } })]
      : [new TextRun({ text: "No request activity in this period.", italics: true, size: 18 })],
  }), gap());

  children.push(h("By model", HeadingLevel.HEADING_2));
  children.push(dataTable(
    ["Model", "Requests", "OK", "~4xx", "~5xx", "Tokens", "Avg lat"],
    data.by_model.map((m) => [m.model, fmtInt(m.requests), fmtInt(m.completed), fmtInt(m.client_cancelled), fmtInt(m.server_error), fmtTok(m.tokens_total), fmtLat(m.avg_latency_s)]),
  ), gap());

  children.push(h("By endpoint", HeadingLevel.HEADING_2));
  children.push(dataTable(
    ["Endpoint", "Requests", "OK", "~5xx"],
    data.by_endpoint.map((e) => [e.endpoint, fmtInt(e.requests), fmtInt(e.completed), fmtInt(e.server_error)]),
  ), gap());

  children.push(h("Top users", HeadingLevel.HEADING_2));
  children.push(dataTable(
    ["User", "Requests", "Tokens"],
    data.by_user.map((u) => [u.username, fmtInt(u.requests), fmtTok(u.tokens_total)]),
  ), gap());

  if (spend) {
    children.push(h(`Resource spend — $${spend.total_cost_usd.toFixed(2)}`, HeadingLevel.HEADING_2));
    children.push(dataTable(
      ["Type", "Count", "GPU-hours", "Cost (USD)"],
      spend.by_type.map((r) => [r.resource_type, fmtInt(r.count), r.gpu_hours == null ? "—" : r.gpu_hours.toFixed(1), `$${r.cost_usd.toFixed(2)}`]),
    ), gap());
  }

  children.push(h("Daily breakdown", HeadingLevel.HEADING_2));
  for (const day of data.daily) {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({
      text: `${day.day_label} — ${fmtInt(day.requests)} reqs · ${day.client_cancelled} ~4xx · ${day.server_error} ~5xx · ${fmtTok(day.tokens_total)} tok`, bold: true,
    })] }));
    if (day.jobs.length) {
      children.push(dataTable(
        ["Time", "Model", "Endpoint", "User", "Outcome", "Status", "Elapsed"],
        day.jobs.map((j) => [j.start_time, j.model, j.endpoint, j.username, j.outcome, j.status, j.elapsed_label]),
      ));
    } else {
      children.push(new Paragraph({ children: [new TextRun({ text: "No requests this day.", italics: true, size: 18 })] }));
    }
    children.push(gap());
  }

  children.push(new Paragraph({
    alignment: AlignmentType.RIGHT,
    children: [new TextRun({ text: data.note, color: "888888", size: 16, italics: true })],
  }));

  const doc = new Document({
    styles: { paragraphStyles: [{ id: "Normal", name: "Normal", basedOn: "Normal", run: { font: "Arial", size: 20 } }] },
    sections: [{ children: children as never[] }],
  });

  const blob = await Packer.toBlob(doc);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `gpuplatform-usage-${data.period.from_date}-to-${data.period.to_date}.docx`;
  a.click();
  URL.revokeObjectURL(url);
}
