/* Build presentation_slides.pptx from deck_data.json (exported from results/tables).
   Run:  node report/build_deck.js  (after report/export_deck_data.py) */

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const DATA = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf-8"));

// ---------- palette (quantum indigo + lunar amber) ----------
const BG_DARK = "0E1116", BG = "F7F8FA";
const PRIMARY = "312E81", PRIMARY_MID = "4F46E5", PRIMARY_TINT = "E0E7FF";
const ACCENT = "F59E0B";
const TEXT = "111827", MUTED = "6B7280", WHITE = "FFFFFF";
const F = "Segoe UI";
const W = 13.33, H = 7.5, M = 0.6;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Quantum Edge Detection Project";
pres.title = "Quantum Edge Detection Under Changing Image Conditions";

const r1 = v => (typeof v === "number" ? v.toFixed(3) : v);
const r2 = v => (typeof v === "number" ? v.toFixed(2) : v);

// ---------- helpers ----------
function slideBase(dark = false) {
  const s = pres.addSlide();
  s.background = { color: dark ? BG_DARK : BG };
  return s;
}
function title(s, txt, sub = null) {
  s.addText(txt, { x: M, y: 0.42, w: W - 2 * M, h: 0.72, fontSize: 30, bold: true,
    color: TEXT, fontFace: F, margin: 0 });
  if (sub) s.addText(sub, { x: M, y: 1.12, w: W - 2 * M, h: 0.4, fontSize: 14,
    color: MUTED, fontFace: F, margin: 0 });
}
const bu = () => ({ code: "25B8", indent: 12 });
function bullets(s, items, opts) {
  s.addText(items.map((t, i) => ({ text: t,
    options: { bullet: bu(), breakLine: i < items.length - 1, color: TEXT } })),
    Object.assign({ fontSize: 14, fontFace: F, paraSpaceAfter: 10, margin: 0,
      valign: "top" }, opts));
}
function statBox(s, x, y, w, num, label, color = PRIMARY) {
  s.addText(num, { x, y, w, h: 1.0, fontSize: 44, bold: true, color, fontFace: F,
    align: "center", margin: 0 });
  s.addText(label, { x, y: y + 1.0, w, h: 0.65, fontSize: 12, color: MUTED,
    fontFace: F, align: "center", margin: 0 });
}
const CHART_BASE = {
  chartArea: { fill: { color: "FFFFFF" } },
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
  catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
  valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
  dataLabelColor: TEXT, dataLabelFontSize: 9,
  catAxisLabelFontFace: F, valAxisLabelFontFace: F, dataLabelFontFace: F,
  showLegend: false,
};
const METHOD_COLORS = { sobel: "1F77B4", canny: "FF7F0E", robinson: "2CA02C",
  qhed: "DC2626", neqr: "7C3AED" };
const DEC2 = { dataLabelFormatCode: "0.00" };

// ================= 1. COVER =================
{
  const s = slideBase(true);
  s.addText("QUANTUM IMAGE PROCESSING  \u00B7  RESEARCH PROJECT", {
    x: M, y: 1.5, w: W - 2 * M, h: 0.4, fontSize: 13, color: ACCENT,
    charSpacing: 4, fontFace: F, margin: 0 });
  s.addText("Quantum Edge Detection Under\nChanging Image Conditions", {
    x: M, y: 2.0, w: W - 2 * M, h: 2.2, fontSize: 44, bold: true, color: WHITE,
    fontFace: F, margin: 0 });
  s.addText("QPIE amplitude encoding + Hadamard interference  vs  NEQR basis-state encoding + quantum arithmetic\nA comparative study with Sobel / Canny / Robinson baselines on grayscale lunar crater imagery", {
    x: M, y: 4.35, w: W - 2 * M, h: 0.9, fontSize: 15, color: "C7CBD4",
    fontFace: F, margin: 0 });
  s.addText("Qiskit 2.x  \u00B7  Qiskit AerSimulator (ideal + exact density-matrix noise)  \u00B7  OpenCV  \u00B7  10 experiments", {
    x: M, y: 6.55, w: W - 2 * M, h: 0.4, fontSize: 12, color: "8B90A0", fontFace: F, margin: 0 });
}

// ================= 2. MOTIVATION =================
{
  const s = slideBase();
  title(s, "Why quantum edge detection on lunar craters?",
      "Crater rims are mission-critical edges \u2014 and illumination changes them daily");
  const rows = [
    ["Mission-critical", "Rim boundaries feed rover navigation, geological dating, and landing-zone selection."],
    ["Illumination stress test", "Solar incidence angle swings crater contrast dramatically \u2014 a natural robustness benchmark."],
    ["Two representations, one question", "QPIE stores intensity in probability amplitudes; NEQR stores exact bytes in the computational basis. Does the choice change edge quality, robustness, and cost?"],
    ["Unexplored domain", "No published quantum image processing study on planetary surface imagery."],
  ];
  let y = 1.8;
  for (const [h, b] of rows) {
    s.addText(h, { x: M, y, w: 3.4, h: 0.85, fontSize: 16, bold: true, color: PRIMARY, fontFace: F, margin: 0 });
    s.addText(b, { x: 4.3, y, w: W - 4.3 - M, h: 0.85, fontSize: 13.5, color: TEXT, fontFace: F, margin: 0, valign: "top" });
    if (y < 5.6) s.addShape(pres.shapes.LINE, { x: M, y: y + 0.95, w: W - 2 * M, h: 0,
      line: { color: "E5E7EB", width: 1 } });
    y += 1.22;
  }
  s.addText("Primary research question: does the quantum image representation itself significantly affect edge-detection quality, robustness, and resource cost?", {
    x: M, y: 6.75, w: W - 2 * M, h: 0.55, fontSize: 13, italic: true, color: PRIMARY_MID, fontFace: F, margin: 0 });
}

// ================= 3. TWO PIPELINES =================
{
  const s = slideBase();
  title(s, "Two independent quantum pipelines", "Validated against analytic target statevectors before any benchmarking");
  const col = (x, head, items, tint) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: 5.9, h: 4.7,
      fill: { color: tint }, rectRadius: 0.08 });
    s.addText(head, { x: x + 0.35, y: 2.0, w: 5.2, h: 0.5, fontSize: 19, bold: true,
      color: PRIMARY, fontFace: F, margin: 0 });
    bullets(s, items, { x: x + 0.35, y: 2.62, w: 5.2, h: 3.6, fontSize: 13 });
  };
  col(M, "Branch A \u2014 QPIE + QHED  (Yao et al., PRX 2017)", [
    "Intensity lives in probability amplitudes:  |\u03C8\u27E9 = \u03A3 I(r,c)|r,c\u27E9 / \u2016I\u2016",
    "Edge operator: Hadamard \u2192 decrement \u2192 Hadamard interference",
    "P(edge) \u221D |a(r,c) \u2212 a(neighbour)|\u00B2 per shot",
    "7 logical qubits at 8\u00D78 (position + 1 ancilla)",
  ], PRIMARY_TINT);
  col(M + 6.23, "Branch B \u2014 NEQR + quantum gradient  (Zhang et al., 2013)", [
    "Exact 8-bit intensity register in the computational basis",
    "Neighbour image loaded by inverted-control state preparation (quantum translation)",
    "CDKM ripple-carry subtractor computes F \u2212 G exactly, with sign carry",
    "24 logical qubits at 8\u00D78 (position + 2 intensity registers + ancillas)",
  ], "FEF3C7");
  s.addText("Both branches emit continuous per-pixel confidence maps \u2014 never hard-thresholded inside the quantum pipeline.", {
    x: M, y: 6.7, w: W - 2 * M, h: 0.5, fontSize: 12.5, color: MUTED, fontFace: F, margin: 0 });
}

// ================= 4. VALIDATION =================
{
  const s = slideBase();
  title(s, "Circuit correctness: proven, not assumed", "Every quantum primitive checked against exhaustive classical references");
  statBox(s, M, 2.1, 3.7, "< 10\u207B\u00B9\u2074", "max |circuit \u2212 analytic| deviation,\nQHED statevector vs Yao-formula (sizes 2/4/8)");
  statBox(s, M + 4.0, 2.1, 3.7, "100%", "NEQR pixel recovery at 4\u00D74:\nevery measured byte exact, uniform 1/16");
  statBox(s, M + 8.0, 2.1, 3.7, "2\u00D7256", "increment/decrement gates verified\nexhaustively over all inputs, widths 1\u20134");
  bullets(s, [
    "Quantum subtractor: 300 random cases across 2/3/4/8-bit widths, operand restoration and ancilla cleanliness asserted",
    "NEQR gradient payload (F, \u0394, sign carry) matches the analytic target state bit-for-bit in both directions",
    "Ideal-sampling path statistically validated against real circuit execution at 200k shots",
  ], { x: M, y: 4.35, w: W - 2 * M, h: 2.2, fontSize: 14 });
}

// ================= 5. PROTOCOL =================
{
  const s = slideBase();
  title(s, "Evaluation protocol", "Independent ground truth \u2014 classical detectors are baselines, never the reference");
  const rows = [
    ["Ground truth", "Analytic pixel masks (synthetic geometry) and annotation-derived rim masks (lunar patches) \u2014 never a detector output."],
    ["Thresholds", "Validation-set sweep per method (0.01\u20130.99), frozen before test evaluation; full PR curves reported threshold-free."],
    ["Statistics", "n = 10 repeated trials per configuration with shot-seed variation \u2192 mean \u00B1 std; classical results deterministic."],
    ["Noise simulation", "Exact density-matrix execution of the real transpiled circuit (hardware basis {rz, sx, x, cx}); no trajectory approximations."],
    ["Metrics", "Boundary F1 at 0/1/2-px tolerance, precision, recall, IoU, Chamfer distance, PR-AUC, robustness AUC."],
  ];
  let y = 1.85;
  for (const [h, b] of rows) {
    s.addText(h, { x: M, y, w: 2.9, h: 0.9, fontSize: 15, bold: true, color: PRIMARY, fontFace: F, margin: 0 });
    s.addText(b, { x: 3.8, y, w: W - 3.8 - M, h: 0.9, fontSize: 13, color: TEXT, fontFace: F, margin: 0, valign: "top" });
    y += 1.0;
  }
}

// ================= 6. EXP1 =================
{
  const s = slideBase();
  title(s, "Experiment 1 \u2014 representation ablation", "Same images, shots, thresholds; only the quantum representation changes");
  const lunar = (DATA.exp1 || []).filter(r => r.dataset === "lunar");
  if (lunar.length) {
    s.addChart(pres.charts.BAR, [{
      name: "Boundary F1 (1-px)",
      labels: lunar.map(r => r.method),
      values: lunar.map(r => +r.f1_1px_mean),
    }], Object.assign({}, CHART_BASE, {
      x: M, y: 1.9, w: 6.4, h: 4.2, barDir: "col",
      chartColors: lunar.map(r => METHOD_COLORS[r.method] || PRIMARY_MID),
      varyColors: true, showValue: true, dataLabelPosition: "outEnd", ...DEC2,
      valAxisMaxVal: 1.0, showTitle: true, title: "Lunar crater patches (F1)",
      titleFontSize: 12, titleColor: TEXT,
    }));
  }
  const synth = {};
  (DATA.exp1 || []).filter(r => r.dataset === "synthetic").forEach(r => synth[r.method] = r.f1_1px_mean);
  statBox(s, 7.6, 2.1, 5.1, `${r2(synth.neqr)} vs ${r2(synth.qhed)}`,
    "F1 on synthetic images: NEQR vs QHED \u2014 identical edge\nsets, radically different cost (next slide)");
  bullets(s, [
    "On real lunar patches NEQR keeps F1 \u2248 0.88 while QHED drops to \u2248 0.42",
    "Cause: QHED\u2019s P \u221D |\u0394amplitude|\u00B2 compresses weak amplitude differences of low-contrast terrain",
    "NEQR\u2019s exact byte differences remain uniform in sensitivity across contrast levels",
  ], { x: 7.6, y: 4.15, w: 5.1, h: 2.4, fontSize: 12.5 });
}

// ================= 7. EXP2/9 RESOURCES =================
{
  const s = slideBase();
  title(s, "Experiments 2 & 9 \u2014 resource scaling", "Qubits grow logarithmically; gates and depth do not (H6 confirmed)");
  const e2 = (DATA.exp2 || []).filter(r => r.geometry === "circle");
  const sizes = ["4", "8", "16"];
  const qh = e2.filter(r => r.method === "qhed"), nq = e2.filter(r => r.method === "neqr");
  s.addChart(pres.charts.BAR, [
    { name: "QHED raw gates", labels: sizes, values: sizes.map(sz => { const f = qh.find(r => String(r.size) === sz); return f ? +f.raw_gates : 0; }) },
    { name: "NEQR raw gates", labels: sizes, values: sizes.map(sz => { const f = nq.find(r => String(r.size) === sz); return f ? +f.raw_gates : 0; }) },
  ], Object.assign({}, CHART_BASE, {
    x: M, y: 1.95, w: 6.6, h: 4.0, barDir: "col",
    chartColors: [METHOD_COLORS.qhed, METHOD_COLORS.neqr],
    showLegend: true, legendPos: "t", legendFontSize: 10, legendColor: TEXT,
    showValue: true, dataLabelPosition: "outEnd", ...DEC2,
    showTitle: true, title: "Raw gate count vs resolution", titleFontSize: 12, titleColor: TEXT,
  }));
  const q9 = nq.find(r => String(r.size) === "16"), q8 = nq.find(r => String(r.size) === "8");
  statBox(s, 7.7, 2.0, 5.0, "22 \u2192 24 \u2192 26",
    "NEQR logical qubits, 4\u21928\u219216 px\n(logarithmic \u2014 QHED: 5 \u2192 7 \u2192 9)");
  bullets(s, [
    `NEQR gates 289 \u2192 1,301 \u2192 6,009: \u2248 ${(6009 / 289).toFixed(0)}\u00D7 growth for a 16\u00D7 pixel increase`,
    "Transpiled depth follows the same law (184 \u2192 648 \u2192 2,408)",
    "Kernel vs E2E time reported separately; state preparation dominates \u2014 no quantum-advantage claims",
  ], { x: 7.7, y: 4.05, w: 5.0, h: 2.4, fontSize: 12.5 });
}

// ================= 8. EXP3 SHOTS =================
{
  const s = slideBase();
  title(s, "Experiment 3 \u2014 shot-count sensitivity", "Binomial variance law visible; F1 saturates by ~1,024 shots (H2 confirmed)");
  const e3 = DATA.exp3 || [];
  const shotsLabels = [...new Set(e3.map(r => r.shots))].map(String);
  const series = ["qhed", "neqr"].map(m => ({
    name: m, labels: shotsLabels,
    values: shotsLabels.map(sh => { const f = e3.find(r => String(r.shots) === sh && r.method === m); return f ? +f.f1_1px_mean : null; }),
  }));
  s.addChart(pres.charts.LINE, series, Object.assign({}, CHART_BASE, {
    x: M, y: 1.95, w: 7.2, h: 4.2, lineSize: 2.5, lineSmooth: false,
    chartColors: [METHOD_COLORS.qhed, METHOD_COLORS.neqr],
    showLegend: true, legendPos: "t", legendFontSize: 10, legendColor: TEXT,
    showTitle: true, title: "Boundary F1 vs shots (8\u00D78 circle, n=10)", titleFontSize: 12, titleColor: TEXT,
    valAxisMinVal: 0.6, valAxisMaxVal: 0.9,
  }));
  statBox(s, 8.3, 2.1, 4.3, "1/shots", "per-pixel map variance follows the\nbinomial law exactly across 512\u219232,768");
  bullets(s, [
    "512-shot runs show the largest variance (\u03C3 \u2248 0.02\u20130.06)",
    "Diminishing returns above ~2,048 shots",
    "F1 plateaus well before the 8,192-shot default",
  ], { x: 8.3, y: 4.1, w: 4.3, h: 2.2, fontSize: 12.5 });
}

// ================= 9. EXP4 ROBUSTNESS =================
{
  const s = slideBase();
  title(s, "Experiment 4 \u2014 degradation robustness", "Robustness AUC = area under the F1-vs-degradation curve (higher = more graceful)");
  const auc = DATA.exp4_auc || [];
  const fams = [...new Set(auc.map(r => r.family))];
  const methods = ["sobel", "canny", "robinson", "qhed", "neqr"];
  s.addChart(pres.charts.BAR, fams.map(fa => ({
    name: fa, labels: methods,
    values: methods.map(m => { const f = auc.find(r => r.family === fa && r.method === m); return f && f.auc != null ? +f.auc : 0; }),
  })), Object.assign({}, CHART_BASE, {
    x: M, y: 1.95, w: 7.6, h: 4.3, barDir: "col",
    chartColors: ["94A3B8", "C084FC", "5EEAD4"],
    showLegend: true, legendPos: "t", legendFontSize: 10, legendColor: TEXT,
    valAxisMaxVal: 1.0,
    showTitle: true, title: "Robustness AUC by degradation family", titleFontSize: 12, titleColor: TEXT,
  }));
  bullets(s, [
    "Sobel is the most degradation-tolerant classical baseline",
    "QHED\u2019s amplitude normalisation makes global contrast changes non-linear \u2014 visible in its contrast AUC",
    "NEQR tracks QHED under blur but degrades under extreme low illumination (wraparound artefact, Exp 10)",
  ], { x: 8.6, y: 2.3, w: 4.1, h: 3.8, fontSize: 12.5 });
}

// ================= 10. EXP5 NOISE =================
{
  const s = slideBase();
  title(s, "Experiment 5 \u2014 quantum hardware noise", "Exact density-matrix execution of the real transpiled circuits");
  const e5 = DATA.exp5 || [];
  // H5 signal lives in the distribution fidelity: the 12-qubit NEQR
  // circuit collapses far faster than the 3-qubit QHED at equal rates.
  const depol = e5.filter(r => r.size === 2 && String(r.noise).startsWith("depol"));
  if (depol.length) {
    const noises = ["ideal", "depol_0.001", "depol_0.005", "depol_0.01", "depol_0.05"];
    const series = ["qhed", "neqr"].map(m => ({
      name: m, labels: noises.map(n => n.replace("depol_", "p=")),
      values: noises.map(n => { const f = depol.find(r => r.noise === n && r.method === m); return f ? +f.fidelity_mean : null; }),
    }));
    s.addChart(pres.charts.LINE, series, Object.assign({}, CHART_BASE, {
      x: M, y: 1.95, w: 7.2, h: 4.2, lineSize: 2.5,
      chartColors: [METHOD_COLORS.qhed, METHOD_COLORS.neqr],
      showLegend: true, legendPos: "t", legendFontSize: 10, legendColor: TEXT,
      valAxisMinVal: 0, valAxisMaxVal: 1.05,
      showTitle: true, title: "Hellinger circuit fidelity vs depolarizing p (2\u00D72 \u2014 H5)", titleFontSize: 12, titleColor: TEXT,
    }));
  }
  bullets(s, [
    "Edge F1 survives noise far longer than circuit fidelity: NEQR keeps F1 = 1.0 even at fidelity 0.003 \u2014 the mean-based reconstruction averages depolarizing noise away",
    "Depth penalty is real: at p = 0.005 the 12-qubit NEQR circuit loses fidelity (0.33) that the 3-qubit QHED keeps (0.999) \u2014 H5 confirmed on fidelity",
    "Coherent over-rotation injects bias invisible to averaged fidelity metrics",
  ], { x: 8.3, y: 2.3, w: 4.4, h: 3.9, fontSize: 12.5 });
}

// ================= 11. EXP6 GEOMETRY =================
{
  const s = slideBase();
  title(s, "Experiment 6 \u2014 geometry ablation with Robinson diagnostics", "Robinson compass separates quantum-specific failures (A) from image-level ones (B)");
  const e6 = DATA.exp6 || [];
  if (e6.length) {
    const geos = e6.map(r => r.geometry);
    const series = ["sobel", "robinson", "qhed", "neqr"].map(m => ({
      name: m, labels: geos, values: e6.map(r => r[m] != null ? +r[m] : null) }));
    s.addChart(pres.charts.BAR, series, Object.assign({}, CHART_BASE, {
      x: M, y: 1.9, w: 7.4, h: 4.4, barDir: "col",
      chartColors: [METHOD_COLORS.sobel, METHOD_COLORS.robinson, METHOD_COLORS.qhed, METHOD_COLORS.neqr],
      showLegend: true, legendPos: "t", legendFontSize: 10, legendColor: TEXT,
      valAxisMaxVal: 1.0, catAxisLabelRotate: 25,
      showTitle: true, title: "F1 by geometry (8\u00D78, identical contrast)", titleFontSize: 12, titleColor: TEXT,
    }));
  }
  bullets(s, [
    "Quantum F1 drops on corner junctions where classical detectors stay strong \u2192 Case A: quantum-specific",
    "Robinson\u2019s dominant-direction map confirms the image itself is well-defined in those cases",
    "Curved rims rotate through all 8 compass directions \u2014 compounding directional limits",
  ], { x: 8.4, y: 2.3, w: 4.3, h: 3.9, fontSize: 12.5 });
}

// ================= 12. EXP7 PREPROCESSING =================
{
  const s = slideBase();
  title(s, "Experiment 7 \u2014 preprocessing strategies", "Does preprocessing help the quantum pipelines on lunar patches? (H7)");
  const e7 = DATA.exp7 || [];
  if (e7.length) {
    const strats = e7.map(r => r.strategy);
    const series = ["sobel", "robinson", "qhed", "neqr"].map(m => ({
      name: m, labels: strats, values: e7.map(r => r[m] != null ? +r[m] : null) }));
    s.addChart(pres.charts.BAR, series, Object.assign({}, CHART_BASE, {
      x: M, y: 1.9, w: 7.6, h: 4.4, barDir: "col",
      chartColors: [METHOD_COLORS.sobel, METHOD_COLORS.robinson, METHOD_COLORS.qhed, METHOD_COLORS.neqr],
      showLegend: true, legendPos: "t", legendFontSize: 10, legendColor: TEXT,
      valAxisMaxVal: 1.0, catAxisLabelRotate: 20,
      showTitle: true, title: "F1 by preprocessing strategy (lunar patches)", titleFontSize: 12, titleColor: TEXT,
    }));
  }
  bullets(s, [
    "CLAHE and histogram stretch help low-contrast lunar patches most",
    "Aggressive smoothing (median, Gaussian) can erase thin rim signals before encoding",
    "Gamma correction trades shadow recovery against highlight saturation",
  ], { x: 8.6, y: 2.3, w: 4.1, h: 3.9, fontSize: 12.5 });
}

// ================= 13. EXP8 LUNAR BENCHMARK =================
{
  const s = slideBase();
  title(s, "Experiment 8 \u2014 lunar crater benchmark", "Annotation-derived rim masks; 8\u00D78 and 16\u00D716 patches; n = 10 trials");
  const e8 = DATA.exp8 || [];
  if (e8.length) {
    const rows = [[
      { text: "size", options: { bold: true, fill: { color: PRIMARY_TINT } } },
      { text: "method", options: { bold: true, fill: { color: PRIMARY_TINT } } },
      { text: "F1 exact", options: { bold: true, fill: { color: PRIMARY_TINT } } },
      { text: "F1 1-px", options: { bold: true, fill: { color: PRIMARY_TINT } } },
      { text: "F1 2-px", options: { bold: true, fill: { color: PRIMARY_TINT } } },
      { text: "IoU", options: { bold: true, fill: { color: PRIMARY_TINT } } },
      { text: "PR-AUC", options: { bold: true, fill: { color: PRIMARY_TINT } } },
    ]];
    for (const r of e8) {
      rows.push([String(r.size), r.method, r2(r.f1_exact), r2(r.f1_1px), r2(r.f1_2px), r2(r.iou), r2(r.pr_auc)]);
    }
    s.addTable(rows, { x: M, y: 2.0, w: 8.2, fontSize: 11, fontFace: F,
      border: { pt: 0.5, color: "E5E7EB" }, color: TEXT, align: "center", valign: "middle",
      fill: { color: "FFFFFF" }, rowH: 0.32 });
  }
  bullets(s, [
    "Full metric battery per patch and method",
    "Illumination grouping available via tile metadata",
    "Rim masks derived from annotations \u2014 independent of every detector",
  ], { x: 9.1, y: 2.3, w: 3.6, h: 3.4, fontSize: 12.5 });
}

// ================= 14. HYPOTHESES =================
{
  const s = slideBase();
  title(s, "Hypothesis scorecard", "Pre-registered in PROJECT_SUMMARY.md \u00A79 before execution");
  const rows = [
    ["H1", "Degradation reduces F1 at different rates per representation", "CONFIRMED"],
    ["H2", "F1 improves with shots, saturating with diminishing returns", "CONFIRMED"],
    ["H3", "QHED weaker on curved/diagonal edges", "CONFIRMED (Exp 6)"],
    ["H4", "Low-contrast rims give lower quantum F1", "CONFIRMED"],
    ["H5", "Equal depolarizing rates hurt deeper NEQR more", "CONFIRMED"],
    ["H6", "Resource cost grows far faster than qubit count", "CONFIRMED"],
    ["H7", "Preprocessing helps low-contrast patches", "PARTIAL"],
    ["H8", "Quantum E2E time exceeds classical at all scales", "CONFIRMED"],
  ];
  const tbl = [[
    { text: "ID", options: { bold: true, fill: { color: PRIMARY_TINT } } },
    { text: "Hypothesis", options: { bold: true, fill: { color: PRIMARY_TINT } } },
    { text: "Verdict", options: { bold: true, fill: { color: PRIMARY_TINT } } }]];
  for (const [id, txt, v] of rows) tbl.push([id, txt, v]);
  s.addTable(tbl, { x: M, y: 1.95, w: W - 2 * M, fontSize: 12.5, fontFace: F,
    border: { pt: 0.5, color: "E5E7EB" }, color: TEXT, valign: "middle",
    fill: { color: "FFFFFF" }, rowH: 0.42, colW: [0.9, 8.4, 2.83],
    align: "left" });
}

// ================= 15. CLOSING =================
{
  const s = slideBase(true);
  s.addText("Conclusions", { x: M, y: 0.9, w: W - 2 * M, h: 0.8, fontSize: 34, bold: true, color: WHITE, fontFace: F, margin: 0 });
  const items = [
    ["Representation matters", "NEQR\u2019s exact arithmetic transfers to low-contrast planetary terrain; QHED\u2019s quadratic amplitude mapping compresses weak edges."],
    ["Resources, not qubits", "Gate count and depth \u2014 not logical qubit count \u2014 dominate the quantum cost of edge detection at NISQ scales."],
    ["Measurement saturates early", "1,024\u20132,048 shots capture nearly all attainable quality; more shots only shrink binomial tails."],
    ["Noise is structural", "Readout error is representation-agnostic; depolarizing error amplifies with depth; coherent bias needs distribution-level diagnostics."],
  ];
  let y = 1.9;
  for (const [h, b] of items) {
    s.addText(h, { x: M, y, w: 3.6, h: 1.0, fontSize: 16, bold: true, color: ACCENT, fontFace: F, margin: 0 });
    s.addText(b, { x: 4.4, y, w: W - 4.4 - M, h: 1.0, fontSize: 13, color: "C7CBD4", fontFace: F, margin: 0, valign: "top" });
    y += 1.12;
  }
  s.addText("Full reproducibility:  tests/ \u2192 experiments/run_all.py \u2192 results/  \u00B7  report/final_report.md + PDF  \u00B7  Streamlit dashboard", {
    x: M, y: 6.7, w: W - 2 * M, h: 0.5, fontSize: 12, color: "8B90A0", fontFace: F, margin: 0 });
}

pres.writeFile({ fileName: path.join(__dirname, "presentation_slides.pptx") })
  .then(() => console.log("deck written: report/presentation_slides.pptx"));
