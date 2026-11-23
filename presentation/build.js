// DSAA2011 Task A presentation, 12 slides for a 12-minute talk.
const pptxgen = require("pptxgenjs");
const path = require("path");
const SCRIPT = require("./script.js");
const { applyTheme } = require(process.env.SKILL_DIR + "/scripts/apply_theme.js");

const FIG = process.env.FIG_DIR;
const fig = (f) => path.join(FIG, f);

const THEME = {
  name: "Climate Zones",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2A30", lt1: "FFFFFF", dk2: "123A40", lt2: "EDF3F2",
    accent1: "0F7C80", accent2: "E3A02F", accent3: "D1495B",
    accent4: "7C5BA6", accent5: "2F78C4", accent6: "3E9A55",
    hlink: "0F7C80", folHlink: "7C5BA6",
  },
};
const HEX = THEME.colors;
const ZONES = [["Tropical", HEX.accent3], ["Arid", HEX.accent2], ["Subtropical", HEX.accent6],
               ["Mediterranean", HEX.accent4], ["Oceanic", HEX.accent5]];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "From Weather Snapshots to Seasonal Climate Signatures";
pres.author = "Group 7";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// Five zone dots: the deck's visual motif.
function dotsObjects(x, y, d, gap) {
  return ZONES.map(([name, col], i) => ({
    ellipse: { x: x + i * (d + gap), y, w: d, h: d, fill: { color: col }, line: { color: col } },
  }));
}

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.75, y: 2.0, w: 11.8, h: 1.5,
        fontFace: THEME.headFontFace, fontSize: 54, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.75, y: 3.55, w: 11.8, h: 1.2,
        fontFace: THEME.bodyFontFace, fontSize: 26, color: "CFE3E1", valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 10.7, h: 0.75,
        fontFace: THEME.headFontFace, fontSize: 34, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.35, y: 7.0, w: 0.5, h: 0.3, fontFace: THEME.bodyFontFace, fontSize: 11, color: "6B7B80", align: "right" },
});

pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.75, y: 0.55, w: 11.8, h: 0.9,
        fontFace: THEME.headFontFace, fontSize: 38, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
});

function addDots(slide, x, y, d, gap) {
  dotsObjects(x, y, d, gap).forEach((o) => slide.addShape(pres.shapes.OVAL, { ...o.ellipse }));
}
const _addSlide = pres.addSlide.bind(pres);
pres.addSlide = (opts) => {
  const sl = _addSlide(opts);
  if (opts.masterName === "CONTENT") addDots(sl, 11.55, 0.62, 0.16, 0.08);
  if (opts.masterName === "TITLE_DARK") addDots(sl, 0.8, 1.35, 0.32, 0.18);
  return sl;
};
const notes = (slide, n) => {
  const s = SCRIPT[n - 1];
  slide.addNotes(`[Speaker ${s.speaker} · ${s.time}]\n${s.en}`);
};

// Takeaway box at the bottom of content slides (same place on every slide).
function takeaway(slide, text) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 6.25, w: 12.1, h: 0.62, rectRadius: 0.1,
    fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "Takeaway box" });
  slide.addText([{ text: "Key message  ", options: { bold: true, color: C.accent1 } }, { text, options: { color: C.text1 } }],
    { x: 0.85, y: 6.25, w: 11.7, h: 0.62, fontSize: 18, valign: "middle", margin: 0, isTextBox: true, objectName: "Takeaway" });
}

function stat(slide, x, y, w, big, label, color) {
  slide.addText(big, { x, y, w, h: 0.85, fontFace: THEME.headFontFace, fontSize: 44, bold: true, color, margin: 0, isTextBox: true });
  slide.addText(label, { x, y: y + 0.85, w, h: 0.45, fontSize: 15, color: C.text1, margin: 0, isTextBox: true });
}

function card(slide, x, y, w, h, head, body, color) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: C.background2 },
    line: { color: C.background2 }, objectName: "Card " + head });
  slide.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.42, h: 0.42, fill: { color }, line: { color } });
  slide.addText(head, { x: x + 0.85, y: y + 0.22, w: w - 1.1, h: 0.6, fontSize: 19, bold: true, color: C.text2, margin: 0,
    valign: "middle", isTextBox: true });
  slide.addText(body, { x: x + 0.3, y: y + 0.95, w: w - 0.6, h: h - 1.15, fontSize: 17, color: C.text1, margin: 0,
    valign: "top", paraSpaceAfter: 6, isTextBox: true });
}

const bullets = (items) => items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } }));

const CHART_TEXT = { catAxisLabelColor: "4A5A60", valAxisLabelColor: "4A5A60", catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  catAxisLabelFontSize: 13, valAxisLabelFontSize: 12, dataLabelFontSize: 11, legendFontSize: 13,
  valGridLine: { color: "D9E2E1", size: 0.75 }, catGridLine: { style: "none" }, dataLabelColor: "1F2A30" };

// ---------------- Slide 1: title ----------------
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("From weather snapshots to climate signatures", { placeholder: "title" });
s.addText("Recovering Australia's climate zones from weather observations", { placeholder: "body" });
s.addText([{ text: "DSAA2011 Course Project · Task A · Rain in Australia dataset", options: { breakLine: true } },
           { text: "Group 7 · QIAO, Xiaolong · MO, Zijun · WANG, Zhifei · XIA, Youyue" }],
  { x: 0.75, y: 5.6, w: 11.8, h: 0.9, fontSize: 18, color: "CFE3E1", margin: 0, isTextBox: true });
notes(s, 1);

// ---------------- Slide 2: data and labels ----------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("From rain data to a climate question", { placeholder: "title" });
stat(s, 0.6, 1.45, 3.0, "142,451", "daily records", C.accent1);
stat(s, 3.7, 1.45, 2.4, "48", "weather stations", C.accent1);
stat(s, 0.6, 2.95, 3.0, "5", "climate groups", C.accent1);
stat(s, 3.7, 2.95, 2.4, "4", "tropical stations only", C.accent3);
s.addText(bullets([
  "Label = Köppen code at each station's coordinates (BoM station list + kgcpy 100″ map)",
  "Groups: Tropical (A), Arid (B), Subtropical (Cfa), Mediterranean (Cs), Oceanic (Cfb)",
  "Norfolk Island removed: it is not on the map",
]), { x: 0.6, y: 4.45, w: 5.8, h: 1.6, fontSize: 15, color: C.text1, margin: 0, paraSpaceAfter: 6, valign: "top" });
s.addImage({ path: fig("p1_station_map.png"), x: 7.05, y: 1.3, w: 5.65, h: 4.54, objectName: "Station map",
  altText: "Map of the 48 stations coloured by climate group" });
takeaway(s, "Can one day of weather tell us the climate zone of its station?");
notes(s, 2);

// ---------------- Slide 3: preprocessing ----------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Preprocessing: do not let the model cheat", { placeholder: "title" });
card(s, 0.6, 1.55, 3.85, 4.1, "Missing = fingerprint",
  bullets(["Some stations never measure sunshine, evaporation or cloud",
           "Missing flags alone find the station 10× better than chance (0.240 vs 0.024)",
           "→ drop these 4 columns, no missing indicators"]), C.accent3);
card(s, 4.72, 1.55, 3.85, 4.1, "Encode",
  bullets(["Wind direction + speed → vector (u, v); calm = (0, 0)",
           "Day of year → sin, cos (no year)",
           "Rainfall → log(1 + x)",
           "RainToday dropped (same as Rainfall > 1 mm)"]), C.accent2);
card(s, 8.84, 1.55, 3.86, 4.1, "No leakage",
  bullets(["Removed: Location, RainTomorrow",
           "Station ID only used to split data",
           "Median imputation + scaling fitted on training folds only",
           "Result: 20 features"]), C.accent1);
takeaway(s, "The label belongs to the station, so any station clue is a leak.");
notes(s, 3);

// ---------------- Slide 4: t-SNE ----------------
pres.addSection({ title: "Mandatory tasks" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Mandatory tasks" });
s.addText("t-SNE: only the tropics stand apart", { placeholder: "title" });
s.addImage({ path: fig("report_tsne.png"), x: 0.6, y: 1.3, w: 12.1, h: 4.05, objectName: "t-SNE figure",
  altText: "t-SNE of 6000 days coloured by climate group, month and maximum temperature" });
s.addText([{ text: "6,000 days (1,200 per group), perplexity 30.   ", options: { color: C.text1 } },
           { text: "Trustworthiness: t-SNE 0.985 vs PCA 0.809.   ", options: { bold: true, color: C.accent1 } },
           { text: "A picture, not a test score.", options: { italic: true, color: C.text1 } }],
  { x: 0.6, y: 5.5, w: 12.1, h: 0.5, fontSize: 15, margin: 0, isTextBox: true });
takeaway(s, "Tropical days form their own region; the other four groups overlap.");
notes(s, 4);

// ---------------- Slide 5: clustering ----------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Mandatory tasks" });
s.addText("Clustering: snapshots versus seasonal signatures", { placeholder: "title" });
s.addImage({ path: fig("p7_seasonal_clustering.png"), x: 0.6, y: 1.3, w: 7.4, h: 2.65, objectName: "Seasonal clustering sweep",
  altText: "K-means and Ward seasonal clustering sweep" });
const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.accent1 } } });
s.addTable([
  [hdr("k = 3"), hdr("Silhouette ↑"), hdr("Davies–Bouldin ↓"), hdr("Stability ↑"), hdr("ARI with zone")],
  [{ text: "Daily K-means", options: { bold: true } }, "0.164", "1.880", "—", "0.098"],
  ["Daily Ward", "0.117", "2.150", "—", "0.114"],
  ["Seasonal K-means", "0.121", "2.205", "7", "0.314"],
  ["Seasonal Ward", "0.106", "2.289", "7", "0.282"],
], { x: 0.6, y: 3.95, w: 7.4, colW: [1.4, 1.5, 1.7, 1.4, 1.4], fontSize: 14, color: C.text1, align: "center",
  border: { type: "solid", pt: 0.75, color: "D9E2E1" }, rowH: 0.42, objectName: "Clustering table" });
s.addText("Two units answer different questions", { x: 8.45, y: 1.3, w: 4.25, h: 0.45, fontSize: 19, bold: true, color: C.text2, margin: 0, isTextBox: true });
s.addText(bullets(["Daily clusters are weather types", "Seasonal clusters are more zone-aligned", "ARI remains descriptive, not a training target"]),
  { x: 8.45, y: 1.8, w: 4.25, h: 1.2, fontSize: 16, color: C.text1, margin: 0, paraSpaceAfter: 4, valign: "top" });
stat(s, 8.45, 3.25, 2.0, "0.098", "daily ARI", C.accent2);
stat(s, 10.6, 3.25, 2.1, "0.314", "best seasonal ARI", C.accent1);
s.addText("Four windows: equinoxes + solstices, 21 days each", { x: 8.45, y: 4.65, w: 4.25, h: 0.7,
  fontSize: 13, italic: true, color: "4A5A60", margin: 0, isTextBox: true });
takeaway(s, "One day looks like weather; many days start to look like climate.");
notes(s, 5);

// ---------------- Slide 6: evaluation design ----------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Mandatory tasks" });
s.addText("Two different exams", { placeholder: "title" });
function stationGrid(x, y, mode) {
  // 8 x 3 grid of station squares; in each square, 6 small 'day' bars
  const cols = 8, rows = 3, sz = 0.52, gap = 0.12;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const idx = r * cols + c;
    const sx = x + c * (sz + gap), sy = y + r * (sz + gap);
    if (mode === "random") {
      s.addShape(pres.shapes.RECTANGLE, { x: sx, y: sy, w: sz, h: sz * 0.62, fill: { color: HEX.accent1 }, line: { color: HEX.accent1 } });
      s.addShape(pres.shapes.RECTANGLE, { x: sx, y: sy + sz * 0.62, w: sz, h: sz * 0.38, fill: { color: HEX.accent2 }, line: { color: HEX.accent2 } });
    } else {
      const test = [2, 5, 9, 14, 19, 22].includes(idx);
      const col = test ? HEX.accent2 : HEX.accent1;
      s.addShape(pres.shapes.RECTANGLE, { x: sx, y: sy, w: sz, h: sz, fill: { color: col }, line: { color: col } });
    }
  }
}
s.addText("Random split (course 70/30)", { x: 0.6, y: 1.35, w: 5.8, h: 0.5, fontSize: 21, bold: true, color: C.text2, margin: 0, isTextBox: true });
stationGrid(0.6, 2.0, "random");
s.addText(bullets(["Test days come from stations the model has seen", "Question: new days, known stations"]),
  { x: 0.6, y: 4.1, w: 5.6, h: 1.0, fontSize: 16, color: C.text1, margin: 0, paraSpaceAfter: 4, valign: "top" });
s.addText("Station split (our main result)", { x: 6.9, y: 1.35, w: 5.8, h: 0.5, fontSize: 21, bold: true, color: C.text2, margin: 0, isTextBox: true });
stationGrid(6.9, 2.0, "station");
s.addText(bullets(["4 folds by station; every fold has all 5 groups",
                   "Tuning, window and feature choices only inside training stations",
                   "Seeds 42 / 0 / 7; metric: five-class macro-F1"]),
  { x: 6.9, y: 4.1, w: 5.8, h: 1.4, fontSize: 16, color: C.text1, margin: 0, paraSpaceAfter: 4, valign: "top" });
// legend
s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 5.62, w: 0.3, h: 0.3, fill: { color: HEX.accent1 }, line: { color: HEX.accent1 } });
s.addText("training", { x: 1.0, y: 5.57, w: 1.4, h: 0.4, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
s.addShape(pres.shapes.RECTANGLE, { x: 2.4, y: 5.62, w: 0.3, h: 0.3, fill: { color: HEX.accent2 }, line: { color: HEX.accent2 } });
s.addText("test", { x: 2.8, y: 5.57, w: 1.0, h: 0.4, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
s.addText("each square = one station", { x: 3.8, y: 5.57, w: 3.0, h: 0.4, fontSize: 14, italic: true, color: "4A5A60", margin: 0, isTextBox: true });
takeaway(s, "For our question, only the unseen-station score is an honest exam.");
notes(s, 6);

// ---------------- Slide 7: Task 4 ----------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Mandatory tasks" });
s.addText("Task 4: the deep tree memorises", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Logistic regression", labels: ["Train", "Test", "Entire"], values: [0.604, 0.599, 0.602] },
  { name: "Decision tree (default)", labels: ["Train", "Test", "Entire"], values: [1.000, 0.652, 0.895] },
], { x: 0.6, y: 1.3, w: 6.2, h: 3.4, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent1, HEX.accent2],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1.1,
  valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00", showLegend: true, legendPos: "t",
  showTitle: true, title: "Accuracy, random 70/30 split", titleFontSize: 15, titleColor: "1F2A30", ...CHART_TEXT,
  objectName: "Train test entire chart" });
s.addText(bullets(["Tree: train 1.00 vs test 0.65 → overfitting",
                   "Entire set = 70% training data → not a fair test",
                   "Unseen stations: tree 0.533 < LR 0.563"]),
  { x: 0.6, y: 4.8, w: 6.2, h: 1.3, fontSize: 16, color: C.text1, margin: 0, paraSpaceAfter: 4, valign: "top" });
s.addImage({ path: fig("report_confusion.png"), x: 7.15, y: 1.25, w: 5.55, h: 4.17, objectName: "Confusion matrices",
  altText: "Row-normalised confusion matrices for LR and tree on train, test and entire sets" });
s.addText("Confusion matrices (row-normalised recall)", { x: 7.15, y: 5.45, w: 5.55, h: 0.4, fontSize: 13, italic: true,
  color: "4A5A60", margin: 0, align: "center", isTextBox: true });
takeaway(s, "A high entire-set score is not evidence; only the test column is.");
notes(s, 7);

// ---------------- Slide 8: Task 5 ----------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Mandatory tasks" });
s.addText("Task 5: metrics, ROC and tuning", { placeholder: "title" });
s.addImage({ path: fig("report_roc.png"), x: 0.6, y: 1.25, w: 5.6, h: 4.75, objectName: "ROC curves",
  altText: "One-vs-rest ROC curves of tuned LR and tree on random and grouped holdouts" });
s.addTable([
  [hdr("Split"), hdr("Model"), hdr("F1 default"), hdr("F1 tuned"), hdr("AUC tuned")],
  ["Random", "LR", "0.615", "0.615", "0.859"],
  ["Random", { text: "Tree", options: { bold: true } }, "0.669", { text: "0.691", options: { bold: true } }, { text: "0.889", options: { bold: true } }],
  ["Station", "LR", "0.561", "0.554", "0.821"],
  ["Station", "Tree", "0.536", "0.558", "0.809"],
], { x: 6.6, y: 1.35, w: 6.1, colW: [1.25, 1.0, 1.3, 1.25, 1.3], fontSize: 14, color: C.text1, align: "center",
  border: { type: "solid", pt: 0.75, color: "D9E2E1" }, rowH: 0.42, objectName: "Tuning table" });
s.addText(bullets(["Precision / recall / F1 computed from each confusion matrix",
                   "ROC one-vs-rest: Tropical AUC > 0.95 in all cases; Mediterranean hardest",
                   "Default tree: 0/1 leaf probabilities → AUC only 0.788",
                   "Tuned tree: depth 16, ≥ 20 days per leaf"]),
  { x: 6.6, y: 3.7, w: 6.1, h: 2.3, fontSize: 16, color: C.text1, margin: 0, paraSpaceAfter: 6, valign: "top" });
takeaway(s, "Tuning reduces overfitting, but it does not solve the new-station problem.");
notes(s, 8);

// ---------------- Slide 9: seasonal feature engineering ----------------
pres.addSection({ title: "Open-ended exploration" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Open-ended exploration" });
s.addText("Open-ended: build a seasonal weather signature", { placeholder: "title" });
s.addImage({ path: fig("p7_seasonal_feature_means.png"), x: 0.6, y: 1.25, w: 7.8, h: 4.85, objectName: "Seasonal feature means",
  altText: "Feature means across four seasonal windows" });
stat(s, 8.85, 1.35, 3.5, "4", "windows per station-year", C.accent1);
stat(s, 8.85, 2.8, 3.5, "21 d", "window width", C.accent2);
stat(s, 8.85, 4.25, 3.5, "73.6%", "complete coverage", C.accent4);
s.addText(bullets(["March equinox, June solstice, September equinox, December solstice",
                   "Means + standard deviations + rainy-day fraction + temperature range",
                   "At least 7 valid days in every window; no coordinates or station IDs"]),
  { x: 8.85, y: 5.05, w: 3.6, h: 1.05, fontSize: 13, color: C.text1, margin: 0, paraSpaceAfter: 4, valign: "top" });
takeaway(s, "Climate is a pattern across the year, so the representation must span the year.");
notes(s, 9);

// ---------------- Slide 10: seasonal prediction ----------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Open-ended exploration" });
s.addText("Seasonal signatures improve the matched baseline", { placeholder: "title" });
s.addImage({ path: fig("p7_seasonal_model_comparison.png"), x: 0.6, y: 1.25, w: 7.2, h: 4.75, objectName: "Seasonal model comparison",
  altText: "Seasonal station-year model comparison" });
stat(s, 8.25, 1.35, 4.1, "0.861", "seasonal LR macro-F1", C.accent1);
stat(s, 8.25, 2.8, 4.1, "0.807", "seasonal HGB macro-F1", C.accent2);
stat(s, 8.25, 4.25, 4.1, "0.759", "four-snapshot daily vote", C.accent4);
s.addText(bullets(["Station-held-out folds; three reused partition seeds",
                   "Within-family matched gains: HGB +0.047; LR +0.159",
                   "LR is strongest: richer representation does not require a more flexible model",
                   "Daily trailing-window curves remain supporting sensitivity analyses"]),
  { x: 8.25, y: 5.05, w: 4.2, h: 1.05, fontSize: 13, color: C.text1, margin: 0, paraSpaceAfter: 4, valign: "top" });
takeaway(s, "Four seasonal summaries beat four isolated daily snapshots in the matched comparison.");
notes(s, 10);

// ---------------- Slide 11: limits ----------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Open-ended exploration" });
s.addText("Limits we tested", { placeholder: "title" });
const lim = [
  ["Coverage", "73.6%", "Only complete station-years with seven valid days in every window are retained", C.accent1],
  ["Space", "0.565", "Removing training stations within 100 km lowers daily HGB transfer", C.accent5],
  ["Labels", "1 map", "Only 4 tropical stations and one manually matched raster annotation", C.accent3],
  ["Unit", "48", "Years repeat within stations; evidence is about this network, not every site", C.accent4],
];
lim.forEach(([head, big, body, col], i) => {
  const x = 0.6 + i * 3.08, w = 2.86;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.55, w, h: 4.1, rectRadius: 0.12, fill: { color: C.background2 },
    line: { color: C.background2 }, objectName: "Limit " + head });
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.85, w: 0.42, h: 0.42, fill: { color: col }, line: { color: col } });
  s.addText(head, { x: x + 0.85, y: 1.77, w: w - 1.0, h: 0.6, fontSize: 18, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
  s.addText(big, { x: x + 0.3, y: 2.6, w: w - 0.5, h: 0.9, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: col, margin: 0, isTextBox: true });
  s.addText(body, { x: x + 0.3, y: 3.6, w: w - 0.55, h: 1.95, fontSize: 17, color: C.text1, margin: 0, valign: "top", isTextBox: true });
});
takeaway(s, "Our results describe this 48-station network, not every place in Australia.");
notes(s, 11);

// ---------------- Slide 12: conclusion ----------------
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
s.addText("Three things to remember", { placeholder: "title" });
const concl = [
  ["1", "A single day is a weather snapshot", "It is a useful baseline, not a climate normal", HEX.accent3],
  ["2", "Temporal coverage improves the representation", "Four seasonal windows reach 0.861 macro-F1 with LR", HEX.accent2],
  ["3", "Scope matters", "Station-held-out results describe this 48-station network", HEX.accent6],
];
concl.forEach(([num, head, body, col], i) => {
  const y = 1.75 + i * 1.3;
  s.addShape(pres.shapes.OVAL, { x: 0.8, y, w: 0.8, h: 0.8, fill: { color: col }, line: { color: col } });
  s.addText(num, { x: 0.8, y, w: 0.8, h: 0.8, fontFace: THEME.headFontFace, fontSize: 28, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(head, { x: 1.95, y: y - 0.05, w: 10.5, h: 0.5, fontSize: 24, bold: true, color: "FFFFFF", margin: 0, isTextBox: true });
  s.addText(body, { x: 1.95, y: y + 0.45, w: 10.5, h: 0.4, fontSize: 18, color: "CFE3E1", margin: 0, isTextBox: true });
});
s.addText("Thank you! Questions?", { x: 0.8, y: 5.75, w: 7.5, h: 0.8, fontFace: THEME.headFontFace, fontSize: 34, bold: true, color: "FFFFFF", margin: 0, isTextBox: true });
s.addText("AI tools were used for coding and writing; every use is recorded in AI_LOG.md",
  { x: 0.8, y: 6.6, w: 11.8, h: 0.45, fontSize: 14, italic: true, color: "CFE3E1", margin: 0, isTextBox: true });
const dots = dotsObjects(10.25, 6.0, 0.32, 0.18);
dots.forEach((d) => s.addShape(pres.shapes.OVAL, { ...d.ellipse }));
notes(s, 12);

(async () => {
  const out = process.env.OUT || "deck.pptx";
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
