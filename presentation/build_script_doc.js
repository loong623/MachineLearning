// Bilingual speaker script + Q&A preparation (Word document).
const fs = require("fs");
const D = require("docx");
const SCRIPT = require("./script.js");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, LevelFormat, BorderStyle, PageBreak } = D;

const FONT = "Calibri";
const ZH = "Microsoft YaHei";
const run = (text, o = {}) => new TextRun({ text, font: { ascii: FONT, hAnsi: FONT, eastAsia: ZH }, size: 22, ...o });
const p = (children, o = {}) => new Paragraph({ children: Array.isArray(children) ? children : [run(children)], spacing: { after: 120 }, ...o });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [run(t, { size: 32, bold: true, color: "123A40" })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [run(t, { size: 26, bold: true, color: "0F7C80" })] });
const bullet = (children) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 80 },
  children: Array.isArray(children) ? children : [run(children)] });
const label = (t, color) => run(t, { bold: true, color, size: 20 });

const W = 9026; // A4 text width with 1" margins
const cell = (text, w, o = {}) => new TableCell({ width: { size: w, type: WidthType.DXA },
  shading: o.head ? { type: ShadingType.CLEAR, fill: "0F7C80", color: "auto" } : undefined,
  margins: { top: 60, bottom: 60, left: 100, right: 100 },
  children: [new Paragraph({ children: [run(text, { bold: !!o.head, color: o.head ? "FFFFFF" : "1F2A30", size: 20 })] })] });
const table = (rows, widths) => new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths,
  rows: rows.map((r, i) => new TableRow({ tableHeader: i === 0, children: r.map((t, j) => cell(t, widths[j], { head: i === 0 })) })) });

const kids = [];
kids.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [run("从天气快照到季节气候画像：演讲稿（中英对照）", { size: 40, bold: true, color: "123A40" })] }));
kids.push(p("DSAA2011 Task A · 12 分钟 · Group 7 · 对应 presentation_7_TaskA.pdf"));

kids.push(h1("1. 时间与分工"));
kids.push(p("英文讲稿共约 1,120 词：按 110 词/分钟约 10 分钟，按非母语常见的 95 词/分钟约 12 分钟，留有少量缓冲。四人每人 3 页、约 3 分钟。如果只有 3 人，C 同时讲 Slide 10。"));
kids.push(table([["Slide", "主讲", "时长", "内容"],
  ...SCRIPT.map((s) => [String(s.n), s.speaker, s.time, s.title])], [900, 900, 1000, 6226]));
kids.push(p(""));
kids.push(h2("给非母语演讲者的建议"));
[ "讲稿刻意用短句和常用词。不要背诵，记住每页底部的 Key message，按它来讲即可。",
  "数字慢读：0.659 读作 “zero point six five nine”；macro-F1 读作 “macro F one”；ARI 读作 “A-R-I”；Köppen 读作 “KER-pen”。",
  "换人时用固定的交接句：“Now [name] will talk about …” / “Thank you. Next, …”。",
  "每页先指图、再说结论，例如 “On the left, … ”, “The key message is …”。",
  "整场排练两次并计时。超时时，Slide 5 和 Slide 9 的最后一两句可以省略。" ].forEach((t) => kids.push(bullet(t)));

kids.push(new Paragraph({ children: [new PageBreak()] }));
kids.push(h1("2. 逐页讲稿"));
SCRIPT.forEach((s) => {
  kids.push(h2(`Slide ${s.n} · ${s.title}（${s.speaker} · ${s.time}）`));
  kids.push(p([label("EN  ", "0F7C80"), run(s.en)]));
  kids.push(p([label("中文  ", "D1495B"), run(s.zh)], { spacing: { after: 240 } }));
});

kids.push(new Paragraph({ children: [new PageBreak()] }));
kids.push(h1("3. 关键术语"));
kids.push(table([["英文", "中文", "一句话解释"],
  ["macro-F1", "宏平均 F1", "每一类单独算 F1 再取平均，五类权重相同，不被大类主导"],
  ["unseen / withheld station", "未见站点", "训练时完全没出现过的站点；我们的主要考题"],
  ["station-grouped split / fold", "按站点分组划分", "同一站点的所有天只会在训练或测试的一边"],
  ["nested (inner) CV", "嵌套交叉验证", "调参、选窗口、选特征都只在训练站点内部再做一次 CV"],
  ["OOF (out-of-fold) prediction", "折外预测", "每个站点由没见过它的模型预测，合起来算一个总分"],
  ["leakage", "数据泄漏", "模型拿到了本不该有的答案线索，例如站点身份"],
  ["overfitting", "过拟合", "训练集很好、测试集差很多，例如树 1.00 对 0.65"],
  ["one-vs-rest ROC / AUC", "一对其余 ROC", "每一类当成“这一类 vs 其他”画一条曲线；AUC 衡量排序好坏"],
  ["trustworthiness", "可信度", "低维图里的邻居是否也是原空间里的邻居（1 最好）"],
  ["silhouette / Davies–Bouldin", "轮廓系数 / DB 指数", "簇内紧、簇间远的程度；silhouette 越大越好，DB 越小越好"],
  ["ARI", "调整兰德指数", "两种分组的一致程度，0 表示随机，1 表示完全一致"],
  ["trailing N-day mean", "过去 N 天均值", "同一站点包含当天在内过去 N 天的平均，不用未来数据"],
  ["HGB / RF / MLP", "梯度提升树 / 随机森林 / 神经网络", "三种较灵活的模型"],
  ["conditional bootstrap interval", "条件自助区间", "整站重抽样得到的区间；不包含重新训练带来的波动"],
], [2600, 2200, 4226]));

kids.push(new Paragraph({ children: [new PageBreak()] }));
kids.push(h1("4. Q&A 准备：别人可能问我们的问题"));
const QA = [
  ["If the map already gives the zone, why predict it from weather?",
   "Because our question is how much climate information is inside weather data. Map lookup is easier, but it does not answer this question.",
   "因为我们研究的是“天气数据里含有多少气候信息”。查地图当然更简单，但回答不了这个问题。"],
  ["Why macro-F1 and not accuracy?",
   "The classes are imbalanced, from 7.6% to 32%. Macro-F1 gives each climate group the same weight, so the small tropical group still counts.",
   "类别不平衡，占比从 7.6% 到 32%。macro-F1 让每一类权重相同，小的热带类也同样重要。"],
  ["Why split by station? The course asks for 70/30.",
   "We did the required 70/30 split too. But the label belongs to the station, so a random split lets the model recognise stations it has already seen. The station split tests new places.",
   "要求的 70/30 我们也做了。但标签属于站点，随机划分会让模型认出见过的站点；按站点划分测试的是新地点。"],
  ["Is the N-day average a leak?",
   "No. It uses only past weather of the same station, never labels or future days. In station-split folds a station is fully in training or fully in test.",
   "不是。它只用同一站点过去的天气，不用标签，也不用未来数据。按站点划分时，一个站点要么全在训练集，要么全在测试集。"],
  ["Why K-means with k = 3, if k = 2 has a higher silhouette?",
   "We required stability of at least 0.9 and then took the best average rank of three metrics. k = 3 had the lowest Davies–Bouldin. We admit this rule is not unique: with only two metrics, k = 2 wins.",
   "我们要求稳定性至少 0.9，再看三个指标的平均名次。k=3 的 Davies–Bouldin 最低。我们也承认规则不唯一：只用两个指标时 k=2 会胜出。"],
  ["Which model is the best?",
   "Random forest has the highest point estimate, 0.665, but HGB and MLP are within 0.01, and the intervals include zero. So we say they are tied.",
   "随机森林点估计最高（0.665），但 HGB 和 MLP 只差不到 0.01，区间包含零，所以我们说它们打平。"],
  ["Why is the Arid group so hard?",
   "Some arid stations, like Nhil, are cool semi-arid, not hot desert. And some subtropical inland stations, like Moree, have hot dry summers. So the groups overlap.",
   "有些干旱站点（如 Nhil）是凉爽的半干旱，不像炎热的沙漠；而有些内陆亚热带站点（如 Moree）夏天又热又干，所以两类重叠。"],
  ["Why does the 100 km buffer lower the score so much?",
   "Two things change at once: nearby similar stations are removed, and the number of training stations falls from 36 to 22–25. So it is not a pure measure of space.",
   "同时变了两件事：附近相似的站点被去掉，训练站点数也从 36 降到 22–25，所以它不是纯粹的空间效应。"],
  ["Why not use t-SNE coordinates as features?",
   "t-SNE cannot place new points, and it used all points at once. It is only a picture for exploration.",
   "t-SNE 无法给新点定位，而且用到了所有样本，所以只用来做探索性的图。"],
  ["How did you use AI?",
   "We used AI tools for coding, debugging and writing. Every use is in our AI log, and we checked every number against the notebook output.",
   "我们用 AI 工具写代码、调试和写作，每次使用都记录在 AI 日志里，所有数字都和 notebook 输出核对过。"],
];
QA.forEach(([q, en, zh], i) => {
  kids.push(p([run(`Q${i + 1}. ${q}`, { bold: true, color: "123A40" })], { spacing: { before: 120, after: 60 } }));
  kids.push(p([label("EN  ", "0F7C80"), run(en)], { spacing: { after: 40 } }));
  kids.push(p([label("中文  ", "D1495B"), run(zh)]));
});

kids.push(h1("5. 我们可以问其他组的问题（至少 2 个）"));
kids.push(p("课程规定：提出 2 个以上“好问题”可以加分。下面这些问题都是在问方法选择，不是在问外观。"));
[ ["How did you handle class imbalance, and why did you choose that metric?", "你们怎么处理类别不平衡？为什么选这个指标？"],
  ["Could the same station, or neighbouring days, be in both your training and test sets?", "同一站点或相邻日期会不会同时出现在训练集和测试集里？"],
  ["Which metric decided your number of clusters, and does the choice change with another metric?", "你们的簇数由哪个指标决定？换一个指标结论会变吗？"],
  ["Is your best model clearly better, or is the difference within the noise?", "你们最好的模型是明显更好，还是差距在误差范围内？"],
  ["Why is your entire-set score higher than your test score, and which one should we trust?", "为什么全集分数比测试集高？应该相信哪一个？"],
].forEach(([en, zh]) => kids.push(bullet([run(en), run("  —  " + zh, { color: "4A5A60" })])));

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true }, paragraph: { spacing: { before: 300, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true }, paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 40, bold: true }, paragraph: { spacing: { before: 0, after: 160 } } },
    ],
  },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: kids }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(process.env.OUT, b); console.log("wrote", process.env.OUT); });
