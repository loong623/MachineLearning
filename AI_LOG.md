# AI Usage Log — DSAA2011 Task A

> 老师要求：可以自由使用 AI，但要有清楚的日志，并且 presentation 中要能证明我们真正理解了项目和代码。
> 每次使用 AI 追加一条。"我能讲清楚的要点"一栏由组员自己填写，用来准备 Q&A。

| 字段 | 说明 |
|---|---|
| 日期 | YYYY-MM-DD |
| 组员 | 谁在使用 |
| 工具 | 例如 Claude (Anthropic) |
| 目的 | 问了什么或让 AI 做什么 |
| AI 产出 | 文件或代码的位置 |
| 我们的修改与验证 | 改了什么，怎么确认是对的 |
| 我能讲清楚的要点 | 组员自己填 |

---

## 2026-11-09 #1 项目要求分析
- **组员**：m1368
- **工具**：Claude (Anthropic)
- **目的**：阅读老师的项目 PDF，用中文解释要求、交付物和评分点，并给出 3 周的时间计划。
- **AI 产出**：聊天中的分析（没有生成文件）。
- **我们的修改与验证**：对照 PDF 原文核对了截止日期（11-30 提交，12-01 展示）和交付物清单。
- **我能讲清楚的要点**：（待填）

## 2026-11-09 #2 设计点头脑风暴
- **组员**：m1368
- **工具**：Claude (Anthropic)
- **目的**：汇总 PDF 中所有需要我们自己设计的地方，每个设计点给出至少 5 个候选，并从多个维度比较。分类目标不想用老师课上举过的 RainTomorrow。
- **AI 产出**：
  - `BRAINSTORM.md`：16 个设计点（D1–D16）和候选方案。
  - `explore/01_target_feasibility.py` 及其输出：统计数据规模、缺失率，以及 9 个候选目标的正例比例。
  - `explore/02_quick_baselines.py` 及其输出：在 4 万条样本上用 LR 和 DT 快速测试各候选目标的难度。
- **我们的修改与验证**：
  - 数据来自 GitHub 镜像的旧版 weatherAUS（142,193 行，含会泄漏答案的 `RISK_MM` 列），只用于可行性评估。正式实验要换成 Kaggle 官方的当前版本。
  - 气候区映射（49 个站点 → 5 个区）是 AI 按经验给出的粗分，**正式使用前要对照 BoM/Köppen 来源逐个核对**。
  - 简化 FFDI 中的 Drought Factor 固定取 8，是一个假设，需要在 report 里说明。
- **我能讲清楚的要点**：（待填）

## 2026-11-09 #3 讨论：要不要加二分类副线
- **组员**：m1368
- **工具**：Claude (Anthropic)
- **目的**：追问"A 主线 + B/C 副线"是什么意思，会不会让故事断开，Task 5 是否一定需要二分类。
- **AI 产出**：聊天解释；`BRAINSTORM.md` 中 D1 的推荐方案改为"只做 A"。
- **我们的修改与验证**：确认 PDF 的 Task 5 要求（metrics、ROC/AUC、validation）都能用多分类完成，ROC 用 one-vs-rest。
- **我能讲清楚的要点**：（待填）

## 2026-11-09 #4 确定方案 A；获取官方数据；建立 Köppen 标签和泄漏审计
- **组员**：m1368
- **工具**：Claude (Anthropic)
- **目的**：根据另一位同学（"高手"）提出的四点意见（标签来源要可复现、排除泄漏列、按天聚类必须做、随机划分和站点泄漏要同时处理），开始正式任务。
- **AI 产出**：
  - 用 `kagglehub` 下载 Kaggle 官方 v2 数据到 `data/raw/weatherAUS.csv`（145,460 × 23，没有 `RISK_MM`，SHA256 记录在 `src/config.py`）。
  - 从 BoM FTP 下载站点列表（`data/external/bom_stations_IDCJMC0014_2026-11-09.zip`）。
  - `data/external/location_station.csv`：AI 为 49 个 Location 选定了对应的 BoM 站号。规则是：优先选覆盖 2007–2017 的机场站或主站；Watsonia 对应 Viewbank，Wollongong 对应 Bellambi，GoldCoast 对应 Seaway。
  - `src/build_zone_labels.py`：用 kgcpy（Rubel et al. 2017 的地图）查询 Köppen 代码，再映射到 5 类，生成 `data/processed/location_zone.csv`。
  - `src/config.py`、`src/data.py`：目标定义、泄漏排除列表，以及带校验的数据加载函数。
  - `DECISIONS.md`（DEC-01 到 DEC-05）、`README.md`。
- **我们的修改与验证**：
  - 脚本里有断言，检查站号能匹配上、站名一致；加载时打印泄漏审计结果。实际结果：删除了 Location 和 RainTomorrow，没有发现 RISK_MM；NorfolkIsland 没有标签（3,009 行）。
  - **待组员人工核对**：49 个站号的选择是否合理。地图分辨率约 3 km，同一城市内换一个站点一般不影响结果，但仍需抽查。
  - **待组员决定**：NorfolkIsland 的处理方式（目前是排除）。
- **我能讲清楚的要点**：（待填。比如：为什么 5 类属于同一个分类层级？为什么年份不能作为特征？missing indicator 为什么会泄漏站点身份？）

## 2026-11-09 #5 Task 1：预处理和 EDA
- **组员**：m1368
- **工具**：Claude (Anthropic)
- **目的**：完成 Task 1（缺失值、非数值编码、标准化和观察总结），并开始写正式的 notebook。
- **AI 产出**：
  - `tools/build_notebook_part1.py`，由它生成 `project_XX_TaskA.ipynb` 的 Part 0–1（共 26 个 cell，在 nbconvert 中从头执行无报错，耗时约 40 秒）。
  - `figures/p1_*.png`，共 7 张图：站点地图、类别分布、缺失热力图、Rainfall 的 log 变换、季节循环、箱线图、相关矩阵。
  - `DECISIONS.md` 新增 DEC-06（预处理方案）和 DEC-07（notebook 自包含），并更正了 DEC-02 中不用年份的理由。
- **我们的修改与验证**：
  - 所有数字都来自 notebook 的实际输出，Markdown 里的结论已经和输出逐条核对。
  - 发现并改正了 AI 自己之前的一个错误假设：年份其实不能识别站点（实测准确率 0.323，基线 0.322）。
  - 箱线图的 x 轴标签重叠，改用普通 subplots 重画。
- **我能讲清楚的要点**：（待填。比如：为什么风矢量比 one-hot 好？为什么 indicator 只在随机划分时有用？什么叫"站点指纹"？）

## 2026-11-10 #6 Part 2：t-SNE
- **组员**：m1368
- **工具**：Claude (Anthropic)
- **目的**：完成 Task 2（t-SNE 的 2D/3D 可视化、着色、模式分析）。
- **AI 产出**：
  - `tools/build_notebook_part2.py`（17 个 cell）；新增 `tools/build_notebook.py` 负责拼接各部分，part1 改为只提供 cells，并补充了相应的 import。
  - `figures/p2_*.png`，共 7 张图：主图（按 zone、月份、温度三种着色）、邻域混淆矩阵、季节对照、perplexity 扫描、随机种子对比、PCA 和 t-SNE 对比、3D 图。
  - `data/processed/tsne/`：10 个 embedding 缓存。
  - `DECISIONS.md` 新增 DEC-08。
- **我们的修改与验证**：
  - 先在临时脚本里试跑，确认各项数字，再写进 notebook。Markdown 里的每个结论都和 notebook 的输出逐条核对过（例如 96%、0.50–0.64、约 15 倍、0.81 和 0.985）。
  - 看过季节对照图后，补充了一条观察：冬季热带完全独立，夏季干旱区单独分到一边。
  - 整个 notebook 共 43 个 cell，有缓存时从头执行无报错，耗时约 50 秒。没有缓存时，t-SNE 部分还要额外跑约 5 分钟。
- **我能讲清楚的要点**：（待填。比如：perplexity 是什么意思？为什么 t-SNE 图上簇之间的距离没有意义？k-NN purity 怎么算？为什么要和 PCA 对比？）

## 2026-11-10 #7 Part 3：聚类
- **组员**：m1368
- **工具**：Claude (Anthropic)
- **目的**：完成 Task 3，包括至少两个算法、多种指标、可视化、选出最优结果并论证，以及算法之间的比较。
- **AI 产出**：
  - `tools/build_notebook_part3.py`（18 个 cell），`build_notebook.py` 的 PARTS 中加入了 part3。
  - `figures/p3_*.png`，共 5 张图：k 选择曲线、t-SNE 上的聚类结果、簇画像加交叉表、站点 dendrogram、站点聚类地图。
  - `DECISIONS.md` 新增 DEC-09。
- **我们的修改与验证**：
  - 先用临时脚本试跑全部 k 值和稳定性，再写进 notebook。
  - 发现 GMM 的 silhouette 接近 0 之后，没有直接放弃这个算法，而是检查了各分量的方差，证实问题来自"无雨日"这个点质量。
  - 根据实际输出修正了几处文字：ARI 的上限写成 0.13（不是笼统的约 0.1）；簇 1 的季节改为"冬季和秋季"；补充写明站点簇 1 包含哪些站点。
  - notebook 共 61 个 cell，从头执行无报错，约 95 秒。
- **我能讲清楚的要点**：（待填。比如：silhouette 和 Davies–Bouldin 的公式含义；ARI 为什么要"adjusted"；Ward linkage 和 K-means 的关系；为什么按天聚类找不到气候区、按站点聚类却可以？）

## 2026-11-12 #8 Part 4、Part 5、Open-ended、requirements、report 初稿
- **组员**：m1368
- **工具**：Claude (Anthropic)
- **目的**：用户要求"一口气做完剩下全部"（presentation 除外）。
- **AI 产出**：
  - `tools/build_notebook_part4.py`、`part5.py`、`part6.py`；notebook 现在共 111 个 cell，从头执行无报错，有缓存时约 85 秒。
  - Part 1 和 Part 3 中耗时的计算改成用 `cached(...)` 缓存；新增 `tools/run_cells.py`。
  - `figures/p4_*`、`p5_*`、`p6_*`，共 14 张新图。
  - `requirements_XX_TaskA.txt`；`report/report_XX_TaskA.tex` 及编译出的 PDF（9 页）。
  - `DECISIONS.md` 新增 DEC-10 到 DEC-12。
- **我们的修改与验证**：
  - 先用临时脚本试跑并计时，再写进 notebook。所有 markdown 和 report 里的数字都和 notebook 输出逐条核对过。核对时改正了 AI 初稿中多处与实际结果不符的说法，例如："Tropical recall ≥ 0.9"（实际是 0.77–0.92）；"Arid 主要被误判成 Mediterranean"（实际主要是 Subtropical）；"树的验证曲线在峰值后下降"（只在 leaf = 1 时成立，所以改为同时画 leaf = 1 和 leaf = 20 两条线）；"多尺度特征最好"（实际是 14 天窗口的 macro-F1 最高）；"polynomial 在未见站点上提升很小"（实际 +0.10）；错误分析中最差站点的名单。
  - 发现 sklearn 1.7.2 的 MLP 在字符串标签加 early_stopping 时会报错，改为不用 early stopping、max_iter = 60。
  - 随机搜索的单次调用超过了沙箱的时间限制，改写成逐个配置缓存的等价循环。
  - **待组员完成**：替换课程的 LaTeX style 文件；填写 groupID 和成员贡献；在本机的干净环境里验证 requirements；人工通读 report。
- **我能讲清楚的要点**：（待填。比如：为什么 entire set 不是独立的估计？为什么默认树的 AUC 比 LR 低？为什么调参要用分组 CV？什么是 distribution shift？为什么 365 天均值反而变差？为什么滚动窗口特征在随机划分下会泄漏？）

## 2026-11-16 #9 核查同学对 report 的审阅意见
- **组员**：m1368
- **工具**：Claude (Anthropic)
- **目的**：逐条评估一位同学对 `report_XX_TaskA.pdf` 的审阅意见（标签可审计性、随机划分的时间相关性、t-SNE 结论过度、聚类选择规则、滚动窗口定义、open-ended 实验细节），这一轮只核查，不改代码。
- **AI 产出**：聊天中的逐条评估，以及下面几项核查结果（都没有改动项目文件）：
  - 地图来源：kgcpy 内置的栅格是 12960×6480，即 100″/像素，已核实。CRAN kgc 包把 Rubel et al. (2017, doi:10.1127/metz/2016/0816) 列为数据来源；Köppen 官网说明全球重分析图采用 Rubel et al. (2017) 的降尺度方法，代表 1986–2010 年。所以引用的论文没错，但 report 需要补引 kgc 包和官网，说明这一点。
  - Norfolk Island：在 3×3 到 41×41 像素的窗口内全是 Ocean，最近的陆地像素在约 790 km 外（新西兰 North Cape）。说明这个岛在栅格里根本不存在，不是坐标匹配出错。
  - t-SNE：用 Cohen's kappa 做机会校正后，t-SNE 图上 zone 是 0.539、season 是 0.550，基本相等。原文说 "season purity 高于 zone purity" 不成立。而且季节特征（doy_sin/cos）本身就是模型输入，季节可分是理所当然的。
  - 分组 CV 的逐折结果：LR 0.483 ± 0.108，DT 0.487 ± 0.092，HGB 0.552 ± 0.113，HGB 14 天均值 0.636 ± 0.119。逐折平均明显低于合并后的 OOF（0.553 / 0.557 / 0.638 / 0.727），原因是有些折缺少 Tropical 等类别，macro-F1 会被拉低。这也解释了 6.3 节中 "grouped CV 0.48 vs 测试 0.63" 的差距，原来那段解释不完整。
- **我们的修改与验证**：待组员决定是否按修改计划执行。
- **我能讲清楚的要点**：（待填。比如：为什么逐折平均的 macro-F1 会低于合并后的 OOF？为什么季节可分不能说明"季节主导"？）

## 2026-11-16 #10 按审阅意见修改 notebook 和 report
- **组员**：m1368
- **工具**：Claude (Anthropic)
- **目的**：执行 #9 中的修改计划第 1–6 项（第 7 项"换一张地图做敏感性检查"未做）。
- **AI 产出**：
  - 修改了 `tools/build_notebook_part1/2/3/5/6.py`。notebook 现在共 125 个 cell，从头执行无报错。新增的缓存放在 `data/processed/cache/`（p0_norfolk、p3_stab_*、p3_gmm_sensitivity、p6_time_*、p6_oof_seed*）。
  - 新增图：`figures/p3_stability.png`、`figures/p6_protocols.png`。
  - `report/report_XX_TaskA.tex` 和 PDF：修改了摘要、引言、2.1–2.5 节和第 3 节，新增 Table 4（三种协议）、Fig. 9，以及附录 Table 5（映射表）、Table 6（超参数）、Fig. 11（k 选择曲线），补充了引用。共 11 页，其中参考文献约 0.75 页。
  - `DECISIONS.md` 新增 DEC-13。
- **我们的修改与验证**：
  - report 里的每个新数字都和 notebook 输出逐条核对过。
  - **更正 #9 的一个数字**：#9 里说 Norfolk 离最近的陆地像素"约 790 km"，那是用粗略方法估算的；改用 haversine 公式按最近陆地像素计算，实际是 736 km。
  - 发现 AI 最初实现的 Ward 稳定性（bootstrap 加最近质心分配）本身就有偏差：只做最近质心这一步，ARI 就降到 0.49。因此三个算法统一改用 80% 子采样、并只在共有的样本上比较。改用新方法后，Ward 依然不稳定（0.26–0.36），这个结论成立。
  - 聚类的选择规则是在第一版分析之后才写成明确规则的。notebook 和 report 都如实写明了这一点；按这条规则选出的结果和原来一致。
  - seed 42 的分组折偏乐观 0.01–0.04，现在以 3 个 seed 的结果作为主要数字。
  - **待组员**：换上课程 style 文件后重新检查页数；目前排除参考文献后约 10 页。
- **我能讲清楚的要点**：（待填。比如：为什么配对比较比单看均值 ± 标准差更有说服力？为什么时间切分要设 embargo？为什么预先计算滚动特征在分组 CV 下不算泄漏？κ 和 purity 有什么区别？）

## 2026-11-16 #11 report 换成课程的 NeurIPS 2026 格式
- **组员**：m1368
- **工具**：Claude (Anthropic)
- **目的**：用老师提供的格式包（`Formatting_Instructions_For_NeurIPS_2026.zip`）重新生成 report。
- **AI 产出**：解压格式包，把 `neurips_2026.sty` 复制到 `report/`（与包内文件逐字节一致）。用课程 style 重新编译 `report/report_XX_TaskA.pdf`（11 页），并做了检查：没有 Overfull box，没有 Type 3 字体，所有字体都已嵌入，首页底部显示的是课程信息。另外新增了 `DECISIONS.md` 中的 DEC-14。
- **我们的修改与验证**：
  - 开始工作时发现，`report_XX_TaskA.tex` 在 22:15 左右**已经被转换成 NeurIPS 格式**（同时生成了备份 `report/old/`）。这次转换没有记录在本日志中，**请组员补记是谁、用什么工具做的**。
  - AI 当时没有发现这一点，先用自己写的导言区覆盖了这份文件，随后察觉，又**恢复了原来的转换版本**。正文部分全程没有被改动。现在的 tex 就是那份转换版本，AI 只负责了编译和检查。
  - AI 新建了 `report/article_draft/`，里面是转换前的 article 版本，内容和 `report/old/` 相同，属于重复备份，可以删除。
- **我能讲清楚的要点**：（待填。比如：为什么用 `[main, final]` 而不用默认选项或 preprint？为什么要改首页底部的说明文字？）

## 2026-11-17 #12 保留主线，按严格评审重新设计并完整重跑
- **工具**：OpenAI Codex。
- **用户授权与范围**：保留“天气观测识别气候分组”的主线，改进此前评审指出的问题，重做实验和报告，完成后统一提交 GitHub。原始数据和人工站点标注不变；保留原有日志、决策记录和报告格式转换备份。
- **主要改动**：
  - 固定五类 pooled OOF macro-F1；站点表分层四折，三个划分种子，内层三折调参；断言类覆盖与站点隔离。
  - 九种日级配置统一做随机、时间和重复站点验证。LR/树/HGB 的参数搜索在每个外层训练集内重新执行。
  - LR/HGB 的 1/3/7/14/30/90/365 天曲线与内层选窗口程序分开评估；补充有效历史与共同历史样本检查。
  - 特征排序和特征数选择分别使用外层训练内部的 A/B/C 站点集，最后重拟合，再预测外层测试站点。
  - 配对整站 bootstrap、站点等权指标、逐站错误、100 km 空间缓冲、边界标注与缺失值策略敏感性。
  - 聚类补充评价规则、自然频率抽样和整站/整年扰动检查；撤回因果归因、独立重复与普遍最优窗口等过强表述。
  - 新建 Python 3.12.6 隔离环境，完整锁定依赖；缓存指纹包含全部可执行单元、数据、标注和实装环境。保存预测、划分、诊断及机器可读结果。
  - 报告从结果文件生成数值和表格；增加适合正文尺寸的图，Notebook 末尾生成实际结果解读。四张不再使用的旧图归入历史目录；停止跟踪 Python 字节码。
- **运行中发现并修复的问题**：
  - Windows 临时文件写入异常：运行器改为原子检查点保存与有限重试。
  - scikit-learn 1.7.2 的 MLP 早停对字符串标签执行 `isnan` 而报错。用可逆整数标签封装解决；小规模对照验证预测、概率列、迭代次数和 estimator cloning 与手动编码完全一致。没有修改第三方库；修复后重新开始完整 `--fresh` 运行。
- **解释范围**：重做属于评审后的分析，不是预注册或独立外部验证；有限站点、单一栅格、历史探索和条件区间的限制继续保留。未编造成员身份或贡献比例。

- **完整执行与核验**：最终新指纹为 `0ed824d64a3cdbe2`，运行 6224.1 秒（约 104 分钟），104 个单元中 60 个代码单元全部执行成功。335 项缓存计算、0 次结果缓存命中。`verify_results.py` 通过代码/数据哈希、各折类覆盖和站点隔离、A/B/C 特征选择隔离、内层选择决策及保存预测的指标重算。126 条日级拟合记录均无捕获警告；MLP 14 次拟合为 92–252 轮。`pip check` 通过。
- **实际结论更正**：日级主分数 RF 0.665、HGB 0.659，不能宣称 HGB 明确更优；seed 42 在九种配置上均最乐观，报告同时显示重复均值和范围。内层选窗口 LR/HGB 为 0.745/0.758；HGB 相对单日的条件配对提升区间为 [0.066, 0.131]。全年 HGB 在共同历史样本上为 0.782，撤回统一长窗口退化论。训练内特征选择为 0.709，低于全部特征的 0.754。100 km 缓冲下日级 HGB 为 0.565；Nhil/Moree/SalmonGums 的历史模型站点准确率分别约 2.0%/11.2%/15.8%。
- **报告交付核验**：按课程 NeurIPS 2026 格式生成 10 页 PDF（正文 9 页，参考文献与 Credit 在末页），规定讨论为 179 词。53 个数值/文本宏及表格从本轮 CSV 生成。Tectonic 编译成功，无 Overfull box、无未解析引用，14 个字体全部嵌入且无 Type 3。渲染并检查正文图表与页面；混淆矩阵正文图使用可读的归一化比例，完整计数保留在 Notebook/CSV。报告绘图重拟合的基础模型指标与实验输出一致。

## 2026-11-19 #13 Presentation slides 与中英讲稿
- **组员**：m1368
- **工具**：Claude (Anthropic)
- **目的**：根据 2026-11-17 重跑后的最新结果（`results/*.csv`、`report/report_XX_TaskA.pdf`）制作 12 分钟的 presentation，并提供中英对照讲稿，帮助组员理解项目。
- **AI 产出**（都在 `presentation/` 文件夹）：
  - `presentation_XX_TaskA.pptx` 和 `.pdf`：共 12 页，备注栏里有英文讲稿和主讲人分配。模型对比图和窗口曲线是 PowerPoint 原生图表，数据取自 `model_comparison.csv` 和 `window_comparison.csv`（seed 42）；其他图片直接用 `figures/` 里的现有图。
  - `讲稿_中英对照_XX_TaskA.docx`：包含时间与分工表、逐页中英讲稿、术语表、10 个可能被问到的问题及中英回答，以及 5 个可以问其他组的问题。
  - 生成脚本：`build.js`、`script.js`、`build_script_doc.js`。
- **我们的修改与验证**：
  - 幻灯片上的所有数字都和 `results/*.csv`、`report_values.json` 以及 report 中的表格核对过。
  - 英文讲稿约 1,120 词：按 110 词/分钟约 10 分钟，按 95 词/分钟约 12 分钟。
  - **待组员**：填写组号和姓名，确认主讲分工（A/B/C/D），按讲稿排练并计时。
- **我能讲清楚的要点**：（待填）

## 2026-11-20 #14 按季节画像主线完成重设计与交付准备

- **用户授权与范围**：用户要求保留单日实验作为基线和动机，把四季窗口天气画像作为主线，重新设计并完整运行实验，最后提交 GitHub。未臆造组员贡献比例。
- **实验与交付**：完成 48 站、142451 条日记录的完整 Notebook；构造 328 个 station-year 季节画像（四个 21 天窗口，平均完整覆盖率 73.6%）；重写季节主线报告和 12 页演示文稿；生成 `submission_7_TaskA.zip`。
- **代码修复**：修复季节模型内层选择的索引隔离问题和 Windows `.npz` 原子写入问题；把工具脚本的默认 Notebook 路径统一为 `project_7_TaskA.ipynb`。
- **验证**：`tools/verify_results.py` 通过代码、数据、站点隔离、特征选择隔离、内层选择和预测指标重算检查；Notebook 代码单元全部执行成功；报告使用 Tectonic 编译成功并完成渲染检查。
- **关键结果**：日级 RF/HGB macro-F1 为 0.665/0.659；季节 LR 为 0.861、macro-AUC 为 0.973；季节 HGB 为 0.807；四季中心日模型投票为 0.759；季节 K-means 最佳外部 ARI 为 0.314。结果和限制已同步到 README、报告和演示文稿。

## 2026-11-20 #15 独立建议复核与报告修订

- **复核结论**：建议 1–4 均有可参考之处。修正后的同模型 matched baseline 是日 LR 0.702 → 季节 LR 0.861（+0.159），日 HGB 0.759 → 季节 HGB 0.807（+0.047）；撤回跨模型的 0.101 like-for-like 说法。
- **报告补充**：加入 Task 4 train/test/entire 表、随机混淆矩阵、二维决策边界和 tuned LR/DT 验证表；补充 12 折季节 LR `0.850±0.085`、范围 `0.680–0.975`，并说明条件 bootstrap 区间的限制。
- **聚类修正**：报告明确 k=7 的季节 ARI 是看过外部标签后的 post-hoc 描述，内部指标更偏好 k=2。
- **可视化修正**：将混合单位季节特征图改为逐特征标准化热图；同时修复 LaTeX 宏后的空格。
- **实现**：新增 `tools/compute_matched_votes.py` 和 `results/seasonal_matched_baseline.csv` 中的 LR 四快照投票行；Notebook 增加同模型比较说明；presentation 更新同模型提升文字；报告重新编译成功。
