# 设计决定记录（Decision Log）

> 这里只记录已经定下来的决定和理由。候选方案的比较见 `BRAINSTORM.md`。每次改动决定都要追加新条目，旧条目不删。

## 总体问题（2026-11-09 定）

**唯一分类目标：根据一天的天气观测，判断它来自哪个气候区（Zone，5 类）。** 不加二分类副线。

故事线分四步：

1. **t-SNE**：天气样本在高维空间里会不会按气候区自然分开？
2. **聚类**：不提供标签时，算法能不能重新发现这些气候区？
3. **分类**：提供标签后，模型能多准确地从一天的天气判断气候区？
4. **泛化**：换到训练时从未见过的站点，结论是否依然成立？

---

## DEC-01 气候区标签：来源和定义

**来源**（完全可复现，脚本是 `src/build_zone_labels.py`）：

- 站点坐标来自澳大利亚气象局（BoM）的站点列表 IDCJMC0014，2026-11-09 下载，原始压缩包存放在 `data/external/`。数据集里的 49 个 Location 对应哪个 BoM 站号，是人工匹配的，见 `data/external/location_station.csv`。
- Köppen-Geiger 代码由 `kgcpy` 1.1.8 查询得到。这个包是 CRAN `kgc` 包的 Python 版本，按包文档的说法，它查询的是 Rubel et al. (2017) 的 100 角秒（约 3 km）分辨率地图，参考时期为 1986–2010。

**5 类的定义**：全部用 Köppen 代码的并集来定义，同属一个分类体系。

| Zone | Köppen 代码 | 含义 | 站点数 | 行数（占比） |
|---|---|---|---|---|
| Tropical | A 组（Am、Aw） | 热带 | 4 | 10,851（7.6%） |
| Arid | B 组（BWh、BSh、BSk） | 干旱和半干旱 | 7 | 18,224（12.8%） |
| Subtropical | Cfa | 湿润亚热带：夏季炎热，全年有雨 | 15 | 45,802（32.2%） |
| Mediterranean | Csa、Csb | 地中海式：夏季干燥 | 11 | 33,526（23.5%） |
| Oceanic | Cfb | 温带海洋性：夏季温和，全年有雨 | 11 | 34,048（23.9%） |

A 组和 B 组整组保留。C 组按第 2 个字母（降水季节性：f 表示全年有雨，s 表示夏季干燥）和第 3 个字母（夏季温度：a 表示炎热，b 表示温和）细分，这正好是 Köppen 体系本身的下一级划分。49 个站点里出现的 C 类代码只有 Cfa、Cfb、Csa、Csb 四种。

**特殊情况**：

- **NorfolkIsland 没有标签，已排除（共 3,009 行）**。这个岛太小，在 3 km 的网格上被识别成 "Ocean"。为了保证标签来源单一、可复现，我们不手动补标签。
- **Wollongong 是边界站点**：中心像素是 Cfa，周围 3×3 窗口内有 Cfb 像素。`zone_boundary` 列标记了这种情况，错误分析时要重点看。
- 和之前的经验粗分相比，有不少站点改了类别：Moree、WaggaWagga、Albury 是 Cfa，不是干旱区或温带；Nhil 是 BSk；Dartmoor、Portland、MountGambier 是 Csb，不是温带。这正说明需要一个权威的来源。

**已知局限**（写进 report）：

1. 地图的参考时期是 1986–2010，而数据是 2007–2017。
2. 不同版本的 Köppen 地图在维州西南和南澳东南一带（Cs 和 Cfb 的交界处）可能给出不同的结论。可以选做一个敏感性检查：换成 Beck et al. (2018) 的地图，看有多少站点的类别会变。

## DEC-02 泄漏审计

见 `src/config.py` 和 `src/data.py`，每次加载数据时都会打印审计结果。

- **硬排除**：`Location`（气候区就是由站点决定的）、`RainTomorrow`（来自第二天的信息）、`RISK_MM`。在 Kaggle v2 版本里没有 `RISK_MM`，但加载时仍然会检查。
- **不能作为原始特征**：`Date`。只允许从日期派生 day-of-year 的 sin/cos，**年份不使用**。（2026-11-09 实测更正：最初的理由是各站点记录时段不同、年份能识别站点，但实测单用年份预测 Zone 的准确率是 0.323，和多数类基线 0.322 几乎一样，因为大多数站点都从 2008–2009 年开始记录。所以不用年份的理由改为：它没有气象意义，也无法推广到别的年份。）
- **站点指纹风险**（不是直接泄漏，但能间接暴露站点）：
  - missing indicator：有些站点整站没有 Sunshine、Evaporation、Cloud 的仪器，所以"是否缺失"这个标记就等于告诉了模型是哪个站点。
  - 按 Location × Month 分组填补缺失值：会把站点特有的数值复制进特征。
  - 这两项都做成开关，在敏感性比较中分别报告开和关的结果。
- **绝对禁止**：按 Zone 分组填补缺失值，因为这等于直接用了标签。
- 站点信息 `groups` 单独保存，只用于按站点分组的划分和错误分析，永远不进入特征表。
- 所有的 imputer、scaler、encoder 都放在 sklearn Pipeline 里面，只在训练集（或 CV 的训练折）上 fit。

## DEC-03 聚类对象

- **必做**：按"天"聚类，也就是对预处理后的样本（抽样）聚类。这对应 PDF 中 Task 3 的原文要求。
- **额外亮点**：用 48 个站点的气候画像（各特征的统计量）做聚类，并和 Köppen 分区对比。

## DEC-04 数据划分：同时回答两个问题

| 划分方式 | 回答的问题 | 地位 |
|---|---|---|
| 随机分层 70/30（`stratify=Zone`） | 对于见过的站点，模型能否识别它的新的一天？ | PDF 的必做要求；train、test、entire 三组都要报告 |
| 按站点分组（GroupShuffleSplit 或 GroupKFold，按 Location 分组） | 对于从未见过的站点，模型能否判断它的气候区？ | 泛化测试，是故事线的第 4 步 |

预期随机划分的分数会明显高于分组划分。两者之间的差距，可以用来衡量模型在多大程度上"记住了站点"。

按站点分组时要注意：测试集的站点在训练集里完全不存在，所以按 Location 分组填补在这种设置下根本没法用。这本身就证明了它是一种站点指纹。

## DEC-05 Task 5 的报告方式

- 每个气候区分别报告 precision、recall 和 F1，并同时给出 macro-F1 和 weighted-F1。
- LR 和 Decision Tree 各画 5 条 one-vs-rest ROC 曲线，报告每类的 AUC 和 macro-AUC，必要时再加 weighted-AUC。
- 用 StratifiedKFold（或 GroupKFold）加 GridSearchCV 改进两个模型。
- 用 validation curve、train/test 差距和错误分析来讨论过拟合。

## DEC-06 Task 1 预处理方案（2026-11-09 定，依据见 notebook 第 1 部分）

| 问题 | 决定 | 依据（实测） |
|---|---|---|
| 高缺失的传感器列（Sunshine、Evaporation、Cloud9am/3pm） | 默认的 `core` 特征集里不包含；只在 `full` 特征集中用于敏感性比较 | 这些列在 12–19 个站点整站缺失。只用缺失标记（不看任何天气数值）训练决策树，预测 Zone 的准确率是 0.439（基线 0.322），预测具体站点的准确率是 0.240（基线 0.024，约为 10 倍） |
| missing indicator | 不使用 | 用 LR 探测：加上 indicator 后，随机划分的准确率从 0.606 升到 0.671，但分组划分只从 0.569 升到 0.572（macro-F1 还从 0.567 降到 0.558）。也就是说，提升全部来自"认出站点" |
| 风向（16 个方位） | 和风速合成风矢量 (u, v)；静风编码为 (0, 0) | 风速为 0 的行 100% 没有风向，说明风向是"无定义"，而不是缺失 |
| RainToday | 编码成 0/1 后删除 | 它和 Rainfall > 1 mm 100% 一致，没有新信息 |
| Rainfall | 取 log1p | 偏度从 9.9 降到 2.1；63% 的值为 0 |
| Date | 取 day-of-year 的 sin/cos | 让季节首尾相接（12 月 31 日和 1 月 1 日相邻） |
| 填补和缩放 | 中位数填补后接 StandardScaler，放在 Pipeline 里，只在训练集上 fit | 防止泄漏 |
| 最终特征 | `core` 共 20 维：12 个原始测量值、6 个风矢量分量、2 个季节特征 | |

## DEC-07 代码组织

提交的 zip 里只有 notebook 和 `data/`，所以 **notebook 必须自包含**，所有代码都写在 cell 里。notebook 放在项目根目录（`project_XX_TaskA.ipynb`，提交前把 XX 改成 groupID），由 `tools/build_notebook_part*.py` 生成。`src/` 里早期的模块已经被 notebook 取代，只作为历史记录保留。

## DEC-08 Part 2 t-SNE 设计（2026-11-10）

| 问题 | 决定 | 理由 |
|---|---|---|
| 样本 | 每个 Zone 均衡抽 1,200 天，共 6,000 天（`sample_idx`），Part 3 复用同一个样本 | t-SNE 跑全量太慢；均衡抽样避免热带（占 7.6%）在图上被淹没 |
| 预处理 | `core` 20 维特征，不加 indicator，preprocessor 在这个样本上 fit | t-SNE 是无监督方法，不涉及测试集 |
| 主参数 | perplexity 30，`init="pca"`，`learning_rate="auto"`，seed 42 | 都是 sklearn 默认或推荐的设置；对 perplexity 5/30/50/100 和 seed 0/7/42 做了敏感性检查 |
| 量化 | k-NN purity（k = 10，5 折）和 trustworthiness（k = 10） | 避免只凭肉眼读图 |
| 对照 | 和 PCA 2-D 对比；分别只用夏季、只用冬季的数据，以及去掉季节特征 | 检验地图的结构是不是"只是季节" |
| 缓存 | 结果存在 `data/processed/tsne/`，文件名包含输入矩阵的哈希 | 每次 embedding 要 20–110 秒；输入一变，缓存就自动失效 |

主要结论（数字见 notebook 2.3–2.5 节）：只有 Tropical 自成一个区域（96% 的邻居同属热带），其余四类严重重叠（0.50–0.64）；地图的主轴是季节；最容易混淆的是 Mediterranean 和 Oceanic；同一站点的样本会聚在一起（station purity 约为随机水平的 15 倍）；PCA 2-D 的 trustworthiness 是 0.81，t-SNE 是 0.985。

## DEC-09 Part 3 聚类设计（2026-11-10）

| 问题 | 决定 | 理由 |
|---|---|---|
| 数据 | 用 Part 2 的同一个样本（6,000 天，20 维 `core` 特征） | 和 t-SNE 图直接对应；Ward 和 silhouette 都是 O(n²)，跑不了全量 |
| 算法 | K-means、Ward 层次聚类、GMM（full covariance），每个都扫描 k = 2–10 | 分别代表质心型、层次型、概率型三类方法，选择理由见 notebook 3.1 节 |
| 指标 | 内部指标：silhouette、Davies–Bouldin、Calinski–Harabasz、inertia、BIC；外部指标：与 Zone 和季节的 ARI、NMI；另做 K-means 的 bootstrap 稳定性（10 次） | 满足 PDF 中 "multiple metrics" 的要求 |
| 最优结果 | **K-means，k = 3** | DB = 1.88，是全部 27 个解中最低的；silhouette 0.164，仅次于 k = 2 的 0.171；稳定性 0.96。k = 2 只能分出暖天和冷天，k = 3 能把冷天进一步分成"静稳高压"和"多风降雨锋面"两种天气 |
| GMM 表现差的原因 | 实测：有一个分量的权重是 0.615，里面 100% 是无雨日，Rainfall 方差为 0 | 63% 的天降雨量正好是 0，形成一个点质量，full covariance 的 GMM 会专门用一个分量去拟合它。所以 BIC 和 silhouette 给出相反的结论 |
| 额外分析 | 48 个站点的气候画像：4 个变量 × 12 个月的均值 = 48 维，标准化后做 Ward 聚类（k = 5）并画 dendrogram | 气候是天气的统计特征。与 Zone 的 ARI 从按天聚类的约 0.1 提升到 0.345；Tropical 和沿海 Subtropical 被完整识别出来，其余站点主要按沿海和内陆分开 |

结论：单独一天的天气聚出来的是"天气类型"，而不是气候区（所有算法、所有 k 的 ARI 都不超过 0.13）。这为 Part 4 设定了预期：热带容易识别，南部各气候区之间容易混淆。

## DEC-10 Part 4–5 预测与评估（2026-11-12）

| 问题 | 决定 | 依据（实测） |
|---|---|---|
| 模型 | 多分类 LR（C = 1）和 CART 决策树（默认、不限深度） | LR 是线性基线，有概率输出，可以解释；树是非线性模型，规则可读。Part 4 先用默认参数，看清过拟合，Part 5 再调参 |
| 划分 | 随机分层 70/30（必做），加按站点分组 70/30（15 个测试站点，5 类都覆盖） | 见 DEC-04 |
| 三组评估 | train / test / entire；随机划分画 6 张 confusion matrix，分组划分画 2 张 | entire = 0.7·train + 0.3·test，不是独立的估计（树的 entire 是 0.895，有误导性） |
| Decision boundary | 用 MinTemp × Humidity3pm 两个特征重新训练，画 LR、默认树、depth-6 树三张图 | 在可解释的特征对里，这一对的 2 特征 LR macro-F1 最高（0.390） |
| 调参 | GridSearchCV（macro-F1）。LR 调 C × class_weight，DT 调 max_depth × min_samples_leaf。随机设置用 StratifiedKFold，未见站点设置用 StratifiedGroupKFold | 两种 CV 都选 depth 16、leaf 20；分组 CV 的分数低约 0.26 |
| ROC | one-vs-rest，每类一条曲线，报告每类 AUC、macro 和 weighted | 默认树的叶子是纯的，概率只有 0/1，所以 AUC 是 0.788，比 LR 的 0.859 还低；调参后升到 0.889 |
| 错误分析 | 用 StratifiedGroupKFold 对全部 48 个站点做 out-of-fold 预测 | 最差的是内陆 Cfa 站点（Wagga Wagga、Moree、Albury）、南澳东南的 Cs 站点和 BSk 站点，和 Part 3 站点聚类的"沿海/内陆"分组一致 |

## DEC-11 Open-ended（2026-11-12）

- 统一的评估协议：每个结果报告两个数字，一个是随机划分的测试集 macro-F1，另一个是对 48 个站点做 StratifiedGroupKFold(5) 的 out-of-fold macro-F1（"unseen"）。
- 7 类模型对比：灵活模型在随机划分下比 LR 高 0.14–0.18，在未见站点下只高 0.06–0.11。HGB 最好（0.795 / 0.638）。
- 随机超参搜索：16 组配置，每组单独缓存（ParameterSampler + cross_validate，等价于 RandomizedSearchCV）。提升不到 0.01。
- "气候记忆"特征：取同一站点过去 N 天的均值。N = 14 时最好（未见站点 0.727）；N = 365 反而下降（0.655），因为一年平均把季节循环抹掉了。随机划分下这类特征几乎完全泄漏（0.99），所以只看未见站点的结果。
- 最终选择：单日数据用 HGB；如果有 14 天的观测，用 HGB 加 14 天均值。

## DEC-12 工程与复现（2026-11-12）

- 所有耗时的步骤都用 `cached(name, fn, **params)` 缓存到 `data/processed/cache/`。文件名里包含数据 SHA 和参数的哈希，参数一变就自动重算。`RECOMPUTE = True` 会全部重新计算（约 15–20 分钟）。有缓存时整个 notebook 约 85 秒跑完。
- `tools/run_cells.py`：开发时分段填充缓存用。
- `requirements_XX_TaskA.txt`：固定了实际运行 notebook 时的版本（Python 3.10.12）。尝试在全新 venv 里安装验证，但沙箱下载速度太慢，没有完成，需要在本机验证一次。
- Report：`report/report_XX_TaskA.tex`。课程 style 文件到位之前，用 article 加 1 英寸页边距的版本作为回退。当前共 9 页，其中正文约 8 页。

## DEC-13 根据同学审阅意见做的修改（2026-11-16）

| 审阅意见 | 决定 / 修改 | 依据（实测） |
|---|---|---|
| 标签可审计性 | notebook 0.2 节显示完整的 49 站审计表；report 附录 Table 5 也放了这张表。五类改称"项目定义的 Köppen 分组"，并写明：只出现 9 种代码，没有 D/E 类，映射规则以外的代码一律无标签 | kgcpy 栅格为 12960×6480，即 100″/像素；kgc 包把 Rubel et al. 2017 列为数据来源；官网说明全球图使用的是该文的降尺度方法，代表 1986–2010 年。report 补引了 kgc（CRAN DOI）和官网 |
| Norfolk Island | 保留排除，并在 notebook 里用缓存的检查结果说明原因 | 站点所在像素是 Ocean，从站点向外第一次出现陆地像素是在第 185 圈，位于新西兰北端，距离 736 km。说明岛屿本身不在栅格上，不是坐标误差 |
| t-SNE 结论过度 | 加入 Cohen's κ；声明这些数字只是可视化诊断；改写"季节主导"和"非线性"两句 | κ：zone 0.54，season 0.55；去掉 doy 特征后，season 降到 0.17，zone 仍有 0.49 |
| 聚类选择规则 | 规则写在代码里：先要求稳定性 ≥ 0.90，再在 silhouette、DB、CH 三项上取平均名次最好的解。三个算法都改用同一种 80% 子采样稳定性 | 选中的仍是 K-means k = 3（平均名次 1.67，稳定性 0.98）。Ward 在所有 k 上只有 0.26–0.36，因此从不入选。原来的 bootstrap 加最近质心分配方法不准确：仅最近质心这一步，就让 Ward 的 ARI 降到 0.49，所以弃用 |
| GMM 细节 | 报告默认参数（reg_covar = 1e-6）和收敛情况；加 reg_covar 的敏感性检查 | 把 reg_covar 调到 0.1 后，零方差分量消失，silhouette 从 0.032 升到 0.100，但仍不如 K-means 的 0.164 |
| 一次分组划分不够稳定 | 新增 6.6 节：三种协议（随机、时间切分加 14 天 embargo、分组 CV 3 个 seed × 5 折）以及逐折配对比较；5.6 节解释合并结果和逐折结果的区别 | 逐折 macro-F1 只在该折实际出现的类别上平均；sklearn 默认的算法会低估约 0.06。15 折的结果：LR 0.509±0.058，DT 0.528±0.056，HGB 0.599±0.069，HGB 14 天 0.683±0.091。HGB 胜过 LR 15/15 次，14 天特征胜过单日 15/15 次，DT 胜过 LR 只有 10/15 次。从随机划分换成时间切分只损失 0.016–0.07，再换成未见站点又损失 0.09–0.21 |
| 滚动窗口定义 | notebook 6.4 节和 report 都写出了公式：右闭窗口、按日历天计算、有效天数至少 ⌊N/3⌋、缺失值在训练折内用中位数填补；并论证了为什么在分组 CV 下预先计算不会泄漏 | 14 天均值的缺失率：中位数 1.84%，最高 10.6%（Pressure9am） |
| Open-ended 细节 | 超参数表直接从拟合好的 pipeline 读取（notebook 6.1 节和 report 附录 Table 6）；HGB 搜索空间也列入表中 | 同时更正了"HGB 原生处理 NaN"的说法：pipeline 里是先做中位数填补的 |

尚未做：用一张独立的 Köppen 地图（例如 Beck et al. 2018）做标签敏感性检查。这一点已在 report 结论中列为局限。
