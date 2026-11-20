# DSAA2011 Task A — From Weather Snapshots to Seasonal Climate Signatures

根据天气观测识别澳大利亚站点的五类 Köppen 气候分组。单日天气是基线，四季窗口站点年份画像是主线。

## 当前版本：2026-11-20 Group 7 seasonal redesign

原始数据和外部站点标注保持不变。实验协议见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，当前数字以 `results/` 和执行后的 Group 7 notebook 为准。早期探索只用于说明设计动机，不把报告割裂成实验轮次。

- 固定五类 pooled macro-F1 为主指标；同时报告站点等权指标、逐类指标和站点错误。
- 站点表分层四折，每折五类齐全；外层重复种子 42/0/7。重复划分不当作独立实验。
- 超参数、窗口和特征选择仅使用外层训练站点；没有全局测试集特征排序。
- LR/HGB × 1/3/7/14/30/90/365 天对照；窗口选择另用内层验证。
- RF、MLP 等全部日级候选采用一致的随机、时间和站点协议。
- 加入条件站点 bootstrap、空间缓冲、边界标签和缺失值策略敏感性。
- 保留单日 t-SNE、聚类和预测作为基线；新增四个 21 天季节窗口（春分、冬至、秋分、夏至）构造站点年份画像。
- 季节画像含均值、标准差、雨天比例和温差统计；每个窗口至少 7 个有效观测，外层仍按站点隔离。
- 季节画像新增 t-SNE、K-means/Ward 聚类、LR/DT/RF/HGB 预测和四快照日模型匹配基线。
- 报告数值/表格由结果文件生成；不手工沿用旧结果。

## 当前主要结果

主指标为预先指定的 seed 42、固定五类 pooled OOF macro-F1。该划分在所有日级模型上都高于另外两次划分，因此必须结合重复均值/范围阅读。

| 配置 | 主分数 | 三次划分均值 |
|---|---:|---:|
| 日级 LR | 0.556 | 0.549 |
| 日级 RF | 0.665 | 0.651 |
| 日级 HGB | 0.659 | 0.647 |
| 内层选窗口 LR | 0.745 | 未重复该选择程序 |
| 内层选窗口 HGB | 0.758 | 未重复该选择程序 |

季节画像主实验保留 328 个站点年份行（48 个站点，平均完整覆盖率 73.6%）。站点外 macro-F1 为：季节 LR 0.861、季节 HGB 0.807；四个季节中心最近日 HGB 投票为 0.759。季节 LR 的 macro-AUC 为 0.973。季节模型的结果是同一站点外划分下的主线比较，不能解释为增加了独立站点。

- HGB 的训练内窗口选择相对单日提升 0.100，条件整站配对区间为 [0.066, 0.131]；不是包含重训练波动的总体区间。
- 不存在统一的“14 天最好”或“365 天必然更差”：LR 与 HGB 的曲线不同，HGB 全年窗口在共同历史样本上达到 0.782。
- 训练内特征数选择得分 0.709，低于全 74 特征的 0.754；保留这一负结果。
- 100 km 缓冲后的日级 HGB 为 0.565，同时训练站点从 36 降为 22–25；Nhil、Moree 等站点仍有严重失败。

## 环境与运行

本次环境：Windows，Python 3.12.6。`requirements_7_TaskA.txt` 是实装环境的完整精确版本列表。

```powershell
 python -m venv .venv
 .\.venv\Scripts\python.exe -m pip install -r requirements_7_TaskA.txt
.\.venv\Scripts\python.exe tools/build_notebook.py
.\.venv\Scripts\python.exe tools/execute_notebook.py --fresh
.\.venv\Scripts\python.exe tools/build_notebook.py --sync-markdown
.\.venv\Scripts\python.exe tools/verify_results.py
.\.venv\Scripts\python.exe tools/build_report_assets.py
.\.venv\Scripts\python.exe tools/build_report.py
```

`--fresh` 强制重算所有结果和 t-SNE。省略时可重用**同一指纹**缓存。完整嵌套实验比历史版本更耗时；运行器逐 cell 记录进度并保存 notebook。所有算法代码均写入最终 notebook，不依赖 `src/`；`tools/` 是生成、运行和校验工具。运行器明确使用调用它的 Python 作为 kernel，并嵌入图像输出。

报告使用老师提供的 `report/neurips_2026.sty`，选项 `[main, final]`。报告源文件为 `report/report_7_TaskA.tex`，由结果文件生成；不要直接修改生成的数字。

## 文件

| 路径 | 用途 |
|---|---|
| `project_7_TaskA.ipynb` | 自包含、完整执行的课程 notebook |
| `report/report_7_TaskA.tex/.pdf` | 根据季节主线生成的报告 |
| `report/report_template.tex` | 报告文字与结构模板 |
| `results/*.csv` | 指标、划分、选参、诊断、条件区间 |
| `results/predictions.npz` | 每日与季节画像真实标签、站点、日期和 OOF 预测 |
| `results/environment.json`, `run_manifest.json` | 代码/数据/标签哈希、实装环境、完成时间、缓存记录 |
| `results/report_values.json` | 报告中自动生成的数值宏 |
| `data/raw/weatherAUS.csv` | 原始 Kaggle v2 文件（不修改） |
| `data/external/`, `data/processed/location_zone.csv` | BoM 原始站表、人工匹配与栅格标签 |
| `data/processed/audit_cache/`, `tsne_audit/` | 新计算缓存，Git 忽略；删除后可重算 |
| `figures/` | notebook 生成的图；报告只引用本轮确认的图 |
| `data/processed/cache/`, `tsne/`, `src/`, `explore/`, `report/old/`, `report/article_draft/` | 历史记录，不是当前实验依据 |
| `AI_LOG.md`, `DECISIONS.md` | 原有记录保留，新条目明确更正旧结论 |

## 解释边界

结果仅适用于当前 48 站网络和一张栅格标注。只有 4 个热带站，手工站点匹配、地图边界及空间相关性仍有限制。多随机种子不是新数据；条件 bootstrap 不包括重新训练和选模波动。局部边界敏感性不是独立地图验证。缺乏新数据的限制通过明确范围处理，不假装已经消除。

## Group 7 提交文件

根目录的 `project_7_TaskA.ipynb`、`report_7_TaskA.pdf`、`presentation_7_TaskA.pdf` 和 `requirements_7_TaskA.txt` 是提交所需的四个文件；`submission_7_TaskA.zip` 另含两份站点标注 CSV。报告 Credit 已列出四名组员，贡献比例需要组员按真实分工确认后补入。
