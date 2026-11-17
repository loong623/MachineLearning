# DSAA2011 Task A — "Where am I?"

根据天气观测识别澳大利亚站点的五类 Köppen 气候分组，并比较单日/多日信息和已知/未见站点评估。

## 当前版本：2026-11-17 评审后重做

主线和原始数据保持不变。新实验协议见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，逐项回应见 [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md)。此前结果是历史探索，当前数字以 `results/` 和执行后的 notebook 为准。

- 固定五类 pooled macro-F1 为主指标；同时报告站点等权指标、逐类指标和站点错误。
- 站点表分层四折，每折五类齐全；外层重复种子 42/0/7。重复划分不当作独立实验。
- 超参数、窗口和特征选择仅使用外层训练站点；没有全局测试集特征排序。
- LR/HGB × 1/3/7/14/30/90/365 天对照；窗口选择另用内层验证。
- RF、MLP 等全部日级候选采用一致的随机、时间和站点协议。
- 加入条件站点 bootstrap、空间缓冲、边界标签和缺失值策略敏感性。
- 报告数值/表格由结果文件生成；不手工沿用旧结果。

## 本轮主要结果

主指标为预先指定的 seed 42、固定五类 pooled OOF macro-F1。该划分在所有日级模型上都高于另外两次划分，因此必须结合重复均值/范围阅读。

| 配置 | 主分数 | 三次划分均值 |
|---|---:|---:|
| 日级 LR | 0.556 | 0.549 |
| 日级 RF | 0.665 | 0.651 |
| 日级 HGB | 0.659 | 0.647 |
| 内层选窗口 LR | 0.745 | 未重复该选择程序 |
| 内层选窗口 HGB | 0.758 | 未重复该选择程序 |

- HGB 的训练内窗口选择相对单日提升 0.100，条件整站配对区间为 [0.066, 0.131]；不是包含重训练波动的总体区间。
- 不存在统一的“14 天最好”或“365 天必然更差”：LR 与 HGB 的曲线不同，HGB 全年窗口在共同历史样本上达到 0.782。
- 训练内特征数选择得分 0.709，低于全 74 特征的 0.754；保留这一负结果。
- 100 km 缓冲后的日级 HGB 为 0.565，同时训练站点从 36 降为 22–25；Nhil、Moree 等站点仍有严重失败。

## 环境与运行

本次环境：Windows，Python 3.12.6。`requirements_XX_TaskA.txt` 是实装环境的完整精确版本列表。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_XX_TaskA.txt
.\.venv\Scripts\python.exe tools/build_notebook.py
.\.venv\Scripts\python.exe tools/execute_notebook.py --fresh
.\.venv\Scripts\python.exe tools/build_notebook.py --sync-markdown
.\.venv\Scripts\python.exe tools/verify_results.py
.\.venv\Scripts\python.exe tools/build_report_assets.py
.\.venv\Scripts\python.exe tools/build_report.py
```

`--fresh` 强制重算所有结果和 t-SNE。省略时可重用**同一指纹**缓存。完整嵌套实验比历史版本更耗时；运行器逐 cell 记录进度并保存 notebook。所有算法代码均写入最终 notebook，不依赖 `src/`；`tools/` 是生成、运行和校验工具。运行器明确使用调用它的 Python 作为 kernel，并嵌入图像输出。

报告使用老师提供的 `report/neurips_2026.sty`，选项 `[main, final]`。在 `report/` 内运行 `pdflatex report_XX_TaskA.tex` 两遍，或使用 Tectonic；图片保留 `../figures/` 相对路径。不要直接修改生成的 report 数字；修改 `report/report_template.tex`，然后运行 `tools/build_report.py`。

## 文件

| 路径 | 用途 |
|---|---|
| `project_XX_TaskA.ipynb` | 自包含、完整执行的课程 notebook |
| `report/report_XX_TaskA.tex/.pdf` | 根据新实验生成的报告 |
| `report/report_template.tex` | 报告文字与结构模板 |
| `results/*.csv` | 指标、划分、选参、诊断、条件区间 |
| `results/predictions.npz` | 每日真实标签、站点、日期和主要 OOF 预测 |
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

## 正式课程提交前

组员仍需填写真实 groupID、姓名和贡献；文件里的 `XX` 不由 AI 猜测。演示文稿尚不属于本轮实验重做交付，需要另行制作。报告与 notebook 必须由组员阅读、核验并能够解释。
