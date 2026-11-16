# DSAA2011 Task A — "Where am I?" Climate-Zone Classification

根据一天的天气观测，判断它来自澳大利亚的哪个气候区（5 类，基于 Köppen-Geiger）。

## 目录结构

```
project_XX_TaskA.ipynb           最终 notebook（111 个 cell，自包含；提交前把 XX 改成 groupID）
requirements_XX_TaskA.txt        固定版本（Python 3.10.12）
report/report_XX_TaskA.tex/.pdf  report 初稿（9 页；课程 style 文件到位前用 article 版本）
data/raw/weatherAUS.csv          Kaggle jsphyg/weather-dataset-rattle-package v2（145,460 行 × 23 列）
data/external/                   BoM 站点列表压缩包；Location → BoM 站号的人工匹配表
data/processed/location_zone.csv 49 个站点 → Köppen 代码 → Zone
data/processed/cache/, tsne/     计算结果缓存（约 15 MB）；没有缓存时 notebook 会自动重算
figures/                         notebook 输出的全部图（31 张），report 和 slides 都用这些
tools/build_notebook*.py         生成 notebook 的脚本（每个 Part 一个文件）
tools/run_cells.py               分段执行 notebook、填充缓存（开发工具）
explore/, src/                   早期试验和旧模块，仅作历史记录
BRAINSTORM.md / DECISIONS.md     候选方案 / 已定决定（DEC-01 到 DEC-12）
AI_LOG.md                        AI 使用日志
```

## 运行

```bash
pip install -r requirements_XX_TaskA.txt
jupyter notebook project_XX_TaskA.ipynb          # 或者：
python -m nbconvert --to notebook --execute --inplace project_XX_TaskA.ipynb
```

- 有缓存时约 1.5 分钟跑完。删除 `data/processed/cache` 和 `tsne`，或者设置 `RECOMPUTE = True` / `RECOMPUTE_TSNE = True`，会全部重新计算，在 2 核机器上约 15–20 分钟。
- 修改 notebook 时请改 `tools/build_notebook_partN.py`，然后运行 `python tools/build_notebook.py` 重新生成。直接编辑 .ipynb 的改动会在下次生成时被覆盖。

## 提交前清单

1. 把文件名里的 `XX` 改成 groupID（notebook、requirements、report、presentation），zip 命名为 `groupID_TaskA.zip`。
2. 下载课程的 LaTeX style 文件，放进 `report/`，把 tex 里的 `COURSE_STYLE` 改成它的文件名（不要用 preprint 选项），然后重新编译。
3. 填写 report 的作者和 Credit 部分（成员贡献、AI 使用比例）。
4. 在干净的虚拟环境里按 requirements 安装，并完整跑一遍 notebook。
5. 每位组员填写 `AI_LOG.md` 中"我能讲清楚的要点"一栏。
