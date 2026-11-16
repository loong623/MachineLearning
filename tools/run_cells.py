"""Execute the notebook's code cells in one Python process, to fill data/processed/cache/.

Used during development when a single run must stay under a time limit: every expensive step in the
notebook is wrapped in `cached(...)`, so repeated calls make progress and finish quickly once all
results exist. Figures are rendered off-screen (Agg). Stops after the cell that crosses --budget seconds.
Run from the project root:  python tools/run_cells.py --budget 150
"""
import argparse
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat

ap = argparse.ArgumentParser()
ap.add_argument("--notebook", default="project_XX_TaskA.ipynb")
ap.add_argument("--budget", type=float, default=150)
args = ap.parse_args()

nb = nbformat.read(args.notebook, as_version=4)
ns = {"display": lambda *a, **k: None}
plt.show = lambda *a, **k: plt.close("all")
t0 = time.time()
code_cells = [c for c in nb.cells if c.cell_type == "code"]
for i, c in enumerate(code_cells):
    t = time.time()
    exec(compile(c.source, f"cell{i}", "exec"), ns)
    plt.close("all")
    print(f"cell {i:3d}/{len(code_cells)} {time.time() - t:6.1f}s  total {time.time() - t0:6.1f}s", flush=True)
    if time.time() - t0 > args.budget and i < len(code_cells) - 1:
        print("budget reached; run again to continue"); sys.exit(2)
print("ALL CELLS DONE")
