"""Assemble the project notebook from its parts.

Run from the project root:
    python tools/build_notebook.py
    python -m nbconvert --to notebook --execute --inplace project_XX_TaskA.ipynb
Each tools/build_notebook_partN.py defines a list `cells`; parts are concatenated in order.
"""
import importlib
import sys
from pathlib import Path

import nbformat as nbf

NB_PATH = Path("project_XX_TaskA.ipynb")   # rename XX to our groupID before submission
PARTS = ["build_notebook_part1", "build_notebook_part2", "build_notebook_part3", "build_notebook_part4",
         "build_notebook_part5", "build_notebook_part6"]


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    cells = []
    for name in PARTS:
        part = importlib.import_module(name)
        cells += part.cells
        print(f"{name}: {len(part.cells)} cells")
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB_PATH)
    print("wrote", NB_PATH, "with", len(cells), "cells")


if __name__ == "__main__":
    main()
