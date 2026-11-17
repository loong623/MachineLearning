"""Assemble the project notebook from its parts.

Run from the project root:
    python tools/build_notebook.py
    python -m nbconvert --to notebook --execute --inplace project_XX_TaskA.ipynb
Each tools/build_notebook_partN.py defines a list `cells`; parts are concatenated in order.
"""
import importlib
import sys
import argparse
from pathlib import Path

import nbformat as nbf

NB_PATH = Path("project_XX_TaskA.ipynb")   # rename XX to our groupID before submission
PARTS = ["build_notebook_part1", "build_notebook_part2", "build_notebook_part3", "build_notebook_part4",
         "build_notebook_part5", "build_notebook_part6"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sync-markdown',action='store_true',help='Update prose only, retaining outputs only if every executable cell is identical.')
    args=parser.parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    cells = []
    for name in PARTS:
        part = importlib.import_module(name)
        cells += part.cells
        print(f"{name}: {len(part.cells)} cells")
    if args.sync_markdown:
        nb=nbf.read(NB_PATH,as_version=4)
        assert len(nb.cells)==len(cells),'Cell structure changed; perform a full rebuild and execution.'
        for old,new in zip(nb.cells,cells):
            assert old.cell_type==new.cell_type
            if old.cell_type=='code':
                assert old.source==new.source,'Executable code changed; preserving outputs would be misleading.'
            else:
                old.source=new.source
        # Add actual findings only after execution; never attach old results to new code.
        import hashlib,json
        manifest_path=Path('results/run_manifest.json')
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            digest=hashlib.sha256('\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest()
            assert digest==manifest['code_sha256'],'Result narrative belongs to different code.'
            from result_narrative import notebook_summary
            next(c for c in reversed(nb.cells) if c.cell_type=='markdown').source+='\n\n'+notebook_summary(Path.cwd())
    else:
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB_PATH)
    print("wrote", NB_PATH, "with", len(cells), "cells")


if __name__ == "__main__":
    main()
