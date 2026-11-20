"""Execute the complete notebook in this Python environment and retain outputs/checkpoints."""
import os
os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '2'
import argparse
import ast
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

parser=argparse.ArgumentParser()
parser.add_argument('--notebook',default='project_7_TaskA.ipynb')
parser.add_argument('--fresh',action='store_true',help='Ignore all computed caches, including t-SNE.')
args=parser.parse_args()
path=Path(args.notebook).resolve()
if args.fresh: os.environ['ML_RECOMPUTE']='1'
nb=nbformat.read(path,as_version=4)
for i,cell in enumerate(nb.cells):
    if cell.cell_type=='code': ast.parse(cell.source,filename=f'cell-{i}')
start=time.monotonic()
def checkpoint():
    # Atomic replacement avoids exposing a partially written notebook to file watchers.
    temp=path.with_suffix('.executing.tmp')
    for attempt in range(6):
        try:
            temp.write_text(nbformat.writes(nb),encoding='utf-8')
            temp.replace(path)
            return
        except OSError:
            if attempt==5: raise
            time.sleep(.5)
def progress(cell,cell_index,**kwargs):
    if cell.cell_type=='code':
        print(f'CELL {cell_index+1}/{len(nb.cells)} START elapsed={time.monotonic()-start:.1f}s {cell.source.splitlines()[0][:100]}',flush=True)
def finished(cell,cell_index,**kwargs):
    for out in cell.get('outputs',[]):
        if out.output_type=='stream': print(out.text[-2500:],flush=True)
    print(f'CELL {cell_index+1} DONE elapsed={time.monotonic()-start:.1f}s',flush=True)
    checkpoint()
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv[0]=sys.executable
client=NotebookClient(nb,km=km,timeout=14400,allow_errors=False,resources={'metadata':{'path':str(path.parent)}},on_cell_start=progress,on_cell_executed=finished)
try:
    client.execute()
finally:
    checkpoint()
print(f'COMPLETE {time.monotonic()-start:.1f}s Python={sys.executable}',flush=True)
