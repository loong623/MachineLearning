"""Validate the experiment outputs and notebook before publication."""
import json
import ast
import re
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
from sklearn.metrics import f1_score

def numeric_literal(text):
    # NumPy 2 scalar repr includes its type; keep parsing restricted to literals.
    return ast.literal_eval(re.sub(r'np\.float(?:32|64)\(([-+0-9.eE]+)\)',r'\1',text))

root=Path(__file__).resolve().parents[1]
r=root/'results'
nb=nbformat.read(root/'project_XX_TaskA.ipynb',as_version=4)
codes=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in codes)
assert not [(i,o) for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if o.output_type=='error']
assert any('image/png' in o.get('data',{}) for c in codes for o in c.get('outputs',[])), 'No embedded figures'
manifest=json.loads((r/'run_manifest.json').read_text())
import hashlib
digest=hashlib.sha256('\n'.join(c.source for c in codes).encode()).hexdigest()
assert manifest['code_sha256']==digest,'Notebook code differs from the executed experiment fingerprint'
assert manifest['cache_hits']==0, 'The requested full fresh execution must not reuse result caches'
env=json.loads((r/'environment.json').read_text())
for name,expected in env['files'].items():
    # A saved Windows manifest can also be checked from a POSIX checkout.
    assert hashlib.sha256((root/name.replace('\\','/')).read_bytes()).hexdigest()==expected, name
folds=pd.read_csv(r/'outer_folds.csv',index_col=0)
for (seed,fold),block in folds.groupby(['seed','fold']):
    assert block.zone.nunique()==5
    tr=set(block.train_stations.iloc[0].split(';'))
    assert not tr & set(block.test_station)
for seed,block in folds.groupby('seed'):
    assert block.test_station.nunique()==len(block)==48
details=pd.read_csv(r/'feature_selection_details.csv',index_col=0)
for _,row in details.iterrows():
    test=set(folds[(folds.seed==42)&(folds.fold==row.fold)].test_station)
    a,b,c=[set(row[k].split(';')) for k in ['A','B','C']]
    assert not (a&b or a&c or b&c or (a|b|c)&test)
    assert len(a|b|c|test)==48
    station_labels=folds.drop_duplicates('test_station').set_index('test_station').zone
    assert all(station_labels.loc[list(part)].nunique()==5 for part in [a,b,c,test])
p=np.load(r/'predictions.npz',allow_pickle=False)
labels=['Tropical','Arid','Subtropical','Mediterranean','Oceanic']
assert len(p['y'])==142451
models=pd.read_csv(r/'model_comparison.csv',index_col=0)
for kind,row in models.iterrows():
    scores=[]
    for seed in manifest['outer_seeds']:
        pred=p[f'daily_{kind}_s{seed}']
        assert set(pred)<=set(labels)
        scores.append(f1_score(p['y'],pred,labels=labels,average='macro',zero_division=0))
    assert np.allclose(scores[0],row.grouped_primary,rtol=0,atol=1e-10)
    assert np.allclose([np.mean(scores),min(scores),max(scores)],
                       [row.grouped_seed_mean,row.grouped_seed_min,row.grouped_seed_max],rtol=0,atol=1e-10)
    station=pd.Series(p['station'])
    weights=1/station.map(station.value_counts()).to_numpy()
    balanced=f1_score(p['y'],p[f'daily_{kind}_s42'],labels=labels,average='macro',zero_division=0,sample_weight=weights)
    assert abs(balanced-row.station_balanced_macro_F1)<1e-10
win=pd.read_csv(r/'window_comparison.csv',index_col=0)
for _,row in win.iterrows():
    pred=p[f'window_{row.model}_{int(row.window)}_s{int(row.seed)}']
    assert abs(f1_score(p['y'],pred,labels=labels,average='macro',zero_division=0)-row.macro_F1)<1e-10
choices=pd.read_csv(r/'nested_window_choices.csv',index_col=0)
for _,row in choices.iterrows():
    candidates=numeric_literal(row.inner_results)
    expected=min(candidates,key=lambda x:(-x['inner_mean_F1'],x['window']))['window']
    assert row.selected_window==expected
for _,row in details.iterrows():
    scores=numeric_literal(row.validation)
    assert row.selected_k==min(scores,key=lambda k:(-scores[k],k))
for file,field,key,pred_key in [('nested_window_scores','model','HGB','HGB-window-selected'),
                                ('feature_selection_scores','k','selected','HGB-features-selected')]:
    score=pd.read_csv(r/f'{file}.csv',index_col=0).set_index(field).loc[key,'macro_F1']
    assert abs(f1_score(p['y'],p[pred_key],labels=labels,average='macro',zero_division=0)-score)<1e-10
assert pd.DataFrame({'station':p['station'],'date':p['date']}).duplicated().sum()==0
assert pd.DataFrame({'station':p['station'],'zone':p['y']}).groupby('station').zone.nunique().eq(1).all()
print(f'PASS: {len(codes)} executed cells; embedded figures; fresh run; hashes; class-covered disjoint folds; feature-selection isolation; inner selection decisions; daily/repeated/station-balanced/window/selected-procedure metrics reconstructed from predictions.')
