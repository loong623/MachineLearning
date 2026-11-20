"""Create legible report-size plots and preserve their numerical inputs.

Training/prediction is repeated only for the inexpensive course illustration.
Scores must match the completed run before an illustration is accepted.
"""
import os
os.environ['MPLBACKEND']='Agg'
import ast, hashlib, json, pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,roc_curve,roc_auc_score
from threadpoolctl import threadpool_limits
import nbformat

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
threadpool_limits(2)
manifest=json.loads(Path('results/run_manifest.json').read_text())
run=manifest['run_id']
nb=nbformat.read('project_7_TaskA.ipynb',as_version=4)
digest=hashlib.sha256('\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest()
assert digest==manifest['code_sha256']
zones=pd.read_csv('data/processed/location_zone.csv')
raw=pd.read_csv('data/raw/weatherAUS.csv',parse_dates=['Date'])
df=raw.merge(zones[['Location','Zone']],on='Location',validate='many_to_one').dropna(subset=['Zone']).reset_index(drop=True)
groups=df.pop('Location');y=df.Zone.to_numpy();RANDOM_STATE=42
# Reuse the exact executed feature definition; no separate implementation to drift.
feature_cell=next(c.source for c in nb.cells if c.cell_type=='code' and c.source.startswith('COMPASS ='))
exec(feature_cell,globals())
X=F[FEATURE_SETS['core']]
labels=['Tropical','Arid','Subtropical','Mediterranean','Oceanic']
short=['Trop.','Arid','Subtrop.','Medit.','Ocean.']
colors=dict(zip(labels,['#d62728','#e6a01e','#2ca02c','#9467bd','#1f77b4']))
idx=np.arange(len(y));rt,rv=train_test_split(idx,test_size=.3,stratify=y,random_state=42)
gt,gv=next(GroupShuffleSplit(1,test_size=.3,random_state=42).split(idx,y,groups))
sns.set_theme(style='whitegrid',context='paper',font_scale=1.25)
fig,axes=plt.subplots(2,3,figsize=(9,6.8))
cm_rows=[]
p4=pd.read_csv('results/p4_metrics.csv',index_col=0)
for row,kind in enumerate(['LR','DT']):
    m=make_pipeline(build_preprocessor(),LogisticRegression(max_iter=3000) if kind=='LR' else DecisionTreeClassifier(random_state=42))
    m.fit(X.iloc[rt],y[rt])
    for col,(part,ids) in enumerate([('train',rt),('test',rv),('entire',idx)]):
        pred=m.predict(X.iloc[ids]);cm=confusion_matrix(y[ids],pred,labels=labels)
        expected=p4[(p4['split']=='random')&(p4.model==kind)&(p4['set']==part)].iloc[0]
        assert abs(f1_score(y[ids],pred,labels=labels,average='macro')-expected['macro-F1'])<1e-10
        assert abs(accuracy_score(y[ids],pred)-expected.accuracy)<1e-10
        cm_rows.extend({'model':kind,'set':part,'true':a,'predicted':b,'count':int(cm[i,j])} for i,a in enumerate(labels) for j,b in enumerate(labels))
        sns.heatmap(cm/cm.sum(1,keepdims=True),annot=True,fmt='.2f',vmin=0,vmax=1,cmap='Blues',cbar=False,
                    linewidths=.4,linecolor='white',xticklabels=short,yticklabels=short,
                    annot_kws={'fontsize':12},ax=axes[row,col])
        axes[row,col].set(title=f'{kind}: {part}',xlabel='Predicted',ylabel='True' if col==0 else '')
        axes[row,col].tick_params(axis='both',labelsize=11,rotation=35)
fig.tight_layout();fig.savefig('figures/report_confusion.png',dpi=300,bbox_inches='tight');plt.close(fig)
pd.DataFrame(cm_rows).to_csv('results/report_confusion_counts.csv',index=False)

hold=pd.read_csv('results/p5_holdout.csv',index_col=0)
fig,axes=plt.subplots(2,2,figsize=(8,6.8))
for row,(split,tr,te) in enumerate([('random',rt,rv),('grouped',gt,gv)]):
    for col,kind in enumerate(['LR-tuned','DT-tuned']):
        info=hold[(hold['split']==split)&(hold.model==kind)].iloc[0]
        est=LogisticRegression(max_iter=3000) if kind.startswith('LR') else DecisionTreeClassifier(random_state=42)
        m=make_pipeline(build_preprocessor(),est).set_params(**ast.literal_eval(info.params)).fit(X.iloc[tr],y[tr])
        prob=m.predict_proba(X.iloc[te]);pred=m.predict(X.iloc[te])
        assert abs(f1_score(y[te],pred,labels=labels,average='macro')-info.macro_F1)<1e-10
        for zone in labels:
            p=prob[:,list(m.classes_).index(zone)];fp,tp,_=roc_curve(y[te]==zone,p)
            axes[row,col].plot(fp,tp,color=colors[zone],label=f'{zone}: {roc_auc_score(y[te]==zone,p):.3f}')
        axes[row,col].plot([0,1],[0,1],'k--',lw=.7)
        axes[row,col].set(title=f'{kind}, {split}',xlabel='False positive rate',ylabel='True positive rate')
        axes[row,col].legend(fontsize=11,loc='lower right')
fig.tight_layout();fig.savefig('figures/report_roc.png',dpi=300,bbox_inches='tight');plt.close(fig)

vis_path=Path('results/visualization_inputs.npz')
sample=df.groupby('Zone',group_keys=False).sample(n=1200,random_state=42).index
xv=build_preprocessor().fit_transform(F.loc[sample,FEATURE_SETS['core']])
key=hashlib.sha256(np.ascontiguousarray(xv).tobytes()).hexdigest()[:12]
embedding_path=Path('data/processed/tsne_audit')/run/f'tsne_{key}_d2_p30_s42.npy'
cache=next((Path('data/processed/audit_cache')/run).glob('p3_sweep_*.pkl'),None)
if embedding_path.exists() and cache is not None:
    # Prefer freshly executed arrays even when the code/environment fingerprint is unchanged.
    emb=np.load(embedding_path)
    with cache.open('rb') as stream: _,labs=pickle.load(stream)
    vis={'run_id':np.array(run),'embedding':emb,'zone':np.array(y[sample],dtype=str),
         'month':df.Date.iloc[sample].dt.month.to_numpy(),'temperature':F.MaxTemp.loc[sample].to_numpy(),
         **{a:labs[(a,3)] for a in ['K-means','Ward','GMM']}}
    np.savez_compressed(vis_path,**vis)
else:
    assert vis_path.exists(),'Run the notebook first or restore its exported visualization inputs.'
    vis=dict(np.load(vis_path,allow_pickle=False))
    assert vis['run_id'].item()==run,'Visualization inputs belong to a different experiment.'
emb=vis['embedding']
fig,axes=plt.subplots(1,3,figsize=(9,3))
for z in labels:
    mask=vis['zone']==z;axes[0].scatter(emb[mask,0],emb[mask,1],s=2,color=colors[z],label=z,alpha=.7,rasterized=True)
for ax,key,cmap,title in [(axes[1],'month','twilight_shifted','Month'),(axes[2],'temperature','inferno','Max. temperature (°C)')]:
    sc=ax.scatter(emb[:,0],emb[:,1],c=vis[key],cmap=cmap,s=2,rasterized=True)
    bar=fig.colorbar(sc,ax=ax,shrink=.8)
    if key=='month':bar.set_ticks([1,3,6,9,12])
    ax.set_title(title)
axes[0].set_title('Climate label')
for ax in axes:ax.set_xticks([]);ax.set_yticks([])
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,markerscale=3,fontsize=12)
fig.tight_layout(rect=[0,.12,1,1]);fig.savefig('figures/report_tsne.png',dpi=300,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(9,3))
for ax,a in zip(axes,['K-means','Ward','GMM']):
    ax.scatter(emb[:,0],emb[:,1],c=vis[a],cmap='Set2',s=2,rasterized=True)
    ax.set(title=a,xticks=[],yticks=[])
fig.tight_layout();fig.savefig('figures/report_clusters.png',dpi=300,bbox_inches='tight');plt.close(fig)

windows=pd.read_csv('results/window_comparison.csv',index_col=0)
common=pd.read_csv('results/window_common_history.csv',index_col=0)
fig,axes=plt.subplots(1,2,figsize=(8,3.8));ns=[1,3,7,14,30,90,365]
for kind,color in [('LR','C0'),('HGB','C1')]:
    d=windows[windows.model==kind].groupby('window').macro_F1.agg(['mean','min','max']).reindex(ns)
    axes[0].plot(ns,d['mean'],marker='o',color=color,label=kind);axes[0].fill_between(ns,d['min'],d['max'],color=color,alpha=.15)
    d=common[common.model==kind].set_index('window').reindex(ns);axes[1].plot(ns,d.macro_F1,marker='o',color=color,label=kind)
for ax in axes:
    ax.set_xscale('log');ax.set_xticks(ns);ax.set_xticklabels(ns);ax.set(xlabel='Trailing calendar days',ylabel='Pooled five-class macro-F1');ax.legend()
axes[0].set_title('Three seeds: mean and range')
axes[1].set_title('Common history-eligible rows')
fig.tight_layout();fig.savefig('figures/report_memory.png',dpi=300,bbox_inches='tight');plt.close(fig)
print('Report assets generated; repeated illustration scores match the complete run.')
