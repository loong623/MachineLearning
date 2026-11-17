"""Generate report source from verified experiment outputs; never hard-code outcome numbers."""
import json
import ast
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results'
def read(name): return pd.read_csv(RESULT/f'{name}.csv',index_col=0)
def escape(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('&',r'\&').replace('%',r'\%')
def f(v): return f'{float(v):.3f}'
def table(caption,label,headers,rows,align=None,colsep=None):
    align=align or 'l'+'r'*(len(headers)-1)
    return ('\\begin{table}[htbp]\n\\centering\\small\n'+
            (f'\\setlength{{\\tabcolsep}}{{{colsep}pt}}\n' if colsep is not None else '')+
            '\\caption{'+caption+'}\\label{'+label+'}\n'
            '\\begin{tabular}{'+align+'}\\toprule\n'+' & '.join(headers)+r' \\ \midrule'+'\n'+
            '\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+
            '\n\\bottomrule\\end{tabular}\n\\end{table}')

manifest=json.loads((RESULT/'run_manifest.json').read_text())
env=json.loads((RESULT/'environment.json').read_text())
models=read('model_comparison')
nested=read('nested_window_scores').set_index('model')
common_history=read('window_common_history').set_index(['model','window'])
paired=read('paired_intervals').set_index('comparison')
selection=read('feature_selection_scores').set_index('k')
selection_details=read('feature_selection_details')
station_errors=read('station_errors')
worst=station_errors.query("model=='HGB-window-selected'").sort_values('accuracy').head(3)
tsne=read('tsne_diagnostics')
cluster=read('clustering_sweep')
station=read('station_clustering')
cluster_controls=read('clustering_sampling_controls')
km_station=cluster_controls.query("scheme=='station-block' and algorithm=='K-means'").ARI_reference
km_year=cluster_controls.query("scheme=='year-block' and algorithm=='K-means'").ARI_reference
policy=read('missingness_policies').set_index('policy')
spatial=read('spatial_buffer_summary').iloc[0]
spatial_folds=read('spatial_buffer_folds')
boundary=read('label_sensitivity').set_index('mode')
mlp=read('fit_audit').query("model=='MLP'")
mlp_iterations=mlp.iterations.map(ast.literal_eval)
mlp_warning_fits=mlp.warnings.map(lambda x:bool(ast.literal_eval(x))).sum()
ranking=cluster[['silhouette','Davies-Bouldin','Calinski-Harabasz']].rank(ascending=False)
ranking['Davies-Bouldin']=cluster['Davies-Bouldin'].rank(ascending=True)
eligible=cluster[cluster.stability>=.9]
winner=eligible.loc[ranking.mean(axis=1).loc[eligible.index].idxmin()]

macros={
 'DailyHGB':f(models.loc['HGB','grouped_primary']), 'DailyLR':f(models.loc['LR','grouped_primary']),
 'DailyRF':f(models.loc['RF','grouped_primary']), 'SelectedWindow':f(nested.loc['HGB','macro_F1']),
 'DailyPoly':f(models.loc['PolyLR','grouped_primary']), 'TunedHGB':f(models.loc['HGB-tuned','grouped_primary']),
 'DailyModelGap':f(models.loc['HGB','grouped_primary']-models.loc['LR','grouped_primary']),
 'HistoryModelGap':f(nested.loc['HGB','macro_F1']-nested.loc['LR','macro_F1']),
 'CommonLRNinety':f(common_history.loc[('LR',90),'macro_F1']),
 'CommonLRAnnual':f(common_history.loc[('LR',365),'macro_F1']),
 'CommonHGBThirty':f(common_history.loc[('HGB',30),'macro_F1']),
 'CommonHGBAnnual':f(common_history.loc[('HGB',365),'macro_F1']),
 'SelectedWindowLR':f(nested.loc['LR','macro_F1']), 'TsneTrust':f(tsne.loc['t-SNE 2-D','trustworthiness']),
 'PcaTrust':f(tsne.loc['PCA 2-D','trustworthiness']), 'TsnePurity':f(tsne.loc['t-SNE 2-D','purity: zone']),
 'ClusterChoice':escape(winner.algorithm)+f' with $k={int(winner.k)}$',
 'NaturalClusterARI':f(cluster_controls.query("scheme=='natural-frequency' and algorithm=='K-means'").ARI_zone.iloc[0]),
 'StationClusterStability':f(km_station.min())+'--'+f(km_station.max()),
 'YearClusterStability':f(km_year.min())+'--'+f(km_year.max()),
 'StationARI':f(station.query("algorithm=='Ward' and k==5")['ARI zone'].iloc[0]),
 'SelectedFeatures':f(selection.loc['selected','macro_F1']), 'FullFeatures':f(selection.loc['74','macro_F1']),
 'TopTwenty':f(selection.loc['20','macro_F1']), 'CorePolicy':f(policy.loc['core','macro_F1']),
 'TopForty':f(selection.loc['40','macro_F1']),
 'SelectedFeatureCounts':', '.join(str(int(k)) for k in selection_details.selected_k),
 'WorstStations':', '.join(escape(r.station)+' ('+f'{r.accuracy*100:.1f}'+r'\%)' for _,r in worst.iterrows()),
 'FullPolicy':f(policy.loc['full','macro_F1']), 'IndicatorPolicy':f(policy.loc['full+flags','macro_F1']),
 'SpatialScore':f(spatial.macro_F1), 'SpatialRows':f'{int(spatial.test_rows):,}',
 'BufferedStations':str(int(spatial_folds.buffered_train_stations.min()))+'--'+str(int(spatial_folds.buffered_train_stations.max())),
 'BoundaryStations':escape(boundary.boundary_stations.iloc[0]),
 'BoundaryExclude':f(boundary.loc['exclude_boundary','macro_F1']),
 'BoundaryExcludeReference':f(boundary.loc['exclude_boundary','original_label_F1_same_rows']),
 'BoundaryAlternative':f(boundary.loc['neighbour_label','macro_F1']),
 'BestDailyName':escape(models.grouped_primary.idxmax()),'BestDailyScore':f(models.grouped_primary.max()),
 'MlpFitCount':str(len(mlp)),'MlpIterationRange':f'{mlp_iterations.min()}--{mlp_iterations.max()}',
 'MlpWarningFits':str(int(mlp_warning_fits)),
 'PythonVersion':escape(env['python']),'CacheHits':str(manifest['cache_hits']),'CacheMisses':str(manifest['cache_misses'])}
for prefix,comparison in [('HgbRf','HGB minus RF'),('HgbMlp','HGB minus MLP'),('History','HGB-14d minus HGB'),
                          ('SelectedHistory','HGB-window-selected minus HGB')]:
    row=paired.loc[comparison]
    macros[prefix+'Difference']=f(row.difference)
    macros[prefix+'Interval']=f'[{f(row.conditional_low)}, {f(row.conditional_high)}]'
macro_text='\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())

c=cluster[cluster.k==3]
cluster_table=table('Daily clustering at $k=3$; stability is mean shared-point ARI after daily subsampling.',
 'tab:cluster',['Model','Silhouette','DB','CH','ARI zone','Stability'],
 [[escape(r.algorithm),f(r.silhouette),f(r['Davies-Bouldin']),f'{r["Calinski-Harabasz"]:.0f}',f(r['ARI zone']),f(r.stability)] for _,r in c.iterrows()])
p4=read('p4_metrics');rows=[]
for split in ['random','grouped']:
    for model in ['LR','DT']:
        row=[split,model]
        for part in ['train','test','entire']:
            r=p4[(p4['split']==split)&(p4.model==model)&(p4['set']==part)].iloc[0]
            row.append(f(r.accuracy)+' / '+f(r['macro-F1']))
        rows.append(row)
basic_table=table('Default models: accuracy / five-class macro-F1. Entire-set scores reuse training data.',
 'tab:basic',['Split','Model','Train','Test','Entire'],rows,align='llccc')
holdout=read('p5_holdout');tuning_rows=[]
for split in ['random','grouped']:
    for kind in ['LR','DT']:
        default=holdout[(holdout['split']==split)&(holdout.model==kind)].iloc[0]
        tuned=holdout[(holdout['split']==split)&(holdout.model==kind+'-tuned')].iloc[0]
        params=ast.literal_eval(tuned.params)
        if kind=='LR':
            setting='$C='+str(params['logisticregression__C'])+'$; '+str(params['logisticregression__class_weight'] or 'unweighted')
        else:
            setting='depth '+str(params['decisiontreeclassifier__max_depth'] or 'unlimited')+', leaf '+str(params['decisiontreeclassifier__min_samples_leaf'])
        tuning_rows.append([split,kind,f(default.macro_F1),f(tuned.macro_F1),f(tuned.macro_AUC),setting])
tuning_table=table('Historical holdouts: validation-selected settings, before/after test macro-F1 and tuned macro-AUC. These illustrations are separate from the nested primary comparison.',
                   'tab:tuning',['Split','Model','Default F1','Tuned F1','AUC','Selected setting'],tuning_rows,align='llrrrl')
model_table=table('Daily models. Primary is seed-42 pooled OOF macro-F1; mean/range describe three reused-station partition seeds. Station-balanced gives every station equal total weight in the primary predictions.',
 'tab:models',['Model','Random','Time','Primary','Mean','Seed range','Station-bal.'],
 [[escape(k),f(r.random),f(r.time),f(r.grouped_primary),f(r.grouped_seed_mean),f(r.grouped_seed_min)+'--'+f(r.grouped_seed_max),f(r.station_balanced_macro_F1)] for k,r in models.iterrows()])
windows=read('window_comparison');choices=read('nested_window_choices');rows=[]
for kind in ['LR','HGB']:
    scores=[windows[(windows.model==kind)&(windows.window==N)&(windows.seed==42)].macro_F1.iloc[0] for N in [1,3,7,14,30,90,365]]
    choice=', '.join(str(int(v)) for v in choices[choices.model==kind].selected_window)
    rows.append([kind]+[f(s) for s in scores]+[f(nested.loc[kind,'macro_F1']),choice])
window_table=table('Primary pooled OOF scores for fixed windows and the inner-selected window procedure. Last column lists selected calendar days in outer folds 0--3; it is not chosen from outer scores.',
 'tab:windows',['Model','1 d','3 d','7 d','14 d','30 d','90 d','365 d','Selected','Choices'],rows,colsep=4)
pc=read('outer_per_class')
class_rows=[]
for z in ['Tropical','Arid','Subtropical','Mediterranean','Oceanic']:
    daily=pc[(pc.model=='HGB')&(pc.zone==z)].iloc[0]
    selected=pc[(pc.model=='HGB-window-selected')&(pc.zone==z)].iloc[0]
    class_rows.append([z,f(daily['f1-score']),f(selected.precision),f(selected.recall),f(selected['f1-score'])])
class_table=table('Primary OOF class metrics. The history model selects its window within training stations; no outer label enters this selection.',
 'tab:classes',['Zone','Daily HGB F1','Selected P','Selected R','Selected F1'],class_rows)
import sys
sys.path.insert(0,str(ROOT/'tools'))
import build_notebook_part5
discussion=build_notebook_part5.cells[-1].source.split('\n\n',1)[1]
assert 100<=len(discussion.split())<=200
template=(ROOT/'report/report_template.tex').read_text(encoding='utf-8-sig')
replacements={'@@MACROS@@':macro_text,'@@CLUSTER_TABLE@@':cluster_table,'@@BASIC_TABLE@@':basic_table,
              '@@MODEL_TABLE@@':model_table,'@@WINDOW_TABLE@@':window_table,'@@CLASS_TABLE@@':class_table,
              '@@TUNING_TABLE@@':tuning_table,'@@DISCUSSION@@':discussion}
for key,value in replacements.items():
    assert key in template,key
    template=template.replace(key,value)
assert '@@' not in template
(ROOT/'report/report_XX_TaskA.tex').write_text(template,encoding='utf-8')
(RESULT/'report_values.json').write_text(json.dumps(macros,indent=2),encoding='utf-8')
print('Generated report with',len(macros),'verified numerical/text macros. Discussion words:',len(discussion.split()))
