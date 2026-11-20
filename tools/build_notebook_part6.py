"""Nested experiments, controls, uncertainty and audit exports."""
import nbformat as nbf
cells=[]
md=lambda s:cells.append(nbf.v4.new_markdown_cell(s.strip()))
code=lambda s:cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
## 6. Open-ended exploration: weather, climate and generalisation

### 6.1 Daily model comparison under three protocols
All daily candidates receive the same random and chronological test sets and three repeats of four station folds. The primary comparison is seed-42 **pooled five-class macro-F1**, not the mean of fold F1s. Tuned candidates repeat their inner search in every outer fold. Fixed candidates are fixed configurations, not optimally tuned algorithm families. HGB early stopping is disabled (100 iterations); MLP has a 500-epoch budget and row-based early stopping within training only. Its iteration count and warnings are exported. A time split is future days of known stations, not spatial extrapolation. These protocol gaps are descriptive and are not subtracted to identify causal contributions.
""")
code(r"""
DAILY_MODELS=["LR","LR-tuned","DT-tuned","PolyLR","kNN","RF","MLP","HGB","HGB-tuned"]
cut=pd.Timestamp("2015-01-01")
time_tr=np.flatnonzero((df.Date<cut).values);time_te=np.flatnonzero((df.Date>=cut+pd.Timedelta(days=14)).values)
assert set(groups.iloc[time_te])<=set(groups.iloc[time_tr])
DAILY_RESULTS={}; OOF={}; FIT_AUDIT=[]; FOLD_RESULTS=[]

def grouped_predictions(kind,X,seed=42,tag="daily"):
    pred=np.empty(len(y),dtype=object);prob=np.zeros((len(y),len(ZONE_ORDER)))
    for fold,(tr,te) in enumerate(OUTER[seed]):
        res=cached(f"outer_{tag}_{kind}_s{seed}_f{fold}",lambda:run_fit(kind,X,tr,te,"grouped",seed),grid=GRIDS.get(kind,{}))
        pred[te]=res["pred"];prob[te]=res["proba"]
        FIT_AUDIT.append({"experiment":tag,"model":kind,"seed":seed,"fold":fold,"params":repr(res["meta"]["selected_params"]),"iterations":repr(res["meta"]["iterations"]),"warnings":repr(res["meta"]["warnings"]),"seconds":res["meta"]["fit_predict_seconds"]})
        FOLD_RESULTS.append({"experiment":tag,"model":kind,"seed":seed,"fold":fold,**res["scores"]})
    assert all(p in ZONE_ORDER for p in pred)
    return {"pred":pred,"proba":prob,"scores":score_predictions(np.arange(len(y)),pred,prob)}

for kind in DAILY_MODELS:
    for protocol,tr,te in [("random",rand_tr,rand_te),("time",time_tr,time_te)]:
        res=cached(f"daily_{protocol}_{kind}",lambda k=kind,a=tr,b=te,p=protocol:run_fit(k,X_all,a,b,p),grid=GRIDS.get(kind,{}))
        DAILY_RESULTS[(kind,protocol)]=res
        FIT_AUDIT.append({"experiment":protocol,"model":kind,"seed":42,"fold":-1,"params":repr(res["meta"]["selected_params"]),"iterations":repr(res["meta"]["iterations"]),"warnings":repr(res["meta"]["warnings"]),"seconds":res["meta"]["fit_predict_seconds"]})
    for seed in OUTER_SEEDS:OOF[(kind,seed)]=grouped_predictions(kind,X_all,seed)

model_rows=[]
for kind in DAILY_MODELS:
    repeated=[OOF[(kind,s)]["scores"]["macro_F1"] for s in OUTER_SEEDS]
    model_rows.append({"model":kind,"random":DAILY_RESULTS[(kind,"random")]["scores"]["macro_F1"],"time":DAILY_RESULTS[(kind,"time")]["scores"]["macro_F1"],"grouped_primary":repeated[0],"grouped_seed_mean":np.mean(repeated),"grouped_seed_min":min(repeated),"grouped_seed_max":max(repeated),**{k:v for k,v in OOF[(kind,42)]["scores"].items() if k!="macro_F1"}})
model_table=export_table("model_comparison",pd.DataFrame(model_rows).set_index("model"))
export_table("fit_audit",pd.DataFrame(FIT_AUDIT));export_table("fold_scores",pd.DataFrame(FOLD_RESULTS))
model_table.round(3)
""")
code(r"""
fig,ax=plt.subplots(figsize=(10,6))
model_table[["random","time","grouped_primary"]].plot.barh(ax=ax)
ax.set(xlabel="Fixed-five macro-F1",title="Daily models: different prediction populations");ax.legend(["Random: known stations","Time: future known stations","Grouped: withheld stations"])
plt.tight_layout();savefig("p6_model_comparison");plt.show()
""")
md(r"""
### 6.2 Climate memory: fixed-window contrasts and nested window selection
Trailing calendar windows include today and past weather from the same station only. Rainfall is averaged after log1p, and wind-vector means describe net components rather than mean direction. A minimum of floor(N/3) observations is required for each feature (at least one); hence an N-day window does not imply N complete observations. Season sine/cosine are retained. Computing these row-local histories before station splitting is safe because stations are disjoint. Random row splits have strongly overlapping histories and do not estimate new-station performance.

The full 1/3/7/14/30/90/365-day curve is descriptive. An additional procedure selects N by inner three-fold station CV and is evaluated on untouched outer stations. Outer scores do not choose N. Three fixed-window repeats describe partition sensitivity; no claim of a uniquely optimal 14-day window is made.
""")
code(r"""
BASE_DAILY=[c for c in FEATURE_SETS["core"] if not c.startswith("doy")]
WINDOWS=[1,3,7,14,30,90,365]
def rolling_features(N):
    t=F[BASE_DAILY].copy();t["_date"]=df.Date.values;t["_station"]=groups.values
    parts=[];counts=[]
    for _,g in t.sort_values(["_station","_date"]).groupby("_station"):
        r=g.set_index("_date")[BASE_DAILY].rolling(f"{N}D",min_periods=max(1,N//3)).mean();r.index=g.index;parts.append(r)
        c=g.set_index("_date")[BASE_DAILY].rolling(f"{N}D",min_periods=1).count();c.index=g.index;counts.append(c)
    out=pd.concat(parts).loc[F.index];out.columns=[f"{c}_mean{N}d" for c in BASE_DAILY]
    return pd.concat([out,F[["doy_sin","doy_cos"]]],axis=1),pd.concat(counts).loc[F.index]
X_WINDOWS={1:X_all};WINDOW_COUNTS={};window_diag=[]
age=(df.Date-df.Date.groupby(groups).transform("min")).dt.days
for N in WINDOWS[1:]:
    X_WINDOWS[N],WINDOW_COUNTS[N]=rolling_features(N)
    window_diag.append({"window":N,"rows_age_at_least_window":int((age>=N-1).sum()),"median_missing_fraction":X_WINDOWS[N].iloc[:,:18].isna().mean().median(),"max_missing_fraction":X_WINDOWS[N].iloc[:,:18].isna().mean().max(),"median_valid_days":WINDOW_COUNTS[N].median().median()})
export_table("window_diagnostics",pd.DataFrame(window_diag))
WINDOW_OOF={};window_rows=[]
for kind in ["LR","HGB"]:
    for N in WINDOWS:
        for seed in OUTER_SEEDS:
            r=OOF[(kind,seed)] if N==1 else grouped_predictions(kind,X_WINDOWS[N],seed,tag=f"window{N}")
            WINDOW_OOF[(kind,N,seed)]=r
            window_rows.append({"model":kind,"window":N,"seed":seed,**r["scores"]})
window_table=export_table("window_comparison",pd.DataFrame(window_rows))
window_table.pivot_table(index=["model","window"],values="macro_F1",aggfunc=["mean","min","max"]).round(3)
""")
code(r"""
NESTED_WINDOW={};nested_window_rows=[];window_choices=[]
def select_window_fold(kind,tr,te,seed):
    candidates=[]
    for N in WINDOWS:
        scores=[]
        for a,b in station_folds(tr,3,seed):
            model=model_factory(kind,seed).fit(X_WINDOWS[N].iloc[a],y[a])
            scores.append(macro5(y[b],model.predict(X_WINDOWS[N].iloc[b])))
        candidates.append({"window":N,"inner_mean_F1":np.mean(scores)})
    choice=sorted(candidates,key=lambda r:(-r["inner_mean_F1"],r["window"]))[0]["window"]
    res=run_fit(kind,X_WINDOWS[choice],tr,te,"grouped",seed)
    res["window"]=choice;res["candidates"]=candidates
    return res
for kind in ["LR","HGB"]:
    pred=np.empty(len(y),dtype=object);prob=np.zeros((len(y),5))
    for fold,(tr,te) in enumerate(OUTER[42]):
        res=cached(f"nested_window_{kind}_f{fold}",lambda:select_window_fold(kind,tr,te,42),windows=WINDOWS)
        pred[te]=res["pred"];prob[te]=res["proba"]
        window_choices.append({"model":kind,"fold":fold,"selected_window":res["window"],"outer_F1":res["scores"]["macro_F1"],"inner_results":repr(res["candidates"])})
    NESTED_WINDOW[kind]={"pred":pred,"proba":prob}
    nested_window_rows.append({"model":kind,**score_predictions(np.arange(len(y)),pred,prob)})
export_table("nested_window_choices",pd.DataFrame(window_choices))
export_table("nested_window_scores",pd.DataFrame(nested_window_rows)).round(3)
""")
code(r"""
# Same target rows for every window. This checks evaluation-population differences;
# training stays unchanged, so it does not isolate the mechanism of a 365-day effect.
common=np.flatnonzero((age>=364).values)
assert set(y[common])==set(ZONE_ORDER)
common_rows=[]
for kind in ["LR","HGB"]:
    for N in WINDOWS:
        r=WINDOW_OOF[(kind,N,42)]
        common_rows.append({"model":kind,"window":N,"rows":len(common),**score_predictions(common,r["pred"][common],r["proba"][common])})
export_table("window_common_history",pd.DataFrame(common_rows))
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for kind,color in [("LR","C0"),("HGB","C1")]:
    d=window_table[window_table.model==kind].groupby("window").macro_F1.agg(['mean','min','max']).reindex(WINDOWS)
    axes[0].plot(WINDOWS,d['mean'],marker='o',color=color,label=kind);axes[0].fill_between(WINDOWS,d['min'],d['max'],color=color,alpha=.15)
    d=pd.DataFrame(common_rows).query("model==@kind").set_index("window")
    axes[1].plot(WINDOWS,d.macro_F1,marker='o',color=color,label=kind)
for ax in axes:
    ax.set_xscale('log');ax.set_xticks(WINDOWS);ax.set_xticklabels(WINDOWS);ax.set(xlabel="Trailing calendar days",ylabel="Pooled five-class macro-F1");ax.legend()
axes[0].set_title("Three partition seeds: mean and range (not CI)")
axes[1].set_title("Same history-eligible test rows, primary seed")
plt.tight_layout();savefig("p6_climate_memory");plt.show()
""")
md(r"""
### 6.3 Feature selection without outer-test feedback
For each outer fold, split training stations into class-covered disjoint sets A/B/C. Fit an importance model on A; rank features using B labels; choose k by training on A+B and validating on C. Refit on all outer training stations with that ranking and chosen k. None of the outer-test labels enter ranking or k selection. Fixed-k curves are descriptive comparisons; only the internally chosen k procedure is a selection-performance estimate. The small A/B/C sets make rankings noisy, so report selection frequency rather than a universal physical ranking.
""")
code(r"""
X_MULTI=pd.concat([X_all]+[X_WINDOWS[N].iloc[:,:18] for N in [7,30,90]],axis=1)
KS=[3,5,10,20,40,74]
SELECTION_OOF={k:np.empty(len(y),dtype=object) for k in KS+["selected"]};selection_details=[];rank_rows=[]
def selected_features_fold(tr,te,seed):
    parts=station_folds(tr,3,seed)
    A,B,C=[b for a,b in parts]
    assert not (set(groups.iloc[A])&set(groups.iloc[B]) or set(groups.iloc[A])&set(groups.iloc[C]) or set(groups.iloc[B])&set(groups.iloc[C]))
    m=model_factory("HGB",seed).fit(X_MULTI.iloc[A],y[A])
    # Sample at most 80 days per ranking station, equal station budget.
    rng=np.random.default_rng(seed)
    sub=np.concatenate([rng.choice(B[groups.iloc[B].values==st],min(80,sum(groups.iloc[B].values==st)),replace=False) for st in sorted(set(groups.iloc[B]))])
    pi=permutation_importance(m,X_MULTI.iloc[sub],y[sub],scoring=SCORER,n_repeats=3,random_state=seed,n_jobs=1)
    order=np.argsort(-pi.importances_mean,kind="stable");ranking=X_MULTI.columns[order].tolist()
    ab=np.concatenate([A,B]);scores={}
    for k in KS:
        cols=ranking[:k];m=model_factory("HGB",seed).fit(X_MULTI.iloc[ab][cols],y[ab])
        scores[k]=macro5(y[C],m.predict(X_MULTI.iloc[C][cols]))
    chosen=min(KS,key=lambda k:(-scores[k],k));predictions={}
    for k in KS:
        cols=ranking[:k];m=model_factory("HGB",seed).fit(X_MULTI.iloc[tr][cols],y[tr])
        predictions[k]=m.predict(X_MULTI.iloc[te][cols])
    return {"predictions":predictions,"chosen":chosen,"validation":scores,"ranking":ranking,"importance":pi.importances_mean[order],"A":sorted(set(groups.iloc[A])),"B":sorted(set(groups.iloc[B])),"C":sorted(set(groups.iloc[C]))}
for fold,(tr,te) in enumerate(OUTER[42]):
    r=cached(f"feature_selection_f{fold}",lambda:selected_features_fold(tr,te,42),ks=KS)
    for k in KS:SELECTION_OOF[k][te]=r["predictions"][k]
    SELECTION_OOF["selected"][te]=r["predictions"][r["chosen"]]
    selection_details.append({"fold":fold,"selected_k":r["chosen"],"validation":repr(r["validation"]),"A":";".join(r["A"]),"B":";".join(r["B"]),"C":";".join(r["C"])})
    rank_rows.extend({"fold":fold,"rank":j+1,"feature":f,"importance":float(r["importance"][j])} for j,f in enumerate(r["ranking"]))
selection_scores=export_table("feature_selection_scores",pd.DataFrame([{"k":k,**score_predictions(np.arange(len(y)),p)} for k,p in SELECTION_OOF.items()]))
export_table("feature_selection_details",pd.DataFrame(selection_details));export_table("feature_rankings",pd.DataFrame(rank_rows))
fig,axes=plt.subplots(1,2,figsize=(12,5))
d=selection_scores.iloc[:len(KS)];axes[0].plot(KS,d.macro_F1,marker='o');axes[0].axhline(selection_scores.iloc[-1].macro_F1,ls='--',label='k selected inside training');axes[0].set(xlabel="Features retained",ylabel="Outer pooled macro-F1",title="Outer-fold-specific feature rankings");axes[0].legend()
ranks=pd.DataFrame(rank_rows);freq=ranks[ranks['rank']<=20].feature.value_counts().head(12).sort_values();freq.plot.barh(ax=axes[1]);axes[1].set(xlabel="Outer training rankings including feature in top 20",title="Ranking variability across four folds")
plt.tight_layout();savefig("p6_importance_selection");plt.show()
selection_scores.round(3)
""")
md(r"""
### 6.4 Conditional uncertainty and station-level errors
Bootstrap whole stations within each of the five climate groups, preserving their days. Paired resamples use identical station multiplicities for both models. These 95% percentile intervals are **conditional on the primary OOF predictions and this station network**: they do not include retraining, model-selection variation or unobserved spatial dependence. No independent-fold t-test or equivalence claim is made. Station-balanced metrics give each observed station equal total weight before calculating the five-class score.
""")
code(r"""
BOOT_MODELS={k:OOF[(k,42)]["pred"] for k in DAILY_MODELS}
BOOT_MODELS.update({"HGB-14d":WINDOW_OOF[("HGB",14,42)]["pred"],"LR-14d":WINDOW_OOF[("LR",14,42)]["pred"],"HGB-window-selected":NESTED_WINDOW["HGB"]["pred"],"HGB-features-selected":SELECTION_OOF["selected"]})
st_names=np.array(sorted(set(groups)));st_zones=station_zone.loc[st_names].values
+def_cm_placeholder=0

def f1_from_cm(cm):
    diag=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(-1)+cm.sum(-2)
    return np.divide(2*diag,den,out=np.zeros_like(diag,dtype=float),where=den>0).mean(-1)
CM_BY_MODEL={k:np.stack([confusion_matrix(y[groups.values==s],p[groups.values==s],labels=ZONE_ORDER) for s in st_names]) for k,p in BOOT_MODELS.items()}
rng=np.random.default_rng(20261117);weights=np.zeros((2000,len(st_names)),dtype=int)
for z in ZONE_ORDER:
    ids=np.flatnonzero(st_zones==z)
    draws=rng.choice(ids,(len(weights),len(ids)),replace=True)
    for b,draw in enumerate(draws):np.add.at(weights[b],draw,1)
BOOT={k:f1_from_cm(np.einsum('bs,sij->bij',weights,cm)) for k,cm in CM_BY_MODEL.items()}
ci_rows=[]
for k,p in BOOT_MODELS.items():
    lo,hi=np.quantile(BOOT[k],[.025,.975]);ci_rows.append({"model":k,"pooled_F1":macro5(y,p),"conditional_low":lo,"conditional_high":hi})
export_table("conditional_intervals",pd.DataFrame(ci_rows))
pairs=[("HGB","LR"),("HGB","RF"),("HGB","MLP"),("DT-tuned","LR-tuned"),("HGB-14d","HGB"),("LR-14d","LR"),("HGB-window-selected","HGB")]
paired_rows=[]
for a,b in pairs:
    lo,hi=np.quantile(BOOT[a]-BOOT[b],[.025,.975]);paired_rows.append({"comparison":a+" minus "+b,"difference":macro5(y,BOOT_MODELS[a])-macro5(y,BOOT_MODELS[b]),"conditional_low":lo,"conditional_high":hi})
export_table("paired_intervals",pd.DataFrame(paired_rows)).round(3)
""".replace('+def_cm_placeholder=0',''))
code(r"""
station_rows=[];class_rows=[]
for kind,pred in BOOT_MODELS.items():
    pc=classification_report(y,pred,labels=ZONE_ORDER,output_dict=True,zero_division=0)
    class_rows.extend({"model":kind,"zone":z,**pc[z]} for z in ZONE_ORDER)
    for s in st_names:
        mask=groups.values==s;station_rows.append({"model":kind,"station":s,"zone":station_zone[s],"days":int(mask.sum()),"accuracy":accuracy_score(y[mask],pred[mask])})
export_table("station_errors",pd.DataFrame(station_rows));export_table("outer_per_class",pd.DataFrame(class_rows))
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,kind in zip(axes,["HGB","HGB-window-selected"]):plot_cm(ax,y,BOOT_MODELS[kind],kind+" | outer OOF")
plt.tight_layout();savefig("p6_final_confusion");plt.show()
""")
md(r"""
### 6.5 Stress tests: missingness, spatial separation and label boundary
These checks keep model/feature policies fixed. They are sensitivities on the same network, not new external evidence. A 100-km training exclusion changes training size and station composition, so it does not isolate spatial dependence. The annotation sensitivity uses alternative neighbouring raster pixels and does not replace validation against an independent climate map.
""")
code(r"""
# Indicator policies are compared, not selected from the resulting outer scores.
POLICY={"core":X_all,"full":F[FEATURE_SETS['full']],"full+flags":pd.concat([F[FEATURE_SETS['full']],F[FEATURE_SETS['full']].isna().astype(float).add_suffix('_missing')],axis=1)}
policy_rows=[]
for name,X in POLICY.items():
    res=OOF[("HGB",42)] if name=="core" else grouped_predictions("HGB",X,42,tag="policy_"+name)
    policy_rows.append({"policy":name,**res['scores']})
export_table("missingness_policies",pd.DataFrame(policy_rows))

def haversine_matrix(a,b):
    la1,lo1=np.deg2rad(a[:,0])[:,None],np.deg2rad(a[:,1])[:,None]
    la2,lo2=np.deg2rad(b[:,0])[None,:],np.deg2rad(b[:,1])[None,:]
    h=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 6371*2*np.arcsin(np.sqrt(np.clip(h,0,1)))
coordinates=zones.set_index('Location')[['lat','lon']]
spatial_rows=[];spatial_pred=np.empty(len(y),dtype=object);spatial_pred[:]=None
for fold,(tr,te) in enumerate(OUTER[42]):
    train_st=sorted(set(groups.iloc[tr]));test_st=sorted(set(groups.iloc[te]))
    distances=haversine_matrix(coordinates.loc[train_st].values,coordinates.loc[test_st].values)
    kept=np.array(train_st)[distances.min(1)>=100]
    a=tr[groups.iloc[tr].isin(kept).values]
    supported=set(y[a])==set(ZONE_ORDER)
    row={"fold":fold,"original_train_stations":len(train_st),"buffered_train_stations":len(kept),"supported":supported}
    if supported:
        res=cached(f"spatial_buffer100_f{fold}",lambda:run_fit("HGB",X_all,a,te))
        spatial_pred[te]=res['pred'];row.update(res['scores'])
    spatial_rows.append(row)
export_table("spatial_buffer_folds",pd.DataFrame(spatial_rows))
valid=np.flatnonzero(pd.notna(spatial_pred))
export_table("spatial_buffer_summary",pd.DataFrame([{"test_rows":len(valid),"complete_coverage":len(valid)==len(y),**score_predictions(valid,spatial_pred[valid])}]))

boundary=zones[zones.zone_boundary & zones.Zone.notna()].Location.tolist()
label_rows=[]
for mode in ['exclude_boundary','neighbour_label']:
    pred=np.empty(len(y),dtype=object);pred[:]=None;alt_y=y.copy()
    if mode=='neighbour_label':
        for station in boundary:
            z=zones.set_index('Location').loc[station];alternatives=sorted({KOPPEN_TO_ZONE[c] for c in z.neighbour_codes.split(';') if c in KOPPEN_TO_ZONE and KOPPEN_TO_ZONE[c]!=z.Zone})
            if alternatives:alt_y[groups.values==station]=alternatives[0]
    for fold,(tr,te) in enumerate(OUTER[42]):
        if mode=='exclude_boundary':
            tr=tr[~groups.iloc[tr].isin(boundary).values];te=te[~groups.iloc[te].isin(boundary).values]
        def fit_alt():
            m=model_factory('HGB').fit(X_all.iloc[tr],alt_y[tr]);return m.predict(X_all.iloc[te])
        pred[te]=cached(f"labels_{mode}_f{fold}",fit_alt,stations=boundary)
    valid=np.flatnonzero(pd.notna(pred))
    label_rows.append({"mode":mode,"boundary_stations":';'.join(boundary),"rows":len(valid),"macro_F1":macro5(alt_y[valid],pred[valid]),"original_label_F1_same_rows":macro5(y[valid],OOF[('HGB',42)]['pred'][valid])})
export_table('label_sensitivity',pd.DataFrame(label_rows)).round(3)
""")
md(r"""
### 6.6 Reproducibility and interpretation
All results belong to this fixed station network and one manually matched raster annotation. More stations, an independent map and spatially independent sampling would address limits that rerunning cannot remove. Paired conditional intervals do not correct prior exploration or prove model equivalence. Selection-performance results describe the complete internal selection procedure; fixed-window and fixed-k curves describe those configurations without claiming an independently verified optimum. A common-history subset controls which rows are evaluated, not all mechanisms behind window effects.

Raw observations are unchanged. The executable-cell hash, annotation hashes, Python and installed distributions identify the run. Predictions, fold membership, timing, convergence diagnostics and machine-readable scores are exported. Old caches are not used by this revision.
""")
code(r"""
export_table('fit_audit',pd.DataFrame(FIT_AUDIT));export_table('fold_scores',pd.DataFrame(FOLD_RESULTS))
export_table('p4_metrics',p4_table);export_table('tsne_diagnostics',quant);export_table('clustering_sweep',sweep);export_table('station_clustering',st_sweep)
export_table('preprocessing_probe',pd.DataFrame(rows) if isinstance(rows,list) and rows and 'features' in rows[0] else sens)
export_table('cache_events',pd.DataFrame(CACHE_EVENTS))
arrays={'y':np.array(y,dtype=str),'station':groups.to_numpy(dtype=str),'date':df.Date.astype(str).to_numpy(dtype=str)}
for k,p in BOOT_MODELS.items():arrays[k]=np.array(p,dtype=str)
for (kind,seed),r in OOF.items():arrays[f'daily_{kind}_s{seed}']=np.array(r['pred'],dtype=str)
for (kind,N,seed),r in WINDOW_OOF.items():arrays[f'window_{kind}_{N}_s{seed}']=np.array(r['pred'],dtype=str)
pred_path = RESULT_DIR / 'predictions.npz'
pred_tmp = RESULT_DIR / 'predictions.base.tmp.npz'
if pred_tmp.exists(): pred_tmp.unlink()
np.savez_compressed(pred_tmp, **arrays)
pred_tmp.replace(pred_path)
manifest={'run_id':RUN_ID,'code_sha256':code_digest,'rows':len(y),'stations':len(st_names),'outer_seeds':OUTER_SEEDS,'outer_folds':4,'metric':'pooled fixed-five macro-F1','cache_hits':sum(e['hit'] for e in CACHE_EVENTS),'cache_misses':sum(not e['hit'] for e in CACHE_EVENTS),'completed_utc':pd.Timestamp.now(tz='UTC').isoformat()}
(RESULT_DIR/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
""")
