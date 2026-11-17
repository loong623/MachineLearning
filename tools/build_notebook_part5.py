"""Fixed-label evaluation and training-only model selection."""
import nbformat as nbf
cells=[]
md=lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code=lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
## 5. Evaluation and model validation

This review-informed protocol is not preregistration. Primary score: pooled OOF macro-F1 over the **fixed five labels**, zero_division=0. Every scorer uses this definition. Four outer folds stratify the station table; three inner folds stratify only the outer training stations. Both sides of every fold cover all five classes. Repeated seeds reuse the same network and are not independent replications.

### 5.1 Metrics, audited splits and training-only selection
""")
code(r"""
import warnings, itertools
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer, classification_report, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, label_binarize, LabelEncoder
from sklearn.inspection import permutation_importance

def macro5(yt, yp, sample_weight=None):
    return f1_score(yt,yp,labels=ZONE_ORDER,average="macro",zero_division=0,sample_weight=sample_weight)
SCORER=make_scorer(macro5)
FOLD_AUDIT=[]
def station_folds(indices,n_splits=4,seed=42,record=None):
    indices=np.asarray(indices)
    st=pd.DataFrame({"station":groups.iloc[indices].values,"zone":y[indices]}).drop_duplicates().sort_values("station")
    assert st.station.is_unique and st.zone.value_counts().min()>=n_splits
    result=[]
    for fold,(a,b) in enumerate(StratifiedKFold(n_splits,shuffle=True,random_state=seed).split(st.station,st.zone)):
        tr=indices[groups.iloc[indices].isin(st.iloc[a].station).values]
        te=indices[groups.iloc[indices].isin(st.iloc[b].station).values]
        assert not set(groups.iloc[tr]) & set(groups.iloc[te])
        assert set(y[tr])==set(ZONE_ORDER)==set(y[te])
        result.append((tr,te))
        if record:
            for station in st.iloc[b].station:
                FOLD_AUDIT.append({"scheme":record,"seed":seed,"fold":fold,"test_station":station,"zone":station_zone[station],"train_stations":";".join(sorted(set(groups.iloc[tr])))})
    assert np.array_equal(np.sort(np.concatenate([b for a,b in result])),np.sort(indices))
    return result

def time_folds(indices):
    indices=np.asarray(indices); dates=df.Date.iloc[indices]
    cuts=[dates.quantile(q).normalize() for q in [.4,.6,.8,1.0]]
    folds=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        tr=indices[(dates<a).values];te=indices[((dates>=a+pd.Timedelta(days=14))&(dates<=b)).values]
        assert len(tr) and len(te) and set(y[tr])==set(y[te])==set(ZONE_ORDER)
        assert df.Date.iloc[tr].max()<df.Date.iloc[te].min()-pd.Timedelta(days=13)
        folds.append((tr,te))
    return folds

def inner_folds(indices,protocol,seed):
    if protocol=="grouped":return station_folds(indices,3,seed)
    if protocol=="time":return time_folds(indices)
    return [(indices[a],indices[b]) for a,b in StratifiedKFold(3,shuffle=True,random_state=seed).split(indices,y[indices])]

class EncodedClassifier(ClassifierMixin, BaseEstimator):
    # Keep sklearn 1.7.2 MLP early stopping numeric; restore external labels.
    def __init__(self,estimator):
        self.estimator=estimator
    def fit(self,X,y):
        self.encoder_=LabelEncoder().fit(y)
        self.estimator_=clone(self.estimator).fit(X,self.encoder_.transform(y))
        self.classes_=self.encoder_.classes_
        self.n_iter_=self.estimator_.n_iter_
        self.n_features_in_=self.estimator_.n_features_in_
        assert np.array_equal(self.estimator_.classes_,np.arange(len(self.classes_)))
        return self
    def predict(self,X):
        return self.encoder_.inverse_transform(self.estimator_.predict(X))
    def predict_proba(self,X):
        return self.estimator_.predict_proba(X)

def model_factory(kind,seed=42):
    pre=build_preprocessor()
    if kind in ["LR","LR-tuned"]:return make_pipeline(pre,LogisticRegression(C=1,max_iter=3000))
    if kind in ["DT","DT-tuned"]:return make_pipeline(pre,DecisionTreeClassifier(random_state=seed))
    if kind=="PolyLR":return make_pipeline(pre,PolynomialFeatures(2,include_bias=False),StandardScaler(),LogisticRegression(C=.1,max_iter=3000))
    if kind=="kNN":return make_pipeline(pre,KNeighborsClassifier(n_neighbors=15,n_jobs=2))
    if kind=="RF":return make_pipeline(pre,RandomForestClassifier(n_estimators=200,min_samples_leaf=5,n_jobs=2,random_state=seed))
    if kind=="MLP":return make_pipeline(pre,EncodedClassifier(MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-3,max_iter=500,early_stopping=True,validation_fraction=.1,n_iter_no_change=20,random_state=seed)))
    if kind in ["HGB","HGB-tuned"]:return make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),HistGradientBoostingClassifier(max_iter=100,early_stopping=False,random_state=seed))
    raise ValueError(kind)
GRIDS={
 "LR-tuned":{"logisticregression__C":[.01,1,100],"logisticregression__class_weight":[None,"balanced"]},
 "DT-tuned":{"decisiontreeclassifier__max_depth":[4,8,16,None],"decisiontreeclassifier__min_samples_leaf":[5,20,100]},
 "HGB-tuned":{"histgradientboostingclassifier__learning_rate":[.05,.1],"histgradientboostingclassifier__max_leaf_nodes":[15,31],"histgradientboostingclassifier__l2_regularization":[0,1]}}

def fit_selected(kind,X,train,protocol="grouped",seed=42):
    train=np.asarray(train);model=model_factory(kind,seed);search=None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning)
        if kind in GRIDS:
            local=pd.Index(train)
            cv=[(local.get_indexer(a),local.get_indexer(b)) for a,b in inner_folds(train,protocol,seed)]
            search=GridSearchCV(model,GRIDS[kind],scoring=SCORER,cv=cv,n_jobs=1,refit=True,return_train_score=True,error_score="raise")
            search.fit(X.iloc[train],y[train]);model=search.best_estimator_
        else:model.fit(X.iloc[train],y[train])
    warning_text=[str(w.message) for w in caught]
    if warning_text:print(kind,"FIT WARNINGS",sorted(set(warning_text)),flush=True)
    meta={"selected_params":search.best_params_ if search else {},"warnings":warning_text,"iterations":getattr(model[-1],"n_iter_",None)}
    if search is not None:meta["search"]=pd.DataFrame(search.cv_results_)
    return model,meta

def score_predictions(indices,pred,proba=None):
    weights=1/groups.iloc[indices].map(groups.iloc[indices].value_counts()).to_numpy()
    result={"macro_F1":macro5(y[indices],pred),"accuracy":accuracy_score(y[indices],pred),"weighted_F1":f1_score(y[indices],pred,labels=ZONE_ORDER,average="weighted",zero_division=0),"station_balanced_macro_F1":macro5(y[indices],pred,weights),"mean_station_accuracy":accuracy_score(y[indices],pred,sample_weight=weights)}
    if proba is not None:result["macro_AUC"]=np.mean([roc_auc_score(y[indices]==z,proba[:,i]) for i,z in enumerate(ZONE_ORDER)])
    return result

def run_fit(kind,X,train,test,protocol="grouped",seed=42):
    t=time.time();model,meta=fit_selected(kind,X,train,protocol,seed)
    pred=model.predict(X.iloc[test]);p=model.predict_proba(X.iloc[test]);p=p[:,[list(model.classes_).index(z) for z in ZONE_ORDER]]
    meta["fit_predict_seconds"]=time.time()-t
    return {"pred":pred,"proba":p,"meta":meta,"scores":score_predictions(test,pred,p),"train_macro_F1":macro5(y[train],model.predict(X.iloc[train]))}
OUTER_SEEDS=[42,0,7]
OUTER={s:station_folds(np.arange(len(y)),4,s,record="outer") for s in OUTER_SEEDS}
export_table("outer_folds",pd.DataFrame(FOLD_AUDIT))
pd.DataFrame([{"seed":s,"fold":j,"train_stations":groups.iloc[a].nunique(),"test_stations":groups.iloc[b].nunique(),"test_rows":len(b),"classes":len(set(y[b]))} for s,folds in OUTER.items() for j,(a,b) in enumerate(folds)])
""")
md(r"""
### 5.2 Confusion-derived metrics, ROC/AUC and simple-model tuning
The historical holdouts remain course illustrations, not new external validation sets. Search sees their training partitions only; later OOF evaluation repeats selection inside every outer fold. Zero denominators yield zero precision/recall/F1.
""")
code(r"""
metric_rows=[]
for split in ["random","grouped"]:
    for kind in ["LR","DT"]:
        te=SPLITS[split][1];pred=preds_p4[(split,kind,"test")];cm=confusion_matrix(y[te],pred,labels=ZONE_ORDER)
        tp=np.diag(cm);p=np.divide(tp,cm.sum(0),out=np.zeros(5),where=cm.sum(0)>0);r=np.divide(tp,cm.sum(1),out=np.zeros(5),where=cm.sum(1)>0)
        f=np.divide(2*p*r,p+r,out=np.zeros(5),where=p+r>0)
        expected=precision_recall_fscore_support(y[te],pred,labels=ZONE_ORDER,zero_division=0)
        assert np.allclose(np.c_[p,r,f],np.array(expected[:3]).T)
        metric_rows.extend({"split":split,"model":kind,"zone":z,"precision":p[i],"recall":r[i],"F1":f[i]} for i,z in enumerate(ZONE_ORDER))
export_table("p5_per_class_default",pd.DataFrame(metric_rows)).round(3)
""")
code(r"""
simple_holdout={};simple_rows=[]
for split,(tr,te) in SPLITS.items():
    for kind in ["LR","LR-tuned","DT","DT-tuned"]:
        res=cached(f"p5_{split}_{kind}",lambda k=kind,a=tr,b=te,s=split:run_fit(k,X_all,a,b,s),grid=GRIDS.get(kind,{}))
        simple_holdout[(split,kind)]=res
        simple_rows.append({"split":split,"model":kind,"train_macro_F1":res["train_macro_F1"],**res["scores"],"params":repr(res["meta"]["selected_params"])})
simple_table=export_table("p5_holdout",pd.DataFrame(simple_rows))
simple_table.round(3)
""")
code(r"""
fig,axes=plt.subplots(2,2,figsize=(12,10));auc_rows=[]
for row,split in enumerate(["random","grouped"]):
    te=SPLITS[split][1]
    for col,kind in enumerate(["LR-tuned","DT-tuned"]):
        res=simple_holdout[(split,kind)]
        for j,z in enumerate(ZONE_ORDER):
            fp,tp,_=roc_curve(y[te]==z,res["proba"][:,j]);auc=roc_auc_score(y[te]==z,res["proba"][:,j])
            axes[row,col].plot(fp,tp,color=ZONE_COLORS[z],label=f"{z}: {auc:.3f}");auc_rows.append({"split":split,"model":kind,"zone":z,"AUC":auc})
        axes[row,col].plot([0,1],[0,1],'k--',lw=.8);axes[row,col].set(title=f"{kind}: {split}",xlabel="False positive rate",ylabel="True positive rate");axes[row,col].legend(fontsize=8)
plt.tight_layout();savefig("p5_roc_tuned");plt.show();export_table("p5_auc",pd.DataFrame(auc_rows))
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,kind,param in zip(axes,["LR-tuned","DT-tuned"],["logisticregression__C","decisiontreeclassifier__max_depth"]):
    for split,style in [("random","-"),("grouped","--")]:
        cv=simple_holdout[(split,kind)]["meta"]["search"].copy();cv["axis"]=cv[f"param_{param}"].map(lambda x:40 if x is None else float(x))
        best=cv.loc[cv.groupby("axis")["mean_test_score"].idxmax()].sort_values("axis")
        ax.plot(best.axis,best.mean_test_score,style,marker='o',label=f"{split}: validation");ax.plot(best.axis,best.mean_train_score,style,alpha=.4,label=f"{split}: train")
    if kind=="LR-tuned":ax.set_xscale('log')
    ax.set(xlabel="C" if kind=="LR-tuned" else "max depth (40 denotes unlimited)",ylabel="Fixed-five macro-F1",title=kind);ax.legend(fontsize=8)
plt.tight_layout();savefig("p5_validation_curves");plt.show()
""")
md(r"""
### 5.3 Discussion of model performance

The unrestricted tree can fit training observations closely, while its test confusion matrix exposes errors. Reducing depth and increasing minimum leaf size constrain this behaviour; validation chooses these constraints without consulting the outer test labels. Logistic regression provides a simpler reference, but a small training/test gap alone does not prove that all errors are bias. Climate labels are based on long-term statistics, and single days may genuinely be ambiguous. ROC curves measure ranking of one zone against the others; accuracy and F1 measure hard decisions, so their model orderings need not agree. Precision and recall also expose different costs of assigning too many or too few observations to a zone. The random holdout asks about new days of known stations; a grouped holdout asks about different stations in this network. Neither score is universally correct for every deployment setting. Differences between them are descriptive, not a causal decomposition of station information. The following experiments report nested station evaluation, fixed-label pooled scores and explicitly conditional uncertainty, and distinguish model improvement from the additional information supplied by a longer weather record.
""")
