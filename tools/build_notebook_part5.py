"""Part 5 of the project notebook: evaluation and choice of prediction model. Assembled by tools/build_notebook.py."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
## 5. Evaluation and Choice of Prediction Model

### 5.1 Metrics computed from the confusion matrix

For a multi-class confusion matrix `C` (rows = true zone, columns = predicted zone), each zone *k* is treated as
"zone *k* vs. the rest":

* `TP_k = C[k, k]`, `FP_k = Σ_i C[i, k] − TP_k` (column sum), `FN_k = Σ_j C[k, j] − TP_k` (row sum)
* precision_k = TP / (TP + FP), recall_k = TP / (TP + FN), F1_k = 2·P·R / (P + R)
* accuracy = trace(C) / ΣC
* **macro** average = plain mean over the 5 zones (each zone counts equally);
  **weighted** average = mean weighted by the number of true days of each zone.

We compute these by hand from the matrices of Part 4 and check them against `sklearn.metrics.classification_report`.
""")
code(r"""
from sklearn.metrics import classification_report, precision_recall_fscore_support

def metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    precision = tp / cm.sum(axis=0)
    recall = tp / cm.sum(axis=1)
    f1 = 2 * precision * recall / (precision + recall)
    support = cm.sum(axis=1)
    per_class = pd.DataFrame({"precision": precision, "recall": recall, "F1": f1, "support": support}, index=ZONE_ORDER)
    w = support / support.sum()
    summary = {"accuracy": tp.sum() / cm.sum(),
               "macro-P": precision.mean(), "macro-R": recall.mean(), "macro-F1": f1.mean(),
               "weighted-P": (precision * w).sum(), "weighted-R": (recall * w).sum(), "weighted-F1": (f1 * w).sum()}
    return per_class, summary

per_class_tables, summaries = {}, {}
for split in ["random", "grouped"]:
    b = SPLITS[split][1]
    for name in ["LR", "DT"]:
        cm = confusion_matrix(y[b], preds_p4[(split, name, "test")], labels=ZONE_ORDER)
        pc, sm = metrics_from_cm(cm)
        # sanity check against sklearn
        p, r, f, _ = precision_recall_fscore_support(y[b], preds_p4[(split, name, "test")], labels=ZONE_ORDER)
        assert np.allclose(pc[["precision", "recall", "F1"]].values, np.c_[p, r, f])
        per_class_tables[(split, name)] = pc; summaries[(split, name)] = sm

pd.concat({f"{n} ({s} test)": per_class_tables[(s, n)][["precision", "recall", "F1"]]
           for s in ["random", "grouped"] for n in ["LR", "DT"]}, axis=1).round(3)
""")
code(r"""
pd.DataFrame(summaries).T.round(3).rename_axis(["split", "model"])
""")
md(r"""
**Reading the per-zone metrics.** Tropical has the highest precision and recall for both models and both splits.
Arid and Mediterranean have the lowest F1 (0.31–0.59), Mediterranean particularly on unseen stations. On the random
split macro-F1 is slightly *higher* than weighted-F1, because the smallest class (Tropical, 7.6%) is the easiest
one and counts as much as the large classes in the macro average. Precision and recall of a zone can differ a lot
(LR, Arid: precision 0.54, recall 0.41): the model rarely predicts Arid, and is often right when it does.

### 5.2 ROC curves and AUC (one-vs-rest)

With 5 classes there is no single ROC curve. For each zone *k* we use the predicted probability `P(zone = k)` as a
score for the binary problem "zone *k* vs. rest" and draw one ROC curve; **macro-AUC** is the mean of the 5 AUCs.
""")
code(r"""
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

def per_class_auc(model, idx):
    proba = model.predict_proba(X_all.iloc[idx])
    classes = list(model.classes_)
    Y = label_binarize(y[idx], classes=classes)
    aucs = {z: roc_auc_score(Y[:, classes.index(z)], proba[:, classes.index(z)]) for z in ZONE_ORDER}
    aucs["macro"] = np.mean([aucs[z] for z in ZONE_ORDER])
    aucs["weighted"] = roc_auc_score(y[idx], proba, multi_class="ovr", average="weighted", labels=classes)
    return aucs, proba, classes

def plot_roc(ax, model, idx, title):
    aucs, proba, classes = per_class_auc(model, idx)
    Y = label_binarize(y[idx], classes=classes)
    for z in ZONE_ORDER:
        fpr, tpr, _ = roc_curve(Y[:, classes.index(z)], proba[:, classes.index(z)])
        ax.plot(fpr, tpr, color=ZONE_COLORS[z], lw=2, label=f"{z} (AUC {aucs[z]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title=f"{title}\nmacro-AUC = {aucs['macro']:.3f}", xlabel="false positive rate", ylabel="true positive rate")
    ax.legend(loc="lower right", fontsize=8.5)
    return aucs

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
auc_p4 = {name: plot_roc(ax, models_p4[("random", name)], rand_te, f"{name} (Part 4 default) | random test set")
          for ax, name in zip(axes, ["LR", "DT"])}
plt.tight_layout(); savefig("p5_roc_default"); plt.show()
""")
md(r"""
The default tree has pure leaves, so its "probabilities" are almost always 0 or 1. Each ROC curve therefore
collapses to a single corner point joined to (0, 0) and (1, 1), and its AUC is much lower than LR's even though
its accuracy is higher. **AUC rewards a good ranking of probabilities, which an overfitted tree cannot provide.**

### 5.3 Improving each model via validation

We tune both models with `GridSearchCV` (5-fold, scoring = macro-F1, the training scores are also recorded):

* **LR**: `C ∈ {0.001, 0.01, 0.1, 1, 10, 100}` (inverse L2 strength) × `class_weight ∈ {None, balanced}`
* **DT**: `max_depth ∈ {2, 4, 6, 8, 10, 13, 16, 20, 25, None}` × `min_samples_leaf ∈ {1, 20, 100, 400}`

The validation folds must imitate the test situation, so we run every search twice:

| Setting | Training data | CV folds |
|---|---|---|
| random | random training set | `StratifiedKFold` — validation days come from the same stations |
| unseen stations | grouped training set (33 stations) | `StratifiedGroupKFold` by station — validation stations are never in the training folds |

The four searches take ~1 min each; results are cached (Section 0). Note that `GridSearchCV` scores each fold with
sklearn's default macro-F1; under grouped CV this is biased downwards (explained in Section 5.6), which affects all
candidates alike, so it is fine for *ranking* configurations but its level should not be compared with test scores.
""")
code(r"""
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

PARAM_GRIDS = {
    "LR": {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
           "logisticregression__class_weight": [None, "balanced"]},
    "DT": {"decisiontreeclassifier__max_depth": [2, 4, 6, 8, 10, 13, 16, 20, 25, None],
           "decisiontreeclassifier__min_samples_leaf": [1, 20, 100, 400]},
}
CV_SCHEMES = {"random": StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
              "grouped": StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)}

def run_search(name, split):
    a = SPLITS[split][0]
    gs = GridSearchCV(make_model(name), PARAM_GRIDS[name], cv=CV_SCHEMES[split],
                      scoring={"macro-F1": "f1_macro", "accuracy": "accuracy"}, refit="macro-F1",
                      return_train_score=True, n_jobs=2)
    fit_kw = {"groups": groups.iloc[a].values} if split == "grouped" else {}
    gs.fit(X_all.iloc[a], y[a], **fit_kw)
    return gs

searches = {(name, split): cached(f"p5_grid_{name}_{split}", lambda n=name, s=split: run_search(n, s),
                                  grid=repr(PARAM_GRIDS[name]), cv=repr(CV_SCHEMES[split]), seed=RANDOM_STATE)
            for name in ["LR", "DT"] for split in ["random", "grouped"]}
pd.DataFrame([{"model": n, "CV setting": s,
               "best parameters": {k.split("__")[1]: v for k, v in gs.best_params_.items()},
               "best CV macro-F1": gs.best_score_,
               "train macro-F1 at best": gs.cv_results_["mean_train_macro-F1"][gs.best_index_]}
              for (n, s), gs in searches.items()]).round(3)
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
# Decision tree: validation curve over max_depth for min_samples_leaf = 1 (thin) and the selected value (thick)
depths = PARAM_GRIDS["DT"]["decisiontreeclassifier__max_depth"]
xpos = np.arange(len(depths))
for split, ls in [("random", "-"), ("grouped", "--")]:
    gs = searches[("DT", split)]; res = pd.DataFrame(gs.cv_results_)
    col = "C3" if split == "grouped" else "C0"
    best_leaf = gs.best_params_["decisiontreeclassifier__min_samples_leaf"]
    for leaf, lw in [(1, 1.0), (best_leaf, 2.5)]:
        r = res[res["param_decisiontreeclassifier__min_samples_leaf"] == leaf]
        axes[0].plot(xpos, r["mean_train_macro-F1"], ls, color="grey", lw=lw,
                     label=f"train, {split} CV, leaf={leaf}")
        axes[0].plot(xpos, r["mean_test_macro-F1"], ls, marker="o", ms=4, color=col, lw=lw,
                     label=f"validation, {split} CV, leaf={leaf}")
        if leaf == best_leaf:
            axes[0].fill_between(xpos, r["mean_test_macro-F1"] - r["std_test_macro-F1"],
                                 r["mean_test_macro-F1"] + r["std_test_macro-F1"], color=col, alpha=0.15)
axes[0].set_xticks(xpos); axes[0].set_xticklabels([str(d) for d in depths])
axes[0].set(title="Decision tree: validation curve", xlabel="max_depth", ylabel="macro-F1"); axes[0].legend(fontsize=7.5, ncol=2)
# Logistic regression: validation curve over C (at the best class_weight)
Cs = PARAM_GRIDS["LR"]["logisticregression__C"]
for split, ls in [("random", "-"), ("grouped", "--")]:
    gs = searches[("LR", split)]; res = pd.DataFrame(gs.cv_results_)
    cw = gs.best_params_["logisticregression__class_weight"]
    r = res[res["param_logisticregression__class_weight"].astype(str) == str(cw)]
    axes[1].semilogx(Cs, r["mean_train_macro-F1"], ls, marker="o", color="grey", label=f"train ({split} CV)")
    axes[1].semilogx(Cs, r["mean_test_macro-F1"], ls, marker="o", color="C3" if split == "grouped" else "C0",
                     label=f"validation ({split} CV), class_weight={cw}")
axes[1].set(title="Logistic regression: validation curve", xlabel="C (larger = weaker regularisation)", ylabel="macro-F1")
axes[1].legend(fontsize=8)
plt.tight_layout(); savefig("p5_validation_curves"); plt.show()
""")
md(r"""
**Interpretation of the validation results.**

* **Decision tree.** With `min_samples_leaf = 1` (thin lines) training macro-F1 climbs to 1.0 while validation
  macro-F1 peaks around depth 13–16 and then declines: left of the peak the tree **underfits**, right of it it
  **overfits**. The selected `min_samples_leaf = 20` (thick lines) is a second regulariser: it stops the tree from
  isolating single days, so the validation curve stays at its peak instead of declining. Both CV schemes select the
  same tree (`max_depth = 16`, `min_samples_leaf = 20`). What differs is the *level*: on unseen stations the
  validation score is about 0.26 lower and varies much more between folds (shaded band), because each fold holds
  out only a few stations per zone.
* **Logistic regression.** Under random CV the train and validation curves coincide and are flat for `C ≥ 0.01`:
  105 parameters cannot overfit 100,000 days, and the limitation is bias (linear boundaries). Under grouped CV even
  LR has a train–validation gap of about 0.16. This is not overfitting to individual days but **distribution
  shift**: new stations differ systematically from the training stations. Strong regularisation (`C = 0.01`) and
  `class_weight = balanced` are selected, but they change the score by less than 0.01.
* **The random-split CV score is an optimistic estimate for new stations.** The gap between the two validation
  curves measures how much of the score comes from recognising stations that were seen during training.

### 5.4 Tuned vs. default models on the held-out test sets
""")
code(r"""
final_rows, tuned = [], {}
for split in ["random", "grouped"]:
    a, b = SPLITS[split]
    for name in ["LR", "DT"]:
        for version, m in [("default (Part 4)", models_p4[(split, name)]), ("tuned (5.3)", searches[(name, split)].best_estimator_)]:
            if version.startswith("tuned"):
                tuned[(split, name)] = m
            tr = evaluate(m, a)[0]; te = evaluate(m, b)[0]
            aucs = per_class_auc(m, b)[0]
            final_rows.append({"split": split, "model": name, "version": version,
                               "train macro-F1": tr["macro-F1"], "test macro-F1": te["macro-F1"],
                               "gap": tr["macro-F1"] - te["macro-F1"], "test accuracy": te["accuracy"],
                               "test weighted-F1": te["weighted-F1"], "test macro-AUC": aucs["macro"]})
p5_table = pd.DataFrame(final_rows).set_index(["split", "model", "version"])
p5_table.round(3)
""")
md(r"""
**Effect of validation.**

* **Decision tree**: tuning cuts the train–test gap on the random split from 0.33 to 0.07 and raises test
  macro-F1 (0.669 → 0.691) and especially macro-AUC (0.788 → 0.889), because leaves with ≥ 20 days give graded
  probabilities. On unseen stations the gain is smaller (macro-F1 0.536 → 0.559) and a gap of about 0.22 remains.
* **Logistic regression**: tuning changes nothing on the random split and slightly lowers the score on the 15
  unseen test stations (0.561 → 0.554), even though it was the best choice in cross-validation on the 33 training
  stations. With so few stations per zone, a difference of 0.01 is within the station-to-station noise.
* After tuning the tree is clearly better than LR on known stations, but **on unseen stations the two models are
  practically equal** (macro-F1 ≈ 0.55–0.56, macro-AUC ≈ 0.81–0.82).
""")
code(r"""
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
auc_tuned = {}
for i, split in enumerate(["random", "grouped"]):
    for j, name in enumerate(["LR", "DT"]):
        what = "random test set" if split == "random" else "15 unseen stations"
        auc_tuned[(split, name)] = plot_roc(axes[i, j], tuned[(split, name)], SPLITS[split][1], f"{name} (tuned) | {what}")
plt.tight_layout(); savefig("p5_roc_tuned"); plt.show()
pd.DataFrame({f"{n} | {s}": auc_tuned[(s, n)] for s in ["random", "grouped"] for n in ["LR", "DT"]}).round(3)
""")

md(r"""
### 5.5 Error analysis: which stations are hard?

A single grouped split tests only 15 stations. To see every station as "unseen" once, we make **out-of-fold
predictions** with `StratifiedGroupKFold(5)` over all 48 stations: each day is predicted by a tuned model (grouped
CV parameters) that was trained without its station.
""")
code(r"""
def oof_predictions(name):
    params = searches[(name, "grouped")].best_params_
    model = make_model(name).set_params(**params)
    return cross_val_predict(model, X_all, y, groups=groups.values, cv=CV_SCHEMES["grouped"], n_jobs=2)

oof = {name: cached(f"p5_oof_{name}", lambda n=name: oof_predictions(n),
                    params=repr(searches[(name, "grouped")].best_params_), cv=repr(CV_SCHEMES["grouped"]))
       for name in ["LR", "DT"]}
station_acc = pd.DataFrame({name: pd.Series(oof[name] == y).groupby(groups.values).mean() for name in oof})
station_acc["Zone"] = station_zone.loc[station_acc.index]
station_acc["koppen"] = zones.set_index("Location").loc[station_acc.index, "koppen"]
for name in oof:
    print(f"{name}: out-of-fold accuracy over all 48 unseen stations = {accuracy_score(y, oof[name]):.3f}, "
          f"macro-F1 = {f1_score(y, oof[name], average='macro'):.3f}")
station_acc.sort_values("LR").round(3).head(10)
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(21, 6.3), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
xy = zones.set_index("Location").loc[station_acc.index]
for ax, name in zip(axes[:2], ["LR", "DT"]):
    sc = ax.scatter(xy["lon"], xy["lat"], c=station_acc[name], cmap="RdYlGn", vmin=0, vmax=1, s=110,
                    edgecolor=[ZONE_COLORS[z] for z in station_acc["Zone"]], linewidths=2.2)
    for loc, r in xy.iterrows():
        if station_acc.loc[loc, name] < 0.35:
            ax.annotate(loc, (r["lon"], r["lat"]), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    ax.set(title=f"{name}: accuracy at each station when it is unseen\n(edge colour = true zone)",
           xlabel="Longitude", ylabel="Latitude"); ax.set_aspect("equal")
plt.colorbar(sc, ax=axes[1], shrink=0.8).set_label("out-of-fold accuracy")
plot_cm(axes[2], y, oof["LR"], "LR out-of-fold, all 48 stations unseen")
plt.tight_layout(); savefig("p5_station_errors"); plt.show()
""")
md(r"""
**What the errors tell us.** Out-of-fold, LR and the tuned tree reach about 0.53–0.55 accuracy over all 48 unseen
stations, but the accuracy per station ranges from below 0.1 to above 0.9:

* **Tropical and coastal Subtropical stations** (Darwin, Cairns, Sydney–Brisbane coast) are recognised well.
* **Inland Subtropical stations** (Wagga Wagga, Moree, Albury) are the worst (LR accuracy 0.06–0.17). Köppen calls
  them Cfa because they have hot summers and no dry season, but their weather is continental — hot dry summers,
  cold nights — and the model calls them Arid, Mediterranean or Oceanic. In the training data "Subtropical"
  means mostly the humid coast.
* **Mediterranean stations of south-east South Australia and western Victoria** (Portland, Dartmoor, Mount Gambier,
  Nuriootpa) are mostly predicted as Oceanic: their summer is only slightly drier than their winter, which is enough
  for Köppen's Cs rule but is a weak signal in a single day.
* **Cool semi-arid stations** (Nhil, Salmon Gums, Köppen BSk) do not look like the hot deserts that dominate the
  Arid class.

These are the same groupings found by the station-profile clustering in Part 3 (coast vs. inland). The labels are
correct by the Köppen definition, but the definition uses thresholds on monthly statistics that a single day hardly
reveals. With only about 38 training stations per fold, the main limit on generalisation is the **number and
diversity of training stations**, not the number of days.

### 5.6 How reliable is one unseen-station number? Pooled vs. per-fold scores

The out-of-fold predictions above can be summarised in two ways:

* **pooled**: concatenate the predictions of all 5 folds and compute one macro-F1 over all 48 stations;
* **per fold**: compute macro-F1 inside each fold (9–11 held-out stations) and report mean ± standard deviation.

Per fold there is a subtlety. With only 4 Tropical stations, some folds contain no Tropical days at all; sklearn's
default `f1_score(average="macro")` still averages over every class that appears in the *predictions*, so a fold
with no Tropical days but a few Tropical predictions gets a Tropical F1 of 0. We therefore compute per-fold
macro-F1 over the classes **present in that fold** and show the default version for comparison.
""")
code(r"""
def per_fold_f1(pred, folds, present_only=True):
    out = []
    for _, te in folds:
        labels = np.unique(y[te]) if present_only else None
        out.append(f1_score(y[te], pred[te], labels=labels, average="macro"))
    return np.array(out)

FOLDS_42 = list(CV_SCHEMES["grouped"].split(X_all, y, groups.values))   # folds depend only on y and the stations
print("held-out stations per fold:", [groups.iloc[te].nunique() for _, te in FOLDS_42])
print("zones present per fold:   ", [len(np.unique(y[te])) for _, te in FOLDS_42])
fold_tab = {}
for name in ["LR", "DT"]:
    pf, pf_default = per_fold_f1(oof[name], FOLDS_42), per_fold_f1(oof[name], FOLDS_42, present_only=False)
    fold_tab[f"{name} (tuned)"] = {"pooled macro-F1": f1_score(y, oof[name], average="macro"),
                                   "per-fold mean (present classes)": pf.mean(), "per-fold std": pf.std(),
                                   "per-fold min": pf.min(), "per-fold max": pf.max(),
                                   "per-fold mean (sklearn default)": pf_default.mean()}
pd.DataFrame(fold_tab).T.round(3)
""")
md(r"""
Two conclusions:

1. sklearn's default per-fold macro-F1 is about 0.06 lower than the other two summaries (LR: 0.483 vs 0.544
   per fold over the classes present and 0.553 pooled) because of the absent-class effect. This is also why the
   grouped-CV scores of the grid search in Section 5.3 look low. Computed over the classes present, the per-fold
   mean is close to the pooled score.
2. Scores vary between groups of held-out stations by about ± 0.07–0.08 (LR: 0.44–0.63). The single 15-station test
   split of Part 4 is one draw from this distribution. Section 6.6 repeats the grouped CV with three fold seeds
   (15 held-out station groups), adds a time-based split, and compares models fold by fold.

### 5.7 Discussion of model performance

*(word count printed below)*
""")
DISCUSSION = (
    "Both simple models recognise tropical days reliably (one-vs-rest AUC above 0.97) but confuse the three southern "
    "zones, which overlap in feature space as already seen with t-SNE. Logistic regression shows almost no gap "
    "between training and test scores: it underfits because the zone boundaries are not linear, and tuning C does "
    "not help. The unconstrained decision tree is the opposite: perfect on the training set, clearly worse on the "
    "test set, and its near-binary probabilities give a poor ROC curve. Limiting depth and leaf size through "
    "cross-validation removes most of this overfitting and raises both macro-F1 and AUC. The most important result "
    "is the comparison of the two splits. On a random split every station is already known, so the tree partly "
    "memorises station-specific weather. On stations never seen in training, all scores drop and vary strongly "
    "between stations, and the tuned tree is no longer clearly better than logistic regression. Accuracy alone would "
    "hide these differences; macro-F1 and per-zone AUC reveal them. Further improvement needs more flexible models "
    "and features that summarise more than one day of weather (Section 6)."
)
md(DISCUSSION)
code(r"""
DISCUSSION_TEXT = '''""" + DISCUSSION + r"""'''
print("discussion word count:", len(DISCUSSION_TEXT.split()))
""")
