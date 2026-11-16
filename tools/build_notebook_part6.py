"""Part 6 of the project notebook: open-ended exploration. Assembled by tools/build_notebook.py."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
## 6. Open-ended Exploration

Parts 4–5 left two open questions: (1) do more flexible model classes help, and do they help on *unseen stations*
or only on known ones? (2) Part 3 showed that a station's *climate profile* identifies its zone much better than a
single day — can we give the classifier "more than one day" of weather?

We explore four directions suggested in the task description:

| Section | Direction | Technique |
|---|---|---|
| 6.1 | Model comparison | 7 model classes compared with cross-validation, random vs. station-grouped |
| 6.2 | Model complexity / regulariser | polynomial features + L2 in logistic regression (inside 6.1) |
| 6.3 | Hyperparameter tuning | randomised search for gradient boosting with station-grouped CV |
| 6.4 | Feature engineering | physically derived features; "climate memory" rolling-window features |
| 6.5 | Feature selection | permutation importance; performance vs. number of selected features |

**Evaluation protocol (all of Section 6).** For every model we report two numbers:

* **random**: macro-F1 on the random 30% test set (model trained on the random 70%);
* **unseen stations**: out-of-fold predictions from `StratifiedGroupKFold(5)` over all 48 stations, so every station
  is predicted once by a model that never saw it. We report the pooled macro-F1 and the per-fold mean ± std
  (classes present in the fold, Section 5.6). This uses all stations and is less noisy than the single 15-station
  test set.

All models use the same 20 `core` features, the same median imputation (fitted inside each training fold), the same
random 70/30 split and the same five station folds. Section 6.6 adds a time-based split and two more fold seeds for the
key models.
""")
code(r"""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import loguniform, randint
import warnings

GCV = CV_SCHEMES["grouped"]          # StratifiedGroupKFold(5, shuffle=True, random_state=42)
best_dt = searches[("DT", "grouped")].best_params_

MODEL_ZOO = {
    "Logistic Regression": lambda: make_pipeline(build_preprocessor(), LogisticRegression(C=1, max_iter=3000)),
    "LR + degree-2 polynomial (C=0.1)": lambda: make_pipeline(build_preprocessor(), PolynomialFeatures(2), StandardScaler(),
                                                             LogisticRegression(C=0.1, max_iter=3000)),
    "Decision Tree (tuned)": lambda: make_model("DT").set_params(**best_dt),
    "k-NN (k=15)": lambda: make_pipeline(build_preprocessor(), KNeighborsClassifier(n_neighbors=15)),
    "Random Forest (200 trees)": lambda: make_pipeline(build_preprocessor(),
                                                       RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                                              n_jobs=2, random_state=RANDOM_STATE)),
    "MLP (64-32)": lambda: make_pipeline(build_preprocessor(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                                             max_iter=60, random_state=RANDOM_STATE)),
    "Gradient Boosting (HGB, default)": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                                              HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
}

def two_protocols(make, X):
    # returns random-split test macro-F1, grouped out-of-fold predictions, and timing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # MLP: max_iter reached (intentional, keeps the run short)
        t0 = time.time()
        m = make().fit(X.iloc[rand_tr], y[rand_tr])
        fit_time = time.time() - t0
        rand_f1 = f1_score(y[rand_te], m.predict(X.iloc[rand_te]), average="macro")
        oof_pred = cross_val_predict(make(), X, y, groups=groups.values, cv=GCV, n_jobs=2)
    return {"random test macro-F1": rand_f1, "oof": oof_pred, "fit time (s)": fit_time}

GROUPED_FOLDS = list(GCV.split(X_all, y, groups.values))    # identical to FOLDS_42 of Section 5.6

def summarise(res):
    pf = per_fold_f1(res["oof"], GROUPED_FOLDS)
    return {"random test macro-F1": res["random test macro-F1"],
            "unseen stations macro-F1": f1_score(y, res["oof"], average="macro"),
            "unseen per-fold mean": pf.mean(), "unseen per-fold std": pf.std(),
            "unseen stations accuracy": accuracy_score(y, res["oof"]),
            "fit time (s)": res["fit time (s)"]}
""")

md(r"""
### 6.1–6.2 Model comparison (7 model classes) and model complexity

Why these models: **k-NN** (instance-based, no training), **Random Forest** (bagging of deep trees, reduces the
variance we saw in Part 4), **Gradient Boosting** (sequentially fitted shallow trees, reduces bias), **MLP**
(non-linear features learned by a neural network), and **LR with degree-2 polynomial features** (the same linear
model with 230 interaction/square features plus a stronger L2 penalty — a direct test of model complexity vs. regularisation).
""")
code(r"""
zoo = {name: cached("p6_zoo_" + "".join(ch for ch in name if ch.isalnum()), lambda mk=mk: two_protocols(mk, X_all),
                    model=name, features="core", best_dt=repr(best_dt))
       for name, mk in MODEL_ZOO.items()}
zoo_table = pd.DataFrame({n: summarise(r) for n, r in zoo.items()}).T
zoo_table["gap (random − unseen)"] = zoo_table["random test macro-F1"] - zoo_table["unseen stations macro-F1"]
zoo_table.sort_values("unseen stations macro-F1", ascending=False).round(3)
""")
md(r"""
**Exact configuration of every model** (read from the fitted pipelines, so the table cannot drift from the code).
Fit time is for one fit on the random 70% training set (~100,000 days) on 2 CPU cores.
""")
code(r"""
KEY_PARAMS = {"LogisticRegression": ["C", "penalty", "solver", "max_iter", "class_weight"],
              "DecisionTreeClassifier": ["criterion", "max_depth", "min_samples_leaf"],
              "KNeighborsClassifier": ["n_neighbors", "weights", "metric"],
              "RandomForestClassifier": ["n_estimators", "min_samples_leaf", "max_features", "bootstrap"],
              "MLPClassifier": ["hidden_layer_sizes", "activation", "solver", "alpha", "batch_size", "max_iter"],
              "HistGradientBoostingClassifier": ["learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf",
                                                 "l2_regularization", "early_stopping"]}
spec_rows = []
for name, mk in MODEL_ZOO.items():
    pipe = mk(); est = pipe[-1]; prm = est.get_params()
    pre = []
    for step in pipe[:-1]:
        pre += [type(t).__name__ for _, t in step.steps] if isinstance(step, Pipeline) else [type(step).__name__]
    spec_rows.append({"model": name, "preprocessing (fitted on training rows only)": " → ".join(pre),
                      "key hyperparameters": ", ".join(f"{k}={prm[k]}" for k in KEY_PARAMS[type(est).__name__]),
                      "fit time (s)": round(zoo[name]["fit time (s)"], 1)})
with pd.option_context("display.max_colwidth", 120):
    display(pd.DataFrame(spec_rows).set_index("model"))
print("HGB early_stopping='auto' is active because n_samples > 10,000 (10% of the training rows are held out "
      "internally; at most max_iter boosting rounds).")
""")
code(r"""
zt = zoo_table.sort_values("unseen stations macro-F1")
fig, ax = plt.subplots(figsize=(11, 5.5))
ypos = np.arange(len(zt))
ax.barh(ypos + 0.2, zt["random test macro-F1"], height=0.4, color="#9ecae1", label="random split (stations known)")
ax.barh(ypos - 0.2, zt["unseen stations macro-F1"], height=0.4, color="#08519c", label="unseen stations (grouped CV)")
for i, (r, u) in enumerate(zip(zt["random test macro-F1"], zt["unseen stations macro-F1"])):
    ax.text(r + 0.005, i + 0.2, f"{r:.3f}", va="center", fontsize=8); ax.text(u + 0.005, i - 0.2, f"{u:.3f}", va="center", fontsize=8)
ax.set_yticks(ypos); ax.set_yticklabels(zt.index); ax.set_xlim(0, 1.2)
ax.set(xlabel="macro-F1", title="Model comparison: the more flexible the model, the larger the gap between known and unseen stations")
ax.legend(loc="lower right"); plt.tight_layout(); savefig("p6_model_comparison"); plt.show()
""")
md(r"""
**Findings.**

* On the random split the flexible models (gradient boosting, random forest, MLP, k-NN, polynomial LR) beat plain
  LR by 0.14–0.18 macro-F1. On unseen stations they still win, in almost the same order, but by only 0.06–0.11.
  The gap column grows with flexibility (LR 0.09, k-NN 0.18): part of what flexible models learn on a random split
  is station identity. k-NN, which literally looks up the most similar training days, has the largest gap.
* **Model complexity vs. regularisation**: degree-2 polynomial features (230 features, `C = 0.1`) raise LR from
  0.615 to 0.752 on the random split and from 0.525 to 0.623 on unseen stations. Interactions such as
  *MinTemp × Humidity* remove much of LR's bias, and this gain does transfer to new stations.
* Gradient boosting is the best model under both protocols and the fastest of the strong models (≈ 4 s per fit);
  MLP and random forest are within 0.01. We tune it in 6.3 and use it for the feature experiments.
""")

md(r"""
### 6.3 Hyperparameter tuning: randomised search for gradient boosting

Randomised search (`ParameterSampler`, the sampler used inside `RandomizedSearchCV`) draws 16 configurations from continuous/discrete distributions instead of a fixed grid,
evaluated with station-grouped 5-fold CV on the grouped training set (so the selection criterion is "performance on
unseen stations"). The tuned model is then tested once on the 15 held-out stations.
""")
code(r"""
HGB_SPACE = {"histgradientboostingclassifier__learning_rate": loguniform(0.02, 0.3),
             "histgradientboostingclassifier__max_leaf_nodes": randint(8, 64),
             "histgradientboostingclassifier__min_samples_leaf": randint(20, 400),
             "histgradientboostingclassifier__l2_regularization": loguniform(1e-3, 10),
             "histgradientboostingclassifier__max_iter": randint(100, 400)}

HGB_DEFAULT = lambda: make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(random_state=RANDOM_STATE))
# Equivalent to RandomizedSearchCV(n_iter=16), written as an explicit loop so that every sampled
# configuration is cached on its own (one configuration = 5 fits, ~20-40 s).
from sklearn.model_selection import ParameterSampler, cross_validate
rs_rows = []
for i, params in enumerate(ParameterSampler(HGB_SPACE, n_iter=16, random_state=RANDOM_STATE)):
    cvres = cached("p6_hgb_cfg", lambda params=params: cross_validate(
        HGB_DEFAULT().set_params(**params), X_all.iloc[grp_tr], y[grp_tr], groups=groups.iloc[grp_tr].values,
        cv=GCV, scoring="f1_macro", return_train_score=True, n_jobs=2), params=repr(sorted(params.items())), cv=repr(GCV))
    rs_rows.append({**{k.split("__")[1]: v for k, v in params.items()},
                    "mean_train_score": cvres["train_score"].mean(), "mean_test_score": cvres["test_score"].mean(),
                    "std_test_score": cvres["test_score"].std(), "_params": params})
rs_res = pd.DataFrame(rs_rows).sort_values("mean_test_score", ascending=False)
best_hgb_params = rs_res["_params"].iloc[0]
hgb_tuned = HGB_DEFAULT().set_params(**best_hgb_params).fit(X_all.iloc[grp_tr], y[grp_tr])
display(rs_res.drop(columns="_params").head(6).round(3))

hgb_default_cv = cached("p6_hgb_default_groupedcv", lambda: cross_val_score(
    HGB_DEFAULT(), X_all.iloc[grp_tr], y[grp_tr], groups=groups.iloc[grp_tr].values, cv=GCV,
    scoring="f1_macro", n_jobs=2).mean(), cv=repr(GCV))
hgb_default = HGB_DEFAULT().fit(X_all.iloc[grp_tr], y[grp_tr])
tuning_cmp = pd.DataFrame({
    "default HGB": {"grouped CV macro-F1 (33 training stations)": hgb_default_cv,
                    "test macro-F1 (15 unseen stations)": f1_score(y[grp_te], hgb_default.predict(X_all.iloc[grp_te]), average="macro")},
    "tuned HGB": {"grouped CV macro-F1 (33 training stations)": rs_res["mean_test_score"].iloc[0],
                  "test macro-F1 (15 unseen stations)": f1_score(y[grp_te], hgb_tuned.predict(X_all.iloc[grp_te]), average="macro")},
}).T
tuning_cmp.round(3)
""")
code(r"""
fig, ax = plt.subplots(figsize=(7.5, 5))
sc = ax.scatter(rs_res["mean_train_score"], rs_res["mean_test_score"],
                c=rs_res["max_leaf_nodes"].astype(float), cmap="viridis", s=60)
plt.colorbar(sc).set_label("max_leaf_nodes")
best = rs_res.iloc[0]
ax.scatter(best["mean_train_score"], best["mean_test_score"], s=250, facecolors="none", edgecolors="r", lw=2, label="selected")
ax.set(xlabel="mean training macro-F1", ylabel="mean validation macro-F1 (unseen stations)",
       title="16 random HGB configurations: training scores vary,\nvalidation on unseen stations hardly changes")
ax.legend(); plt.tight_layout(); savefig("p6_hgb_search"); plt.show()
""")
md(r"""
**Findings.** All 16 configurations fit the training stations very well (macro-F1 0.93–0.97) but score
0.47–0.48 on validation stations; the best is only 0.009 above the default model, and on the 15 test stations the
gain is 0.004. The hyperparameters are not what limits gradient boosting here: the validation stations simply
differ from the training stations. (The grouped-CV level, 0.48, is lower than the 15-station test score, 0.63, for
two reasons. First, the search uses sklearn's default per-fold macro-F1, which counts a class with F1 = 0 in folds
where it is predicted but absent (Section 5.6). Second, each fold holds out only 6–7 of 33 stations, so the score
depends strongly on which stations are held out. Configurations are therefore compared only with each other.)
""")

md(r"""
### 6.4 Feature engineering

**(a) Physically derived features.** Meteorologists describe a climate by daily *ranges and changes*, not only
levels. We add four features: diurnal temperature range `MaxTemp − MinTemp` (large in dry climates), humidity
drop `Humidity9am − Humidity3pm`, pressure change `Pressure3pm − Pressure9am`, and warming `Temp3pm − Temp9am`.

**(b) "Climate memory": how many days do you need to know where you are?** Part 3 showed that a station's
long-term profile reveals its zone far better than a single day. We therefore replace each day's measurements by
their mean over the last N calendar days at the same station. For station *s*, day *t* and daily feature *j*:

$$\bar x^{(N)}_{s,t,j} \;=\; \frac{1}{|W|}\sum_{\tau \in W} x_{s,\tau,j}, \qquad
W = \{\tau : t-N < \tau \le t,\; x_{s,\tau,j}\ \text{observed}\},$$

defined only if $|W| \ge \max(1, \lfloor N/3 \rfloor)$, otherwise missing.

* The window contains day *t* itself and the N − 1 days before it, and **never a later day**
  (pandas time-based `rolling("ND")`, which is right-closed).
* It is measured in **calendar days**: a gap in the record shortens the window's count instead of reaching further
  back. Missing values inside the window are skipped.
* At the start of a station's record (fewer than ⌊N/3⌋ observed days) the value is missing; like every other missing
  value it is filled by the median imputer that is fitted inside each training fold.
* The 18 daily features are averaged; `doy_sin` / `doy_cos` of day *t* are kept unchanged.

**Why computing the windows once on the full table does not leak in grouped CV.** A row's window uses only the same
station's own weather (no labels, no statistics learned across stations), and under station-grouped CV every station
is entirely in either the training or the validation fold. No information can pass from validation rows to
training rows. Under a **random** split this is not true: neighbouring days of the same station share most of their
window and land on both sides, so almost every test day has a near-copy in the training set. We show the
random-split number once as a warning; the time-based split in 6.6 uses a 14-day embargo for the same reason.

The question becomes: *"If you were dropped at an unknown Australian weather station and watched the weather for
N days, how well could you tell its climate zone?"*
""")
code(r"""
BASE_DAILY = [c for c in FEATURE_SETS["core"] if not c.startswith("doy")]

def derived_features(Fr):
    out = Fr[FEATURE_SETS["core"]].copy()
    out["TempRange"] = Fr["MaxTemp"] - Fr["MinTemp"]
    out["HumidityDrop"] = Fr["Humidity9am"] - Fr["Humidity3pm"]
    out["PressureChange"] = Fr["Pressure3pm"] - Fr["Pressure9am"]
    out["Warming"] = Fr["Temp3pm"] - Fr["Temp9am"]
    return out

def rolling_features(window_days, cols=BASE_DAILY):
    # trailing N-day mean per station (past + today only); row order is restored at the end
    t = F[cols].copy(); t["_date"] = df["Date"].values; t["_st"] = groups.values
    t = t.sort_values(["_st", "_date"])
    parts = []
    for _, s in t.groupby("_st"):
        r = s.set_index("_date")[cols].rolling(f"{window_days}D", min_periods=max(1, window_days // 3)).mean()
        r.index = s.index; parts.append(r)
    out = pd.concat(parts).loc[F.index]
    out.columns = [f"{c}_mean{window_days}d" for c in cols]
    return out

X_derived = derived_features(F)
WINDOWS = [1, 3, 7, 14, 30, 90, 365]
X_windows = {N: (X_all if N == 1 else pd.concat([rolling_features(N), F[["doy_sin", "doy_cos"]]], axis=1)) for N in WINDOWS}
X_multi = pd.concat([X_all] + [rolling_features(N) for N in (7, 30, 90)], axis=1)   # today + 7/30/90-day means
print("derived:", X_derived.shape[1], "features | rolling:", X_windows[30].shape[1], "| multi-scale:", X_multi.shape[1])
miss14 = X_windows[14][[f"{c}_mean14d" for c in BASE_DAILY]].isna().mean()
print(f"14-day means missing before imputation: median {miss14.median():.2%}, max {miss14.max():.2%} ({miss14.idxmax()})")
""")
code(r"""
HGB = MODEL_ZOO["Gradient Boosting (HGB, default)"]
fe_results = {"core (20 daily features)": zoo["Gradient Boosting (HGB, default)"]}
fe_results["core + 4 derived"] = cached("p6_fe_derived", lambda: two_protocols(HGB, X_derived), version=1)
for N in WINDOWS[1:]:
    fe_results[f"{N}-day trailing mean"] = cached(f"p6_fe_roll{N}", lambda N=N: two_protocols(HGB, X_windows[N]), window=N, version=1)
fe_results["multi-scale: today + 7/30/90-day means"] = cached("p6_fe_multi", lambda: two_protocols(HGB, X_multi), windows=(7, 30, 90), version=1)
fe_table = pd.DataFrame({k: summarise(v) for k, v in fe_results.items()}).T.drop(columns="fit time (s)")
fe_table.round(3)
""")
code(r"""
curve = fe_table.loc[["core (20 daily features)"] + [f"{N}-day trailing mean" for N in WINDOWS[1:]]]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(WINDOWS, curve["unseen stations macro-F1"], marker="o", lw=2.5, color="#08519c", label="unseen stations (grouped CV)")
ax.plot(WINDOWS, curve["random test macro-F1"], marker="o", ls="--", color="#9ecae1", label="random split (leaks: windows overlap)")
ax.axhline(fe_table.loc["multi-scale: today + 7/30/90-day means", "unseen stations macro-F1"], color="C2", ls=":",
           label="multi-scale (today + 7/30/90 d), unseen stations")
ax.set_xscale("log"); ax.set_xticks(WINDOWS); ax.set_xticklabels(WINDOWS)
ax.set(xlabel="days of weather observed at the station (window length N)", ylabel="macro-F1 (HGB)",
       title="How many days of weather do you need to tell the climate zone?")
ax.legend(); plt.tight_layout(); savefig("p6_climate_memory"); plt.show()
""")
md(r"""
**Findings.**

* The four derived features change little (+0.003): gradient boosting can already combine two inputs through
  successive splits.
* **Averaging the weather over a few days helps strongly on unseen stations**: macro-F1 rises from 0.638 (one day)
  to 0.710 (one week) and peaks at 0.727 for two weeks. A single day is dominated by the current weather situation
  (Part 3's "weather regimes"); averaging cancels the day-to-day noise and leaves the climate signal.
* **Longer windows do not help, and a one-year mean is clearly worse (0.655).** A one-year mean removes the seasonal
  cycle — exactly the information (summer vs. winter rain, seasonal temperature range) that Köppen uses to define
  the zones.
* Combining several time scales (today + 7/30/90-day means, 74 features) gives the best accuracy (0.730) but not a
  better macro-F1 (0.718) than the 14-day window alone; the differences between 7 and 90 days are within noise.
* The random-split curve reaches 0.99 for windows of 30 days or more: with overlapping windows, a test day has an
  almost identical neighbour in the training set. This is the strongest example of station leakage in the project
  and the reason we only trust the unseen-station curve.

### 6.5 Feature selection: what does the model use?

Permutation importance of the best model (multi-scale HGB trained on the grouped training set): how much
macro-F1 on the 15 unseen test stations drops when one feature is randomly shuffled (3 repeats, 10,000 random test days).
We then retrain using only the top-k features to see how many are actually needed.
""")
code(r"""
def importance():
    model = HGB().fit(X_multi.iloc[grp_tr], y[grp_tr])
    sub = np.random.default_rng(RANDOM_STATE).choice(grp_te, 10000, replace=False)
    pi = permutation_importance(model, X_multi.iloc[sub], y[sub], scoring="f1_macro", n_repeats=3,
                                random_state=RANDOM_STATE, n_jobs=2)
    return pd.DataFrame({"mean": pi.importances_mean, "std": pi.importances_std},
                        index=X_multi.columns).sort_values("mean", ascending=False)

imp = cached("p6_importance", importance, version=1, cols=tuple(X_multi.columns))
sel_rows = []
for k in [3, 5, 10, 20, 40]:
    cols = list(imp.index[:k])
    p = cached(f"p6_select_top{k}", lambda cols=cols: cross_val_predict(HGB(), X_multi[cols], y, groups=groups.values,
                                                                        cv=GCV, n_jobs=2), cols=tuple(cols))
    sel_rows.append({"top-k features": k, "unseen stations macro-F1": f1_score(y, p, average="macro"),
                     "features": ", ".join(cols[:5]) + (" ..." if k > 5 else "")})
selection = pd.DataFrame(sel_rows)
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.4, 1]})
top = imp.head(15)[::-1]
axes[0].barh(top.index, top["mean"], xerr=top["std"], color="#3182bd")
axes[0].set(title="Permutation importance (drop in macro-F1 on unseen stations), top 15", xlabel="Δ macro-F1")
full_score = fe_table.loc["multi-scale: today + 7/30/90-day means", "unseen stations macro-F1"]
axes[1].plot(selection["top-k features"], selection["unseen stations macro-F1"], marker="o", lw=2)
axes[1].axhline(full_score, ls="--", color="grey", label=f"all {X_multi.shape[1]} features ({full_score:.3f})")
axes[1].set(xscale="log", xticks=selection["top-k features"], xticklabels=selection["top-k features"],
            xlabel="number of top-ranked features kept", ylabel="unseen stations macro-F1", title="Feature selection")
axes[1].legend(); plt.tight_layout(); savefig("p6_importance_selection"); plt.show()
selection.round(3)
""")
md(r"""
**Findings.** The most important features are **90-day means**: mean morning wind speed, the east–west wind component
in the afternoon, afternoon humidity and rainfall, then the day of the year and the mean minimum temperature.
These are the long-term "climate" variables. Wind ranks first; part of this may be a site effect (how exposed the
anemometer is), but because the importance is measured on unseen stations, it also reflects real wind regimes such
as the south-east trade winds in the tropics and the westerlies in the south. Ten features reach 0.70 and twenty
match all 74 features (0.718), so most of the multi-scale features are redundant.
""")

md(r"""
### 6.6 Three evaluation protocols for the key models

A single number per model can depend on the split. For the four models that carry our conclusions we compare:

| Protocol | Train | Test | Question |
|---|---|---|---|
| random 70/30 | random 70% of days | other 30% | new days, known stations (adjacent days on both sides) |
| **time 70/30** | days before the cut date | days ≥ cut + 14 days (embargo) | **future** days, known stations |
| **station-grouped, 3 × 5 folds** | 4/5 of the stations | the other 1/5 | **unseen stations**; `StratifiedGroupKFold` with seeds 42, 0, 7 → 15 held-out station groups |

The cut date is the first day of the month that contains the 70% quantile of all dates. The 14-day embargo removes
test days whose 14-day window would reach into the training period.
""")
code(r"""
cut = df["Date"].quantile(0.7).to_period("M").to_timestamp()
time_tr = np.where(df["Date"] < cut)[0]
time_te = np.where(df["Date"] >= cut + pd.Timedelta(days=14))[0]
print(f"time split: train {df['Date'].min().date()} → {(cut - pd.Timedelta(days=1)).date()} ({len(time_tr):,} days), "
      f"test {(cut + pd.Timedelta(days=14)).date()} → {df['Date'].max().date()} ({len(time_te):,} days); "
      f"stations in test: {groups.iloc[time_te].nunique()}, also in train: "
      f"{len(set(groups.iloc[time_te]) & set(groups.iloc[time_tr]))}")

KEY_MODELS = {"Logistic Regression": (MODEL_ZOO["Logistic Regression"], X_all, zoo["Logistic Regression"]),
              "Decision Tree (tuned)": (MODEL_ZOO["Decision Tree (tuned)"], X_all, zoo["Decision Tree (tuned)"]),
              "HGB, daily features": (HGB, X_all, zoo["Gradient Boosting (HGB, default)"]),
              "HGB, 14-day trailing mean": (HGB, X_windows[14], fe_results["14-day trailing mean"])}
FOLD_SEEDS = [RANDOM_STATE, 0, 7]

def time_split_f1(make, X):
    m = make().fit(X.iloc[time_tr], y[time_tr])
    return f1_score(y[time_te], m.predict(X.iloc[time_te]), average="macro")

def grouped_oof(make, X, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(make(), X, y, groups=groups.values, cv=cv, n_jobs=2)

proto_rows, fold_scores = [], {}
for name, (make, X, res42) in KEY_MODELS.items():
    key = "".join(ch for ch in name if ch.isalnum())
    t_f1 = cached(f"p6_time_{key}", lambda make=make, X=X: time_split_f1(make, X), model=name, cut=str(cut), embargo=14)
    scores, pooled = [], []
    for seed in FOLD_SEEDS:
        pred = res42["oof"] if seed == RANDOM_STATE else cached(f"p6_oof_seed{seed}_{key}",
                                                                lambda make=make, X=X, seed=seed: grouped_oof(make, X, seed),
                                                                model=name, seed=seed)
        folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X_all, y, groups.values))
        scores.append(per_fold_f1(pred, folds)); pooled.append(f1_score(y, pred, average="macro"))
    fold_scores[name] = np.concatenate(scores)
    proto_rows.append({"model": name, "random 70/30": res42["random test macro-F1"], "time 70/30": t_f1,
                       "grouped: pooled (mean of 3 seeds)": np.mean(pooled),
                       "grouped: per-fold mean": fold_scores[name].mean(), "grouped: per-fold std": fold_scores[name].std(),
                       "grouped: min fold": fold_scores[name].min(), "grouped: max fold": fold_scores[name].max()})
protocols = pd.DataFrame(proto_rows).set_index("model")
protocols.round(3)
""")
code(r"""
pairs = [("HGB, daily features", "Logistic Regression"), ("HGB, daily features", "Decision Tree (tuned)"),
         ("Decision Tree (tuned)", "Logistic Regression"), ("HGB, 14-day trailing mean", "HGB, daily features")]
paired = pd.DataFrame([{"comparison": f"{a}  −  {b}", "mean difference": (fold_scores[a] - fold_scores[b]).mean(),
                        "std of difference": (fold_scores[a] - fold_scores[b]).std(),
                        "folds where first is better": f"{int((fold_scores[a] > fold_scores[b]).sum())} / {len(fold_scores[a])}"}
                       for a, b in pairs]).set_index("comparison")

fig, ax = plt.subplots(figsize=(10, 4.8))
names = list(fold_scores)
ax.boxplot([fold_scores[n] for n in names], widths=0.5, showmeans=True)
for i, n in enumerate(names, start=1):
    ax.scatter(np.full(15, i) + np.random.default_rng(i).uniform(-0.12, 0.12, 15), fold_scores[n], s=14, color="k", alpha=0.6)
    ax.scatter(i + 0.33, protocols.loc[n, "time 70/30"], marker="s", color="C1", s=45, label="time 70/30" if i == 1 else None)
    ax.scatter(i + 0.33, protocols.loc[n, "random 70/30"], marker="^", color="C0", s=45, label="random 70/30" if i == 1 else None)
ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels([n.replace(", ", ",\n") for n in names])
ax.set(ylabel="macro-F1", title="Unseen-station macro-F1 of 15 held-out station groups (box) vs. known-station protocols")
ax.legend(loc="lower right"); plt.tight_layout(); savefig("p6_protocols"); plt.show()
paired.round(3)
""")
md(r"""
**Findings.**

* **Neighbouring days explain only a small part of the random-split optimism.** Going from the random to the time
  split (future days of known stations, 14-day embargo) lowers macro-F1 by 0.016–0.033 for single-day models and by
  0.07 for 14-day means. Going from the time split to unseen stations lowers it by a further 0.09–0.21. Most of the
  optimism of a random split comes from knowing the station, not from adjacent days.
* **The seed-42 folds used in 6.1–6.5 were slightly favourable.** Over 15 held-out station groups (3 seeds × 5 folds)
  the per-fold means are 0.51 ± 0.06 (LR), 0.53 ± 0.06 (tree), 0.60 ± 0.07 (HGB) and 0.68 ± 0.09 (HGB, 14-day
  means), 0.01–0.04 below the seed-42 values. Comparisons *within* one seed are unaffected because all models share
  the same folds; the 3-seed values are our headline estimates.
* **Paired comparisons are far more precise than the spread suggests.** Station groups differ a lot in difficulty,
  but they affect all models alike. HGB beats LR in 15 of 15 folds (+0.091 ± 0.025) and the tuned tree in 15 of 15;
  14-day means beat single days in 15 of 15 (+0.083 ± 0.039). The tuned tree beats LR in only 10 of 15 folds
  (+0.019 ± 0.028): on unseen stations the two simple models are not reliably different.

### 6.7 Final model choice
""")
code(r"""
final_choice = pd.DataFrame({
    "Part 4 LR (default)": summarise(zoo["Logistic Regression"]),
    "Part 5 DT (tuned)": summarise(zoo["Decision Tree (tuned)"]),
    "HGB, daily features": summarise(zoo["Gradient Boosting (HGB, default)"]),
    "HGB, 14-day trailing mean": summarise(fe_results["14-day trailing mean"]),
    "HGB, multi-scale features": summarise(fe_results["multi-scale: today + 7/30/90-day means"]),
}).T.drop(columns="fit time (s)")
final_choice.round(3)
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
plot_cm(axes[0], y, zoo["Gradient Boosting (HGB, default)"]["oof"], "HGB, single day | all stations unseen")
plot_cm(axes[1], y, fe_results["14-day trailing mean"]["oof"], "HGB, 14-day trailing mean | all stations unseen")
plt.tight_layout(); savefig("p6_final_confusion"); plt.show()
""")
md(r"""
**Choice.** For classifying a **single day** we select **gradient boosting (HGB)**: it is best under every
protocol (0.795 random, 0.765 time split, 0.60 ± 0.07 on unseen stations over 15 station groups), beats LR and the
tree in every one of the 15 folds, and trains in seconds. If two weeks of observations from the same station are
available, **HGB on 14-day trailing means** is clearly better on unseen stations (0.68 ± 0.09; better than single
days in 15 of 15 folds) with the same 20 features; the 74-feature multi-scale model is not better. Among the two
"simple" models of Parts 4–5, the tuned tree is better on known stations, while on new stations the two are not
reliably different (tree better in 10 of 15 folds). The confusion matrices show where the remaining errors are: Arid is still confused with
Subtropical, and Mediterranean with Oceanic.

### 6.8 Summary of the open-ended exploration

1. **Model comparison.** Seven model classes; flexible models are better, but their advantage over LR shrinks from
   0.14–0.18 (random split) to 0.06–0.11 macro-F1 (unseen stations), and the random − unseen gap grows with flexibility.
2. **Complexity and regularisation.** Degree-2 polynomial features with a stronger L2 penalty remove much of LR's
   bias, and the gain carries over to unseen stations (+0.10 macro-F1).
3. **Hyperparameter tuning.** Randomised search with station-grouped CV improves HGB by less than 0.01: the
   limitation is station diversity, not hyperparameters.
4. **Feature engineering.** "Climate memory" features are the most effective change in the project: two weeks of
   observations raise unseen-station macro-F1 from 0.64 to 0.73, while a one-year mean destroys the seasonal
   signal that defines the zones. Under a random split the same features leak almost perfectly.
5. **Feature selection.** 90-day means of wind, humidity, rainfall and minimum temperature carry most of the
   information; 20 of 74 features are enough.
6. **Evaluation protocols.** A time-based split costs only 0.02–0.07 compared with a random split, while unseen
   stations cost a further 0.09–0.21: the station, not temporal adjacency, is the main source of optimism. Over 15
   held-out station groups the headline scores are LR 0.51 ± 0.06, tree 0.53 ± 0.06, HGB 0.60 ± 0.07 and HGB with
   14-day means 0.68 ± 0.09, and the HGB improvements hold in every fold.
""")
