"""Seasonal weather signatures: the main open-ended experiment."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
## 7. Seasonal climate signatures: from weather snapshots to annual patterns

The daily experiments are retained as the baseline and motivation. A single day is a first-impression test: it can
contain a climate signal, but it is also strongly affected by synoptic weather. The main open-ended question is whether
four seasonal observations recover the target more faithfully. We therefore construct one **station-year** observation
from four 21-day windows centred on the March equinox, June solstice, September equinox and December solstice. The
windows use dates available in that calendar year, and require at least seven observed days in each window. A station
year is labelled by its station's fixed climate group; the year is never used as a predictor.

This is a seasonal-weather signature, not a new climate ground truth. Köppen labels describe long-term climate normals,
whereas the supplied data cover roughly a decade. The purpose is to test how much of the supplied external label can be
recovered from increasingly complete weather information. Every station-year model is evaluated using the same withheld
station folds as the daily study, so years from one station never appear on both sides of an outer fold.
""")

code(r"""
SEASON_SPECS = {
    "mar_equinox": (3, 20),
    "jun_solstice": (6, 21),
    "sep_equinox": (9, 22),
    "dec_solstice": (12, 21),
}
SEASON_LABELS = {
    "mar_equinox": "March equinox (autumn)",
    "jun_solstice": "June solstice (winter)",
    "sep_equinox": "September equinox (spring)",
    "dec_solstice": "December solstice (summer)",
}
SEASON_HALF_WINDOW = 10
SEASON_MIN_DAYS = 7
SEASON_BASE = [c for c in FEATURE_SETS["core"] if c not in ["doy_sin", "doy_cos"]]
SEASON_STATS = [c for c in SEASON_BASE if c not in ["Rainfall_log1p"]]

def build_seasonal_signatures():
    # Return station-year rows and diagnostics from four fixed seasonal windows.
    records = []
    diagnostics = []
    local = F.copy()
    local["station"] = groups.to_numpy()
    local["date"] = df["Date"].to_numpy()
    local["year"] = df["Date"].dt.year.to_numpy()
    local["rain_raw"] = df["Rainfall"].to_numpy()
    local["temp_range"] = df["MaxTemp"].to_numpy() - df["MinTemp"].to_numpy()

    for station, sg in local.groupby("station", sort=True):
        zone = station_zone.loc[station]
        years = sorted(sg["year"].unique())
        for year in years:
            row = {"station": station, "year": int(year), "Zone": zone}
            all_ok = True
            diag = {"station": station, "year": int(year), "Zone": zone}
            for key, (month, day) in SEASON_SPECS.items():
                centre = pd.Timestamp(year=int(year), month=month, day=day)
                lo, hi = centre - pd.Timedelta(days=SEASON_HALF_WINDOW), centre + pd.Timedelta(days=SEASON_HALF_WINDOW)
                w = sg[(sg["date"] >= lo) & (sg["date"] <= hi)]
                valid = int(w[SEASON_BASE].notna().any(axis=1).sum())
                diag[f"{key}_rows"] = int(len(w))
                diag[f"{key}_valid_days"] = valid
                all_ok &= valid >= SEASON_MIN_DAYS
                for col in SEASON_STATS:
                    row[f"{key}_{col}_mean"] = w[col].mean()
                    row[f"{key}_{col}_std"] = w[col].std(ddof=0)
                row[f"{key}_Rainfall_log1p_mean"] = w["Rainfall_log1p"].mean()
                row[f"{key}_rain_day_fraction"] = (w["rain_raw"] > 1).mean()
                row[f"{key}_temp_range_mean"] = w["temp_range"].mean()
                row[f"{key}_temp_range_std"] = w["temp_range"].std(ddof=0)
            diag["complete_signature"] = bool(all_ok)
            diagnostics.append(diag)
            if all_ok:
                records.append(row)
    out = pd.DataFrame(records).sort_values(["station", "year"]).reset_index(drop=True)
    diag = pd.DataFrame(diagnostics).sort_values(["station", "year"]).reset_index(drop=True)
    assert len(out) > 300 and out.station.nunique() == 48
    assert out.groupby("station").Zone.nunique().eq(1).all()
    return out, diag

seasonal, seasonal_diag = build_seasonal_signatures()
SEASON_META = ["station", "year", "Zone"]
SEASON_FEATURES = [c for c in seasonal.columns if c not in SEASON_META]
season_y = seasonal["Zone"].to_numpy()
season_groups = seasonal["station"].copy()
X_season = seasonal[SEASON_FEATURES]
export_table("seasonal_feature_diagnostics", seasonal_diag)
print("station-year signatures:", seasonal.shape, "| features:", len(SEASON_FEATURES),
      "| stations:", seasonal.station.nunique(), "| complete coverage:", f"{len(seasonal) / len(seasonal_diag):.1%}")
seasonal_diag.groupby("Zone").complete_signature.mean().reindex(ZONE_ORDER)
""")

md(r"""
### 7.1 What the four windows contain

The window summaries retain both level and variability. Means describe the seasonal state; standard deviations and rainy-
day fractions retain weather variability that a single mean would erase. We report coverage before fitting because a
station-year with too few observations is not equivalent to a complete seasonal signature. The feature table contains no
coordinates, station identifiers or labels. Imputation and scaling are learned inside each training-fold pipeline.
""")

code(r"""
coverage = seasonal_diag.assign(
    complete=seasonal_diag.complete_signature.astype(int),
    min_valid=seasonal_diag[[f"{k}_valid_days" for k in SEASON_SPECS]].min(axis=1),
)
coverage_table = coverage.groupby("Zone").agg(
    station_years=("complete", "sum"),
    candidate_years=("complete", "size"),
    median_min_valid=("min_valid", "median"),
).reindex(ZONE_ORDER)
coverage_table["coverage"] = coverage_table.station_years / coverage_table.candidate_years
export_table("seasonal_coverage", coverage_table)
coverage_table.round(3)
""")

code(r"""
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
for ax, (key, label) in zip(axes, SEASON_LABELS.items()):
    vals = seasonal[[c for c in SEASON_FEATURES if c.startswith(key + "_") and c.endswith("_mean")]].mean()
    vals.index = [v.replace(key + "_", "").replace("_mean", "") for v in vals.index]
    vals.sort_values().plot.barh(ax=ax, color=ZONE_COLORS["Subtropical"])
    ax.set_title(label, fontsize=10); ax.set_xlabel("feature mean (mixed units)")
axes[0].set_ylabel("seasonal feature")
plt.tight_layout(); savefig("p7_seasonal_feature_means"); plt.show()
""")

md(r"""
### 7.2 Seasonal t-SNE

We embed the station-year signatures after median imputation and scaling. This visualization is descriptive: the same
station contributes multiple years, so it is not a held-out prediction estimate. The relevant comparison is whether the
seasonal signature shows more coherent zone structure than the single-day embedding, not whether t-SNE distances are
physical distances.
""")

code(r"""
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
season_matrix = build_preprocessor().fit_transform(X_season)
season_matrix = np.asarray(season_matrix)
season_tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(seasonal)//12)), init="pca",
                   learning_rate="auto", random_state=RANDOM_STATE).fit_transform(season_matrix)
fig, ax = plt.subplots(figsize=(8.5, 6.5))
for zone in ZONE_ORDER:
    mask = season_y == zone
    ax.scatter(season_tsne[mask, 0], season_tsne[mask, 1], s=18, alpha=.65,
               color=ZONE_COLORS[zone], label=zone)
ax.set(title="t-SNE of four-season station-year weather signatures", xlabel="t-SNE 1", ylabel="t-SNE 2")
ax.legend(ncol=2, fontsize=9)
plt.tight_layout(); savefig("p7_seasonal_tsne"); plt.show()
season_tsne_diag = pd.DataFrame([{
    "embedding": "seasonal station-year t-SNE",
    "rows": len(seasonal),
    "perplexity": min(30, max(5, len(seasonal)//12)),
    "silhouette_zone_original": silhouette_score(season_matrix, season_y),
    "silhouette_zone_tsne": silhouette_score(season_tsne, season_y),
}])
export_table("seasonal_tsne_diagnostics", season_tsne_diag)
season_tsne_diag.round(3)
""")

md(r"""
### 7.3 Clustering: weather types versus seasonal climate signatures

The daily clustering section already shows that one-day clusters are mainly weather types. We now repeat a smaller,
transparent comparison on the four-season signatures. K-means and Ward are fitted without labels; silhouette,
Davies--Bouldin and ARI with the external zone labels are reported for $k=2,\ldots,7$. The labels are used only for
external evaluation, never to fit or select the clusters.
""")

code(r"""
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
season_cluster_rows = []
for algorithm in ["K-means", "Ward"]:
    for k in range(2, 8):
        if algorithm == "K-means":
            lab = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(season_matrix)
        else:
            lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(season_matrix)
        season_cluster_rows.append({
            "algorithm": algorithm, "k": k,
            "silhouette": silhouette_score(season_matrix, lab),
            "Davies-Bouldin": davies_bouldin_score(season_matrix, lab),
            "ARI zone": adjusted_rand_score(season_y, lab),
            "cluster_count": len(np.unique(lab)),
        })
season_cluster_table = export_table("seasonal_clustering_sweep", pd.DataFrame(season_cluster_rows))
season_best = season_cluster_table.sort_values(["ARI zone", "silhouette"], ascending=[False, False]).iloc[0]
season_cluster_table.round(3)
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for algorithm, ax in zip(["K-means", "Ward"], axes):
    s = season_cluster_table[season_cluster_table.algorithm == algorithm]
    ax.plot(s.k, s.silhouette, marker="o", label="silhouette")
    ax.plot(s.k, s["ARI zone"], marker="s", label="ARI with zone")
    ax.set(title=f"Seasonal {algorithm}", xlabel="number of clusters", ylabel="score")
    ax.legend()
plt.tight_layout(); savefig("p7_seasonal_clustering"); plt.show()
export_table("seasonal_clustering_summary", pd.DataFrame([{
    "best_by_ARI_algorithm": season_best.algorithm,
    "best_by_ARI_k": int(season_best.k),
    "best_by_ARI": season_best["ARI zone"],
    # ``station`` is a loop variable used in the feature-construction cells;
    # the earlier station-profile clustering sweep is stored as ``st_sweep``.
    "daily_reference_station_profile_ARI": st_sweep.query("algorithm=='Ward' and k==5")['ARI zone'].iloc[0],
}]))
""")

md(r"""
### 7.4 Main prediction experiment: station-year signatures

The main prediction unit is a station-year, while the outer split remains a station split. This prevents years from the
same station appearing in both training and test. We compare a linear model, a simple tree, random forest and histogram
gradient boosting. The tuned tree and HGB perform their searches inside the outer training stations only. Scores are
pooled over station-year predictions; station-balanced scores are reported separately because years do not create new
stations.
""")

code(r"""
def seasonal_outer(seed):
    result = []
    for fold, (daily_tr, daily_te) in enumerate(OUTER[seed]):
        tr_st = set(groups.iloc[daily_tr]); te_st = set(groups.iloc[daily_te])
        tr = np.flatnonzero(season_groups.isin(tr_st).to_numpy())
        te = np.flatnonzero(season_groups.isin(te_st).to_numpy())
        assert not set(season_groups.iloc[tr]) & set(season_groups.iloc[te])
        assert set(season_y[tr]) == set(ZONE_ORDER) == set(season_y[te])
        result.append((tr, te))
    return result

def seasonal_inner(indices, seed):
    indices = np.asarray(indices)
    station_table = seasonal.iloc[indices][["station", "Zone"]].drop_duplicates().sort_values("station")
    # GridSearchCV receives X_season.iloc[indices], so its cv indices must be
    # relative to that training subset. Keep the absolute arrays for the
    # station-disjointness and class-coverage checks, then map them back.
    position = {int(idx): pos for pos, idx in enumerate(indices)}
    out = []
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(station_table.station, station_table.Zone):
        a_st = set(station_table.iloc[a].station); b_st = set(station_table.iloc[b].station)
        ia_abs = indices[season_groups.iloc[indices].isin(a_st).to_numpy()]
        ib_abs = indices[season_groups.iloc[indices].isin(b_st).to_numpy()]
        assert set(season_y[ia_abs]) == set(ZONE_ORDER) == set(season_y[ib_abs])
        out.append((np.asarray([position[int(i)] for i in ia_abs], dtype=int),
                    np.asarray([position[int(i)] for i in ib_abs], dtype=int)))
    return out

def seasonal_score(indices, pred, proba=None):
    indices = np.asarray(indices)
    counts = season_groups.iloc[indices].value_counts()
    weights = 1 / season_groups.iloc[indices].map(counts).to_numpy()
    result = {
        "macro_F1": macro5(season_y[indices], pred),
        "accuracy": accuracy_score(season_y[indices], pred),
        "weighted_F1": f1_score(season_y[indices], pred, labels=ZONE_ORDER, average="weighted", zero_division=0),
        "station_balanced_macro_F1": macro5(season_y[indices], pred, weights),
        "mean_station_accuracy": accuracy_score(season_y[indices], pred, sample_weight=weights),
    }
    if proba is not None:
        result["macro_AUC"] = np.mean([roc_auc_score(season_y[indices] == z, proba[:, j])
                                        for j, z in enumerate(ZONE_ORDER)])
    return result

def fit_season_model(kind, train, test, seed):
    model = model_factory(kind, seed)
    search = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if kind in GRIDS:
            cv = seasonal_inner(train, seed)
            search = GridSearchCV(model, GRIDS[kind], scoring=SCORER, cv=cv, n_jobs=1,
                                  refit=True, return_train_score=True, error_score="raise")
            search.fit(X_season.iloc[train], season_y[train]); model = search.best_estimator_
        else:
            model.fit(X_season.iloc[train], season_y[train])
    pred = model.predict(X_season.iloc[test])
    proba = model.predict_proba(X_season.iloc[test])
    proba = proba[:, [list(model.classes_).index(z) for z in ZONE_ORDER]]
    meta = {
        "selected_params": search.best_params_ if search is not None else {},
        "warnings": [str(w.message) for w in caught],
    }
    if search is not None: meta["search"] = pd.DataFrame(search.cv_results_)
    return {"pred": pred, "proba": proba, "meta": meta,
            "scores": seasonal_score(test, pred, proba),
            "train_macro_F1": macro5(season_y[train], model.predict(X_season.iloc[train]))}

SEASON_MODELS = ["LR", "DT-tuned", "RF", "HGB-tuned"]
SEASON_OOF = {}; season_rows = []; season_fold_rows = []; season_audit = []
for kind in SEASON_MODELS:
    for seed in OUTER_SEEDS:
        pred = np.empty(len(seasonal), dtype=object); proba = np.zeros((len(seasonal), 5))
        for fold, (tr, te) in enumerate(seasonal_outer(seed)):
            res = cached(f"seasonal_{kind}_s{seed}_f{fold}",
                         lambda k=kind,a=tr,b=te,s=seed: fit_season_model(k,a,b,s),
                         features=len(SEASON_FEATURES), model=kind, seed=seed, fold=fold)
            pred[te] = res["pred"]; proba[te] = res["proba"]
            season_fold_rows.append({"model": kind, "seed": seed, "fold": fold, **res["scores"]})
            season_audit.append({"experiment": "seasonal", "model": kind, "seed": seed, "fold": fold,
                                 "params": repr(res["meta"]["selected_params"]),
                                 "warnings": repr(res["meta"]["warnings"])})
        SEASON_OOF[(kind, seed)] = {"pred": pred, "proba": proba}
        season_rows.append({"model": kind, "seed": seed,
                            **seasonal_score(np.arange(len(seasonal)), pred, proba)})
season_table = pd.DataFrame(season_rows)
season_summary_rows = []
for kind in SEASON_MODELS:
    block = season_table[season_table.model == kind]
    primary = block[block.seed == 42].iloc[0]
    season_summary_rows.append({
        "model": kind, "grouped_primary": primary.macro_F1,
        "grouped_seed_mean": block.macro_F1.mean(), "grouped_seed_min": block.macro_F1.min(),
        "grouped_seed_max": block.macro_F1.max(),
        "accuracy": primary.accuracy, "station_balanced_macro_F1": primary.station_balanced_macro_F1,
        "mean_station_accuracy": primary.mean_station_accuracy, "macro_AUC": primary.macro_AUC,
    })
season_summary = export_table("seasonal_model_comparison", pd.DataFrame(season_summary_rows).set_index("model"))
export_table("seasonal_fold_scores", pd.DataFrame(season_fold_rows))
export_table("seasonal_fit_audit", pd.DataFrame(season_audit))
season_summary.round(3)
""")

code(r"""
fig, ax = plt.subplots(figsize=(9, 5))
season_summary[["grouped_primary", "grouped_seed_mean"]].plot.bar(ax=ax, color=["#0f7c80", "#9cc7c5"])
ax.set(title="Four-season station-year prediction", ylabel="fixed-five macro-F1", xlabel="model")
ax.legend(["seed 42 primary", "mean of reused station partitions"])
plt.xticks(rotation=25); plt.tight_layout(); savefig("p7_seasonal_model_comparison"); plt.show()
""")

code(r"""
season_class_rows = []
for kind in ["HGB-tuned", "LR"]:
    pred = SEASON_OOF[(kind, 42)]["pred"]
    p, r, f, s = precision_recall_fscore_support(season_y, pred, labels=ZONE_ORDER, zero_division=0)
    season_class_rows.extend({"model": kind, "zone": z, "precision": p[i], "recall": r[i],
                              "F1": f[i], "support": s[i]} for i, z in enumerate(ZONE_ORDER))
export_table("seasonal_per_class", pd.DataFrame(season_class_rows))
""")

md(r"""
### 7.5 Matched baseline and interpretation

The daily baseline remains the primary first-impression result. For a more comparable station-year check, we also take
the daily HGB out-of-fold predictions in the four seasonal windows and use a deterministic majority vote across those
snapshots. This is still a daily model; it does not receive seasonal means or standard deviations. The comparison is
descriptive because the seasonal model predicts station-years while the headline daily score predicts individual days.
""")

code(r"""
def nearest_snapshot_indices():
    lookup = []
    for _, row in seasonal.iterrows():
        station, year = row.station, int(row.year)
        sg = df.index[(groups == station) & (df.Date.dt.year == year)]
        chosen = []
        for month, day in SEASON_SPECS.values():
            centre = pd.Timestamp(year=year, month=month, day=day)
            candidates = sg[(df.loc[sg, "Date"] - centre).abs().dt.days <= SEASON_HALF_WINDOW]
            if len(candidates):
                chosen.append(candidates[np.argmin((df.loc[candidates, "Date"] - centre).abs().dt.days)])
        lookup.append(chosen)
    return lookup

snapshot_rows = nearest_snapshot_indices()
daily_pred = np.asarray(OOF[("HGB", 42)]["pred"], dtype=object)
snapshot_pred = []
for inds in snapshot_rows:
    votes = pd.Series(daily_pred[inds]).value_counts()
    snapshot_pred.append(sorted(votes.index, key=lambda z: (-votes[z], ZONE_ORDER.index(z)))[0])
snapshot_pred = np.asarray(snapshot_pred, dtype=object)
matched = pd.DataFrame([{
    "representation": "daily HGB majority over four nearest seasonal snapshots",
    "rows": len(seasonal), "macro_F1": macro5(season_y, snapshot_pred),
    "accuracy": accuracy_score(season_y, snapshot_pred),
}, {
    "representation": "seasonal HGB-tuned signature",
    "rows": len(seasonal), "macro_F1": seasonal_score(np.arange(len(seasonal)), SEASON_OOF[("HGB-tuned",42)]["pred"])["macro_F1"],
    "accuracy": accuracy_score(season_y, SEASON_OOF[("HGB-tuned",42)]["pred"]),
}])
export_table("seasonal_matched_baseline", matched)
matched.round(3)
""")

md(r"""
### 7.6 Seasonal conclusions and limits

The seasonal signature is the main open-ended representation, but it is not automatically superior because it uses a
different unit and fewer observations than the daily table. The relevant evidence is the station-held-out comparison,
its repeated partition range, class-level errors and coverage diagnostics. A positive result supports the claim that
seasonal weather patterns contain more climate information than a snapshot. A weak or negative result is also informative:
four fixed dates may miss onset timing, extreme-event frequency or the longer climatological baseline.

The four dates are a transparent design choice rather than a physical law. We therefore retain the existing trailing-window
curves as a sensitivity analysis and do not claim that equinoxes or solstices are universally optimal. The external label
still comes from one map, only four stations are Tropical, and station-year rows do not create independent geographic
replicates.
""")

code(r"""
# Append seasonal arrays to the main prediction archive and include this section in the run audit.
pred_path = RESULT_DIR / "predictions.npz"
with np.load(pred_path, allow_pickle=False) as old:
    arrays = {k: old[k] for k in old.files}
for (kind, seed), obj in SEASON_OOF.items():
    arrays[f"seasonal_{kind}_s{seed}"] = np.asarray(obj["pred"], dtype=str)
arrays["seasonal_y"] = np.asarray(season_y, dtype=str)
arrays["seasonal_station"] = seasonal.station.to_numpy(dtype=str)
arrays["seasonal_year"] = seasonal.year.to_numpy(dtype=int)
pred_tmp = pred_path.with_name("predictions.seasonal.tmp.npz")
if pred_tmp.exists(): pred_tmp.unlink()
np.savez_compressed(pred_tmp, **arrays)
pred_tmp.replace(pred_path)
manifest = json.loads((RESULT_DIR / "run_manifest.json").read_text())
manifest.update({"seasonal_rows": int(len(seasonal)), "seasonal_stations": int(seasonal.station.nunique()),
                 "seasonal_windows": list(SEASON_SPECS), "cache_hits": int(sum(e["hit"] for e in CACHE_EVENTS)),
                 "cache_misses": int(sum(not e["hit"] for e in CACHE_EVENTS))})
(RESULT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
""")
