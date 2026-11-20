"""Build the Group 7 report from the executed notebook's exported results."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results"


def read(name, index_col=0):
    return pd.read_csv(RESULT / f"{name}.csv", index_col=index_col)


def esc(value):
    return (str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#"))


def f(value, digits=3):
    return f"{float(value):.{digits}f}"


def table(caption, label, headers, rows, align=None, sep=4):
    align = align or "l" + "r" * (len(headers) - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\setlength{{\tabcolsep}}{{{sep}pt}}",
             rf"\caption{{{caption}}}\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}\toprule",
             " & ".join(headers) + r" \\ \midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


manifest = json.loads((RESULT / "run_manifest.json").read_text())
env = json.loads((RESULT / "environment.json").read_text())
models = read("model_comparison")
season = read("seasonal_model_comparison")
season_cls = read("seasonal_per_class")
season_cov = read("seasonal_coverage")
season_cluster = read("seasonal_clustering_sweep")
season_match = read("seasonal_matched_baseline")
p4 = read("p4_metrics")
p5 = read("p5_holdout")
daily_cluster = read("clustering_sweep")
station_cluster = read("station_clustering")
tsne = read("tsne_diagnostics")
season_tsne = read("seasonal_tsne_diagnostics")
nested = read("nested_window_scores").set_index("model")
outer_cls = read("outer_per_class")
policy = read("missingness_policies")
spatial = read("spatial_buffer_summary").iloc[0]

daily_hgb = models.loc["HGB", "grouped_primary"]
season_lr = season.loc["LR", "grouped_primary"]
season_hgb = season.loc["HGB-tuned", "grouped_primary"]
matched_daily = season_match.loc[0, "macro_F1"]
matched_hgb = season_match.loc[1, "macro_F1"]
matched_lr = season_match.loc[0, "macro_F1"]
season_fold_lr = read("seasonal_fold_scores")
season_fold_lr = season_fold_lr.loc[season_fold_lr.model == "LR", "macro_F1"]

def conditional_station_bootstrap(pred_key, n=10000, seed=20261120):
    """Conditional interval over saved station-year predictions; no refitting."""
    import numpy as np
    from sklearn.metrics import f1_score
    archive = np.load(RESULT / "predictions.npz", allow_pickle=False)
    y = archive["seasonal_y"].astype(str)
    pred = archive[pred_key].astype(str)
    stations = archive["seasonal_station"].astype(str)
    labels = ["Tropical", "Arid", "Subtropical", "Mediterranean", "Oceanic"]
    unique = np.unique(stations)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(stations == s) for s in sampled])
        scores.append(f1_score(y[idx], pred[idx], labels=labels, average="macro", zero_division=0))
    return np.quantile(scores, [0.025, 0.975])

season_lr_boot = conditional_station_bootstrap("seasonal_LR_s42")

macros = {
    "DailyHGB": f(daily_hgb), "DailyRF": f(models.loc["RF", "grouped_primary"]),
    "DailyAUC": f(models.loc["HGB", "macro_AUC"]),
    "SeasonLR": f(season_lr), "SeasonHGB": f(season_hgb),
    "SeasonLRAUC": f(season.loc["LR", "macro_AUC"]),
    "SeasonHGBAUC": f(season.loc["HGB-tuned", "macro_AUC"]),
    "MatchedDaily": f(matched_hgb), "SeasonGain": f(season_hgb - matched_hgb),
    "MatchedLR": f(matched_lr), "SeasonLRGain": f(season_lr - matched_lr),
    "SeasonFoldMean": f(season_fold_lr.mean()), "SeasonFoldSD": f(season_fold_lr.std(ddof=1)),
    "SeasonFoldMin": f(season_fold_lr.min()), "SeasonFoldMax": f(season_fold_lr.max()),
    "SeasonBootLow": f(season_lr_boot[0]), "SeasonBootHigh": f(season_lr_boot[1]),
    "SeasonRows": f'{int(manifest["seasonal_rows"]):,}',
    "SeasonStations": str(manifest["seasonal_stations"]),
    "SeasonCoverage": f(season_cov.coverage.mean() * 100, 1) + r"\%",
    "DailyTsnePurity": f(tsne.loc["t-SNE 2-D", "purity: zone"]),
    "DailyTsneTrust": f(tsne.loc["t-SNE 2-D", "trustworthiness"]),
    "SeasonTsneSil": f(season_tsne.loc[0, "silhouette_zone_tsne"]),
    "DailyClusterARI": f(daily_cluster.query("algorithm=='K-means' and k==3")["ARI zone"].iloc[0]),
    "SeasonClusterARI": f(season_cluster["ARI zone"].max()),
    "StationClusterARI": f(station_cluster.query("algorithm=='Ward' and k==5")["ARI zone"].iloc[0]),
    "DailyLR": f(models.loc["LR", "grouped_primary"]), "DailyDT": f(models.loc["DT-tuned", "grouped_primary"]),
    "DailyPoly": f(models.loc["PolyLR", "grouped_primary"]),
    "HistoryHGB": f(nested.loc["HGB", "macro_F1"]), "HistoryLR": f(nested.loc["LR", "macro_F1"]),
    "SpatialHGB": f(spatial.macro_F1), "CacheHits": str(manifest["cache_hits"]),
    "CacheMisses": str(manifest["cache_misses"]), "PythonVersion": esc(env["python"]),
    "RunID": esc(manifest["run_id"]),
}
macro_text = "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items())

daily_rows = []
for name in ["LR", "DT-tuned", "RF", "HGB", "HGB-tuned", "PolyLR"]:
    r = models.loc[name]
    daily_rows.append([esc(name), f(r.grouped_primary), f(r.grouped_seed_mean),
                       f(r.grouped_seed_min) + "--" + f(r.grouped_seed_max), f(r.station_balanced_macro_F1), f(r.macro_AUC)])
daily_table = table("Daily weather baseline and model comparison. Primary is pooled station-held-out macro-F1; seed range reuses the same station network.",
                     "tab:daily", ["Model", "Primary F1", "Seed mean", "Seed range", "Station-bal.", "AUC"], daily_rows, align="lrrrrr")

p4_rows = []
for name in ["LR", "DT"]:
    for split in ["train", "test", "entire"]:
        r = p4.query("split == 'random' and model == @name and set == @split").iloc[0]
        p4_rows.append([esc(name), esc(split), f(r.accuracy), f(r["macro-F1"]), f(r["weighted-F1"])])
p4_table = table("Required train/test/entire metrics under the course-style random split.",
                 "tab:required", ["Model", "Set", "Accuracy", "Macro-F1", "Weighted-F1"], p4_rows, align="llrrr")

validation_rows = []
for name in ["LR-tuned", "DT-tuned"]:
    rr = p5.query("split == 'random' and model == @name").iloc[0]
    gg = p5.query("split == 'grouped' and model == @name").iloc[0]
    validation_rows.append([esc(name), f(rr.macro_F1), f(gg.macro_F1), f(rr.macro_AUC), f(gg.macro_AUC)])
validation_table = table("Held-out results after inner-fold hyperparameter selection; tuning never uses the outer test stations.",
                         "tab:validation", ["Model", "Random F1", "Grouped F1", "Random AUC", "Grouped AUC"], validation_rows, align="lrrrr")

season_rows = []
for name in ["LR", "DT-tuned", "RF", "HGB-tuned"]:
    r = season.loc[name]
    season_rows.append([esc(name), f(r.grouped_primary), f(r.grouped_seed_mean),
                        f(r.grouped_seed_min) + "--" + f(r.grouped_seed_max), f(r.station_balanced_macro_F1), f(r.macro_AUC)])
season_table = table("Four-season station-year signature models. Every outer fold holds out complete stations.",
                      "tab:season", ["Model", "Primary F1", "Seed mean", "Seed range", "Station-bal.", "AUC"], season_rows, align="lrrrrr")

cluster_rows = []
for alg in ["K-means", "Ward"]:
    d = daily_cluster.query("algorithm == @alg and k == 3").iloc[0]
    internal = season_cluster.query("algorithm == @alg").sort_values("silhouette", ascending=False).iloc[0]
    posthoc = season_cluster.query("algorithm == @alg").sort_values("ARI zone", ascending=False).iloc[0]
    cluster_rows.append([esc(alg), f(d["silhouette"]), f(d["ARI zone"]), str(int(internal.k)), f(internal.silhouette),
                         str(int(posthoc.k)), f(posthoc["ARI zone"])])
cluster_table = table("Daily clusters versus seasonal clusters. Seasonal internal metrics choose $k=2$; the ARI-maximising $k=7$ row is a post-hoc label comparison.",
                      "tab:cluster", ["Algorithm", "Daily sil.", "Daily ARI", "Season $k$ (sil.)", "Season sil.", "Post-hoc $k$", "Post-hoc ARI"], cluster_rows, align="lrrrrrr", sep=2)

coverage_rows = []
for zone, row in season_cov.iterrows():
    coverage_rows.append([esc(zone), str(int(row.station_years)), str(int(row.candidate_years)), f(row.coverage)])
coverage_table = table("Coverage of complete four-window station-year signatures; at least seven valid days are required in every window.",
                       "tab:coverage", ["Zone", "Complete", "Candidate", "Coverage"], coverage_rows, align="lrrr")

class_rows = []
for zone in ["Tropical", "Arid", "Subtropical", "Mediterranean", "Oceanic"]:
    r = season_cls.query("model == 'LR' and zone == @zone").iloc[0]
    class_rows.append([esc(zone), f(r.precision), f(r.recall), f(r.F1), str(int(r.support))])
class_table = table("Per-class precision, recall and F1 for the primary seasonal LR model.",
                    "tab:class", ["Zone", "Prec.", "Recall", "F1", "Support"], class_rows, align="lrrrr")

discussion = ("The daily weather baseline contains real but limited climate information. "
              f"Under the station-held-out protocol, HGB reaches {daily_hgb:.3f} macro-F1 and "
              f"{models.loc['HGB','macro_AUC']:.3f} macro-AUC, while LR reaches {models.loc['LR','grouped_primary']:.3f}. "
              "The random row split is easier because the same stations occur on both sides, so it is not used as the "
              "headline estimate. The required train/test/entire-set tables show the default tree's large fit gap, "
              "whereas LR has a smaller gap. Precision, recall and F1 remain uneven: Arid and Mediterranean sites "
              "are confused most often. The four-season signature changes the unit to station-year and retains only "
              f"{season_cov.coverage.mean()*100:.1f}\\% of candidate years, but seasonal LR reaches {season_lr:.3f} macro-F1 "
              f"and {season.loc['LR','macro_AUC']:.3f} macro-AUC on held-out stations. This supports the motivation "
              "that temporal coverage matters, while the coverage filter, station-year dependence and one external "
              "label map limit how broadly the result can be generalised.")
assert 100 <= len(discussion.split()) <= 200

template = (ROOT / "report/report_template.tex").read_text(encoding="utf-8-sig")
replacements = {"@@MACROS@@": macro_text, "@@DAILY_TABLE@@": daily_table,
                "@@SEASON_TABLE@@": season_table, "@@CLUSTER_TABLE@@": cluster_table,
                "@@COVERAGE_TABLE@@": coverage_table, "@@CLASS_TABLE@@": class_table,
                "@@P4_TABLE@@": p4_table, "@@VALIDATION_TABLE@@": validation_table,
                "@@DISCUSSION@@": discussion}
for key, value in replacements.items():
    if key not in template:
        raise AssertionError(key)
    template = template.replace(key, value)
if "@@" in template:
    raise AssertionError("unreplaced report placeholder")
(ROOT / "report/report_7_TaskA.tex").write_text(template, encoding="utf-8")
(RESULT / "report_values.json").write_text(json.dumps(macros, indent=2), encoding="utf-8")
print("Generated integrated seasonal report; discussion words:", len(discussion.split()))
