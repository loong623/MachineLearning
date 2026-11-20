"""Outcome text generated from completed results, for notebook markdown only."""
from pathlib import Path
import pandas as pd


def notebook_summary(root):
    result = Path(root) / "results"
    read = lambda name: pd.read_csv(result / f"{name}.csv", index_col=0)
    models = read("model_comparison")
    season = read("seasonal_model_comparison")
    matched = read("seasonal_matched_baseline")
    coverage = read("seasonal_coverage")
    cluster = read("seasonal_clustering_sweep")
    nested = read("nested_window_scores").set_index("model")
    buffer = read("spatial_buffer_summary").iloc[0]
    best_daily = models.grouped_primary.idxmax()
    best_season = season.grouped_primary.idxmax()
    best_cluster = cluster.sort_values(["ARI zone", "silhouette"], ascending=[False, False]).iloc[0]
    candidate_years = int(coverage.candidate_years.sum())
    return f"""### Completed-run findings

The daily snapshot baseline is intentionally retained as the first-impression reference. The predefined daily HGB
reaches {models.loc['HGB', 'grouped_primary']:.3f} pooled station-held-out macro-F1 and {models.loc['HGB', 'macro_AUC']:.3f}
macro-AUC; the highest daily point estimate in the tested comparison is {best_daily},
{models.loc[best_daily, 'grouped_primary']:.3f}. These scores are pooled over stations withheld from training, and
the three reused station partitions span {models.loc['HGB', 'grouped_seed_min']:.3f}--{models.loc['HGB', 'grouped_seed_max']:.3f}
for HGB. Random row splits are easier because locations occur on both sides.

The main representation is a four-window station-year signature centred on the March equinox, June solstice, September
equinox and December solstice. It contains {candidate_years} candidate station-years, retains {coverage.coverage.mean():.1%}
on average, and covers 48 stations. The strongest seasonal model is {best_season}, with
{season.loc[best_season, 'grouped_primary']:.3f} macro-F1 and {season.loc[best_season, 'macro_AUC']:.3f} macro-AUC. A matched
majority vote over four nearest daily HGB snapshots reaches {matched.loc[0, 'macro_F1']:.3f}, so the seasonal signature
adds temporal aggregation beyond four isolated observations in this station-held-out comparison.

Seasonal clustering remains exploratory: the best ARI in the $k=2,\\ldots,7$ sweep is {best_cluster['ARI zone']:.3f}
({best_cluster.algorithm}, $k={int(best_cluster.k)}$). The seasonal t-SNE is descriptive because years from one station
repeat. Nested trailing-history selection gives LR {nested.loc['LR', 'macro_F1']:.3f} and HGB {nested.loc['HGB', 'macro_F1']:.3f},
so the fixed-window history sensitivity remains supporting evidence rather than the paper's headline.

The 100-km training exclusion buffer lowers daily HGB to {buffer.macro_F1:.3f} on {int(buffer.test_rows):,} rows. This
network-transfer stress test, four Tropical stations, the manually matched raster and station-year dependence limit
generalisation. No independent climate map or external station network has been added.
"""
