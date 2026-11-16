"""Part 3 of the project notebook: clustering analysis. Assembled by tools/build_notebook.py."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
## 3. Clustering Analysis

**Question.** Without labels, can clustering rediscover the climate zones?

We cluster the same balanced sample of 6,000 days as in Part 2 (preprocessed 20-D `core` features). Labels are
never given to the algorithms; `Zone` (and the season) are only used afterwards as *external* references.

### 3.1 Algorithms and why we chose them

| Algorithm | Model of a cluster | Why it is useful here |
|---|---|---|
| **K-means** | a centroid; points go to the nearest one (spherical, similar-size clusters) | the standard baseline; fast enough for repeated runs (stability test) |
| **Agglomerative, Ward linkage** | merges the pair of clusters that increases within-cluster variance least | deterministic (no random initialisation); gives a hierarchy and a dendrogram; Ward optimises the same criterion as K-means, so differences come from greedy merging vs iterative refinement |
| **Gaussian Mixture (full covariance)** | each cluster is a Gaussian with its own covariance; soft assignments | allows elliptical, differently-shaped clusters; has a likelihood, so BIC can choose the number of components |

### 3.2 Choosing the number of clusters

For k = 2 … 10 we record internal metrics (no labels needed) and external metrics (agreement with zone / season):

* **Silhouette** (−1 … 1, higher is better): how much closer a point is to its own cluster than to the next one.
* **Davies–Bouldin** (≥ 0, lower is better): average ratio of within-cluster spread to between-cluster distance.
* **Calinski–Harabasz** (higher is better): between-cluster variance / within-cluster variance.
* **Inertia** (K-means, elbow) and **BIC** (GMM, lower is better).
* **ARI** and **NMI** with `Zone` and with the season (0 = no agreement beyond chance, 1 = identical partition).
""")
code(r"""
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.mixture import GaussianMixture

def fit_cluster(alg, k, X, seed=RANDOM_STATE):
    # returns (labels, fitted model)
    if alg == "K-means":
        m = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X); return m.labels_, m
    if alg == "Ward":
        m = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X); return m.labels_, m
    if alg == "GMM":
        m = GaussianMixture(n_components=k, covariance_type="full", n_init=2, random_state=seed).fit(X)
        return m.predict(X), m
    raise ValueError(alg)

ALGS = ["K-means", "Ward", "GMM"]
def cluster_sweep():
    rows, labels = [], {}
    for k in range(2, 11):
        for alg in ALGS:
            lab, m = fit_cluster(alg, k, X_vis)
            labels[(alg, k)] = lab
            rows.append({"algorithm": alg, "k": k,
                         "silhouette": silhouette_score(X_vis, lab),
                         "Davies-Bouldin": davies_bouldin_score(X_vis, lab),
                         "Calinski-Harabasz": calinski_harabasz_score(X_vis, lab),
                         "ARI zone": adjusted_rand_score(y_vis, lab), "NMI zone": normalized_mutual_info_score(y_vis, lab),
                         "ARI season": adjusted_rand_score(season_vis, lab),
                         "inertia": m.inertia_ if alg == "K-means" else np.nan,
                         "BIC": m.bic(X_vis) if alg == "GMM" else np.nan})
    return pd.DataFrame(rows), labels

sweep, labels_k = cached("p3_sweep", cluster_sweep, n_per_zone=N_PER_ZONE, seed=RANDOM_STATE, ks=(2, 10), algs=tuple(ALGS))
sweep.pivot(index="k", columns="algorithm", values=["silhouette", "Davies-Bouldin", "ARI zone"]).round(3)
""")
code(r"""
ALG_COLORS = {"K-means": "#1f77b4", "Ward": "#2ca02c", "GMM": "#d62728"}
panels = [("silhouette", "Silhouette (higher = better)"), ("Davies-Bouldin", "Davies–Bouldin (lower = better)"),
          ("Calinski-Harabasz", "Calinski–Harabasz (higher = better)"), ("ARI zone", "ARI with Zone (external)"),
          ("inertia", "K-means inertia (elbow)"), ("BIC", "GMM BIC (lower = better)")]
fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
for ax, (col, title) in zip(axes.ravel(), panels):
    for alg in ALGS:
        s = sweep[sweep["algorithm"] == alg]
        if s[col].notna().any():
            ax.plot(s["k"], s[col], marker="o", color=ALG_COLORS[alg], label=alg)
    ax.set(title=title, xlabel="k"); ax.set_xticks(range(2, 11))
axes[0, 0].legend()
plt.suptitle("Day-level clustering: model selection over k (6,000 days)"); plt.tight_layout()
savefig("p3_k_selection"); plt.show()
""")

md(r"""
**Stability (all three algorithms).** A good clustering should not change much when the data change slightly.
For every algorithm and every k we refit on 10 random subsamples containing 80% of the 6,000 days (drawn without
replacement) and compare the subsample's partition with the reference partition **on the same days** (ARI;
1 = identical). This needs no rule for assigning left-out points, so it treats K-means, Ward (which has no
`predict`) and GMM identically. Each (algorithm, k) result is cached separately.
""")
code(r"""
from scipy.cluster.hierarchy import fcluster
N_SUB, SUB_FRAC = 10, 0.8
SUB_IDX = [np.random.default_rng(RANDOM_STATE + b).choice(len(X_vis), size=int(SUB_FRAC * len(X_vis)), replace=False)
           for b in range(N_SUB)]

def stability_kmeans_gmm(alg, k):
    ref, scores = labels_k[(alg, k)], []
    for b, idx in enumerate(SUB_IDX):
        lab, _ = fit_cluster(alg, k, X_vis[idx], seed=b)
        scores.append(adjusted_rand_score(ref[idx], lab))
    return scores

def stability_ward():
    # one Ward tree per subsample, cut at every k
    out = {k: [] for k in range(2, 11)}
    for idx in SUB_IDX:
        Z_sub = linkage(X_vis[idx], method="ward")
        for k in out:
            out[k].append(adjusted_rand_score(labels_k[("Ward", k)][idx], fcluster(Z_sub, t=k, criterion="maxclust")))
    return out

STAB_PARAMS = dict(n_sub=N_SUB, frac=SUB_FRAC, seed=RANDOM_STATE, scheme="subsample")
stab_scores = {("Ward", k): v for k, v in cached("p3_stab_ward", stability_ward, **STAB_PARAMS).items()}
for alg in ["K-means", "GMM"]:
    for k in range(2, 11):
        stab_scores[(alg, k)] = cached(f"p3_stab_{alg}_{k}", lambda a=alg, k=k: stability_kmeans_gmm(a, k), **STAB_PARAMS)
stab = pd.DataFrame([{"algorithm": a, "k": k, "stability": np.mean(v), "stability std": np.std(v)}
                     for (a, k), v in stab_scores.items()])
sweep = sweep.drop(columns=[c for c in ["stability", "stability std"] if c in sweep]).merge(stab, on=["algorithm", "k"])
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for alg in ALGS:
    st_ = sweep[sweep["algorithm"] == alg]
    ax.errorbar(st_["k"], st_["stability"], yerr=st_["stability std"], marker="o", capsize=3, color=ALG_COLORS[alg], label=alg)
ax.axhline(0.9, ls=":", color="grey"); ax.set_xticks(range(2, 11))
ax.set(xlabel="k", ylabel="ARI with reference partition", title="Subsampling stability (10 × 80% subsamples, mean ± std)")
ax.legend(); plt.tight_layout(); savefig("p3_stability"); plt.show()
sweep.pivot(index="k", columns="algorithm", values="stability").round(3).T
""")

md(r"""
**Why does GMM score so badly on the internal metrics?** With full covariance it can put a component on any
low-dimensional "spike" in the data. Rainfall is exactly 0 on 63% of days, so a single standardised value is
shared by most of the sample; GMM can give one component (almost) zero variance in that direction and an
arbitrarily high likelihood. We check the variance of `Rainfall_log1p` inside each component:
""")
code(r"""
_, gmm5 = fit_cluster("GMM", 5, X_vis)
ri = FEATURE_SETS["core"].index("Rainfall_log1p")
dry = X_vis[:, ri] == X_vis[:, ri].min()
gmm_lab = gmm5.predict(X_vis)
pd.DataFrame({
    "weight": gmm5.weights_,
    "var(Rainfall_log1p) inside component": gmm5.covariances_[:, ri, ri],
    "share of dry days (Rainfall = 0)": [dry[gmm_lab == c].mean() for c in range(5)],
}).round(4)
""")
md(r"""
One component contains essentially all dry days and has zero rainfall variance: GMM is splitting the data into
"rain / no rain" rather than into compact weather groups. BIC keeps preferring more components because each extra
component can exploit another spike (calm wind = 0, imputed medians). This is a property of the data (point masses
from zeros and imputation), not a bug, and it explains why the likelihood-based criterion (BIC) and the
geometry-based criteria (silhouette, Davies–Bouldin) disagree for GMM.

**GMM settings and a check of the degeneracy.** All GMMs above use scikit-learn defaults except `n_init = 2`:
`covariance_type = "full"`, `reg_covar = 1e-6` (added to the covariance diagonals), `max_iter = 100`, `tol = 1e-3`,
k-means initialisation. Below we report convergence and refit with a larger `reg_covar`, which puts a floor under
every variance and should remove the zero-variance "dry" component if the degeneracy is the problem.
""")
code(r"""
def gmm_sensitivity():
    rows = []
    for k in [3, 5]:
        for reg in [1e-6, 1e-3, 1e-2, 1e-1]:
            g = GaussianMixture(n_components=k, covariance_type="full", n_init=2, reg_covar=reg,
                                random_state=RANDOM_STATE).fit(X_vis)
            lab = g.predict(X_vis)
            rows.append({"k": k, "reg_covar": reg, "converged": g.converged_, "EM iterations": g.n_iter_,
                         "largest weight": g.weights_.max(),
                         "min var(Rainfall_log1p)": g.covariances_[:, ri, ri].min(),
                         "silhouette": silhouette_score(X_vis, lab) if len(np.unique(lab)) > 1 else np.nan,
                         "Davies-Bouldin": davies_bouldin_score(X_vis, lab) if len(np.unique(lab)) > 1 else np.nan,
                         "ARI zone": adjusted_rand_score(y_vis, lab), "BIC": g.bic(X_vis)})
    return pd.DataFrame(rows)

gmm_sens = cached("p3_gmm_sensitivity", gmm_sensitivity, n_per_zone=N_PER_ZONE, seed=RANDOM_STATE, version=1)
gmm_sens.round(4)
""")
md(r"""
All eight fits in the table converged (24–40 EM iterations). With the default `reg_covar = 1e-6` the "dry" component keeps zero
rainfall variance and 61–63% of the weight. Raising `reg_covar` to 0.1 gives every component a rainfall variance of
at least 0.13, lowers the largest weight to 0.33–0.41 and roughly triples the silhouette at k = 3 (0.032 → 0.100).
GMM nevertheless stays below K-means (silhouette 0.164, Davies–Bouldin 1.88 vs 2.70), and its ARI with the zones
stays ≤ 0.14. The degeneracy explains part of GMM's poor scores; the rest is that these weather data do not form
well-separated Gaussian groups. The sweep keeps the default GMM; this table is a sensitivity check.

### 3.3 Choosing the best day-level clustering
""")
code(r"""
# Selection rule, applied to all 27 (algorithm, k) solutions:
#   1. eligible = subsampling stability >= 0.90 (the partition must be reproducible);
#   2. score    = mean of the ranks on silhouette (high), Davies-Bouldin (low) and Calinski-Harabasz (high);
#   3. choose the eligible solution with the best (lowest) mean rank.
# External metrics (ARI/NMI with zone or season) are NOT used for the choice: they measure agreement with our
# question, not clustering quality, and are reported only for interpretation.
rank = sweep[["algorithm", "k"]].copy()
rank["rank silhouette"] = sweep["silhouette"].rank(ascending=False)
rank["rank DB"] = sweep["Davies-Bouldin"].rank(ascending=True)
rank["rank CH"] = sweep["Calinski-Harabasz"].rank(ascending=False)
rank["mean rank"] = rank[["rank silhouette", "rank DB", "rank CH"]].mean(axis=1)
rank["stability"] = sweep["stability"]
rank["eligible"] = rank["stability"] >= 0.90
rank = rank.sort_values("mean rank")
best_row = rank[rank["eligible"]].iloc[0]
ALG_BEST, K_BEST = best_row["algorithm"], int(best_row["k"])
print(f"selected: {ALG_BEST}, k = {K_BEST}")
display(rank.head(8).round(3))

best = sweep[sweep["k"] == K_BEST].set_index("algorithm")[
    ["silhouette", "Davies-Bouldin", "Calinski-Harabasz", "BIC", "ARI zone", "NMI zone", "ARI season", "stability"]]
print(f"k = {K_BEST}: comparison of the three algorithms")
best.round(3)
""")
md(r"""
**Decision: K-means with k = 3.** The rule in the code cell above was written down after the first version of this
analysis (it formalises the reasoning we had used) and it reproduces the original choice:

* Eligible solutions (stability ≥ 0.90) are K-means with k = 2–4 and GMM with k = 3 and 5. Ward is never eligible.
* K-means k = 3 has the best mean rank (1.67): second on silhouette (0.164 vs 0.171 at k = 2), **first on
  Davies–Bouldin among all 27 solutions** (1.88), second on Calinski–Harabasz, stability 0.98. K-means k = 2 is
  first on two metrics but only sixth on Davies–Bouldin (mean rank 2.67).
* **Ward is the least stable algorithm** (0.26–0.36 for every k): removing 20% of the days changes which large
  groups are merged first, and a greedy merge can never be undone. It is deterministic for fixed data, but not
  robust to small changes in the data.
* GMM is stable at k = 3 and 5 but degenerate (above), and ranks far behind on the internal metrics.
* The extra cluster at k = 3 is physically meaningful (next section): it splits cool days into two different
  weather regimes that k = 2 merges.

### 3.4 Visualising and interpreting the clusters
""")
code(r"""
fig, axes = plt.subplots(1, 4, figsize=(23, 5.8))
scatter_by_zone(axes[0], E2); axes[0].set_title("reference: Köppen zone"); axes[0].legend(markerscale=5, fontsize=8)
cl_cmap = plt.get_cmap("Set2")
for ax, alg in zip(axes[1:], ALGS):
    lab = labels_k[(alg, K_BEST)]
    for c in np.unique(lab):
        m = lab == c
        ax.scatter(E2[m, 0], E2[m, 1], s=3, color=cl_cmap(c), label=f"cluster {c}", linewidths=0)
    r = sweep[(sweep["algorithm"] == alg) & (sweep["k"] == K_BEST)].iloc[0]
    ax.set_title(f"{alg}, k={K_BEST} | sil {r['silhouette']:.3f} | ARI(zone) {r['ARI zone']:.3f}")
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(markerscale=5, fontsize=8)
plt.suptitle("Clusters drawn on the Part 2 t-SNE map (clustering itself was done in 20-D)")
plt.tight_layout(); savefig("p3_clusters_on_tsne"); plt.show()
""")
code(r"""
km_lab = labels_k[("K-means", K_BEST)]
profile_cols = ["MaxTemp", "MinTemp", "Temp3pm", "Humidity9am", "Humidity3pm", "Pressure3pm",
                "Rainfall_log1p", "WindGustSpeed", "WindSpeed3pm", "doy_cos"]
raw_vis = F.loc[sample_idx]
fig, axes = plt.subplots(1, 3, figsize=(21, 4.8), gridspec_kw={"width_ratios": [2.2, 1.3, 1]})
z_means = pd.DataFrame(X_vis, columns=FEATURE_SETS["core"]).groupby(km_lab)[profile_cols].mean()
annot = raw_vis.groupby(km_lab)[profile_cols].mean().round(1)        # annotate with real units
sns.heatmap(z_means, annot=annot, fmt="", cmap="RdBu_r", center=0, vmin=-1.3, vmax=1.3, ax=axes[0],
            cbar_kws={"label": "mean (standardised)"})
axes[0].set(title="Cluster profile (colour = z-score, number = mean in original units)", ylabel="K-means cluster")
axes[0].tick_params(axis="x", rotation=35)
ct_zone = pd.crosstab(km_lab, y_vis, normalize="index")[ZONE_ORDER]
sns.heatmap(ct_zone, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1, ax=axes[1], cbar=False)
axes[1].set(title="zone mix of each cluster", xlabel="", ylabel=""); axes[1].tick_params(axis="x", rotation=35)
ct_season = pd.crosstab(km_lab, season_vis, normalize="index")[["Summer", "Autumn", "Winter", "Spring"]]
sns.heatmap(ct_season, annot=True, fmt=".2f", cmap="Oranges", vmin=0, vmax=1, ax=axes[2], cbar=False)
axes[2].set(title="season mix of each cluster", xlabel="", ylabel="")
plt.tight_layout(); savefig("p3_cluster_profiles"); plt.show()
print("cluster sizes:", np.bincount(km_lab))
""")
md(r"""
**Interpretation (K-means, k = 3).** The clusters are *weather regimes*, not climate zones:

* **Warm days** — high max/min temperature, slightly lower humidity, mostly summer and autumn. This cluster
  contains almost all Tropical days (and is about 40% Tropical), because tropical days are warm all year.
* **Cool, calm days** — the lowest temperatures, low wind and little rain, mostly winter and autumn: settled high-pressure weather.
* **Cool, windy, wet days** — much higher gusts and rainfall: the passage of fronts and storms, most common in the
  southern zones (Oceanic, Mediterranean).

Every non-tropical zone appears in all three clusters, because every zone has warm days, calm winter days and
stormy days. This is why ARI with `Zone` never exceeds 0.13 for any algorithm or k (ARI with the season is
at most about 0.1 as well): a **single day** of weather does not carry a clear climate signature.

### 3.5 Extra: clustering *stations* instead of days

Climate is the long-term statistics of weather. If we describe each of the 48 stations by its **average seasonal
cycle** — mean MaxTemp, MinTemp, Humidity3pm and Rainfall for each of the 12 months (48 numbers per station,
computed from all its days) — the climate signal should no longer be hidden by day-to-day weather.

Pressure is not used because 4 stations never record it. Each of the 48 columns is standardised. This analysis
uses `Location` only to *aggregate* days; the algorithms still never see the zone labels.
""")
code(r"""
PROFILE_VARS = ["MaxTemp", "MinTemp", "Humidity3pm", "Rainfall"]
station_profile = (df.assign(month=df["Date"].dt.month, Location=groups)
                     .pivot_table(index="Location", columns="month", values=PROFILE_VARS, aggfunc="mean"))
assert station_profile.notna().all().all()
S = StandardScaler().fit_transform(station_profile.values)
station_zone_true = station_zone.loc[station_profile.index].values
print("station profile matrix:", S.shape)

st_rows = []
for k in range(3, 8):
    for alg in ["K-means", "Ward"]:
        lab = (KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(S) if alg == "K-means"
               else AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(S))
        st_rows.append({"algorithm": alg, "k": k, "silhouette": silhouette_score(S, lab),
                        "ARI zone": adjusted_rand_score(station_zone_true, lab),
                        "NMI zone": normalized_mutual_info_score(station_zone_true, lab)})
st_sweep = pd.DataFrame(st_rows)
st_sweep.pivot(index="k", columns="algorithm", values=["silhouette", "ARI zone", "NMI zone"]).round(3)
""")
code(r"""
Z = linkage(S, method="ward")
st_lab = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(S)
fig, ax = plt.subplots(figsize=(16, 5.5))
dendrogram(Z, labels=station_profile.index.tolist(), leaf_rotation=90, leaf_font_size=9,
           color_threshold=Z[-5, 2] + 1e-9, above_threshold_color="grey", ax=ax)
for t in ax.get_xticklabels():                         # leaf labels coloured by the Köppen zone
    t.set_color(ZONE_COLORS[station_zone[t.get_text()]]); t.set_fontweight("bold")
ax.axhline(Z[-5, 2] + 1e-9, ls="--", color="k", lw=1)
ax.set(title="Ward dendrogram of 48 station climate profiles (labels coloured by Köppen zone; dashed line = 5 clusters)",
       ylabel="Ward merge distance")
plt.tight_layout(); savefig("p3_station_dendrogram"); plt.show()
""")
code(r"""
loc_xy = zones.set_index("Location").loc[station_profile.index, ["lon", "lat"]]
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for zone in ZONE_ORDER:
    m = station_zone_true == zone
    axes[0].scatter(loc_xy["lon"][m], loc_xy["lat"][m], s=70, color=ZONE_COLORS[zone], edgecolor="k", label=zone)
axes[0].set_title("Köppen zone (our label)")
for c in range(5):
    m = st_lab == c
    axes[1].scatter(loc_xy["lon"][m], loc_xy["lat"][m], s=70, color=cl_cmap(c), edgecolor="k", label=f"cluster {c}")
axes[1].set_title("Ward clusters of station climate profiles (k = 5)")
for ax in axes:
    ax.set(xlabel="Longitude", ylabel="Latitude"); ax.set_aspect("equal"); ax.legend(loc="lower left", fontsize=9)
plt.tight_layout(); savefig("p3_station_clusters_map"); plt.show()

print(pd.crosstab(pd.Series(st_lab, name="cluster"), pd.Series(station_zone_true, name="zone"))[ZONE_ORDER])
for c in range(5):
    print(f"cluster {c}: " + ", ".join(station_profile.index[st_lab == c]))
""")
code(r"""
compare = pd.DataFrame({
    "unit clustered": ["one day (6,000 days)", "one day (6,000 days)", "station profile (48 stations)"],
    "algorithm": ["K-means k=3 (best by internal metrics)", "Ward k=5 (same k as zones)", "Ward k=5"],
    "ARI with Zone": [sweep.query("algorithm=='K-means' and k==3")["ARI zone"].item(),
                      sweep.query("algorithm=='Ward' and k==5")["ARI zone"].item(),
                      st_sweep.query("algorithm=='Ward' and k==5")["ARI zone"].item()],
    "NMI with Zone": [sweep.query("algorithm=='K-means' and k==3")["NMI zone"].item(),
                      sweep.query("algorithm=='Ward' and k==5")["NMI zone"].item(),
                      st_sweep.query("algorithm=='Ward' and k==5")["NMI zone"].item()],
}).round(3)
compare
""")
md(r"""
**Interpretation of the station clusters (Ward, k = 5).**

* Two Köppen zones are recovered exactly: all 4 **Tropical** stations form one cluster, and a second cluster
  holds exactly the 9 **coastal Subtropical** stations from Brisbane to Wollongong.
* The other three clusters follow **distance from the sea** more than Köppen's rainfall-season rule: a hot, dry
  *inland* cluster (6 of the 7 Arid stations, plus the Perth area, Adelaide, Moree and western Sydney), a cool
  *south-east* cluster (10 of 11 Oceanic stations plus Albury, Wagga Wagga, Nhil and the Mediterranean stations of
  south-east South Australia / western Victoria), and a small cool, wet, windy
  *south-west coast* cluster (Albany, Walpole, Witchcliffe, Portland, and the alpine MountGinini).
* The Mediterranean zone is split across three clusters. Köppen defines it by a *ratio* (the driest summer month
  has less than a third of the wettest winter month's rain), which is a small part of our 48-number profile,
  whereas temperature and humidity differences between coast and inland dominate the Euclidean distance.
* ARI with the zones rises from about 0.1 for days to about 0.35 for stations (and to 0.42–0.46 for k = 7).

### 3.6 Summary of Part 3

1. **Algorithms**: K-means, Ward agglomerative and GMM (full covariance), each for k = 2–10, compared with three
   internal metrics, two external metrics, subsampling stability (all three algorithms), inertia and BIC.
2. **Best day-level result** (stability ≥ 0.90, best mean rank of silhouette / Davies–Bouldin / Calinski–Harabasz):
   K-means, k = 3 (lowest Davies–Bouldin of all solutions, near-best silhouette, stability 0.98). Its clusters are
   weather regimes (warm / cool-calm / cool-windy-wet), not climate zones.
3. **Algorithm comparison**: K-means ≥ Ward on all internal metrics, and Ward is unstable under subsampling
   (≤ 0.36); GMM with full covariance degenerates by placing a zero-variance component on the 63% of dry days, so
   BIC and silhouette disagree; a larger `reg_covar` reduces but does not remove the gap to K-means.
4. **Can clustering rediscover the zones?** From single days, no (ARI ≤ 0.13 for every algorithm and k). From
   station climate profiles, partly (ARI ≈ 0.35): the tropics and the subtropical coast are found exactly, and the
   rest of the map is organised by coast vs. inland. For Part 4 this means a classifier must combine many weak
   signals of a single day; we should expect Tropical to be easy and the southern zones to be confused.
""")
