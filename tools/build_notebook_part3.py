"""Self-contained notebook cells; assemble with tools/build_notebook.py."""
import nbformat as nbf
cells=[]
md=lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code=lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
## 3. Clustering Analysis

**Question.** Without labels, can clustering rediscover the climate zones?

We cluster the same balanced sample of 6,000 days as in Part 2 (preprocessed 20-D `core` features). Zone labels determine balanced sampling, but are not supplied to the clustering algorithms. They are external references for interpretation; this is not a wholly label-free sampling pipeline.

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
* **ARI** and **NMI** with `Zone` and season. ARI is chance-adjusted; NMI is not. Both equal 1 for identical partitions.
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
Inspect the table for components concentrated on dry days and near the covariance regularisation floor. Point masses from dry weather and imputation can be poorly described by continuous Gaussians. BIC and compactness metrics answer different questions; neither alone establishes a true weather taxonomy.
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
Use the convergence and covariance table above to distinguish numerical convergence from model suitability. Covariance regularisation prevents an exactly singular fitted Gaussian; very small variance around dry/imputed values can nevertheless dominate likelihood. This sensitivity check does not prove a unique physical cause for all differences between algorithms.

### 3.3 Choosing a descriptive day-level partition
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
The stability threshold and rank rule were introduced after the earlier exploratory analysis, not preregistered. They choose a descriptive partition under a specific geometry, not a uniquely true number of weather regimes. Low absolute silhouette remains a limitation even when repeatability is high. Additional rule and sampling sensitivity checks follow below.

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
The temperature, wind, rain and pressure profiles permit descriptive labels such as warm, cool with weaker winds, and cool with stronger winds and more rain. They do not independently verify synoptic fronts. Low agreement with Zone means these particular representations and clustering methods do not closely reproduce the climate labels; it is not a proof that single days contain no climate information.

### 3.5 Extra: station climate profiles

Aggregate each station to twelve monthly means of four variables. This changes the unit, features, sample size and clustering setup. Differences from day-level ARI therefore cannot be attributed solely to temporal averaging. Coordinates and labels are not clustering inputs; k=5 is an exploratory comparison with the five-label task.
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
### 3.6 Scope of the station-profile comparison

Read cluster membership and the cross-tabulation above. A coastal Subtropical subset is not an entire Köppen class. Coast/inland interpretations are hypotheses based on station names and coordinates, not a tested distance-to-coast mechanism. The dataset has only 48 labelled stations; ARI comparisons across day and station units are descriptive.
""")

md(r"""
### 3.7 Selection-rule and sampling sensitivity
The earlier ranking rule was post-hoc. Vary its stability threshold and remove one compactness criterion at a time. Natural-frequency sampling checks the influence of label-balanced sampling. Perturb entire stations or years rather than scattered days, using the same reference scaling; these diagnostics examine composition sensitivity, not prediction on unobserved stations. Low silhouette is not cured by high stability.
""")

code(r"""
rule_rows=[]
for threshold in [.80,.85,.90,.95]:
    for metrics in [["silhouette","Davies-Bouldin","Calinski-Harabasz"],["silhouette","Davies-Bouldin"],["silhouette","Calinski-Harabasz"],["Davies-Bouldin","Calinski-Harabasz"]]:
        eligible=sweep[sweep.stability>=threshold].copy()
        if len(eligible):
            ranking=sum(sweep[m].rank(ascending=(m=="Davies-Bouldin")) for m in metrics)/len(metrics)
            winner=eligible.loc[ranking.loc[eligible.index].idxmin()]
            rule_rows.append({"threshold":threshold,"metrics":";".join(metrics),"algorithm":winner.algorithm,"k":int(winner.k)})
export_table("clustering_rule_sensitivity",pd.DataFrame(rule_rows))

def cluster_sampling_controls():
    out=[]
    natural=df.sample(n=len(sample_idx),random_state=42).index
    xn=build_preprocessor().fit_transform(F.loc[natural,FEATURE_SETS["core"]])
    for alg in ALGS:
        lab,_=fit_cluster(alg,3,xn)
        out.append({"scheme":"natural-frequency","algorithm":alg,"repeat":0,"n":len(xn),"ARI_reference":np.nan,"ARI_zone":adjusted_rand_score(y[natural],lab),"silhouette":silhouette_score(xn,lab)})
    year=df.loc[sample_idx,"Date"].dt.year.to_numpy()
    for scheme,block in [("station-block",g_vis),("year-block",year)]:
        unique=np.unique(block)
        for rep in range(5):
            kept=np.random.default_rng(100+rep).choice(unique,max(2,int(.8*len(unique))),replace=False)
            idx=np.flatnonzero(np.isin(block,kept))
            for alg in ALGS:
                lab,_=fit_cluster(alg,3,X_vis[idx],seed=42+rep)
                out.append({"scheme":scheme,"algorithm":alg,"repeat":rep,"n":len(idx),"ARI_reference":adjusted_rand_score(labels_k[(alg,3)][idx],lab),"ARI_zone":adjusted_rand_score(y_vis[idx],lab),"silhouette":silhouette_score(X_vis[idx],lab)})
    return pd.DataFrame(out)
cluster_controls=cached("cluster_sampling_controls",cluster_sampling_controls)
export_table("clustering_sampling_controls",cluster_controls)
cluster_controls.groupby(["scheme","algorithm"])[["ARI_reference","ARI_zone","silhouette"]].mean().round(3)

""")
