"""Part 2 of the project notebook: t-SNE visualisation. Assembled by tools/build_notebook.py."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
## 2. Data Visualization with t-SNE

**Question.** Do days from different climate zones separate naturally in the 20-dimensional feature space?

### 2.1 Sample and caching

t-SNE costs roughly O(n log n) per iteration with a large constant, so we embed a **balanced sample of 1,200 days per zone
(6,000 days)**. Balancing keeps the small Tropical class (7.6% of rows) from being drowned by Subtropical (32%) in the
picture. The same sample is reused for clustering in Part 3.

The preprocessor (median imputation + standardisation, no missing indicators) is fitted on this sample. No labels
are used anywhere in t-SNE; `Zone` is only used afterwards to colour the points.

Each embedding takes 20–30 s. Embeddings are cached in `data/processed/tsne/`; the file name contains a hash of
the input matrix and the parameters, so a changed input can never reuse a stale result. Set `RECOMPUTE_TSNE = True`
to ignore the cache.
""")
code(r"""
N_PER_ZONE = 1200
sample_idx = df.groupby("Zone", group_keys=False).sample(n=N_PER_ZONE, random_state=RANDOM_STATE).index
X_vis = build_preprocessor().fit_transform(F.loc[sample_idx, FEATURE_SETS["core"]])
y_vis = df.loc[sample_idx, "Zone"].values
g_vis = groups.loc[sample_idx].values
month_vis = df.loc[sample_idx, "Date"].dt.month.values
SEASON_OF_MONTH = {12: "Summer", 1: "Summer", 2: "Summer", 3: "Autumn", 4: "Autumn", 5: "Autumn",
                   6: "Winter", 7: "Winter", 8: "Winter", 9: "Spring", 10: "Spring", 11: "Spring"}
season_vis = np.array([SEASON_OF_MONTH[m] for m in month_vis])
print("sample:", X_vis.shape, "| stations in sample:", len(set(g_vis)))

TSNE_DIR = DATA_DIR / "processed" / "tsne"; TSNE_DIR.mkdir(parents=True, exist_ok=True)
RECOMPUTE_TSNE = False

def tsne_cached(X, n_components=2, perplexity=30, seed=RANDOM_STATE):
    key = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()[:12]
    path = TSNE_DIR / f"tsne_{key}_d{n_components}_p{perplexity}_s{seed}.npy"
    if path.exists() and not RECOMPUTE_TSNE:
        return np.load(path)
    E = TSNE(n_components=n_components, perplexity=perplexity, init="pca", learning_rate="auto",
             max_iter=1000, random_state=seed).fit_transform(X)
    np.save(path, E)
    return E

def knn_purity(E, labels, k=10):
    # 5-fold accuracy of a k-NN classifier on the embedding = how often a point's neighbours share its label
    return cross_val_score(KNeighborsClassifier(n_neighbors=k), E, labels, cv=5).mean()

def knn_kappa(E, labels, k=10):
    # the same k-NN predictions, scored with Cohen's kappa = (accuracy - chance agreement) / (1 - chance agreement),
    # so that label sets with different numbers of classes / class shares can be compared
    pred = cross_val_predict(KNeighborsClassifier(n_neighbors=k), E, labels, cv=5)
    return cohen_kappa_score(labels, pred)
""")

md(r"""
### 2.2 Main embedding (perplexity 30)

We colour the same embedding three ways: by zone (our target), by month, and by maximum temperature.
""")
code(r"""
E2 = tsne_cached(X_vis, perplexity=30)

def scatter_by_zone(ax, E, labels=y_vis, s=3):
    for zone in ZONE_ORDER:
        m = labels == zone
        ax.scatter(E[m, 0], E[m, 1], s=s, color=ZONE_COLORS[zone], label=zone, alpha=0.7, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
scatter_by_zone(axes[0], E2)
axes[0].legend(markerscale=5, loc="upper left", fontsize=9); axes[0].set_title("coloured by climate zone")
sc = axes[1].scatter(E2[:, 0], E2[:, 1], s=3, c=month_vis, cmap="twilight_shifted", vmin=0.5, vmax=12.5, linewidths=0)
cb = plt.colorbar(sc, ax=axes[1], ticks=range(1, 13)); cb.set_label("month (Jan = summer, Jul = winter)")
axes[1].set_title("coloured by month"); axes[1].set_xticks([]); axes[1].set_yticks([])
sc = axes[2].scatter(E2[:, 0], E2[:, 1], s=3, c=F.loc[sample_idx, "MaxTemp"], cmap="inferno", linewidths=0)
plt.colorbar(sc, ax=axes[2]).set_label("MaxTemp (°C)")
axes[2].set_title("coloured by max temperature"); axes[2].set_xticks([]); axes[2].set_yticks([])
fig.suptitle("t-SNE of 6,000 days (1,200 per zone), 20 core features, perplexity 30", fontsize=13)
plt.tight_layout(); savefig("p2_tsne_main"); plt.show()
""")

md(r"""
### 2.3 Quantifying what the picture shows

A t-SNE plot is easy to over-read, so we add three descriptive numbers:

* **k-NN purity** — 5-fold accuracy of a 10-nearest-neighbour classifier on the coordinates. It answers
  "how often do a point's neighbours have the same label?" With 5 balanced classes, chance is 0.20.
* **Cohen's κ** of the same predictions. Zone (5 balanced classes), season (4 classes) and station (48 classes)
  have different chance levels, so raw purities cannot be compared across label sets; κ subtracts chance agreement.
* **Trustworthiness** (`sklearn.manifold.trustworthiness`, k = 10) — how well the 2-D neighbourhoods reflect
  the true 20-D neighbourhoods (1 = perfect).

PCA to 2-D is the linear baseline. **These are diagnostics of the picture, not estimates of classification
performance**: t-SNE reshapes neighbourhoods, it used all 6,000 points at once (no train/test separation), and days
of the same station appear on both sides of every k-NN fold. Generalisation is measured in Parts 4–6.
""")
code(r"""
pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_vis)
P2 = pca2.transform(X_vis)
spaces = {"original 20-D": X_vis, "PCA 2-D": P2, "t-SNE 2-D": E2}
quant = pd.DataFrame({
    "purity: zone": {k: knn_purity(v, y_vis) for k, v in spaces.items()},
    "κ: zone": {k: knn_kappa(v, y_vis) for k, v in spaces.items()},
    "purity: season": {k: knn_purity(v, season_vis) for k, v in spaces.items()},
    "κ: season": {k: knn_kappa(v, season_vis) for k, v in spaces.items()},
    "purity: station (48)": {k: knn_purity(v, g_vis) for k, v in spaces.items()},
    "κ: station": {k: knn_kappa(v, g_vis) for k, v in spaces.items()},
    "trustworthiness": {"original 20-D": 1.0, "PCA 2-D": trustworthiness(X_vis, P2, n_neighbors=10),
                        "t-SNE 2-D": trustworthiness(X_vis, E2, n_neighbors=10)},
}).round(3)
print(f"chance level: zone 0.20, season ~0.25, station ~{1 / 48:.3f}; "
      f"PCA explains {pca2.explained_variance_ratio_.sum():.1%} of the variance in 2-D")
quant
""")
code(r"""
# Which zones are mixed with which? k-NN predictions on the t-SNE map, row-normalised
pred = cross_val_predict(KNeighborsClassifier(n_neighbors=10), E2, y_vis, cv=5)
mix = pd.crosstab(pd.Series(y_vis, name="true zone"), pd.Series(pred, name="majority zone of the 10 neighbours"),
                  normalize="index").loc[ZONE_ORDER, ZONE_ORDER]
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(mix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set_title("Neighbourhood mixing on the t-SNE map")
plt.tight_layout(); savefig("p2_tsne_neighbour_mixing"); plt.show()
""")

md(r"""
### 2.4 Is the structure just the season?

The month colouring shows a strong summer–winter gradient. If season alone drove the layout, zones would
separate much better once the season is held fixed. We check this by (a) embedding summer days and winter days
separately, and (b) removing the two day-of-year features.
""")
code(r"""
feat = FEATURE_SETS["core"]
no_season_cols = [i for i, f in enumerate(feat) if not f.startswith("doy")]
variants = {
    "all days, all 20 features": (X_vis, y_vis),
    "summer days only (Dec–Feb)": (X_vis[season_vis == "Summer"], y_vis[season_vis == "Summer"]),
    "winter days only (Jun–Aug)": (X_vis[season_vis == "Winter"], y_vis[season_vis == "Winter"]),
    "all days, without doy_sin/doy_cos": (X_vis[:, no_season_cols], y_vis),
}
rows, embs = [], {}
for name, (Xv, yv) in variants.items():
    Ev = E2 if name.startswith("all days, all") else tsne_cached(Xv, perplexity=30)
    embs[name] = (Ev, yv)
    sv = season_vis if name.startswith("all days") else None            # season is constant in the 1-season variants
    rows.append({"variant": name, "n": len(yv), "zone purity (20-D)": knn_purity(Xv, yv),
                 "zone purity (t-SNE)": knn_purity(Ev, yv), "zone κ (t-SNE)": knn_kappa(Ev, yv),
                 "season κ (t-SNE)": knn_kappa(Ev, sv) if sv is not None else np.nan})
pd.DataFrame(rows).set_index("variant").round(3)
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, name in zip(axes, list(embs)[1:]):
    Ev, yv = embs[name]
    scatter_by_zone(ax, Ev, yv, s=5); ax.set_title(name)
axes[0].legend(markerscale=4, fontsize=8)
plt.tight_layout(); savefig("p2_tsne_season_control"); plt.show()
""")

md(r"""
### 2.5 Sensitivity: perplexity and random seed

Perplexity is roughly the number of effective neighbours each point considers: small values emphasise local
detail, large values preserve more global structure. t-SNE is also non-convex, so different seeds give different
layouts; conclusions should hold for all of them.
""")
code(r"""
perplexities = [5, 30, 50, 100]
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
perp_rows = []
for ax, p in zip(axes, perplexities):
    Ep = tsne_cached(X_vis, perplexity=p)
    pur, tru = knn_purity(Ep, y_vis), trustworthiness(X_vis, Ep, n_neighbors=10)
    perp_rows.append({"perplexity": p, "zone purity": pur, "trustworthiness": tru})
    scatter_by_zone(ax, Ep); ax.set_title(f"perplexity {p}\npurity {pur:.3f} | trust {tru:.3f}")
axes[0].legend(markerscale=5, fontsize=8)
plt.tight_layout(); savefig("p2_tsne_perplexity"); plt.show()
pd.DataFrame(perp_rows).set_index("perplexity").round(3).T
""")
code(r"""
seeds = [0, 7, RANDOM_STATE]
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for ax, s in zip(axes, seeds):
    Es = tsne_cached(X_vis, perplexity=30, seed=s)
    scatter_by_zone(ax, Es); ax.set_title(f"seed {s} | zone purity {knn_purity(Es, y_vis):.3f}")
axes[0].legend(markerscale=5, fontsize=8)
plt.tight_layout(); savefig("p2_tsne_seeds"); plt.show()
""")

md("### 2.6 PCA (linear) vs t-SNE, and a 3-D t-SNE view")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
scatter_by_zone(axes[0], P2)
axes[0].set_title(f"PCA 2-D ({pca2.explained_variance_ratio_.sum():.0%} variance) | purity {knn_purity(P2, y_vis):.3f}")
scatter_by_zone(axes[1], E2); axes[1].set_title(f"t-SNE 2-D | purity {knn_purity(E2, y_vis):.3f}")
axes[0].legend(markerscale=5, fontsize=9)
plt.tight_layout(); savefig("p2_pca_vs_tsne"); plt.show()
""")
code(r"""
E3 = tsne_cached(X_vis, n_components=3, perplexity=30)
fig = plt.figure(figsize=(13, 6))
for i, (elev, azim) in enumerate([(20, 35), (20, 125)]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    for zone in ZONE_ORDER:
        m = y_vis == zone
        ax.scatter(E3[m, 0], E3[m, 1], E3[m, 2], s=2, color=ZONE_COLORS[zone], label=zone, alpha=0.6)
    ax.view_init(elev=elev, azim=azim); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f"3-D t-SNE, view azim={azim}°")
fig.axes[0].legend(markerscale=5, fontsize=8, loc="upper left")
plt.tight_layout(); savefig("p2_tsne_3d"); plt.show()
print("zone purity in 3-D t-SNE:", round(knn_purity(E3, y_vis), 3))
""")

md(r"""
### 2.7 Observations and insights

The numbers below refer to the tables printed above.

1. **Only the Tropical zone forms its own region.** Tropical days occupy a separate band of the map, and 96% of
   them have a Tropical majority among their 10 nearest neighbours. The four other zones overlap heavily: their
   neighbourhood agreement is only about 0.50–0.64. Hot nights (MinTemp ≈ 20–25 °C) and low pressure make tropical
   days unlike anything else, while a mild day in Adelaide can look exactly like one in Melbourne.
2. **The season gradient on the map comes mostly from the calendar features.** The month colouring shows summer
   and winter days on opposite sides. With all 20 features, neighbourhoods agree about equally well on season and on
   zone (κ 0.55 vs 0.54; the raw purities 0.663 vs 0.631 are not comparable because the chance levels differ).
   But `doy_sin`/`doy_cos` are inputs, so part of this is by construction: without them, season agreement falls to
   κ 0.17 while zone agreement stays at 0.49. The weather itself carries more information about the zone than about
   the season. Holding the season fixed (summer-only or winter-only embeddings) changes zone κ only slightly
   (0.57 and 0.55 vs 0.54), so season does not explain the overlap between zones.
   The season-specific maps do show *which* zone is distinctive when: in winter the Tropical days form a
   completely separate island (warm, humid dry season vs. cool south), while in summer the Arid days move to
   their own edge (hottest and driest), opposite the hot-and-humid Tropical days.
3. **The confusions follow climate similarity.** Mediterranean and Oceanic are the most mixed pair
   (both southern, temperate, wet winter); Arid days mix most with Mediterranean (hot, dry summers in
   inland South Australia and Victoria). This is the confusion pattern we expect the classifiers in Part 4 to show.
4. **Neighbours also share stations.** Station purity on the map is 0.31 (κ 0.29; chance 0.02): days from the
   same station tend to have similar neighbours. This shows that station-specific information is present in the
   weather variables. It does not by itself show what a classifier will learn; it is a warning that a random
   train/test split *could* reward recognising stations, which Parts 4–6 test directly.
5. **t-SNE vs PCA.** A 2-D PCA projection keeps 42% of the variance and preserves neighbourhoods poorly
   (trustworthiness 0.81 vs 0.985 for t-SNE; zone κ 0.22 vs 0.54). So t-SNE preserves *local neighbourhoods* much
   better than a 2-D linear projection. This alone does not prove that the zone structure is non-linear: two
   principal components discard more than half of the variance.
6. **Robustness.** Zone purity varies by about ±0.01 across perplexities 5–100 and across three random seeds,
   so the conclusions do not depend on these settings. The 2-D maps keep almost all of the 20-D neighbourhood
   information: the 20-D k-NN purity (0.683) is only about 5 points higher than on the t-SNE map (0.631).

**Limitations of reading t-SNE.** Distances between groups, group sizes and the empty space on the map are not
meaningful; t-SNE preserves local neighbourhoods only. The embedding also cannot place new points, so it is used
here for exploration, not as input to the classifiers.
""")
