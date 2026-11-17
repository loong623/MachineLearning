"""Self-contained notebook cells; assemble with tools/build_notebook.py."""
import nbformat as nbf
cells=[]
md=lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code=lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
## 2. Data Visualization with t-SNE

**Question.** Do days from different climate zones separate naturally in the 20-dimensional feature space?

### 2.1 Sample and caching

t-SNE costs roughly O(n log n) per iteration with a large constant, so we embed a **balanced sample of 1,200 days per zone
(6,000 days)**. Balancing keeps the small Tropical class (7.6% of rows) from being drowned by Subtropical (32%) in the
picture. The same sample is reused for clustering in Part 3.

The preprocessor (median imputation + standardisation, no missing indicators) is fitted on this sample. Zone labels determine the balanced sampling; the t-SNE optimisation itself receives no labels. The picture describes this deliberately reweighted sample, not the natural data frequency.

Each embedding takes 20–30 s. Embeddings are cached under `data/processed/tsne_audit/<run_id>/`; the file name contains a hash of
the input matrix and the parameters, and the full experiment fingerprint, so code/data/environment changes invalidate results. Set `RECOMPUTE_TSNE = True`
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

TSNE_DIR = DATA_DIR / "processed" / "tsne_audit" / RUN_ID; TSNE_DIR.mkdir(parents=True, exist_ok=True)
RECOMPUTE_TSNE = RECOMPUTE

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

md(r"""
### 2.6 PCA (linear) vs t-SNE, and a 3-D t-SNE view
""")

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
### 2.7 Interpretation and limits

Read the new tables and plots above. Compare zone mixing, season controls, seed/perplexity sensitivity and neighbourhood preservation. High local label agreement is descriptive: labels influenced sampling and station days occur on both sides of the diagnostic kNN folds. Kappa adjusts chance agreement but is not an information-theoretic comparison between different prediction targets. t-SNE distances between islands and apparent island sizes should not be interpreted physically; classifiers below use the original features, not t-SNE coordinates.
""")
