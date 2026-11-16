"""Part 4 of the project notebook: prediction (training and testing). Assembled by tools/build_notebook.py."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
## 4. Prediction: Training and Testing

**Question.** Given the labels, how well can simple models classify the climate zone of a single day?

### 4.1 Target, models and why we chose them

* **Target**: `Zone` (5 classes, Köppen-derived, Section 0.2). Classes are moderately imbalanced
  (Tropical 7.6% … Subtropical 32.2%), so besides accuracy we always report **macro-F1** (every zone counts equally).
* **Features**: the 20 `core` features from Part 1 (no station-specific sensors, no missing indicators).
* **Logistic Regression (multinomial)** — a linear model: one weight vector per zone, softmax over the five scores.
  It is the natural linear baseline, gives calibrated-looking probabilities (needed for ROC in Part 5), and its
  coefficients are interpretable. Default regularisation `C = 1`.
* **Decision Tree (CART, Gini)** — a non-linear model made of axis-aligned threshold rules such as
  "MinTemp > 18.5 °C". It can represent interactions that LR cannot (e.g. *hot and dry* vs *hot and humid*), and
  its rules are readable. We first train it with **default settings (no depth limit)** to see what an
  unconstrained model does; Part 5 tunes it.

Both models are sklearn Pipelines `median imputation → standardisation → model`, so the imputation medians and the
scaling statistics are learned from the training set only.

### 4.2 Two 70/30 splits, two questions

| Split | How | Question it answers |
|---|---|---|
| **Random** (required by the task) | stratified by `Zone`, 70% / 30% of the rows | Can the model classify *new days* of stations it has already seen? |
| **Station-grouped** | 70% / 30% of the **stations** (`GroupShuffleSplit`); no station appears on both sides | Can it classify days from *stations it has never seen*? |

Both splits were created in Section 1.6 and are reused here. The grouped test set contains 15 stations from all
5 zones.
""")
code(r"""
X_all = F[FEATURE_SETS["core"]]
ALL = np.arange(len(df))
SPLITS = {"random": (rand_tr, rand_te), "grouped": (grp_tr, grp_te)}
print("grouped test stations:", ", ".join(sorted(groups.iloc[grp_te].unique())))
print(pd.DataFrame({s: pd.Series(y[b]).value_counts() for s, (a, b) in SPLITS.items()}).loc[ZONE_ORDER]
      .rename(columns=lambda c: f"{c} test rows"))

def make_model(name, **params):
    if name == "LR":
        est = LogisticRegression(max_iter=3000, **params)
    elif name == "DT":
        est = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    else:
        raise ValueError(name)
    return make_pipeline(build_preprocessor(), est)
""")

md(r"""
### 4.3 Training

Training is a single `fit` on the training rows. For LR, `lbfgs` minimises the multinomial cross-entropy plus an
L2 penalty `‖w‖² / (2C)`. For the tree, CART greedily chooses at every node the feature and threshold that most
reduce Gini impurity, and keeps splitting until every leaf is pure (default settings).
""")
code(r"""
import time
models_p4, fit_info = {}, []
for split, (a, b) in SPLITS.items():
    for name in ["LR", "DT"]:
        t0 = time.time()
        m = make_model(name).fit(X_all.iloc[a], y[a])
        info = {"split": split, "model": name, "fit time (s)": round(time.time() - t0, 1)}
        if name == "DT":
            info.update(depth=m[-1].get_depth(), leaves=m[-1].get_n_leaves())
        fit_info.append(info); models_p4[(split, name)] = m
pd.DataFrame(fit_info)
""")
md(r"""
The unconstrained tree grows to a depth of about 35 with roughly 18,000 leaves for ~100,000 training days, i.e.
on average about 5–6 training days per leaf. This already suggests memorisation.

### 4.4 Evaluation on the training set, test set and entire set
""")
code(r"""
from sklearn.metrics import confusion_matrix

def evaluate(model, idx):
    pred = model.predict(X_all.iloc[idx])
    return {"accuracy": accuracy_score(y[idx], pred),
            "macro-F1": f1_score(y[idx], pred, average="macro"),
            "weighted-F1": f1_score(y[idx], pred, average="weighted")}, pred

eval_rows, preds_p4 = [], {}
for (split, name), m in models_p4.items():
    a, b = SPLITS[split]
    for part, idx in [("train", a), ("test", b), ("entire", ALL)]:
        res, pred = evaluate(m, idx)
        preds_p4[(split, name, part)] = pred
        eval_rows.append({"split": split, "model": name, "set": part, **res})
p4_table = pd.DataFrame(eval_rows)
p4_table.pivot_table(index=["split", "model"], columns="set", values=["accuracy", "macro-F1"]).round(3)[
    [("accuracy", "train"), ("accuracy", "test"), ("accuracy", "entire"),
     ("macro-F1", "train"), ("macro-F1", "test"), ("macro-F1", "entire")]]
""")
code(r"""
def plot_cm(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=ZONE_ORDER)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{p:.2f}\n({c:,})" for p, c in zip(rp, rc)] for rp, rc in zip(cm_norm, cm)])
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                xticklabels=ZONE_ORDER, yticklabels=ZONE_ORDER, annot_kws={"fontsize": 8})
    ax.set(title=title, xlabel="predicted", ylabel="true")
    ax.tick_params(axis="x", rotation=35); ax.tick_params(axis="y", rotation=0)
    return cm

fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
for i, name in enumerate(["LR", "DT"]):
    for j, part in enumerate(["train", "test", "entire"]):
        idx = {"train": rand_tr, "test": rand_te, "entire": ALL}[part]
        acc = accuracy_score(y[idx], preds_p4[("random", name, part)])
        plot_cm(axes[i, j], y[idx], preds_p4[("random", name, part)], f"{name} | random split | {part} set | acc {acc:.3f}")
plt.suptitle("Confusion matrices, random 70/30 split (row-normalised = recall; counts in brackets)", fontsize=13)
plt.tight_layout(); savefig("p4_confusion_random"); plt.show()
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, name in zip(axes, ["LR", "DT"]):
    acc = accuracy_score(y[grp_te], preds_p4[("grouped", name, "test")])
    plot_cm(ax, y[grp_te], preds_p4[("grouped", name, "test")], f"{name} | 15 unseen stations | acc {acc:.3f}")
plt.suptitle("Confusion matrices on the station-grouped test set", fontsize=13)
plt.tight_layout(); savefig("p4_confusion_grouped"); plt.show()
""")

md(r"""
### 4.5 Visualising the decision boundary

The models live in 20 dimensions, so we draw the boundary of the same model classes trained on **two features
only**. We picked the pair with the best two-feature LR test macro-F1 among the interpretable candidates:
`MinTemp` (night temperature) and `Humidity3pm` (afternoon humidity). The decision tree is shown twice: with
default settings, and with `max_depth = 6` to show what limiting complexity does. Points are 1,500 random test days.
""")
code(r"""
pair = ["MinTemp", "Humidity3pm"]
X2_tr, X2_te = F.iloc[rand_tr][pair], F.iloc[rand_te][pair]
boundary_models = {"Logistic Regression": make_model("LR"), "Decision Tree (default)": make_model("DT"),
                   "Decision Tree (max_depth=6)": make_model("DT", max_depth=6)}
xx, yy = np.meshgrid(np.linspace(-6, 34, 400), np.linspace(0, 100, 400))
grid = pd.DataFrame({pair[0]: xx.ravel(), pair[1]: yy.ravel()})
zone_idx = {z: i for i, z in enumerate(ZONE_ORDER)}
from matplotlib.colors import ListedColormap
cmap5 = ListedColormap([ZONE_COLORS[z] for z in ZONE_ORDER])
pts = np.random.default_rng(RANDOM_STATE).choice(len(rand_te), 1500, replace=False)

fig, axes = plt.subplots(1, 3, figsize=(21, 6.3))
for ax, (title, m) in zip(axes, boundary_models.items()):
    m.fit(X2_tr, y[rand_tr])
    Z = np.vectorize(zone_idx.get)(m.predict(grid)).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=np.arange(-0.5, 5.5), cmap=cmap5, alpha=0.30)
    ax.scatter(X2_te.iloc[pts][pair[0]], X2_te.iloc[pts][pair[1]], s=6, linewidths=0,
               c=[ZONE_COLORS[z] for z in y[rand_te][pts]])
    f1 = f1_score(y[rand_te], m.predict(X2_te), average="macro")
    ax.set(title=f"{title}\n2-feature test macro-F1 = {f1:.3f}", xlabel="MinTemp (°C)", ylabel="Humidity3pm (%)",
           xlim=(-6, 34), ylim=(0, 100))
handles = [plt.Line2D([], [], marker="s", ls="", color=ZONE_COLORS[z], label=z) for z in ZONE_ORDER]
axes[0].legend(handles=handles, loc="upper left", fontsize=9)
plt.tight_layout(); savefig("p4_decision_boundary"); plt.show()
""")
md(r"""
* **LR** draws straight boundaries: warm nights (MinTemp > ~22 °C) → Tropical, low afternoon humidity → Arid,
  cold nights → Oceanic, and a wide middle band → Subtropical. **No region is assigned to Mediterranean at all**:
  with only these two features, Mediterranean days are never the most probable class anywhere.
* **The default tree** cuts the plane into thousands of tiny axis-aligned rectangles; many of them contain a single
  training day, which is overfitting made visible.
* **The depth-6 tree** keeps the large structures (the tropical corner, the dry strip) with at most 64 leaves,
  carves out a Mediterranean box (mild nights, humid afternoons) that LR cannot represent, and has the highest
  test score of the three (0.474 vs 0.412 for the default tree and 0.390 for LR).

### 4.6 Interpretation of the results on the training, test and entire sets

*All numbers refer to the tables above.*

**Logistic regression.** Training, test and entire-set scores are almost the same (accuracy ≈ 0.60 on the
random split). A linear model with 105 parameters cannot memorise 100,000 days: its errors come from **bias**
(the zones are not linearly separable), not from variance.

**Decision tree.** It is perfect on the training set (accuracy 1.000) but only ≈ 0.65 on the random test set:
a classic case of **overfitting**. Even so, its test score is higher than LR's, because the tree can model
non-linear interactions.

**The entire set is not an independent estimate.** It is 70% training rows and 30% test rows, so
`entire ≈ 0.7 · train + 0.3 · test`. For LR this changes nothing; for the tree it produces a misleading 0.895.
Only the test column measures generalisation.

**Unseen stations.** On the grouped split both models lose accuracy, but the tree loses far more (test accuracy
falls from 0.65 to 0.53) and drops *below* LR (0.56). Part of what the tree learned on the random split was
**station-specific** (it recognised the typical days of each station), and that knowledge does not transfer to new
stations. LR, which cannot memorise, is the more robust model here.

**Which zones are confused?** Tropical is recognised best in every test matrix (recall 0.77–0.92); its errors
go almost only to Subtropical (the warm, humid Queensland coast). The largest confusion is between Mediterranean
and Oceanic (up to 41% of Mediterranean test days on unseen stations are called Oceanic), as expected from the
t-SNE map (Part 2) and the clusters (Part 3). Arid has the lowest recall (0.34–0.50); its days are most often called
Subtropical, because the inland Subtropical stations (Moree, Wagga Wagga) have hot, dry summers too.

### 4.7 Comparison of the two models

| | Logistic Regression | Decision Tree (default) |
|---|---|---|
| Random test accuracy / macro-F1 | ≈ 0.60 / 0.62 | ≈ 0.65 / 0.67 |
| Unseen-station test accuracy / macro-F1 | ≈ 0.56 / 0.56 | ≈ 0.53 / 0.54 |
| Train − test gap | ≈ 0 (high bias, low variance) | ≈ 0.35 (low bias, high variance) |
| Boundary shape | linear | axis-aligned, very fragmented |

Neither model is good enough yet. Part 5 computes the full metrics, draws the ROC curves and improves both models
with validation.
""")
