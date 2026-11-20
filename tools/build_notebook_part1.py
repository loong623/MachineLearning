"""Self-contained notebook cells; assemble with tools/build_notebook.py."""
import nbformat as nbf
cells=[]
md=lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code=lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# DSAA2011 Course Project (Task A) — *"From Weather Snapshots to Seasonal Climate Signatures"*
## Recovering Australia's climate zones from weather observations alone

**Question.** How much of an externally assigned climate zone can be recovered from weather when a single day is
treated as a baseline and four seasonal windows are combined into an annual weather-pattern proxy?

| Step | Question it answers |
|---|---|
| Parts 1–5 Mandatory tasks | Preprocess, visualise, cluster and predict with the required daily-weather baseline. |
| Part 6 History sensitivity | How does the answer change with trailing history, feature engineering and model flexibility? |
| Part 7 Main exploration | Do four seasonal weather windows recover climate-zone information more faithfully? |

**Target (`Zone`, 5 classes)** is derived from the Köppen-Geiger classification of each station
(groups A and B kept whole, group C split into Cfa / Cs / Cfb). See Section 0.2.
""")

md(r"""
## 0. Setup
""")

code(r"""
import hashlib
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42                      # one seed for every split / sample / model in the notebook
DATA_DIR = Path("data")
FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
RAW_CSV = DATA_DIR / "raw" / "weatherAUS.csv"   # Kaggle: jsphyg/weather-dataset-rattle-package, version 2
RAW_SHA256 = "573fd715cd69fcacc4df32024d823b450ae3edaae7e8ff2eeb623adbed424014"

ZONE_ORDER = ["Tropical", "Arid", "Subtropical", "Mediterranean", "Oceanic"]
ZONE_COLORS = {"Tropical": "#d62728", "Arid": "#e6a01e", "Subtropical": "#2ca02c",
               "Mediterranean": "#9467bd", "Oceanic": "#1f77b4"}
sns.set_theme(style="whitegrid", context="notebook")

def savefig(name):
    # every figure is also written to figures/ so the report and slides use the same image
    plt.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")

# Cache identity includes executable notebook code, data/annotation and environment.
import pickle, json, os, platform, importlib.metadata, time
from threadpoolctl import threadpool_limits
_thread_limit = threadpool_limits(limits=2)
nb_file = Path("project_7_TaskA.ipynb")
nb_source = json.loads(nb_file.read_text(encoding="utf-8"))
code_digest = hashlib.sha256("\n".join("".join(c["source"]) for c in nb_source["cells"] if c["cell_type"] == "code").encode()).hexdigest()
file_digests = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW_CSV, DATA_DIR / "processed/location_zone.csv", DATA_DIR / "external/location_station.csv"]}
environment = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
identity = {"schema": "review-v2", "code_sha256": code_digest, "files": file_digests, "environment": environment, "python": platform.python_version()}
RUN_ID = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
CACHE_DIR = DATA_DIR / "processed/audit_cache" / RUN_ID
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = Path("results"); RESULT_DIR.mkdir(exist_ok=True)
RECOMPUTE = os.environ.get("ML_RECOMPUTE", "0") == "1"
CACHE_EVENTS = []
def cached(name, compute, **params):
    key = hashlib.sha256(repr(sorted(params.items())).encode()).hexdigest()[:12]
    path = CACHE_DIR / f"{name}_{key}.pkl"
    hit = path.exists() and not RECOMPUTE
    t0 = time.time()
    if hit:
        with path.open("rb") as stream: result = pickle.load(stream)
    else:
        result = compute()
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as stream: pickle.dump(result, stream)
        temp.replace(path)
    CACHE_EVENTS.append({"name": name, "hit": hit, "seconds": time.time()-t0})
    print(f"cache {'hit' if hit else 'computed'}: {name} ({time.time()-t0:.1f}s)", flush=True)
    return result
def export_table(name, table):
    table.to_csv(RESULT_DIR / f"{name}.csv", index=True)
    return table
(RESULT_DIR / "environment.json").write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
print("Experiment fingerprint:", RUN_ID)
print("numpy", np.__version__, "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__)
""")

md(r"""
### 0.2 Building the target: station → Köppen code → `Zone`

The dataset has no climate label, so we construct one from two documented sources:

1. **Station coordinates** — Bureau of Meteorology station list (product IDCJMC0014, downloaded 2026-11-09,
   stored in `data/external/`). Each `Location` was matched by hand to a BoM site number
   (`data/external/location_station.csv`).
2. **Köppen-Geiger code** — looked up with `kgcpy` 1.1.8, the Python port of the CRAN package `kgc`
   (Bryant, Wheeler, Rubel & French, 2017). Both packages ship a global 12960 × 6480 raster, i.e. 100 arc-seconds
   (≈ 3 km) per pixel, and cite Rubel et al. (2017, *Meteorol. Z.* 26:115–125, doi:10.1127/metz/2016/0816) as the
   data source. That paper's title refers to the European Alps, but it also describes the downscaling method that
   the authors used for their re-analysed **global** map, which according to the map's website
   (koeppen-geiger.vu-wien.ac.at) represents the period **1986–2010**. The lookup takes the pixel whose centre is
   nearest to the station coordinates.

The five classes are **project-defined groups of Köppen codes** (we call them "zones" for short). Groups A and B
are kept whole; group C is split by Köppen's own second letter (f = no dry season, s = dry summer) and third letter
(a = hot summer, b = warm summer). Only Am, Aw, BWh, BSh, BSk, Cfa, Csa, Csb and Cfb occur at our stations; there
are no D (snow) or E (polar) climates in the data. The code below defines the complete mapping, including additional codes not observed at these sites.

| Zone | Köppen | Meaning |
|---|---|---|
| Tropical | A (Am, Aw) | hot all year, summer wet season |
| Arid | B (BWh, BSh, BSk) | desert / steppe |
| Subtropical | Cfa | hot summer, rain all year |
| Mediterranean | Csa, Csb | dry summer, wet winter |
| Oceanic | Cfb | mild summer, rain all year |

The table is cached in `data/processed/location_zone.csv`; set `REBUILD_LABELS = True` to regenerate it.
""")

code(r"""
KOPPEN_TO_ZONE = {
    "Af": "Tropical", "Am": "Tropical", "Aw": "Tropical", "As": "Tropical",
    "BWh": "Arid", "BWk": "Arid", "BSh": "Arid", "BSk": "Arid",
    "Cfa": "Subtropical", "Cwa": "Subtropical",
    "Csa": "Mediterranean", "Csb": "Mediterranean",
    "Cfb": "Oceanic", "Cfc": "Oceanic",
}
ZONE_CSV = DATA_DIR / "processed" / "location_zone.csv"
REBUILD_LABELS = False  # annotation is fixed; sensitivity is reported separately

def read_bom_stations(zip_path):
    # BoM station list is a fixed-width text file; the first 4 lines are a header
    text = zipfile.ZipFile(zip_path).read("stations.txt").decode("latin-1")
    rows = []
    for line in text.splitlines()[4:]:
        if len(line) >= 90 and line[:7].strip().isdigit():
            rows.append({"bom_site": line[0:7].strip(), "bom_name_list": line[14:55].strip(),
                         "lat": float(line[71:80]), "lon": float(line[80:90])})
    return pd.DataFrame(rows)

def build_location_zone():
    from kgcpy import lookupCZ, nearbyCZ          # only needed when rebuilding
    loc = pd.read_csv(DATA_DIR / "external" / "location_station.csv", dtype=str)
    st = read_bom_stations(DATA_DIR / "external" / "bom_stations_IDCJMC0014_2026-11-09.zip")
    z = loc.merge(st, on="bom_site", how="left", validate="one_to_one")
    assert z["lat"].notna().all() and (z["bom_name"] == z["bom_name_list"]).all()
    z["koppen"] = [lookupCZ(a, b) for a, b in zip(z["lat"], z["lon"])]
    near = [nearbyCZ(a, b, 1) for a, b in zip(z["lat"], z["lon"])]     # 3x3 pixel window
    z["neighbour_match"] = [round(n[1], 2) for n in near]
    z["neighbour_codes"] = [";".join(sorted(set(n[2]))) for n in near]
    z["Zone"] = z["koppen"].map(KOPPEN_TO_ZONE)                          # "Ocean" -> NaN (no label)
    # boundary station = a neighbouring LAND pixel belongs to a different zone
    z["zone_boundary"] = [any(KOPPEN_TO_ZONE.get(c, c) != zz for c in codes.split(";") if c and c != "Ocean")
                          if isinstance(zz, str) else True
                          for codes, zz in zip(z["neighbour_codes"], z["Zone"])]
    z = z[["Location", "bom_site", "bom_name", "lat", "lon", "koppen",
           "neighbour_match", "neighbour_codes", "zone_boundary", "Zone"]]
    ZONE_CSV.parent.mkdir(parents=True, exist_ok=True)
    z.to_csv(ZONE_CSV, index=False)
    return z

zones = build_location_zone() if (REBUILD_LABELS or not ZONE_CSV.exists()) else pd.read_csv(ZONE_CSV, dtype={"bom_site": str})
for zone, s in zones.groupby(zones["Zone"].fillna("(no label)"))["Location"]:
    print(f"{zone:14s} {len(s):2d}: " + ", ".join(s))
print("\nboundary / unlabeled stations:")
zones[zones["zone_boundary"] | zones["Zone"].isna()][["Location", "koppen", "neighbour_codes", "Zone"]]
""")

md(r"""
**Full audit table** (all 49 stations): dataset `Location` → BoM site number and name → coordinates → Köppen code
at that pixel → codes found in the surrounding 3 × 3 pixels → project group. Anyone can check a row by looking up
the site number in the BoM list and the coordinates on the Köppen map.
""")

code(r"""
with pd.option_context("display.max_rows", 60, "display.width", 200):
    display(zones[["Location", "bom_site", "bom_name", "lat", "lon", "koppen", "neighbour_codes", "zone_boundary", "Zone"]]
            .sort_values(["Zone", "Location"], na_position="last").reset_index(drop=True))
""")

md(r"""
**Norfolk Island.** Its station (BoM 200288, Norfolk Island Aero) gets the code "Ocean". To rule out a coordinate or
rounding error we search outward from its pixel for the a nearby *land* pixel on the first occupied square ring of the raster. This needs `kgcpy`; the
result is cached so the notebook runs without it.
""")

code(r"""
def norfolk_check():
    import kgcpy
    img, codes = kgcpy.img, kgcpy.kg_zoneNum_df
    ocean = set(codes.loc[codes["kg_zone"] == "Ocean", "zoneNum"])
    lat, lon = zones.set_index("Location").loc["NorfolkIsland", ["lat", "lon"]]
    x0 = round((lon + 180) * img.size[0] / 360 - 0.5); y0 = round(-(lat - 90) * img.size[1] / 180 - 0.5)
    best = None
    for r in range(1, 400):                      # square rings of growing radius (in pixels)
        ring = [(dx, dy) for dx in range(-r, r + 1) for dy in (-r, r)] + [(dx, dy) for dx in (-r, r) for dy in range(-r + 1, r)]
        land = [(np.hypot(dx, dy), dx, dy) for dx, dy in ring if img.getpixel((x0 + dx, y0 + dy)) not in ocean]
        if land:
            d, dx, dy = min(land)
            v = img.getpixel((x0 + dx, y0 + dy))
            lat2 = 90 - (y0 + dy + 0.5) * 180 / img.size[1]; lon2 = (x0 + dx + 0.5) * 360 / img.size[0] - 180
            p1, p2, dl = np.deg2rad(lat), np.deg2rad(lat2), np.deg2rad(lon2 - lon)      # haversine distance
            km = 2 * 6371 * np.arcsin(np.sqrt(np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))
            best = {"pixel size (arcsec)": 360 * 3600 / img.size[0], "code at station pixel": "Ocean",
                    "first ring with land (pixels)": r, "nearest land pixel lat": round(lat2, 2),
                    "nearest land pixel lon": round(lon2, 2),
                    "code there": codes.loc[codes["zoneNum"] == v, "kg_zone"].item(), "distance (km)": round(km)}
            break
    return best

pd.Series(cached("p0_norfolk_check", norfolk_check, version=1))
""")

md(r"""
The island is simply not on the raster (it is about 8 km across and the map's land mask omits it): the first land
pixel appears 185 pixels away, at the northern tip of New Zealand, 736 km from the station. The "Ocean" code is
therefore not a coordinate or rounding error. Rather than assign a label by hand, we drop the station (3,009 rows, 2.1% of the
data). Because it had no label, dropping it does not change the class proportions of the remaining data.
""")

code(r"""
# Where are the stations? (plain lon/lat scatter, no basemap needed)
z = zones.dropna(subset=["Zone"])
fig, ax = plt.subplots(figsize=(8, 6.5))
for zone in ZONE_ORDER:
    s = z[z["Zone"] == zone]
    ax.scatter(s["lon"], s["lat"], s=60, color=ZONE_COLORS[zone], edgecolor="k", label=f"{zone} ({len(s)})")
for r in z.itertuples():
    ax.annotate(r.Location, (r.lon, r.lat), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
ax.set(xlabel="Longitude", ylabel="Latitude", title="48 labelled stations coloured by Köppen-derived zone")
ax.set_aspect("equal"); ax.legend(loc="lower left")
savefig("p1_station_map"); plt.show()
""")

md(r"""
## 1. Data Preprocessing

### 1.1 Loading and leakage audit

For this target, three columns must never be features:

* `Location` — the zone is a deterministic function of the station (this would be the answer itself);
* `RainTomorrow` — information from the next day;
* `RISK_MM` — next-day rainfall in older versions of the dataset (we check rather than assume it is absent).

`Location` is kept **outside** the feature table as `groups`; it is used for splits, within-station trailing windows and error analysis, never as a predictor.
`Date` is only used to derive the day of year; the year itself is not used.
""")

code(r"""
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

assert sha256(RAW_CSV) == RAW_SHA256, "data file differs from the documented Kaggle v2 file"
raw = pd.read_csv(RAW_CSV, parse_dates=["Date"])
print("raw shape:", raw.shape, "| dates:", raw["Date"].min().date(), "->", raw["Date"].max().date(),
      "| stations:", raw["Location"].nunique())

LEAKAGE_DROP = ["Location", "RainTomorrow", "RISK_MM"]
print("leakage columns present:", [c for c in LEAKAGE_DROP if c in raw.columns],
      "| absent:", [c for c in LEAKAGE_DROP if c not in raw.columns])

df = raw.merge(zones[["Location", "Zone"]], on="Location", how="left", validate="many_to_one")
print("rows without a zone label (dropped):", df["Zone"].isna().sum(), sorted(df.loc[df["Zone"].isna(), "Location"].unique()))
df = df[df["Zone"].notna()].reset_index(drop=True)

groups = df["Location"].copy()                     # station id, NEVER a model input
df = df.drop(columns=[c for c in LEAKAGE_DROP if c in df.columns])
y = df["Zone"].values
print("final shape:", df.shape, "| duplicated (Date, station):", pd.DataFrame({"d": df["Date"], "g": groups}).duplicated().sum())
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
cnt = df["Zone"].value_counts().reindex(ZONE_ORDER)
axes[0].bar(cnt.index, cnt.values, color=[ZONE_COLORS[z] for z in cnt.index])
for i, v in enumerate(cnt.values):
    axes[0].text(i, v, f"{v:,}\n({v / cnt.sum():.1%})", ha="center", va="bottom", fontsize=9)
axes[0].set(title="Rows per zone (class balance)", ylabel="rows"); axes[0].set_ylim(0, cnt.max() * 1.25)

per_station = groups.value_counts()
axes[1].hist(per_station.values, bins=20, color="grey")
axes[1].set(title="Rows per station", xlabel="days recorded", ylabel="stations")
plt.tight_layout(); savefig("p1_class_balance"); plt.show()
print("majority-class baseline accuracy:", round(cnt.max() / cnt.sum(), 3))
""")

md(r"""
### 1.2 Missing values: the structure matters more than the rate

We first look at *where* values are missing, not just how many.
""")

code(r"""
RAW_FEATURES = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustDir", "WindGustSpeed",
                "WindDir9am", "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
                "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"]
miss_station = df[RAW_FEATURES].isna().groupby(groups).mean()          # station x column missing rate
summary = pd.DataFrame({
    "missing_%": (df[RAW_FEATURES].isna().mean() * 100).round(1),
    "stations_fully_missing(>95%)": (miss_station > 0.95).sum(),
}).sort_values("missing_%", ascending=False)
summary
""")

code(r"""
# station x column heatmap, stations sorted by zone; tick labels coloured by zone
station_zone = zones.set_index("Location")["Zone"]
order = sorted(miss_station.index, key=lambda s: (ZONE_ORDER.index(station_zone[s]), s))
cols = summary.index.tolist()
fig, ax = plt.subplots(figsize=(12, 11))
sns.heatmap(miss_station.loc[order, cols], cmap="Reds", vmin=0, vmax=1, cbar_kws={"label": "missing rate"}, ax=ax)
for t in ax.get_yticklabels():
    t.set_color(ZONE_COLORS[station_zone[t.get_text()]])
ax.set(title="Missing rate per station and column (station names coloured by zone)", xlabel="", ylabel="")
savefig("p1_missing_heatmap"); plt.show()
""")

md(r"""
**Observation.** Missingness is a *station property*: Sunshine, Evaporation and Cloud are absent for entire stations
(12–19 of 48) possibly reflecting instrument availability or reporting practices. This creates a hidden risk for our target:
the *pattern* of missing values may identify the station, and the station determines the zone.
We test this directly: train a decision tree on **only the 20 is-missing flags** (no weather values at all).
""")

code(r"""
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
X_missing = df[RAW_FEATURES].isna().astype(int)
def fingerprint_test():
    out = {}
    for name, target in [("Zone", df["Zone"]), ("Location", groups)]:
        tree = cross_val_score(DecisionTreeClassifier(random_state=RANDOM_STATE), X_missing, target, cv=cv).mean()
        dummy = cross_val_score(DummyClassifier(strategy="most_frequent"), X_missing, target, cv=cv).mean()
        out[name] = {"tree on missing flags": round(tree, 3), "majority baseline": round(dummy, 3)}
    year_acc = cross_val_score(DecisionTreeClassifier(random_state=RANDOM_STATE),
                               df[["Date"]].apply(lambda d: d.dt.year), df["Zone"], cv=cv).mean()
    out["Zone (from year only)"] = {"tree on missing flags": round(year_acc, 3),
                                    "majority baseline": out["Zone"]["majority baseline"]}
    return out

fingerprint = cached("p1_fingerprint", fingerprint_test, seed=RANDOM_STATE, cols=tuple(RAW_FEATURES))
pd.DataFrame(fingerprint).T
""")

md(r"""
Missingness-only prediction is an exploratory diagnostic of station-specific information. It does not establish that all such information is non-climatic or harmful under every deployment target. The core feature policy is retained from the earlier exploratory work; its sensitivity is re-evaluated inside the outer-fold analysis.
""")

code(r"""
for d, s in [("WindDir9am", "WindSpeed9am"), ("WindDir3pm", "WindSpeed3pm")]:
    calm = df[s] == 0
    print(f"{d}: {df[d].isna().sum():6d} missing | rows with {s}==0: {calm.sum():5d}, "
          f"of which direction missing: {df.loc[calm, d].isna().mean():.0%}")
r = df.dropna(subset=["Rainfall", "RainToday"])
print("RainToday == (Rainfall > 1 mm) for", f"{((r['Rainfall'] > 1) == (r['RainToday'] == 'Yes')).mean():.1%}", "of rows")
""")

md(r"""
* **Calm wind.** Every row with wind speed 0 has no direction: the direction is *undefined*, not unknown.
  Encoding wind as a vector `(u, v)` handles this exactly — calm is `(0, 0)`.
* **RainToday** is a deterministic function of `Rainfall` (threshold 1 mm). After encoding it as 0/1 it adds no
  information, so we drop it and keep `Rainfall`.

### 1.4 Handling non-numeric values and further processing

| Raw column(s) | Treatment | Reason |
|---|---|---|
| `WindGustDir`, `WindDir9am`, `WindDir3pm` + speeds | 16-point compass → angle θ → vector `u = −s·sinθ`, `v = −s·cosθ` | keeps the circular order (N is next to NNW), merges direction and strength, calm = (0,0); 6 numbers instead of 48 one-hot columns |
| `Date` | day of year → `sin`, `cos` | 31 Dec and 1 Jan are neighbours; year not used |
| `RainToday` | Yes/No → 1/0, then dropped | identical to `Rainfall > 1` |
| `Rainfall`, `Evaporation` | `log1p` | strongly right-skewed, 63% zeros |
| all numeric | median imputation → `StandardScaler` | fitted **inside the pipeline on training data only** |
| Sunshine, Evaporation, Cloud9am/3pm | excluded from the default (`core`) feature set | missing for whole stations → station fingerprint; kept in a `full` set for a sensitivity check |

All transformations in `make_features` are row-wise (no statistics are learned), so applying them before the split is safe.
""")

code(r"""
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
DIR_TO_DEG = {d: i * 22.5 for i, d in enumerate(COMPASS)}
WIND_PAIRS = {"Gust": ("WindGustDir", "WindGustSpeed"),
              "Wind9am": ("WindDir9am", "WindSpeed9am"),
              "Wind3pm": ("WindDir3pm", "WindSpeed3pm")}

def wind_vector(direction, speed):
    theta = np.deg2rad(direction.map(DIR_TO_DEG))
    u, v = -speed * np.sin(theta), -speed * np.cos(theta)
    calm = speed.eq(0)                       # calm: direction undefined, vector is exactly zero
    u[calm], v[calm] = 0.0, 0.0
    return u, v

def make_features(data):
    out = pd.DataFrame(index=data.index)
    for c in ["MinTemp", "MaxTemp", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
              "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm", "Sunshine", "Cloud9am", "Cloud3pm"]:
        out[c] = data[c]
    out["Rainfall_log1p"] = np.log1p(data["Rainfall"])
    out["Evaporation_log1p"] = np.log1p(data["Evaporation"])
    for prefix, (dcol, scol) in WIND_PAIRS.items():
        out[f"{prefix}_u"], out[f"{prefix}_v"] = wind_vector(data[dcol], data[scol])
    doy = data["Date"].dt.dayofyear
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return out

CORE_NUMERIC = ["MinTemp", "MaxTemp", "Rainfall_log1p", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
                "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"]
WIND_VECTOR = [f"{p}_{c}" for p in WIND_PAIRS for c in ("u", "v")]
SEASON = ["doy_sin", "doy_cos"]
STATION_SENSOR = ["Sunshine", "Evaporation_log1p", "Cloud9am", "Cloud3pm"]
FEATURE_SETS = {"core": CORE_NUMERIC + WIND_VECTOR + SEASON,
                "full": CORE_NUMERIC + WIND_VECTOR + SEASON + STATION_SENSOR}

def build_preprocessor(add_indicator=False):
    # learned statistics (medians, means, stds) -> always inside a Pipeline, fitted on training data only
    return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
                     ("scale", StandardScaler())])

F = make_features(df)
X_core = F[FEATURE_SETS["core"]]
print("features:", F.shape[1], "| core set:", X_core.shape[1], "columns")
print("remaining missing rate in core set (top 5):")
print((X_core.isna().mean() * 100).round(1).sort_values(ascending=False).head(5).to_string())
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].hist(df["Rainfall"].dropna(), bins=80, color="steelblue"); axes[0].set_yscale("log")
axes[0].set(title=f"Rainfall (skew = {df['Rainfall'].skew():.1f})", xlabel="mm", ylabel="rows (log scale)")
axes[1].hist(F["Rainfall_log1p"].dropna(), bins=80, color="steelblue"); axes[1].set_yscale("log")
axes[1].set(title=f"log1p(Rainfall) (skew = {F['Rainfall_log1p'].skew():.1f})", xlabel="log(1 + mm)")
plt.tight_layout(); savefig("p1_rainfall_log"); plt.show()
""")

md(r"""
### 1.5 Patterns in the data: what distinguishes the zones?

The **seasonal cycle** is the textbook definition of the Köppen classes, so we expect it to be visible in the data.
""")

code(r"""
monthly = df.assign(month=df["Date"].dt.month).groupby(["Zone", "month"])[
    ["MaxTemp", "Humidity3pm", "Rainfall", "Pressure3pm"]].mean()
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
titles = {"MaxTemp": "Max temperature (°C)", "Humidity3pm": "Humidity at 3pm (%)",
          "Rainfall": "Mean daily rainfall (mm)", "Pressure3pm": "Pressure at 3pm (hPa)"}
for ax, col in zip(axes.ravel(), titles):
    for zone in ZONE_ORDER:
        ax.plot(range(1, 13), monthly.loc[zone, col], marker="o", color=ZONE_COLORS[zone], label=zone)
    ax.set(title=titles[col]); ax.set_xticks(range(1, 13))
for ax in axes[1]:
    ax.set_xlabel("month (southern hemisphere: Dec–Feb = summer)")
axes[0, 0].legend(ncol=2, fontsize=9)
plt.suptitle("Seasonal cycle by zone"); plt.tight_layout(); savefig("p1_seasonal_cycle"); plt.show()
""")

code(r"""
show = ["MaxTemp", "MinTemp", "Humidity3pm", "Pressure3pm", "WindGustSpeed", "Rainfall_log1p"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), show):
    sns.boxplot(x=df["Zone"], y=F[col], order=ZONE_ORDER, hue=df["Zone"], hue_order=ZONE_ORDER,
                palette=ZONE_COLORS, showfliers=False, legend=False, ax=ax)
    ax.set(title=col, xlabel="", ylabel="")
    ax.tick_params(axis="x", rotation=30)
fig.suptitle("Feature distributions by zone (outliers hidden)")
plt.tight_layout(); savefig("p1_boxplots"); plt.show()
""")

code(r"""
corr = X_core.corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True, ax=ax, cbar_kws={"shrink": .7})
ax.set_title("Correlation of the 20 core features"); savefig("p1_correlation"); plt.show()
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().sort_values(key=abs, ascending=False)
print("most correlated pairs:\n", pairs.head(6).round(3).to_string())
""")

md(r"""
### 1.6 Sensitivity check: does the feature-set / indicator choice create a station shortcut?

A quick logistic-regression probe (full evaluation follows in Parts 4–5). Two splits answer two different questions:

* **random stratified 70/30** — can the model recognise a *new day* of stations it has seen?
* **station-grouped 70/30** — can it recognise the zone of *stations it has never seen*?

A protocol-dependent gain is consistent with station-specific information, but is not by itself proof of its cause.
""")

code(r"""
idx = np.arange(len(df))
rand_tr, rand_te = train_test_split(idx, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
grp_tr, grp_te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE).split(idx, y, groups))
print("grouped split: test stations =", groups.iloc[grp_te].nunique(), "| overlap with train stations =",
      len(set(groups.iloc[grp_te]) & set(groups.iloc[grp_tr])))

def sensitivity_probe():
    rows = []
    for fs, ind in [("core", False), ("full", False), ("full", True)]:
        X = F[FEATURE_SETS[fs]]
        for split, (a, b) in {"random": (rand_tr, rand_te), "grouped": (grp_tr, grp_te)}.items():
            model = make_pipeline(build_preprocessor(add_indicator=ind), LogisticRegression(max_iter=3000))
            model.fit(X.iloc[a], y[a]); pred = model.predict(X.iloc[b])
            rows.append({"features": fs, "missing indicator": ind, "split": split,
                         "accuracy": accuracy_score(y[b], pred), "macro-F1": f1_score(y[b], pred, average="macro")})
    return rows

rows = cached("p1_sensitivity", sensitivity_probe, seed=RANDOM_STATE, sets=repr(FEATURE_SETS))
sens = pd.DataFrame(rows).pivot_table(index=["features", "missing indicator"], columns="split",
                                      values=["accuracy", "macro-F1"]).round(3)
sens
""")

md(r"""
This is an exploratory preprocessing comparison, not an independent assessment of a selected policy. A difference between protocols suggests sensitivity to station composition; it does not causally separate instrumentation from climate. Part 6 repeats the comparison with fixed policies and outer station folds. The core policy is not reselected from test scores.
""")
