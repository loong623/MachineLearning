"""Create a compact, reproducible Group 7 presentation PDF from final figures."""
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "presentation" / "presentation_7_TaskA.pdf"
W, H = 13.333, 7.5
NAVY, TEAL, GOLD, RED, MUTED = "#123A40", "#0F7C80", "#E3A02F", "#D1495B", "#53666B"


def slide(pdf, title, subtitle=None, number=None):
    fig = plt.figure(figsize=(W, H), facecolor="white")
    fig.text(.06, .90, title, fontsize=26, fontweight="bold", color=NAVY, va="top")
    if subtitle:
        fig.text(.06, .85, subtitle, fontsize=13, color=MUTED, va="top")
    if number is not None:
        fig.text(.95, .035, str(number), fontsize=10, color=MUTED, ha="right")
    return fig


def bullets(fig, items, x=.08, y=.73, width=.42, size=17, color="#1F2A30", gap=.07):
    for item in items:
        wrapped = textwrap.fill(item, width=58)
        fig.text(x, y, "• " + wrapped, fontsize=size, color=color, va="top", linespacing=1.25)
        y -= gap + .045 * (wrapped.count("\n") + 1)


def image(fig, name, rect):
    ax = fig.add_axes(rect)
    ax.imshow(imread(FIG / name))
    ax.axis("off")
    return ax


def stat(fig, x, y, value, label, color=TEAL):
    fig.text(x, y, value, fontsize=31, fontweight="bold", color=color, va="top")
    fig.text(x, y - .07, label, fontsize=12, color=MUTED, va="top")


with PdfPages(OUT) as pdf:
    fig = slide(pdf, "From weather snapshots to seasonal climate signatures", "DSAA2011 Task A · Group 7 · MO, Zijun · QIAO, Xiaolong · WANG, Zhifei · XIA, Youyue", 1)
    fig.text(.06, .63, "How much climate-zone information can weather provide?", fontsize=25, color=TEAL, fontweight="bold")
    bullets(fig, ["A single day is the required baseline and first impression.", "Four seasonal windows form the main station-year weather signature.", "Coordinates, station IDs and target labels are withheld from predictors."], y=.49, width=.8, size=19, gap=.06)
    fig.text(.06, .08, "Rain in Australia · five external climate groups · station-held-out evaluation", fontsize=12, color=MUTED)
    pdf.savefig(fig); plt.close(fig)

    fig = slide(pdf, "Data and target construction", "External Köppen-Geiger annotation is used only as the prediction target.", 2)
    stat(fig, .08, .70, "142,451", "daily rows after filtering")
    stat(fig, .32, .70, "48", "labelled stations")
    stat(fig, .52, .70, "5", "project climate groups", GOLD)
    stat(fig, .72, .70, "4", "Tropical stations", RED)
    image(fig, "p1_station_map.png", [.57, .18, .36, .48])
    bullets(fig, ["Tropical, Arid, Subtropical, Mediterranean, Oceanic.", "Norfolk Island is excluded because it is not assigned to the five-group map.", "The Mediterranean class is spatially discontinuous; geography remains a prior, not a predictor."], x=.08, y=.43, width=.43, size=16, gap=.05)
    pdf.savefig(fig); plt.close(fig)

    fig = slide(pdf, "Preprocessing and leakage control", "The label belongs to the station, so station fingerprints must be treated carefully.", 3)
    bullets(fig, ["Remove Location, RainTomorrow and unavailable future information.", "Wind direction + speed → circular vector components; rainfall → log(1+x).", "Day of year → sine/cosine; year is excluded.", "Median imputation and scaling are learned inside each training fold.", "Daily core: 20 weather-derived features; no coordinates or station ID."], x=.08, y=.75, width=.48, size=17, gap=.045)
    image(fig, "p1_missing_heatmap.png", [.60, .18, .33, .52])
    fig.text(.60, .12, "Missingness diagnostics motivate the core policy.", fontsize=12, color=MUTED)
    pdf.savefig(fig); plt.close(fig)

    fig = slide(pdf, "Daily t-SNE: a useful baseline, not climate itself", "Balanced sample: 1,200 days per class; labels do not optimize the embedding.", 4)
    image(fig, "p2_tsne_main.png", [.06, .20, .58, .62])
    stat(fig, .73, .66, "0.985", "t-SNE trustworthiness")
    stat(fig, .73, .47, "0.631", "diagnostic zone purity", GOLD)
    bullets(fig, ["Tropical days separate most clearly.", "Arid, Subtropical, Mediterranean and Oceanic days overlap.", "This motivates adding temporal coverage."], x=.70, y=.34, width=.25, size=15, gap=.04)
    pdf.savefig(fig); plt.close(fig)

    fig = slide(pdf, "Clustering: weather types versus seasonal signatures", "K-means and Ward are fitted without target labels.", 5)
    image(fig, "p7_seasonal_clustering.png", [.06, .22, .57, .58])
    stat(fig, .72, .66, "0.098", "daily K-means ARI", GOLD)
    stat(fig, .72, .47, "0.314", "best seasonal ARI", TEAL)
    bullets(fig, ["Daily clusters mainly describe weather types.", "Seasonal ARI improves at k=7, but remains exploratory.", "ARI is an external comparison, not a training objective."], x=.70, y=.30, width=.26, size=16, gap=.04)
    pdf.savefig(fig); plt.close(fig)

    fig = slide(pdf, "The honest evaluation exam", "Station-held-out outer folds are the primary evidence.", 6)
    bullets(fig, ["Random row split: new days at known stations; useful for the course illustration.", "Primary: four folds by station; no station appears in both train and test.", "Each fold contains all five classes.", "Inner tuning, window choice and feature selection use training stations only.", "Seeds 42, 0 and 7; headline metric is five-class macro-F1."], x=.08, y=.75, width=.78, size=20, gap=.055)
    fig.text(.08, .12, "This split asks whether weather transfers to an unseen station, not whether the model memorises station identity.", fontsize=14, color=TEAL, fontweight="bold")
    pdf.savefig(fig); plt.close(fig)

    fig = slide(pdf, "Required prediction and metrics", "Daily station-held-out baseline.", 7)
    image(fig, "p4_confusion_grouped.png", [.06, .20, .48, .58])
    stat(fig, .63, .68, "0.659", "daily HGB macro-F1", TEAL)
    stat(fig, .63, .49, "0.883", "daily HGB macro-AUC", GOLD)
    bullets(fig, ["LR grouped macro-F1: 0.556.", "Default tree shows a large random train/test gap.", "We report accuracy, precision, recall, F1, ROC/AUC and confusion matrices."], x=.63, y=.32, width=.30, size=16, gap=.04)
    pdf.savefig(fig); plt.close(fig)

    fig = slide(pdf, "Main feature engineering: four seasonal windows", "Four 21-day windows centred on equinoxes and solstices.", 8)
    image(fig, "p7_seasonal_feature_means.png", [.06, .20, .58, .62])
    stat(fig, .73, .68, "328", "complete station-year rows", TEAL)
    stat(fig, .73, .50, "73.6%", "mean complete coverage", GOLD)
    bullets(fig, ["Means + standard deviations.", "Rainy-day fraction + temperature-range summaries.", "At least seven valid days in every window.", "All 48 stations represented; station-year remains the unit."], x=.70, y=.33, width=.25, size=15, gap=.035)
    pdf.savefig(fig); plt.close(fig)

    fig = slide(pdf, "Seasonal t-SNE and model comparison", "The same station-held-out folds evaluate the main signature.", 9)
    image(fig, "p7_seasonal_model_comparison.png", [.06, .22, .56, .58])
    stat(fig, .72, .69, "0.861", "seasonal LR macro-F1", TEAL)
    stat(fig, .72, .51, "0.807", "seasonal HGB macro-F1", GOLD)
    stat(fig, .72, .33, "0.759", "four daily snapshots vote", "#7C5BA6")
    fig.text(.72, .17, "Seasonal LR macro-AUC: 0.973", fontsize=14, color=NAVY, fontweight="bold")
    pdf.savefig(fig); plt.close(fig)

    fig = slide(pdf, "Why the comparison is meaningful", "Representation and model flexibility are separated.", 10)
    image(fig, "p7_seasonal_tsne.png", [.06, .20, .50, .62])
    bullets(fig, ["A daily HGB prediction nearest each seasonal centre is voted across four snapshots.", "HGB: 0.759 to 0.807 (+0.047); LR: 0.702 to 0.861 (+0.159).", "Same-model comparisons support the value of seasonal summaries.", "Seasonal t-SNE is descriptive because years repeat within stations."], x=.62, y=.72, width=.32, size=18, gap=.055)
    pdf.savefig(fig); plt.close(fig)

    fig = slide(pdf, "Limits and conclusion", "Evidence is strong for the tested network and representation, not universal.", 11)
    bullets(fig, ["100-km exclusion buffer lowers daily HGB to 0.565.", "Only four Tropical stations; years repeat within stations.", "One manually matched raster provides the external target.", "Incomplete seasonal years are filtered at 73.6% mean coverage.", "Four dates are a transparent design choice, not a physical law."], x=.08, y=.75, width=.82, size=19, gap=.055)
    fig.text(.08, .14, "Takeaway: a day is a weather snapshot; four seasonal windows provide a stronger annual-pattern proxy.", fontsize=18, color=TEAL, fontweight="bold")
    pdf.savefig(fig); plt.close(fig)

    fig = slide(pdf, "Thank you", "Questions?", 12)
    fig.text(.08, .62, "Weather snapshots → seasonal climate signatures", fontsize=28, color=TEAL, fontweight="bold")
    fig.text(.08, .50, "Group 7 · MO, Zijun · QIAO, Xiaolong · WANG, Zhifei · XIA, Youyue", fontsize=16, color=NAVY)
    fig.text(.08, .38, "All code, figures, AI usage and the executed notebook are included in the project package.", fontsize=15, color=MUTED)
    pdf.savefig(fig); plt.close(fig)

print(f"wrote {OUT}")
