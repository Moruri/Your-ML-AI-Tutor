"""
Lesson 045 - Dimensionality reduction: PCA

Yesterday we grouped rides with k-means using six columns. Six is already
more than a screen can show at once: a scatter plot has two axes, and our
eyes give up somewhere around three. Principal Component Analysis (PCA)
finds a few new axes that carry most of the spread in the data, so we can
compress many columns into a handful and actually *look* at the result.

Same cleaned September bike-share rides as lessons 029 to 044, in
rides_clean.csv next to this file. We use seven numeric traits of each
ride and keep rider type aside, to colour the picture and to test whether
compression throws away anything a model needs.

Run it with (inside the venv from lesson 013, after `pip install -r
requirements.txt`):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes one picture, output/pca_rides.png, next to this file.
Every random step has a fixed seed, so your numbers will match the README.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"
OUTPUT = HERE / "output"

# Seven traits of a ride. Rider type stays out: we use it to colour the plot
# and as a label in section 5, never as an input to PCA.
FEATURES = ["distance_km", "duration_min", "temp_c", "rain_mm",
            "weekend", "hour", "electric"]

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        print(f"  {label}:")
        for line in value.to_string().splitlines():
            print(f"      {line}")
    else:
        print(f"  {label:<46} -> {value!r}")


def r3(x):
    return round(float(x), 3)


def load_rides(path):
    rides = pd.read_csv(path, parse_dates=["started_at"])
    rides["weekend"] = (rides["started_at"].dt.dayofweek >= 5).astype(int)
    rides["hour"] = rides["started_at"].dt.hour
    rides["electric"] = (rides["bike_type"] == "electric").astype(int)
    rides["casual"] = (rides["rider_type"] == "casual").astype(int)
    return rides


# ---------------------------------------------------------------------------
# 1. Seven columns, two eyes
# ---------------------------------------------------------------------------

def section_setup(rides):
    heading("1. Seven columns, two eyes")
    show("rides / features", (len(rides), len(FEATURES)))
    show("feature columns", FEATURES)
    corr = rides[FEATURES].corr().round(2)
    show("correlations between the features", corr)
    print("  distance and duration move together (about 0.9): two columns")
    print("  telling one story. PCA is built to notice exactly that kind of")
    print("  overlap and fold it into a single new axis.")


# ---------------------------------------------------------------------------
# 2. Scale, fit PCA, read explained variance
# ---------------------------------------------------------------------------

def section_explained(rides):
    heading("2. Scale, fit PCA, read explained variance")
    X = StandardScaler().fit_transform(rides[FEATURES])
    pca = PCA().fit(X)
    ratio = pd.Series(pca.explained_variance_ratio_,
                      index=[f"PC{i + 1}" for i in range(len(FEATURES))])
    table = pd.DataFrame({"share": ratio, "cumulative": ratio.cumsum()})
    show("explained variance ratio by component", table)
    k80 = int(np.searchsorted(ratio.cumsum().values, 0.80) + 1)
    show("components needed for 80% of the spread", k80)

    unscaled = PCA().fit(rides[FEATURES])
    show("PC1 share WITHOUT scaling",
         r3(unscaled.explained_variance_ratio_[0]))
    print("  Without scaling, duration (minutes, std ~10) grabs most of the")
    print("  'variance' just because of its units. Same lesson as k-means:")
    print("  scale first, so every feature starts on the same ruler.")
    return X, pca


# ---------------------------------------------------------------------------
# 3. What the new axes mean
# ---------------------------------------------------------------------------

def section_loadings(pca):
    heading("3. What the new axes mean")
    loadings = pd.DataFrame(pca.components_[:3].T, index=FEATURES,
                            columns=["PC1", "PC2", "PC3"])
    show("loadings (how much each feature feeds each component)",
         loadings.round(2))
    for pc in loadings.columns:
        top = loadings[pc].abs().sort_values(ascending=False).index[:2].tolist()
        show(f"biggest ingredients of {pc}", top)
    print("  Each component is a weighted blend of the original features. The")
    print("  sign of a whole component is arbitrary (PCA could flip it), so")
    print("  read which features move together, not whether PC1 is 'positive'.")


# ---------------------------------------------------------------------------
# 4. Look at 1,176 rides in 2D
# ---------------------------------------------------------------------------

def section_plot(rides, X, pca):
    heading("4. Look at 1,176 rides in 2D")
    Z = pca.transform(X)[:, :2]
    casual = rides["casual"].to_numpy() == 1
    show("2D shape of the projected data", Z.shape)
    means = pd.DataFrame({"PC1": [Z[~casual, 0].mean(), Z[casual, 0].mean()],
                          "PC2": [Z[~casual, 1].mean(), Z[casual, 1].mean()]},
                         index=["member", "casual"])
    show("average position by rider type", means.round(2))

    OUTPUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(Z[~casual, 0], Z[~casual, 1], s=10, alpha=0.5, label="member")
    ax.scatter(Z[casual, 0], Z[casual, 1], s=10, alpha=0.5, label="casual")
    share = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({share[0]:.0%} of the spread)")
    ax.set_ylabel(f"PC2 ({share[1]:.0%} of the spread)")
    ax.set_title("September rides, seven features squeezed into two")
    ax.legend()
    fig.tight_layout()
    path = OUTPUT / "pca_rides.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    show("picture written to", str(path.relative_to(HERE)))
    print("  Rider type was never an input, yet casual riders drift to one")
    print("  side of PC1. Two axes cannot show everything, though: the")
    print("  colours overlap a lot, and the remaining components hold the rest.")


# ---------------------------------------------------------------------------
# 5. Compress, then model: what does it cost?
# ---------------------------------------------------------------------------

def section_compress(rides, X):
    heading("5. Compress, then model: what does it cost?")
    rows = {}
    for k in [1, 2, 3, 5, 7]:
        pca = PCA(n_components=k).fit(X)
        back = pca.inverse_transform(pca.transform(X))
        rows[f"{k} comp."] = {"kept": r3(pca.explained_variance_ratio_.sum()),
                              "recon_error": r3(((X - back) ** 2).mean())}
    show("keep k components, then rebuild the 7 columns", pd.DataFrame(rows).T)

    y = rides["casual"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = {}
    for k in [1, 2, 3, 5]:
        model = make_pipeline(StandardScaler(), PCA(n_components=k),
                              LogisticRegression(max_iter=1000))
        scores[f"PCA({k})"] = r3(cross_val_score(model, rides[FEATURES], y,
                                                 cv=cv, scoring="roc_auc").mean())
    full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    scores["all 7 features"] = r3(cross_val_score(full, rides[FEATURES], y,
                                                  cv=cv, scoring="roc_auc").mean())
    show("casual-vs-member ROC AUC, 5-fold CV", pd.Series(scores))
    print("  PCA sits *inside* the pipeline, so each fold learns its axes from")
    print("  training rows only (lesson 036). Fewer components means less")
    print("  information; whether that hurts depends on whether the dropped")
    print("  directions mattered for the label. PCA never looks at the label.")


# ---------------------------------------------------------------------------
# 6. When PCA helps, and when it doesn't
# ---------------------------------------------------------------------------

def section_limits():
    heading("6. When PCA helps, and when it doesn't")
    print("  - Great for looking: a 2D picture of many columns, for spotting")
    print("    groups, outliers and mistakes.")
    print("  - Useful for compressing many correlated columns (sensors, pixels,")
    print("    word counts) before a simple model.")
    print("  - It finds directions of *spread*, not of *usefulness*. A small")
    print("    component can still carry the signal your label depends on.")
    print("  - Components are blends, so you lose the plain 'one column, one")
    print("    meaning' story. Interpreting a model gets harder.")
    print("  - It is linear. Curved structure needs other tools (t-SNE, UMAP),")
    print("    which are for pictures only, not for feeding models.")


def main():
    print("=" * 66)
    print("  Lesson 045: Dimensionality reduction: PCA")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    section_setup(rides)
    X, pca = section_explained(rides)
    section_loadings(pca)
    section_plot(rides, X, pca)
    section_compress(rides, X)
    section_limits()
    print()
    print("PCA folds overlapping columns into a few new axes. Scale first, read")
    print("explained variance to decide how many to keep, read the loadings to")
    print("see what each axis means, and remember: spread is not the same as")
    print("signal.")
    print()


if __name__ == "__main__":
    main()
