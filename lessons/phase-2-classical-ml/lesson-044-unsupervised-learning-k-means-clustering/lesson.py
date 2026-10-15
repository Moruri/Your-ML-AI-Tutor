"""
Lesson 044 - Unsupervised learning: k-means clustering

Every lesson since 029 handed the model a label to predict: duration,
casual-or-member, long-or-not. Today we put the labels away. Clustering
asks a different question: given only the features, are there natural
groups in the rides? k-means is the simplest answer — pick k centres,
assign every ride to the nearest one, move the centres, repeat — and it
is a good place to learn the habits of unsupervised work: scale first,
choose k with care, and never treat a cluster id as ground truth.

Same cleaned September bike-share rides as lessons 029 to 043, in
rides_clean.csv next to this file. We use the numeric columns (and a
couple of flags) with no train/test split: there is no label to hold
out against.

Run it with (inside the venv from lesson 013, after `pip install -r
requirements.txt`):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files. Every random step has a fixed seed, so your
numbers will match the README.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

# Numeric + flags only. Categories would need encoding; we keep the picture
# simple so the cluster centres stay readable in human units.
FEATURES = ["distance_km", "duration_min", "temp_c", "rain_mm", "weekend", "casual"]

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


def r2(x):
    return round(float(x), 2)


def r3(x):
    return round(float(x), 3)


def load_rides(path):
    rides = pd.read_csv(path, parse_dates=["started_at"])
    rides["weekend"] = (rides["started_at"].dt.dayofweek >= 5).astype(int)
    rides["casual"] = (rides["rider_type"] == "casual").astype(int)
    return rides


def scale(rides):
    scaler = StandardScaler()
    X = scaler.fit_transform(rides[FEATURES])
    return X, scaler


# ---------------------------------------------------------------------------
# 1. Put the labels away
# ---------------------------------------------------------------------------

def section_setup(rides, X):
    heading("1. Put the labels away")
    show("rides / features used for clustering", (len(rides), len(FEATURES)))
    show("feature columns", FEATURES)
    show("feature means (raw units)", rides[FEATURES].mean().round(2))
    show("feature stds (raw units)", rides[FEATURES].std(ddof=0).round(2))
    print("  No duration target, no casual label as the thing to predict.")
    print("  We keep casual and weekend as *features* — traits of the ride —")
    print("  not as answers. Clustering only sees the numbers above.")


# ---------------------------------------------------------------------------
# 2. Scale first, then ask k-means for groups
# ---------------------------------------------------------------------------

def section_scale_and_fit(rides, X, scaler):
    heading("2. Scale first, then ask k-means for groups")
    # Deliberately wrong: cluster on unscaled data so distance_km (0-16) and
    # duration_min (2-76) dominate temp_c and the 0/1 flags.
    raw = KMeans(n_clusters=3, n_init=10, random_state=0).fit(rides[FEATURES])
    scaled = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)

    raw_centres = pd.DataFrame(raw.cluster_centers_, columns=FEATURES)
    scaled_centres = pd.DataFrame(
        scaler.inverse_transform(scaled.cluster_centers_), columns=FEATURES,
    )
    show("cluster sizes without scaling (k=3)",
         pd.Series(raw.labels_).value_counts().sort_index().to_dict())
    show("cluster sizes with StandardScaler (k=3)",
         pd.Series(scaled.labels_).value_counts().sort_index().to_dict())
    show("centres in raw units, AFTER scaling was used",
         scaled_centres.round(2))
    print("  k-means uses Euclidean distance. Without scaling, a 10-minute")
    print("  gap in duration looks ten times bigger than a 1 km gap in")
    print("  distance, and the 0/1 flags barely register. Scale first,")
    print("  always, then transform the centres back to human units to read")
    print("  them.")
    return scaled


# ---------------------------------------------------------------------------
# 3. Choosing k: elbow and silhouette
# ---------------------------------------------------------------------------

def section_choose_k(X):
    heading("3. Choosing k: elbow and silhouette")
    rows = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        sil = silhouette_score(X, km.labels_)
        rows[f"k={k}"] = {
            "inertia": r2(km.inertia_),
            "silhouette": r3(sil),
        }
    table = pd.DataFrame(rows).T
    show("inertia (lower) and silhouette (higher) by k", table)

    inertias = [rows[f"k={k}"]["inertia"] for k in range(2, 9)]
    # drops[i] = inertia(k=i+2) - inertia(k=i+3), so the drop *into* k = i+3
    drops = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    into_k = 3 + int(np.argmax(drops))
    best_sil_k = 2 + int(np.argmax(
        [rows[f"k={k}"]["silhouette"] for k in range(2, 9)]))
    show("k with the best silhouette", best_sil_k)
    show("biggest inertia drop happens into k", into_k)
    print("  Inertia always falls as k rises — more centres fit the points")
    print("  better by construction. Look for the *elbow*, where the next")
    print("  centre helps much less. Silhouette (roughly \"how much closer")
    print("  am I to my own cluster than to the next?\") peaks at a sensible")
    print("  k. The two often disagree a little; here the elbow points at")
    print(f"  k={into_k} and silhouette at k={best_sil_k}. For a first readable")
    print(f"  sketch we will read the k={into_k} centres next.")
    return into_k


def section_interpret(rides, X, scaler, k):
    heading("4. Read the clusters in human terms")
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    centres = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_), columns=FEATURES,
    )
    centres["size"] = pd.Series(km.labels_).value_counts().sort_index().values
    # Order clusters by mean duration so the story reads cleanly
    order = centres["duration_min"].sort_values().index.tolist()
    centres = centres.loc[order].reset_index(drop=True)
    centres.index = [f"cluster {i}" for i in range(k)]
    show(f"centres for k={k}, sorted by duration", centres.round(2))

    # A short hand-written gloss based on the numbers (printed for the README)
    print("  Reading the centres (your numbers, plain English):")
    for i, row in centres.iterrows():
        vibe = []
        if row["duration_min"] < 12 and row["distance_km"] < 2.5:
            vibe.append("short city hop")
        elif row["duration_min"] >= 25 or row["distance_km"] >= 5:
            vibe.append("long leisure ride")
        else:
            vibe.append("medium everyday ride")
        if row["weekend"] >= 0.45:
            vibe.append("often weekend")
        if row["casual"] >= 0.45:
            vibe.append("often casual")
        if row["rain_mm"] >= 3:
            vibe.append("wetter days")
        print(f"    {i}: {', '.join(vibe)} "
              f"(n={int(row['size'])}, "
              f"{row['distance_km']:.1f} km, "
              f"{row['duration_min']:.0f} min)")

    # Sanity: how do clusters line up with the long-ride flag from 043?
    long = (rides["duration_min"] >= 30).astype(int)
    labels = pd.Series(km.labels_, index=rides.index)
    # remap labels to the duration-sorted order used above
    remap = {old: new for new, old in enumerate(order)}
    labels = labels.map(remap)
    crosstab = pd.crosstab(labels, long, rownames=["cluster"], colnames=["long>=30"])
    show("clusters vs the long-ride flag from lesson 043", crosstab)
    print("  Notice: we never told k-means about \"long\". One cluster still")
    print("  soaks up most of the long rides, because duration and distance")
    print("  are in the features. That is a *check*, not a proof. Clusters")
    print("  are a useful sketch of the data, not ground-truth labels.")


# ---------------------------------------------------------------------------
# 5. What unsupervised is not
# ---------------------------------------------------------------------------

def section_limits():
    heading("5. What unsupervised is not")
    print("  - Cluster ids are arbitrary. Cluster 0 today might be cluster 2")
    print("    tomorrow after a shuffle; only the *grouping* matters.")
    print("  - k-means assumes round-ish blobs of similar size. Elongated or")
    print("    nested groups need different tools (later lessons).")
    print("  - Scaling, feature choice and k all change the answer. Two")
    print("    honest analysts can get different sketches of the same CSV.")
    print("  - Never score a clustering with accuracy against a label you")
    print("    hid, then claim you \"discovered\" that label. Use labels only")
    print("    as a sanity check, the way we did with long rides above.")


def main():
    print("=" * 66)
    print("  Lesson 044: Unsupervised learning: k-means clustering")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    X, scaler = scale(rides)
    section_setup(rides, X)
    section_scale_and_fit(rides, X, scaler)
    best_k = section_choose_k(X)
    section_interpret(rides, X, scaler, best_k)
    section_limits()
    print()
    print("Unsupervised learning finds structure without answers. Scale the")
    print("features, pick k with elbow and silhouette, read the centres in")
    print("human units, and treat every cluster id as a sketch — not a fact.")
    print()


if __name__ == "__main__":
    main()
