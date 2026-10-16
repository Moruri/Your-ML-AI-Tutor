"""
Lesson 046 - Anomaly detection

Yesterday PCA squeezed every ride onto one flat picture, and a few dots
sat far away from the crowd. Today we hunt those on purpose. Anomaly
detection asks: which rows don't look like the rest? Sometimes that's a
broken sensor or a typo, sometimes fraud, sometimes the most interesting
ride of the month. The model can't tell you which. It can only point.

Same cleaned September bike-share rides as lessons 029 to 045, in
rides_clean.csv next to this file. To check our detectors honestly we
plant a handful of impossible rides with a fixed seed, so we know where
the needles are before we search the haystack.

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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

FEATURES = ["distance_km", "duration_min", "speed_kmh", "temp_c", "rain_mm", "hour"]

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


def load_rides(path):
    rides = pd.read_csv(path, parse_dates=["started_at"])
    rides["hour"] = rides["started_at"].dt.hour
    rides["speed_kmh"] = rides["distance_km"] / (rides["duration_min"] / 60)
    rides["planted"] = 0
    return rides


def plant_oddities(rides, seed=46):
    """Add eight rides that should not exist, copied from real rows and bent."""
    rng = np.random.default_rng(seed)
    base = rides.sample(8, random_state=seed).copy()
    base["ride_id"] = [f"X{i:05d}" for i in range(1, 9)]
    # 1-3: far too fast for a bike (a GPS jump or a lift in a van)
    base.iloc[0:3, base.columns.get_loc("distance_km")] = rng.uniform(9, 12, 3).round(2)
    base.iloc[0:3, base.columns.get_loc("duration_min")] = rng.uniform(6, 9, 3).round(1)
    # 4-5: bike never docked, clock kept running
    base.iloc[3:5, base.columns.get_loc("distance_km")] = rng.uniform(0.5, 1.0, 2).round(2)
    base.iloc[3:5, base.columns.get_loc("duration_min")] = rng.uniform(150, 200, 2).round(1)
    # 6-8: no single column is extreme, but a 3am start in steady rain
    base.iloc[5:8, base.columns.get_loc("hour")] = 3
    base.iloc[5:8, base.columns.get_loc("rain_mm")] = rng.uniform(9, 12, 3).round(1)
    base["speed_kmh"] = base["distance_km"] / (base["duration_min"] / 60)
    base["planted"] = 1
    return pd.concat([rides, base], ignore_index=True)


def caught(flags, rides):
    """How many planted rides are flagged, and how many flags are real rides."""
    hit = int((flags & (rides["planted"] == 1)).sum())
    return {"flagged": int(flags.sum()), "planted caught": f"{hit} / 8"}


# ---------------------------------------------------------------------------
# 1. What counts as unusual?
# ---------------------------------------------------------------------------

def section_setup(rides):
    heading("1. What counts as unusual?")
    real = rides[rides["planted"] == 0]
    show("real rides", len(real))
    show("speed_kmh in real rides", real["speed_kmh"].describe().round(2))
    show("rides with planted oddities", len(rides))
    show("the eight planted rides",
         rides.loc[rides["planted"] == 1,
                   ["ride_id", "distance_km", "duration_min", "speed_kmh",
                    "rain_mm", "hour"]].round(2))
    print("  We know exactly which eight rows are fake. Real data never")
    print("  hands you that list, so planting a few is how you test a")
    print("  detector before you trust it.")


# ---------------------------------------------------------------------------
# 2. The one-column baseline: z-scores
# ---------------------------------------------------------------------------

def section_zscore(rides):
    heading("2. The one-column baseline: z-scores")
    z = (rides[FEATURES] - rides[FEATURES].mean()) / rides[FEATURES].std(ddof=0)
    flags = (z.abs() > 3).any(axis=1)
    show("rows with any |z| > 3", caught(flags, rides))
    show("which columns triggered most", (z.abs() > 3).sum())
    print("  Cheap and readable: 'more than three standard deviations from")
    print("  the mean' in any single column. It catches the obvious ones,")
    print("  but it also flags every honestly long ride and every rainy day,")
    print("  and it can't see a row that is only odd as a *combination*.")
    return flags


# ---------------------------------------------------------------------------
# 3. Isolation Forest: how quickly can a row be cut off?
# ---------------------------------------------------------------------------

def section_iforest(rides):
    heading("3. Isolation Forest: how quickly can a row be cut off?")
    forest = IsolationForest(n_estimators=300, contamination=0.01, random_state=0)
    forest.fit(rides[FEATURES])
    score = -forest.score_samples(rides[FEATURES])  # higher = stranger
    flags = pd.Series(forest.predict(rides[FEATURES]) == -1, index=rides.index)
    show("contamination=1%", caught(flags, rides))
    top = rides.assign(score=score).nlargest(10, "score")
    show("ten strangest rows by Isolation Forest",
         top[["ride_id", "distance_km", "duration_min", "speed_kmh",
              "rain_mm", "hour", "planted", "score"]].round(2))
    print("  Each tree splits the data on random columns at random points.")
    print("  An odd row gets isolated in very few splits; an ordinary one")
    print("  hides in the crowd for many. Short average path = high score.")
    print("  No scaling needed: the splits are thresholds, like any tree.")
    return score


# ---------------------------------------------------------------------------
# 4. Local Outlier Factor: is my neighbourhood emptier than my neighbours'?
# ---------------------------------------------------------------------------

def section_lof(rides, if_score):
    heading("4. Local Outlier Factor: density, neighbour by neighbour")
    X = StandardScaler().fit_transform(rides[FEATURES])
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    flags = pd.Series(lof.fit_predict(X) == -1, index=rides.index)
    lof_score = -lof.negative_outlier_factor_
    show("contamination=1%", caught(flags, rides))
    top = rides.assign(score=lof_score).nlargest(8, "score")
    show("eight strangest rows by LOF",
         top[["ride_id", "distance_km", "duration_min", "speed_kmh",
              "rain_mm", "hour", "planted", "score"]].round(2))
    ranks = pd.Series(lof_score).rank(ascending=False).astype(int)
    show("LOF rank of the three 3am rides",
         ranks[rides["ride_id"].isin(["X00006", "X00007", "X00008"])].tolist())
    top_if = set(pd.Series(if_score).nlargest(12).index)
    top_lof = set(pd.Series(lof_score).nlargest(12).index)
    show("rows in BOTH top-12 lists", len(top_if & top_lof))
    print("  LOF compares how crowded each row's neighbourhood is with how")
    print("  crowded its neighbours' neighbourhoods are. It works on")
    print("  distances, so scale first (lesson 037 again). The impossible")
    print("  speeds land in empty space and float to the top. The 3am rides")
    print("  sit close to plenty of ordinary rainy rides, so they slip by.")
    return lof_score


# ---------------------------------------------------------------------------
# 5. Choosing how many to flag
# ---------------------------------------------------------------------------

def section_threshold(rides, if_score, lof_score):
    heading("5. Choosing how many to flag")
    rows = {}
    for n in [5, 8, 12, 20, 40]:
        row = {}
        for name, score in [("IF", if_score), ("LOF", lof_score)]:
            order = pd.Series(score).sort_values(ascending=False).index
            row[f"{name} planted found"] = int(rides.loc[order[:n], "planted"].sum())
        rows[f"top {n}"] = row
    show("review the top n rows (out of 8 planted)", pd.DataFrame(rows).T)
    print("  'contamination' is just a guess at the share of odd rows, and")
    print("  it only moves the cut-off. The scores are the real output.")
    print("  Pick n by how many rows a human can actually check, exactly")
    print("  like the threshold trade-off in lesson 043.")


# ---------------------------------------------------------------------------
# 6. What an anomaly score is not
# ---------------------------------------------------------------------------

def section_limits():
    heading("6. What an anomaly score is not")
    print("  - 'Unusual' is not 'wrong'. A 70-minute ride in the sun may be")
    print("    the best ride of someone's month. A human decides.")
    print("  - The detector only knows the features you gave it. Leave out")
    print("    the hour and a 3am ride in the rain looks perfectly normal.")
    print("  - Fit on data that is mostly clean. If a quarter of the rows")
    print("    are broken, broken becomes 'normal'.")
    print("  - When you have labelled examples of the bad thing, use them:")
    print("    that is a (very imbalanced) classification problem, lesson 043.")


def main():
    print("=" * 66)
    print("  Lesson 046: Anomaly detection")
    print("=" * 66)
    rides = plant_oddities(load_rides(RIDES_CSV))
    section_setup(rides)
    section_zscore(rides)
    if_score = section_iforest(rides)
    lof_score = section_lof(rides, if_score)
    section_threshold(rides, if_score, lof_score)
    section_limits()
    print()
    print("Anomaly detection ranks rows by how out of place they look. Plant")
    print("a few known oddities to test it, compare two detectors, flag only")
    print("as many as someone can review, and let a human decide what an")
    print("anomaly means.")
    print()


if __name__ == "__main__":
    main()
