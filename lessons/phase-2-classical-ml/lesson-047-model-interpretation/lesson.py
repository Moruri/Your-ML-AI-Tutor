"""
Lesson 047 - Model interpretation

For a dozen lessons we've been asking models *what* they predict. Today we
ask them *why*. Which columns does a model actually lean on? And when one
of those columns goes up, which way does the prediction move? Two tools
answer most of that, permutation importance and partial dependence, and
both come with traps worth knowing before you put a chart in a report.

Same cleaned September bike-share rides as lessons 029 to 046, in
rides_clean.csv next to this file. We train a random forest (lesson 039)
to tell casual riders from members, then interrogate it. One extra column,
`coin`, is pure random noise with a fixed seed: a column we *know* means
nothing, so we can see which tools get fooled by it.

Run it with (inside the venv from lesson 013, after `pip install -r
requirements.txt`):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes one picture, output/partial_dependence.png, next to this
file. Every random step has a fixed seed, so your numbers will match the
README. It takes about ten seconds.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"
OUTPUT = HERE / "output"

# Seven real traits of a ride, plus one column of pure noise.
FEATURES = ["distance_km", "duration_min", "temp_c", "rain_mm",
            "weekend", "hour", "electric", "coin"]

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")


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
    # A column we know is meaningless: random numbers, nothing to do with riders.
    rides["coin"] = np.random.default_rng(47).normal(size=len(rides)).round(3)
    return rides


def make_forest():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                  random_state=0, n_jobs=-1)


def auc(model, X, y):
    return r3(roc_auc_score(y, model.predict_proba(X)[:, 1]))


# ---------------------------------------------------------------------------
# 1. A model worth explaining
# ---------------------------------------------------------------------------

def section_setup(rides):
    heading("1. A model worth explaining")
    X, y = rides[FEATURES], rides["casual"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0)
    forest = make_forest().fit(X_train, y_train)
    show("train rows / test rows", (len(X_train), len(X_test)))
    show("test ROC AUC", auc(forest, X_test, y_test))
    show("first three rows of the inputs", X_test.head(3))
    print("  A decent model: it ranks a random casual rider above a random")
    print("  member about 84% of the time. Now, what is it looking at?")
    return forest, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. The built-in importance, and the coin that fooled it
# ---------------------------------------------------------------------------

def section_impurity(forest):
    heading("2. The built-in importance, and the coin that fooled it")
    imp = pd.Series(forest.feature_importances_, index=FEATURES)
    show("forest.feature_importances_ (sums to 1)", imp.sort_values(ascending=False))
    print("  These come free with every tree model: how much each column's")
    print("  splits tidied up the training data. But 'coin' is pure noise")
    print("  and it ranks third. Columns with many distinct values give the")
    print("  trees lots of split points to memorise the training rows with,")
    print("  and this score rewards that. It also only ever sees training data.")


# ---------------------------------------------------------------------------
# 3. Permutation importance: shuffle a column, watch the score drop
# ---------------------------------------------------------------------------

def section_permutation(forest, X_test, y_test):
    heading("3. Permutation importance: shuffle a column, watch the score drop")
    base = auc(forest, X_test, y_test)
    # By hand first, for one column, so the idea is concrete.
    rng = np.random.default_rng(0)
    shuffled = X_test.copy()
    shuffled["duration_min"] = rng.permutation(shuffled["duration_min"].to_numpy())
    show("test AUC as it is", base)
    show("test AUC with duration_min shuffled", auc(forest, shuffled, y_test))

    result = permutation_importance(forest, X_test, y_test, scoring="roc_auc",
                                    n_repeats=20, random_state=0, n_jobs=-1)
    table = pd.DataFrame({"mean_drop": result.importances_mean,
                          "std": result.importances_std}, index=FEATURES)
    show("AUC drop when each column is shuffled (20 repeats, test set)",
         table.sort_values("mean_drop", ascending=False))
    print("  Shuffle one column's values across the rows and the model loses")
    print("  that column's information, while everything else stays put. The")
    print("  drop in test score is how much the model needed it. The coin")
    print("  drops the score by about nothing, as it should.")
    return table


# ---------------------------------------------------------------------------
# 4. Twins share the credit
# ---------------------------------------------------------------------------

def section_twins(rides, forest, X_train, X_test, y_train, y_test, table):
    heading("4. Twins share the credit")
    show("correlation of distance_km and duration_min",
         r3(rides["distance_km"].corr(rides["duration_min"])))

    rng = np.random.default_rng(0)
    pair = ["distance_km", "duration_min"]
    base = roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])
    drops = []
    for _ in range(20):
        shuffled = X_test.copy()
        order = rng.permutation(len(shuffled))
        shuffled[pair] = shuffled[pair].to_numpy()[order]
        drops.append(base - roc_auc_score(y_test, forest.predict_proba(shuffled)[:, 1]))
    show("drop for distance alone + duration alone",
         r3(table.loc["distance_km", "mean_drop"] + table.loc["duration_min", "mean_drop"]))
    show("drop when both are shuffled together", r3(np.mean(drops)))

    without = [f for f in FEATURES if f != "duration_min"]
    forest2 = make_forest().fit(X_train[without], y_train)
    result = permutation_importance(forest2, X_test[without], y_test,
                                    scoring="roc_auc", n_repeats=20,
                                    random_state=0, n_jobs=-1)
    show("test AUC after dropping duration_min", auc(forest2, X_test[without], y_test))
    show("distance_km importance, with duration in", r3(table.loc["distance_km", "mean_drop"]))
    show("distance_km importance, duration dropped", r3(result.importances_mean[0]))
    print("  Distance and duration tell almost the same story. Shuffle one and")
    print("  the model limps along on the other, so each looks less important")
    print("  than the pair really is. Drop duration and distance's importance")
    print("  quadruples, while the AUC barely moves. Importance is shared")
    print("  between correlated columns, not owned by either.")


# ---------------------------------------------------------------------------
# 5. Partial dependence: which way does the prediction move?
# ---------------------------------------------------------------------------

def partial_dependence_by_hand(model, X, column, values):
    out = {}
    for v in values:
        imagined = X.copy()
        imagined[column] = v          # pretend every ride had this value
        out[v] = r3(model.predict_proba(imagined)[:, 1].mean())
    return out


def section_partial_dependence(forest, X_test):
    heading("5. Partial dependence: which way does the prediction move?")
    show("avg P(casual) if every ride lasted ... min",
         partial_dependence_by_hand(forest, X_test, "duration_min", [5, 10, 15, 20, 30, 45, 60]))
    show("avg P(casual) if every ride were weekend=...",
         partial_dependence_by_hand(forest, X_test, "weekend", [0, 1]))
    show("avg P(casual) if every ride started at ...h",
         partial_dependence_by_hand(forest, X_test, "hour", [8, 12, 17]))

    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "partial_dependence.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # The plotting helper wants float columns (hour is stored as whole numbers).
    PartialDependenceDisplay.from_estimator(
        forest, X_test.astype(float), ["duration_min", "hour"], kind="both",
        subsample=60, random_state=0, ax=axes,
        ice_lines_kw={"alpha": 0.25, "linewidth": 0.6},
        pd_line_kw={"color": "black", "linewidth": 2.5},
    )
    fig.suptitle("Predicted chance of a casual rider (black = average, thin = single rides)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    show("picture written to", str(path.relative_to(HERE)))
    print("  Partial dependence: set one column to a value for *every* row,")
    print("  predict, average. Repeat across values and you get a curve. It")
    print("  answers 'which way, and how much?' where importance only says")
    print("  'how much'. The thin lines in the picture are single rides (ICE).")


# ---------------------------------------------------------------------------
# 6. What these tools can't tell you
# ---------------------------------------------------------------------------

def section_limits(X_test):
    heading("6. What these tools can't tell you")
    speed_if_60_min = X_test["distance_km"] / 1.0  # km covered in one hour
    show("share of rides that become < 4 km/h at 60 min",
         r3((speed_if_60_min < 4).mean()))
    print("  - Partial dependence imagines rides that never happen: forcing")
    print("    every ride to 60 minutes turns most short hops into walking-")
    print("    pace trips. With correlated columns, the far ends of the")
    print("    curve are the model guessing about an empty part of the map.")
    print("  - Importance describes the *model*, not the world. Electric bikes")
    print("    score near zero here because they don't separate riders in this")
    print("    data, not because bike type never matters.")
    print("  - None of this is causation. Making members ride longer won't")
    print("    turn them casual. Lesson 024 still applies.")
    print("  - Use test data, and a score you care about. On training rows a")
    print("    model that memorised noise will happily report that it's useful.")


def main():
    print("=" * 66)
    print("  Lesson 047: Model interpretation")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    forest, X_train, X_test, y_train, y_test = section_setup(rides)
    section_impurity(forest)
    table = section_permutation(forest, X_test, y_test)
    section_twins(rides, forest, X_train, X_test, y_train, y_test, table)
    section_partial_dependence(forest, X_test)
    section_limits(X_test)
    print()
    print("Permutation importance on test data says how much a model leans on")
    print("each column; partial dependence says which way it pushes. Both")
    print("describe the model, share credit between correlated columns, and")
    print("never prove cause and effect.")
    print()


if __name__ == "__main__":
    main()
