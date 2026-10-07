"""
Lesson 032 - Cross-validation and honest evaluation

Lesson 031 showed that one train/test split is partly luck. Today we ask
several splits instead of one: k-fold cross-validation. We write the loop by
hand, then let scikit-learn do it, use it to compare four models fairly, and
finish with the full honest routine: lock away a test set, choose with
cross-validation, and score the winner on the test set once.

Same cleaned September bike-share rides as lessons 029 to 031, in
rides_clean.csv next to this file.

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
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

FEATURES = ["distance_km", "electric", "casual", "temp_c", "rain_mm"]
TARGET = "duration_min"

# The four candidates we'll compare. None means "ignore the features and
# always predict the training average", our baseline.
CANDIDATES = {
    "guess the mean": None,
    "distance only": ["distance_km"],
    "five features": FEATURES,
    "e-bike slope": ["distance_km", "electric", "casual", "e_km"],
}

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)


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
    rides["electric"] = (rides["bike_type"] == "electric").astype(int)
    rides["casual"] = (rides["rider_type"] == "casual").astype(int)
    # the interaction from lesson 030's exercise: e-bikes get their own slope
    rides["e_km"] = rides["electric"] * rides["distance_km"]
    return rides


def rounded(values, places=2):
    return [round(float(v), places) for v in values]


def cv_mae(model, X, y, cv):
    """Cross-validated MAE, one number per fold, in minutes (positive)."""
    return -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")


def build(columns):
    return DummyRegressor(strategy="mean") if columns is None else LinearRegression()


def columns_for(columns):
    # the dummy model ignores X, but scikit-learn still wants one
    return ["distance_km"] if columns is None else columns


# ---------------------------------------------------------------------------
# 1. k-fold by hand
# ---------------------------------------------------------------------------

def section_by_hand(rides, folds):
    heading("1. k-fold by hand")
    X, y = rides[FEATURES], rides[TARGET]
    scores = []
    for fold, (train_idx, val_idx) in enumerate(folds.split(X), start=1):
        model = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
        score = mean_absolute_error(y.iloc[val_idx], model.predict(X.iloc[val_idx]))
        scores.append(score)
        print(f"  fold {fold}: train on {len(train_idx)} rides, "
              f"score on {len(val_idx)}  -> MAE {score:.2f}")
    show(f"mean MAE across {folds.get_n_splits()} folds (min)", round(float(np.mean(scores)), 2))
    show("spread, standard deviation (min)", round(float(np.std(scores)), 2))
    print("  Every ride is scored exactly once, by a model that never saw it.")
    print("  Five opinions instead of one, and the spread tells you how much")
    print("  any single split could have fooled you.")
    return scores


# ---------------------------------------------------------------------------
# 2. cross_val_score does the loop
# ---------------------------------------------------------------------------

def section_cross_val_score(rides, folds, by_hand):
    heading("2. cross_val_score does the loop")
    scores = cv_mae(LinearRegression(), rides[FEATURES], rides[TARGET], folds)
    show("per-fold MAE from cross_val_score", rounded(scores))
    show("same as the hand-written loop?", bool(np.allclose(scores, by_hand)))
    print("  scoring='neg_mean_absolute_error' is negative because scikit-learn")
    print("  always treats bigger as better. Flip the sign to get minutes back.")


# ---------------------------------------------------------------------------
# 3. cross_validate: more than one number
# ---------------------------------------------------------------------------

def section_cross_validate(rides, folds):
    heading("3. cross_validate: more than one number")
    result = cross_validate(
        LinearRegression(), rides[FEATURES], rides[TARGET], cv=folds,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_train_score=True,
    )
    table = pd.DataFrame({
        "train": [-result["train_neg_mean_absolute_error"].mean(),
                  -result["train_neg_root_mean_squared_error"].mean()],
        "validation": [-result["test_neg_mean_absolute_error"].mean(),
                       -result["test_neg_root_mean_squared_error"].mean()],
    }, index=["MAE", "RMSE"]).round(2)
    show("five features, averaged over the folds (min)", table)
    print("  Train and validation side by side, fold by fold, is the overfitting")
    print("  check from lesson 031 done properly. Here the gap is tiny, so this")
    print("  model is learning a pattern, not memorising rides.")


# ---------------------------------------------------------------------------
# 4. Comparing models fairly
# ---------------------------------------------------------------------------

def section_compare(rides, folds):
    heading("4. Comparing models fairly")
    y = rides[TARGET]
    per_fold = {}
    for name, columns in CANDIDATES.items():
        per_fold[name] = cv_mae(build(columns), rides[columns_for(columns)], y, folds)
    table = pd.DataFrame({
        "mean MAE": {n: s.mean() for n, s in per_fold.items()},
        "std": {n: s.std() for n, s in per_fold.items()},
    }).round(2)
    show("5-fold MAE per model (min)", table)
    gain = per_fold["five features"] - per_fold["e-bike slope"]
    show("five features minus e-bike slope, per fold", rounded(gain))
    print("  Same folds for every model, so it's a fair race. The e-bike slope")
    print("  model wins in every single fold, not just on average: a real")
    print("  improvement, not a lucky split.")
    assert (gain > 0).all()
    return table


# ---------------------------------------------------------------------------
# 5. The honest routine: choose with CV, test once
# ---------------------------------------------------------------------------

def section_honest_routine(rides, folds):
    heading("5. The honest routine: choose with CV, test once")
    dev, test = train_test_split(rides, test_size=0.2, random_state=7)
    show("development rides, locked-away test rides", (len(dev), len(test)))
    choices = {}
    for name, columns in CANDIDATES.items():
        if columns is None:
            continue
        choices[name] = cv_mae(LinearRegression(), dev[columns], dev[TARGET], folds).mean()
    show("CV MAE on development rides only", pd.Series(choices).round(2))
    winner = min(choices, key=choices.get)
    show("chosen model", winner)
    final = LinearRegression().fit(dev[CANDIDATES[winner]], dev[TARGET])
    test_mae = mean_absolute_error(test[TARGET], final.predict(test[CANDIDATES[winner]]))
    show("MAE on the test rides, looked at once (min)", round(float(test_mae), 2))
    print("  The test set played no part in choosing, so its score is honest.")
    print("  It came in under the CV estimate because 236 rides is still one")
    print("  sample, and this one was kind. Expect roughly 1.8 to 2.1 minutes on")
    print("  new rides, not a guaranteed 1.81. And don't go back and re-choose")
    print("  now: the test set has done its one job.")


def main():
    print("=" * 66)
    print("  Lesson 032: Cross-validation and honest evaluation")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    by_hand = section_by_hand(rides, folds)
    section_cross_val_score(rides, folds, by_hand)
    section_cross_validate(rides, folds)
    section_compare(rides, folds)
    section_honest_routine(rides, folds)
    print()
    print("Many folds to choose, one test set to confirm. That's honest")
    print("evaluation. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
