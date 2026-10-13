"""
Lesson 040 - Gradient boosting

A forest trains many deep trees side by side and averages them. Boosting
trains small trees one after another, and each new tree's only job is to fix
what the trees before it still get wrong. Today we build boosting by hand on
ride durations so you can watch the errors shrink, then use scikit-learn's
HistGradientBoostingClassifier on the casual-vs-member rides, see how the
learning rate and the number of trees trade off, let early stopping pick the
number of trees, and tune a few dials with cross-validation.

Same cleaned September bike-share rides as lessons 029 to 039, in
rides_clean.csv next to this file, and the same train/test split.

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

NUMERIC = ["distance_km", "duration_min", "temp_c", "rain_mm"]
CATEGORICAL = ["start_station", "bike_type"]
FLAGS = ["weekend"]
TARGET = "casual"

FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

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
    rides["weekend"] = (rides["started_at"].dt.dayofweek >= 5).astype(int)
    rides["casual"] = (rides["rider_type"] == "casual").astype(int)
    return rides


def split(rides):
    # identical to lessons 033 to 039, so the numbers line up
    return train_test_split(rides, test_size=0.2, random_state=0, stratify=rides[TARGET])


def r3(x):
    return round(float(x), 3)


def make_preprocess():
    # Boosted trees don't need scaling either. Instead of one-hot, we turn each
    # category into an integer code and tell the model which columns are
    # categories, so it can split on them natively.
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         CATEGORICAL),
        ("flags", "passthrough", FLAGS),
    ])


CAT_COLUMNS = [len(NUMERIC), len(NUMERIC) + 1]  # positions after the transformer


def make_boost(**kwargs):
    params = {"categorical_features": CAT_COLUMNS, "early_stopping": False,
              "random_state": 0}
    params.update(kwargs)
    return Pipeline([
        ("prep", make_preprocess()),
        ("model", HistGradientBoostingClassifier(**params)),
    ])


# ---------------------------------------------------------------------------
# 1. Boosting by hand: fix the last model's mistakes
# ---------------------------------------------------------------------------

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def section_by_hand(train, test):
    heading("1. Boosting by hand: fix the last model's mistakes")
    # predict duration (minutes) from distance and the weekend flag
    cols = ["distance_km", "weekend"]
    X_tr, X_te = train[cols].to_numpy(), test[cols].to_numpy()
    y_tr, y_te = train["duration_min"].to_numpy(), test["duration_min"].to_numpy()

    lr = 0.3
    pred_tr = np.full(len(y_tr), y_tr.mean())  # round 0: guess the average
    pred_te = np.full(len(y_te), y_tr.mean())
    rows = {"start (mean)": {"train RMSE": rmse(y_tr, pred_tr), "test RMSE": rmse(y_te, pred_te)}}
    for step in range(1, 51):
        residual = y_tr - pred_tr                     # what we still get wrong
        stump = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X_tr, residual)
        pred_tr += lr * stump.predict(X_tr)           # nudge towards the fix
        pred_te += lr * stump.predict(X_te)
        if step in (1, 2, 5, 10, 25, 50):
            rows[f"after {step} trees"] = {"train RMSE": rmse(y_tr, pred_tr),
                                           "test RMSE": rmse(y_te, pred_te)}
    show("learning rate", lr)
    show("each tree", "DecisionTreeRegressor(max_depth=2)")
    show("errors shrink, round by round (minutes)", pd.DataFrame(rows).T.round(2))
    print("  Start with a dumb guess. Fit a tiny tree to the leftovers (the")
    print("  residuals), add a fraction of its answer, repeat. No single tree")
    print("  is any good. Together they chip the error away, one small fix at a")
    print("  time. That's boosting.")


# ---------------------------------------------------------------------------
# 2. HistGradientBoosting on our classification task
# ---------------------------------------------------------------------------

def section_hgb(train, test):
    heading("2. HistGradientBoosting on our classification task")
    boost = make_boost()
    forest = Pipeline([
        ("prep", make_preprocess()),
        ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                         random_state=0, n_jobs=-1)),
    ])
    rows = {}
    for label, model in [("random forest (lesson 039)", forest), ("boosting, defaults", boost)]:
        cv = cross_val_score(model, train, train[TARGET], cv=FOLDS)
        fitted = model.fit(train, train[TARGET])
        rows[label] = {"train": fitted.score(train, train[TARGET]),
                       "CV mean": cv.mean(), "test": fitted.score(test, test[TARGET])}
    show("forest vs boosting", pd.DataFrame(rows).T.round(3))
    hgb = boost.named_steps["model"]
    show("default learning_rate / max_iter", (hgb.learning_rate, hgb.max_iter))
    show("default max_leaf_nodes / min_samples_leaf",
         (hgb.max_leaf_nodes, hgb.min_samples_leaf))
    print("  'Hist' means it first buckets each numeric column into at most 255")
    print("  bins, which makes it fast even on millions of rows. Out of the box")
    print("  it fits the training rides hard: 100 rounds of 31-leaf trees is a")
    print("  lot of capacity for about 900 rides. Dials coming up.")


# ---------------------------------------------------------------------------
# 3. Learning rate vs number of trees
# ---------------------------------------------------------------------------

def section_learning_rate(train):
    heading("3. Learning rate vs number of trees")
    rows = {}
    for lr in [1.0, 0.3, 0.1, 0.03]:
        row = {}
        for n in [20, 100, 300]:
            model = make_boost(learning_rate=lr, max_iter=n, max_leaf_nodes=8)
            row[f"{n} trees"] = cross_val_score(model, train, train[TARGET], cv=FOLDS).mean()
        rows[f"lr={lr}"] = row
    show("CV mean, max_leaf_nodes=8", pd.DataFrame(rows).T.round(3))
    print("  A big learning rate takes big steps: good fast, then overshoots")
    print("  and starts fitting noise. A small one takes careful steps and")
    print("  needs more trees to get anywhere. The two dials move together:")
    print("  halve the rate, roughly double the trees.")


# ---------------------------------------------------------------------------
# 4. Early stopping: let validation pick the number of trees
# ---------------------------------------------------------------------------

def section_early_stopping(train, test):
    heading("4. Early stopping: let validation pick the number of trees")
    model = make_boost(learning_rate=0.05, max_iter=1000, max_leaf_nodes=8,
                       early_stopping=True, validation_fraction=0.2,
                       n_iter_no_change=20, scoring="loss")
    model.fit(train, train[TARGET])
    hgb = model.named_steps["model"]
    val = hgb.validation_score_
    show("trees allowed / trees actually built", (1000, int(hgb.n_iter_)))
    show("best validation round", int(np.argmax(val)))
    show("test accuracy", r3(model.score(test, test[TARGET])))
    cv = cross_val_score(model, train, train[TARGET], cv=FOLDS).mean()
    show("CV mean with early stopping", r3(cv))
    print("  The model holds back 20% of the training rides, watches its loss")
    print("  on them after every tree, and stops once 20 trees in a row fail to")
    print("  improve it. You set a generous ceiling and let the data decide.")
    print("  The test set is still untouched by any of this.")


# ---------------------------------------------------------------------------
# 5. Tuning the shape of each tree
# ---------------------------------------------------------------------------

def section_tune(train, test):
    heading("5. Tuning the shape of each tree")
    configs = [
        ("defaults", {}),
        ("leaves=8", {"max_leaf_nodes": 8}),
        ("leaves=4", {"max_leaf_nodes": 4}),
        ("leaves=8, min_leaf=40", {"max_leaf_nodes": 8, "min_samples_leaf": 40}),
        ("leaves=8, l2=1.0", {"max_leaf_nodes": 8, "l2_regularization": 1.0}),
    ]
    rows = {}
    for label, extra in configs:
        params = {"learning_rate": 0.05, "max_iter": 200}
        params.update(extra)
        cv = cross_val_score(make_boost(**params), train, train[TARGET], cv=FOLDS)
        rows[label] = {"CV mean": cv.mean(), "CV std": cv.std()}
    table = pd.DataFrame(rows).T
    show("lr=0.05, 200 trees, CV on training rides", table.round(3))
    best = table["CV mean"].idxmax()
    params = {"learning_rate": 0.05, "max_iter": 200}
    params.update(dict(configs)[best])
    final = make_boost(**params).fit(train, train[TARGET])
    show("picked by CV", best)
    show("test accuracy, looked at once", r3(final.score(test, test[TARGET])))
    show("'always member' baseline", r3((test[TARGET] == 0).mean()))
    print("  Small trees, a floor on leaf size, and a little L2 penalty all")
    print("  keep each correction modest. Choose with CV, then look at the")
    print("  test set once at the end. Lesson 042 automates this search.")


def main():
    print("=" * 66)
    print("  Lesson 040: Gradient boosting")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    train, test = split(rides)
    section_by_hand(train, test)
    section_hgb(train, test)
    section_learning_rate(train)
    section_early_stopping(train, test)
    section_tune(train, test)
    print()
    print("A forest is a crowd voting at once. Boosting is a relay: each tree")
    print("fixes what the last one missed. Small steps, small trees, and stop")
    print("when validation says you're done.")
    print()


if __name__ == "__main__":
    main()
