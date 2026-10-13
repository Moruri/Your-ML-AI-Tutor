"""
Lesson 039 - Random forests and bagging

One decision tree is readable but jumpy: change the training rides a little
and the tree changes a lot. Today we measure that jumpiness, tame it by
training many trees on bootstrap samples and letting them vote (bagging),
add a pinch of feature randomness to get a random forest, check it with the
free out-of-bag score, and read feature importances with healthy suspicion.

Same cleaned September bike-share rides as lessons 029 to 038, in
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
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

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
    # identical to lessons 033 to 038, so the numbers line up
    return train_test_split(rides, test_size=0.2, random_state=0, stratify=rides[TARGET])


def r3(x):
    return round(float(x), 3)


def make_preprocess(numeric=NUMERIC):
    # trees (and so forests) don't need scaling; categories still need one-hot
    return ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("flags", "passthrough", FLAGS),
    ])


def make_model(estimator, numeric=NUMERIC):
    return Pipeline([("prep", make_preprocess(numeric)), ("model", estimator)])


# ---------------------------------------------------------------------------
# 1. One tree is jumpy
# ---------------------------------------------------------------------------

def section_jumpy(train, test):
    heading("1. One tree is jumpy")
    rng = np.random.default_rng(0)
    scores, first_splits, preds = [], [], []
    for _ in range(5):
        # keep a random 80% of the training rides each time
        idx = rng.choice(len(train), size=int(0.8 * len(train)), replace=False)
        part = train.iloc[idx]
        model = make_model(DecisionTreeClassifier(random_state=0)).fit(part, part[TARGET])
        tree = model.named_steps["model"]
        names = list(model.named_steps["prep"].get_feature_names_out())
        scores.append(model.score(test, test[TARGET]))
        first_splits.append(f"{names[tree.tree_.feature[0]]} <= {tree.tree_.threshold[0]:.2f}")
        preds.append(model.predict(test))
    preds = np.array(preds)
    disagree = (preds.min(axis=0) != preds.max(axis=0)).mean()
    show("test accuracy of 5 unlimited trees", tuple(r3(s) for s in scores))
    show("spread (max - min)", r3(max(scores) - min(scores)))
    show("first splits", tuple(dict.fromkeys(first_splits)))
    show("test rides where the 5 trees split votes", r3(disagree))
    print("  Same kind of model, slightly different rides, noticeably different")
    print("  answers. That sensitivity is called variance, and deep trees have")
    print("  lots of it. Averaging many of them is the cure.")


# ---------------------------------------------------------------------------
# 2. Bagging: bootstrap, train, vote
# ---------------------------------------------------------------------------

def section_bagging(train, test):
    heading("2. Bagging: bootstrap, train, vote")
    rng = np.random.default_rng(1)
    n = len(train)
    sample = rng.integers(0, n, size=n)
    show("bootstrap sample size", n)
    show("share of rides that made it in at least once", r3(len(np.unique(sample)) / n))

    # bagging by hand: 25 trees, each on its own bootstrap sample, majority vote
    votes = []
    for _ in range(25):
        idx = rng.integers(0, n, size=n)
        part = train.iloc[idx]
        model = make_model(DecisionTreeClassifier(random_state=0)).fit(part, part[TARGET])
        votes.append(model.predict(test))
    votes = np.array(votes)
    majority = (votes.mean(axis=0) > 0.5).astype(int)
    single = make_model(DecisionTreeClassifier(random_state=0)).fit(train, train[TARGET])
    show("single unlimited tree, test accuracy", r3(single.score(test, test[TARGET])))
    show("25 bagged trees by hand, test accuracy", r3((majority == test[TARGET]).mean()))

    bag = make_model(BaggingClassifier(DecisionTreeClassifier(random_state=0),
                                       n_estimators=100, random_state=0, n_jobs=-1))
    cv_single = cross_val_score(single, train, train[TARGET], cv=FOLDS).mean()
    cv_bag = cross_val_score(bag, train, train[TARGET], cv=FOLDS).mean()
    show("CV mean: one tree / BaggingClassifier(100)", (r3(cv_single), r3(cv_bag)))
    print("  Each bootstrap sample draws n rides *with replacement*, so about")
    print("  63% of the rides appear and the rest sit out. Every tree sees a")
    print("  slightly different world, makes different mistakes, and the vote")
    print("  cancels a lot of them out.")


# ---------------------------------------------------------------------------
# 3. Random forests: bagging plus random columns
# ---------------------------------------------------------------------------

def section_forest(train, test):
    heading("3. Random forests: bagging plus random columns")
    rows = {}
    for label, mf in [("all features (bagging)", None), ("sqrt (forest default)", "sqrt"),
                      ("a third", 0.33)]:
        model = make_model(RandomForestClassifier(n_estimators=200, max_features=mf,
                                                  random_state=0, n_jobs=-1))
        cv = cross_val_score(model, train, train[TARGET], cv=FOLDS)
        fitted = model.fit(train, train[TARGET])
        rows[label] = {"CV mean": cv.mean(), "test": fitted.score(test, test[TARGET])}
    show("max_features at each split", pd.DataFrame(rows).T.round(3))

    rows = {}
    for n_trees in [1, 10, 50, 200, 500]:
        model = make_model(RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=5,
                                                  random_state=0, n_jobs=-1))
        cv = cross_val_score(model, train, train[TARGET], cv=FOLDS)
        rows[f"{n_trees} trees"] = {"CV mean": cv.mean(), "CV std": cv.std()}
    show("more trees, min_samples_leaf=5", pd.DataFrame(rows).T.round(3))
    print("  At each split a forest only considers a random handful of columns,")
    print("  so the trees stop all asking the same first question. Less alike")
    print("  trees means a better average. More trees never hurts accuracy; it")
    print("  just stops helping and costs time.")


# ---------------------------------------------------------------------------
# 4. The out-of-bag score: a free validation set
# ---------------------------------------------------------------------------

def section_oob(train, test):
    heading("4. The out-of-bag score: a free validation set")
    forest = make_model(RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                               oob_score=True, random_state=0, n_jobs=-1))
    forest.fit(train, train[TARGET])
    rf = forest.named_steps["model"]
    cv = cross_val_score(forest, train, train[TARGET], cv=FOLDS).mean()
    show("out-of-bag accuracy", r3(rf.oob_score_))
    show("5-fold CV mean", r3(cv))
    show("test accuracy", r3(forest.score(test, test[TARGET])))
    show("'always member' baseline", r3((test[TARGET] == 0).mean()))
    print("  Every ride sat out of roughly a third of the trees. Asking only")
    print("  those trees about it gives an honest score without a separate")
    print("  split. It lands close to CV, for the price of one fit.")
    return forest


# ---------------------------------------------------------------------------
# 5. Feature importances, with suspicion
# ---------------------------------------------------------------------------

def section_importance(train, test):
    heading("5. Feature importances, with suspicion")
    rng = np.random.default_rng(7)
    train = train.copy()
    test = test.copy()
    # a column of pure noise, with lots of distinct values
    train["noise"] = rng.normal(size=len(train))
    test["noise"] = rng.normal(size=len(test))
    numeric = NUMERIC + ["noise"]
    forest = make_model(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
                        numeric=numeric).fit(train, train[TARGET])
    names = list(forest.named_steps["prep"].get_feature_names_out())
    impurity = pd.Series(forest.named_steps["model"].feature_importances_, index=names)

    cols = numeric + CATEGORICAL + FLAGS
    perm = permutation_importance(forest, test[cols], test[TARGET], n_repeats=10,
                                  random_state=0, n_jobs=-1)
    perm = pd.Series(perm.importances_mean, index=cols)

    show("impurity importance (top 6)", impurity.sort_values(ascending=False).head(6).round(3))
    show("permutation importance on test", perm.sort_values(ascending=False).round(3))
    show("noise rank: impurity / permutation",
         (int(impurity.rank(ascending=False)["num__noise"]),
          int(perm.rank(ascending=False)["noise"])))
    print("  Impurity importance is computed on the training rides and loves")
    print("  columns with many distinct values, so pure noise can look useful.")
    print("  Permutation importance shuffles one column on held-out rides and")
    print("  measures how much accuracy drops. Noise barely moves it. Trust that")
    print("  one more, but notice distance looks useless too: duration already")
    print("  carries the same information, so correlated columns share credit.")


def main():
    print("=" * 66)
    print("  Lesson 039: Random forests and bagging")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    train, test = split(rides)
    section_jumpy(train, test)
    section_bagging(train, test)
    section_forest(train, test)
    section_oob(train, test)
    section_importance(train, test)
    print()
    print("One tree has opinions. A forest has a vote. Shake up the rows,")
    print("shake up the columns, and let the crowd smooth out the noise.")
    print()


if __name__ == "__main__":
    main()
