"""
Lesson 038 - Decision trees you can read

k-NN votes by distance. A decision tree asks a sequence of yes/no questions
about one column at a time, and you can literally read the questions. Today
we train a tree on the same casual-vs-member rides, see how impurity picks
the splits, watch depth turn into overfitting, prune with max_depth and
min_samples_leaf, and print the tree as text so it stays readable headless.

Same cleaned September bike-share rides as lessons 029 to 037, in
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

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
    # identical to lessons 033 to 037, so the numbers line up
    return train_test_split(rides, test_size=0.2, random_state=0, stratify=rides[TARGET])


def r3(x):
    return round(float(x), 3)


def make_preprocess():
    # trees don't need scaling; one-hot still turns categories into columns
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("flags", "passthrough", FLAGS),
    ])


def make_tree(**tree_kwargs):
    defaults = {"random_state": 0}
    defaults.update(tree_kwargs)
    return Pipeline([
        ("prep", make_preprocess()),
        ("tree", DecisionTreeClassifier(**defaults)),
    ])


def feature_names(prep):
    return list(prep.get_feature_names_out())


# ---------------------------------------------------------------------------
# 1. A tree of yes/no questions
# ---------------------------------------------------------------------------

def section_first_tree(train, test):
    heading("1. A tree of yes/no questions")
    model = make_tree(max_depth=3).fit(train, train[TARGET])
    tree = model.named_steps["tree"]
    names = feature_names(model.named_steps["prep"])
    show("test accuracy, depth 3", r3(model.score(test, test[TARGET])))
    show("'always member' baseline", r3((test[TARGET] == 0).mean()))
    show("number of leaves", int(tree.get_n_leaves()))
    show("depth", int(tree.get_depth()))
    text = export_text(tree, feature_names=names, max_depth=3)
    print("  readable tree (depth 3):")
    for line in text.splitlines():
        print(f"    {line}")
    print("  Each line is a question about one column. Follow the answers")
    print("  down to a leaf, and you have the prediction. No distances, no")
    print("  scaling, just thresholds you can read aloud.")


# ---------------------------------------------------------------------------
# 2. Impurity: why this split?
# ---------------------------------------------------------------------------

def section_impurity(train):
    heading("2. Impurity: why this split?")
    y = train[TARGET].to_numpy()
    p = y.mean()
    gini_root = 1 - (p ** 2 + (1 - p) ** 2)
    ent_root = 0.0 if p in (0, 1) else float(
        -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    )
    show("share casual in training rides", r3(p))
    show("Gini impurity at the root", r3(gini_root))
    show("entropy at the root", r3(ent_root))

    # first split the depth-3 tree actually chose
    model = make_tree(max_depth=3).fit(train, train[TARGET])
    tree = model.named_steps["tree"]
    names = feature_names(model.named_steps["prep"])
    feat = names[tree.tree_.feature[0]]
    thr = tree.tree_.threshold[0]
    show("first split feature", feat)
    show("first split threshold", round(float(thr), 3))

    # Gini after splitting on weekend by hand (the usual first question)
    left = train.loc[train["weekend"] == 0, TARGET]
    right = train.loc[train["weekend"] == 1, TARGET]

    def gini(series):
        q = series.mean()
        return 1 - (q ** 2 + (1 - q) ** 2)

    w_left, w_right = len(left) / len(train), len(right) / len(train)
    after = w_left * gini(left) + w_right * gini(right)
    show("Gini if we split on weekend == 0", r3(after))
    show("impurity drop (information gained)", r3(gini_root - after))
    show("share casual on weekdays / weekends",
         (r3(left.mean()), r3(right.mean())))
    print("  Gini is 0 when a node is pure (all casual or all member) and")
    print("  highest when it's a 50/50 mix. The tree tries every feature and")
    print("  threshold, and picks the split that drops impurity the most.")
    print("  Entropy tells the same story on a log scale; Gini is the default.")


# ---------------------------------------------------------------------------
# 3. Depth, leaves and overfitting
# ---------------------------------------------------------------------------

def section_depth(train, test):
    heading("3. Depth, leaves and overfitting")
    rows = {}
    for depth in [1, 2, 3, 5, 8, None]:
        label = "unlimited" if depth is None else f"depth={depth}"
        model = make_tree(max_depth=depth)
        cv = cross_val_score(model, train, train[TARGET], cv=FOLDS)
        fitted = model.fit(train, train[TARGET])
        tree = fitted.named_steps["tree"]
        rows[label] = {
            "leaves": int(tree.get_n_leaves()),
            "train": fitted.score(train, train[TARGET]),
            "CV mean": cv.mean(),
            "test": fitted.score(test, test[TARGET]),
        }
    show("deeper trees remember more", pd.DataFrame(rows).T.round(3))
    print("  A stump (depth 1) underfits. An unlimited tree can memorise the")
    print("  training rides (train score near 1.0) and then stumble on new")
    print("  ones. Cross-validation picks a depth that still generalises.")


# ---------------------------------------------------------------------------
# 4. Pruning without the drama
# ---------------------------------------------------------------------------

def section_prune(train, test):
    heading("4. Pruning without the drama")
    rows = {}
    configs = [
        ("depth=3", {"max_depth": 3}),
        ("depth=5", {"max_depth": 5}),
        ("min_leaf=20", {"min_samples_leaf": 20}),
        ("min_leaf=50", {"min_samples_leaf": 50}),
        ("depth=5, min_leaf=20", {"max_depth": 5, "min_samples_leaf": 20}),
    ]
    for label, kwargs in configs:
        model = make_tree(**kwargs)
        cv = cross_val_score(model, train, train[TARGET], cv=FOLDS)
        fitted = model.fit(train, train[TARGET])
        tree = fitted.named_steps["tree"]
        rows[label] = {
            "leaves": int(tree.get_n_leaves()),
            "CV mean": cv.mean(),
            "test": fitted.score(test, test[TARGET]),
        }
    show("prune with max_depth and min_samples_leaf", pd.DataFrame(rows).T.round(3))
    # keep a readable winner for the next section
    winner = make_tree(max_depth=3, min_samples_leaf=20).fit(train, train[TARGET])
    show("chosen: depth=3, min_leaf=20, test accuracy",
         r3(winner.score(test, test[TARGET])))
    print("  max_depth caps how many questions you may ask. min_samples_leaf")
    print("  refuses to carve a leaf from a handful of rides. Either one")
    print("  (or both) stops the tree from memorising quirks. Cost-complexity")
    print("  pruning (ccp_alpha) is another dial; we'll meet it when forests")
    print("  and boosting need finer control.")


# ---------------------------------------------------------------------------
# 5. Reading the pruned tree
# ---------------------------------------------------------------------------

def section_read(train, test):
    heading("5. Reading the pruned tree")
    model = make_tree(max_depth=3, min_samples_leaf=20).fit(train, train[TARGET])
    tree = model.named_steps["tree"]
    names = feature_names(model.named_steps["prep"])
    text = export_text(tree, feature_names=names)
    print("  pruned tree:")
    for line in text.splitlines():
        print(f"    {line}")
    # feature importances: how much impurity each feature dropped in total
    imp = pd.Series(tree.feature_importances_, index=names)
    top = imp.sort_values(ascending=False).head(5).round(3)
    show("top feature importances", top)
    # predict two contrasting rides
    new = pd.DataFrame({
        "ride_id": ["NEW1", "NEW2"],
        "started_at": pd.to_datetime(["2026-10-11 14:20", "2026-10-13 08:05"]),
        "start_station": ["Riverside", "Harbour"],
        "bike_type": ["classic", "electric"],
        "distance_km": [7.5, 1.6],
        "duration_min": [38.0, 7.0],
        "temp_c": [19.5, 15.0],
        "rain_mm": [0.0, 1.2],
    })
    new["weekend"] = (new["started_at"].dt.dayofweek >= 5).astype(int)
    preds = model.predict(new)
    probs = model.predict_proba(new)[:, 1]
    show("Saturday long ride / Tuesday hop -> class", tuple(int(p) for p in preds))
    show("chance each is casual", tuple(round(float(p), 2) for p in probs))
    print("  Importances credit the columns that did the impurity-dropping.")
    print("  Treat them as a hint, not a proof: correlated columns share the")
    print("  credit awkwardly. Next lesson grows many trees and averages them.")


def main():
    print("=" * 66)
    print("  Lesson 038: Decision trees you can read")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    train, test = split(rides)
    section_first_tree(train, test)
    section_impurity(train)
    section_depth(train, test)
    section_prune(train, test)
    section_read(train, test)
    print()
    print("A tree is a checklist you can read. Cap its depth, insist on")
    print("decent leaf sizes, and you'll trust the checklist on new rides.")
    print()


if __name__ == "__main__":
    main()
