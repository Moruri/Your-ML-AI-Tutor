"""
Lesson 042 - Hyperparameter search

Every model so far has had dials: k in k-NN, depth in trees, the learning rate
in boosting, alpha and C in regularised models. Turning them one at a time by
hand gets old fast, and it tempts you into the worst habit in machine
learning: checking each setting against the test set. Today we see why that
habit lies, then hand the search to scikit-learn properly: GridSearchCV for
small, tidy grids, RandomizedSearchCV for big ones, a search that even picks
the model, and nested cross-validation for an honest final number.

Same cleaned September bike-share rides as lessons 029 to 041, in
rides_clean.csv next to this file, and the same train/test split.

Run it with (inside the venv from lesson 013, after `pip install -r
requirements.txt`):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files. Every random step has a fixed seed, so your
numbers will match the README. It takes about a minute, because searching
means training a lot of models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, ParameterGrid, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

NUMERIC = ["distance_km", "duration_min", "temp_c", "rain_mm"]
CATEGORICAL = ["start_station", "bike_type"]
FLAGS = ["weekend"]
TARGET = "casual"

FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)
pd.set_option("display.max_colwidth", 60)


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
    # identical to lessons 033 to 041, so the numbers line up
    return train_test_split(rides, test_size=0.2, random_state=0, stratify=rides[TARGET])


def r3(x):
    return round(float(x), 3)


def make_knn():
    # the scaled k-NN pipeline from lessons 036 and 037
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("flags", "passthrough", FLAGS),
        ])),
        ("model", KNeighborsClassifier()),
    ])


def make_boost():
    # the HistGradientBoosting pipeline from lesson 040
    prep = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         CATEGORICAL),
        ("flags", "passthrough", FLAGS),
    ])
    cat_columns = [len(NUMERIC), len(NUMERIC) + 1]
    return Pipeline([
        ("prep", prep),
        ("model", HistGradientBoostingClassifier(categorical_features=cat_columns,
                                                 early_stopping=False, random_state=0)),
    ])


def results_table(search, params, top=5):
    res = pd.DataFrame(search.cv_results_)
    cols = [f"param_{p}" for p in params] + ["mean_test_score", "std_test_score",
                                             "rank_test_score"]
    table = res[cols].sort_values("rank_test_score").head(top)
    table.columns = [c.replace("param_model__", "").replace("_test_score", "")
                     for c in table.columns]
    return table.round(3).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Why you never tune on the test set
# ---------------------------------------------------------------------------

def section_test_set_trap(train, test):
    heading("1. Why you never tune on the test set")
    # Shuffle the labels. Now "casual" is pure coin-flipping: no setting of
    # any model can genuinely beat a ROC-AUC of 0.5.
    rng = np.random.default_rng(42)
    y_train = rng.permutation(train[TARGET].to_numpy())
    y_test = rng.permutation(test[TARGET].to_numpy())
    scores = {}
    for k in range(1, 81, 2):
        model = make_knn().set_params(model__n_neighbors=k).fit(train, y_train)
        scores[k] = roc_auc_score(y_test, model.predict_proba(test)[:, 1])
    best_k = max(scores, key=scores.get)
    honest = cross_val_score(make_knn().set_params(model__n_neighbors=best_k), train,
                             y_train, cv=FOLDS, scoring="roc_auc").mean()
    show("labels shuffled, so the true skill is", 0.5)
    show("settings tried against the test set", len(scores))
    show("best k, picked by test ROC-AUC", best_k)
    show("its test ROC-AUC", r3(scores[best_k]))
    show("the same k, CV on training rides", r3(honest))
    show("average test ROC-AUC over all 40", r3(np.mean(list(scores.values()))))
    show("worst test ROC-AUC of the 40", r3(min(scores.values())))
    print("  Try enough settings and one gets lucky on the test rides. Pick it")
    print("  by its test score and you've quietly trained on the test set.")
    print("  Choose with CV on training rides; look at the test set once.")


# ---------------------------------------------------------------------------
# 2. GridSearchCV: every combination, cross-validated
# ---------------------------------------------------------------------------

def section_grid(train, test):
    heading("2. GridSearchCV: every combination, cross-validated")
    grid = {
        "model__n_neighbors": [5, 15, 31, 51],
        "model__weights": ["uniform", "distance"],
    }
    search = GridSearchCV(make_knn(), grid, cv=FOLDS, scoring="accuracy")
    search.fit(train, train[TARGET])
    show("combinations x folds = models trained",
         (len(ParameterGrid(grid)), FOLDS.get_n_splits(), len(ParameterGrid(grid)) * 5))
    show("top five by CV mean", results_table(search, grid))
    show("best_params_", search.best_params_)
    show("best_score_ (CV mean of the winner)", r3(search.best_score_))
    show("refit on all training rides, test once", r3(search.score(test, test[TARGET])))
    print("  The parameter names are step name, two underscores, argument:")
    print("  model__n_neighbors reaches into the 'model' step. With refit=True")
    print("  (the default) the winner is retrained on every training ride, so")
    print("  the search object is itself a ready-to-use model.")
    return search


# ---------------------------------------------------------------------------
# 3. RandomizedSearchCV: when the grid gets too big
# ---------------------------------------------------------------------------

def section_random(train, test):
    heading("3. RandomizedSearchCV: when the grid gets too big")
    big_grid = {
        "model__learning_rate": [0.01, 0.03, 0.1, 0.3],
        "model__max_iter": [50, 100, 200, 400],
        "model__max_leaf_nodes": [4, 8, 16, 31],
        "model__min_samples_leaf": [5, 20, 40, 80],
        "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
    }
    n = len(ParameterGrid(big_grid))
    show("full grid for boosting: combinations", n)
    show("models to train with 5 folds", n * 5)
    space = {
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_iter": randint(50, 401),
        "model__max_leaf_nodes": randint(4, 32),
        "model__min_samples_leaf": randint(5, 81),
        "model__l2_regularization": loguniform(1e-3, 10),
    }
    search = RandomizedSearchCV(make_boost(), space, n_iter=25, cv=FOLDS,
                                scoring="accuracy", random_state=0)
    search.fit(train, train[TARGET])
    show("random draws tried / models trained", (25, 25 * 5))
    table = results_table(search, space)
    table["learning_rate"] = table["learning_rate"].astype(float).round(3)
    table["l2_regularization"] = table["l2_regularization"].astype(float).round(3)
    show("top five by CV mean", table)
    show("best_score_ (CV mean of the winner)", r3(search.best_score_))
    show("test accuracy, looked at once", r3(search.score(test, test[TARGET])))
    show("lesson 040's hand-tuned CV mean, for reference", 0.784)
    print("  25 random draws from sensible ranges found a booster about as good")
    print("  as a careful hand search, for 125 fits instead of 5,120. Use")
    print("  log-scales (loguniform) for dials like learning rates that matter")
    print("  in ratios, and plain ranges for counts.")


# ---------------------------------------------------------------------------
# 4. Searching over whole models
# ---------------------------------------------------------------------------

def section_models(train, test):
    heading("4. Searching over whole models")
    pipe = make_knn()
    grid = [
        {"model": [LogisticRegression(max_iter=5000)], "model__C": [0.1, 1, 10]},
        {"model": [KNeighborsClassifier()], "model__n_neighbors": [15, 31]},
        {"model": [RandomForestClassifier(n_estimators=150, random_state=0)],
         "model__min_samples_leaf": [3, 10]},
    ]
    search = GridSearchCV(pipe, grid, cv=FOLDS, scoring="accuracy")
    search.fit(train, train[TARGET])
    res = pd.DataFrame(search.cv_results_)
    res["candidate"] = [
        type(p["model"]).__name__ + ", " + ", ".join(
            f"{k.split('__')[1]}={v}" for k, v in p.items() if k != "model")
        for p in res["params"]]
    table = (res.set_index("candidate")[["mean_test_score", "std_test_score"]]
             .sort_values("mean_test_score", ascending=False).round(3))
    table.columns = ["CV mean", "CV std"]
    show("one search, three kinds of model", table)
    show("winner", table.index[0])
    show("test accuracy, looked at once", r3(search.score(test, test[TARGET])))
    print("  A grid can be a list of grids, and the model step itself can be a")
    print("  parameter. Same preprocessing, same folds, a fair contest. The top")
    print("  candidates sit within one CV std of each other, so on this data")
    print("  they're about equally good; choosing the simpler one is fine too.")


# ---------------------------------------------------------------------------
# 5. Nested cross-validation: an honest score for the whole search
# ---------------------------------------------------------------------------

def section_nested(train, grid_search):
    heading("5. Nested cross-validation: an honest score for the whole search")
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    picks, scores = [], []
    for fit_idx, check_idx in outer.split(train, train[TARGET]):
        fit_part, check_part = train.iloc[fit_idx], train.iloc[check_idx]
        search = GridSearchCV(make_knn(), grid_search.param_grid, cv=FOLDS,
                              scoring="accuracy").fit(fit_part, fit_part[TARGET])
        picks.append((search.best_params_["model__n_neighbors"],
                      search.best_params_["model__weights"]))
        scores.append(search.score(check_part, check_part[TARGET]))
    show("best_score_ from section 2's search", r3(grid_search.best_score_))
    show("nested CV: mean over 5 outer folds", r3(np.mean(scores)))
    show("outer fold scores", tuple(r3(s) for s in scores))
    show("setting each outer fold picked", tuple(picks))
    print("  best_score_ is the winner's score on the same folds that chose it,")
    print("  so it's a little flattering. Nested CV runs the whole search inside")
    print("  each outer fold and scores it on rides the search never saw. It")
    print("  answers 'how good is my tuning recipe?', not 'which k is best?'.")


def main():
    print("=" * 66)
    print("  Lesson 042: Hyperparameter search")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    train, test = split(rides)
    section_test_set_trap(train, test)
    grid_search = section_grid(train, test)
    section_random(train, test)
    section_models(train, test)
    section_nested(train, grid_search)
    print()
    print("Search on training rides, judge with CV, and open the test set once")
    print("at the very end. The dials are yours to try; the test set is not.")
    print()


if __name__ == "__main__":
    main()
