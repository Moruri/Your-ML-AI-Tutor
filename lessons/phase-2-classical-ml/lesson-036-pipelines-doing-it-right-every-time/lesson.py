"""
Lesson 036 - Pipelines: doing it right every time

Lesson 035 scaled and encoded the rides by hand, and confessed to a small
leak along the way. Today we stop doing it by hand. A Pipeline glues
preprocessing and a model into one object with a single fit and predict;
a ColumnTransformer sends each column to the right preprocessing step. Then
cross-validation re-fits everything inside every fold, so nothing leaks, and
we prove why that matters with a leak that looks like a breakthrough.

Same cleaned September bike-share rides as lessons 029 to 035, in
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    # identical to lessons 033 to 035, so the numbers line up
    return train_test_split(rides, test_size=0.2, random_state=0, stratify=rides[TARGET])


def r3(x):
    return round(float(x), 3)


def make_preprocess():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("flags", "passthrough", FLAGS),
    ])


def make_model():
    return Pipeline([
        ("prep", make_preprocess()),
        ("knn", KNeighborsClassifier(n_neighbors=15)),
    ])


# ---------------------------------------------------------------------------
# 1. A two-step pipeline
# ---------------------------------------------------------------------------

def section_two_steps(train, test):
    heading("1. A two-step pipeline")
    scaler = StandardScaler().fit(train[NUMERIC])
    knn = KNeighborsClassifier(n_neighbors=15).fit(scaler.transform(train[NUMERIC]), train[TARGET])
    by_hand = knn.score(scaler.transform(test[NUMERIC]), test[TARGET])
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    pipe.fit(train[NUMERIC], train[TARGET])
    show("by hand: scaler, then k-NN, test accuracy", r3(by_hand))
    show("pipeline, test accuracy", r3(pipe.score(test[NUMERIC], test[TARGET])))
    show("step names", tuple(pipe.named_steps))
    show("the scaler inside learned these means", tuple(round(float(m), 2) for m in pipe[0].mean_))
    print("  Same answer, one object. fit runs fit_transform on each step and")
    print("  fit on the last; predict and score run transform on each step and")
    print("  then the model. You can no longer forget to scale the test set.")


# ---------------------------------------------------------------------------
# 2. Different columns, different treatment
# ---------------------------------------------------------------------------

def section_column_transformer(train):
    heading("2. Different columns, different treatment")
    prep = make_preprocess().fit(train)
    names = prep.get_feature_names_out()
    show("columns in", len(NUMERIC + CATEGORICAL + FLAGS))
    show("columns out", len(names))
    show("first few names out", tuple(names[:6]))
    show("last few names out", tuple(names[-4:]))
    out = prep.transform(train.head(2))
    out = out.toarray() if hasattr(out, "toarray") else out
    show("first training ride, transformed", tuple(round(float(v), 2) + 0.0 for v in out[0]))
    print("  Numbers get standardised, categories get one-hot columns, the")
    print("  weekend flag passes through untouched, and any column you didn't")
    print("  name (ride_id, the timestamp) is dropped. The names tell you")
    print("  which step made each column.")


# ---------------------------------------------------------------------------
# 3. Cross-validating the whole thing
# ---------------------------------------------------------------------------

def section_cross_validate(train, test):
    heading("3. Cross-validating the whole thing")
    rows = {}
    for name, last in [("logistic", LogisticRegression(max_iter=1000)),
                       ("k-NN", KNeighborsClassifier(n_neighbors=15))]:
        model = Pipeline([("prep", make_preprocess()), ("model", last)])
        scores = cross_val_score(model, train, train[TARGET], cv=FOLDS)
        rows[name] = {"mean": scores.mean(), "spread": scores.std()}
    show("5-fold CV accuracy, preprocessing inside each fold", pd.DataFrame(rows).T.round(3))
    for name, last in [("logistic", LogisticRegression(max_iter=1000)),
                       ("k-NN", KNeighborsClassifier(n_neighbors=15))]:
        model = Pipeline([("prep", make_preprocess()), ("model", last)]).fit(train, train[TARGET])
        show(f"{name} pipeline on the test rides, once", r3(model.score(test, test[TARGET])))
    print("  cross_val_score clones the whole pipeline for every fold, so the")
    print("  scaler and encoder only ever learn from that fold's training part.")
    print("  The validation part is treated exactly like new rides. That's the")
    print("  fix for lesson 035's confession, and it cost one line.")


# ---------------------------------------------------------------------------
# 4. A leak that looks like a breakthrough
# ---------------------------------------------------------------------------

def section_leak(train):
    heading("4. A leak that looks like a breakthrough")
    rng = np.random.default_rng(36)
    noise = pd.DataFrame(rng.normal(size=(len(train), 1000)), index=train.index,
                         columns=[f"noise_{i}" for i in range(1000)])
    y = train[TARGET]
    show("rides x columns of pure random noise", noise.shape)

    # the wrong way: pick the 20 noise columns that best match the answers,
    # using every training ride, then cross-validate on just those columns
    picked = SelectKBest(f_classif, k=20).fit(noise, y).get_support()
    wrong = cross_val_score(LogisticRegression(max_iter=1000), noise.loc[:, picked], y,
                            cv=FOLDS, scoring="roc_auc")
    show("select on all rides, then CV (ROC-AUC)", r3(wrong.mean()))

    # the right way: selection is a step in the pipeline, redone in every fold
    pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
    right = cross_val_score(pipe, noise, y, cv=FOLDS, scoring="roc_auc")
    show("select inside the pipeline, CV (ROC-AUC)", r3(right.mean()))
    print("  The noise knows nothing about riders, so an honest AUC is about")
    print("  0.5, a coin toss. Chosen with the answers in view, twenty columns")
    print("  look predictive by luck, and CV is fooled because the validation")
    print("  rides helped choose them. Inside the pipeline, each fold picks its")
    print("  own twenty from its own training part, and the score falls back")
    print("  to what noise deserves.")


# ---------------------------------------------------------------------------
# 5. One object, raw rides in, answers out
# ---------------------------------------------------------------------------

def section_use(train):
    heading("5. One object, raw rides in, answers out")
    model = make_model().fit(train, train[TARGET])
    new = pd.DataFrame({
        "ride_id": ["NEW1", "NEW2"],
        "started_at": pd.to_datetime(["2026-10-10 14:20", "2026-10-12 08:05"]),
        "start_station": ["Riverside", "Harbour"],
        "bike_type": ["classic", "electric"],
        "distance_km": [7.5, 1.6],
        "duration_min": [38.0, 7.0],
        "temp_c": [19.5, 15.0],
        "rain_mm": [0.0, 1.2],
    })
    new["weekend"] = (new["started_at"].dt.dayofweek >= 5).astype(int)
    p = model.predict_proba(new)[:, 1]
    print("  two raw rides: a Saturday 38-minute ride and a Monday 7-minute hop")
    show("chance each new ride is casual", tuple(round(float(v), 2) for v in p))
    show("k before", model.get_params()["knn__n_neighbors"])
    model.set_params(knn__n_neighbors=31).fit(train, train[TARGET])
    show("k after set_params(knn__n_neighbors=31)", model.named_steps["knn"].n_neighbors)
    print("  The second ride starts at Harbour, a station the encoder never")
    print("  saw, and nothing breaks. Settings live under step__setting names,")
    print("  which is how lesson 042 will search over them, and the whole")
    print("  fitted object is what you'd save and ship.")


def main():
    print("=" * 66)
    print("  Lesson 036: Pipelines: doing it right every time")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    train, test = split(rides)
    section_two_steps(train, test)
    section_column_transformer(train)
    section_cross_validate(train, test)
    section_leak(train)
    section_use(train)
    print()
    print("Anything that learns from data goes inside the pipeline. Then fit,")
    print("cross-validate and predict can't get it wrong. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
