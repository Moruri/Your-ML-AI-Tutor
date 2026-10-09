"""
Lesson 035 - Feature scaling and encoding

So far every feature we've fed a model was already a number, and we never
asked whether those numbers were on comparable scales. Today we do. We see
how a distance-based model gets bullied by whichever column has the biggest
numbers, fix it with standardisation (fitted on the training rides only),
turn the start station into columns a model can use with one-hot encoding,
and finish with which models care and which shrug.

Same cleaned September bike-share rides as lessons 029 to 034, in
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

NUMERIC = ["distance_km", "duration_min", "temp_c", "rain_mm"]
TARGET = "casual"

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
    rides["weekend"] = (rides["started_at"].dt.dayofweek >= 5).astype(int)
    rides["casual"] = (rides["rider_type"] == "casual").astype(int)
    return rides


def split(rides):
    # identical to lessons 033 and 034, so the numbers line up
    return train_test_split(rides, test_size=0.2, random_state=0, stratify=rides[TARGET])


def rounded(values, digits=2):
    # plain floats, and no "-0.0"
    return tuple(round(float(v), digits) + 0.0 for v in values)


def acc(model, X_train, y_train, X_test, y_test):
    return round(float(model.fit(X_train, y_train).score(X_test, y_test)), 3)


# ---------------------------------------------------------------------------
# 1. Same rides, very different scales
# ---------------------------------------------------------------------------

def section_scales(train):
    heading("1. Same rides, very different scales")
    summary = train[NUMERIC].agg(["mean", "std", "min", "max"]).T.round(2)
    show("the four numeric columns, training rides", summary)
    a, b = train.iloc[0], train.iloc[1]
    gaps = (a[NUMERIC] - b[NUMERIC]).astype(float).abs().round(2)
    show("gap between the first two training rides", gaps)
    var = train[NUMERIC].var()
    show("share of a typical squared distance", (var / var.sum()).round(2))
    print("  Square the gaps and add them up, and you have the (squared)")
    print("  distance between two rides. Averaged over all pairs, duration")
    print("  supplies three quarters of it, just because it's counted in")
    print("  minutes and spreads widely. Distance in km barely gets a say.")


# ---------------------------------------------------------------------------
# 2. A model that measures distance gets fooled
# ---------------------------------------------------------------------------

def section_knn(train, test):
    heading("2. A model that measures distance gets fooled")
    knn = KNeighborsClassifier(n_neighbors=15)
    show("k-NN, raw numbers", acc(knn, train[NUMERIC], train[TARGET], test[NUMERIC], test[TARGET]))
    big = train[NUMERIC].copy()
    big_test = test[NUMERIC].copy()
    big["temp_c"] = big["temp_c"] * 1000
    big_test["temp_c"] = big_test["temp_c"] * 1000
    show("k-NN, temperature in milli-degrees", acc(knn, big, train[TARGET], big_test, test[TARGET]))
    scaler = StandardScaler().fit(train[NUMERIC])
    X_train = scaler.transform(train[NUMERIC])
    X_test = scaler.transform(test[NUMERIC])
    show("k-NN, standardised", acc(knn, X_train, train[TARGET], X_test, test[TARGET]))
    show("'always member' baseline", round(float((test[TARGET] == 0).mean()), 3))
    print("  Changing the units of one column changed the model's answers.")
    print("  That should never happen to a model you trust. Standardising")
    print("  puts every column on the same footing first.")


# ---------------------------------------------------------------------------
# 3. Standardising by hand, then with StandardScaler
# ---------------------------------------------------------------------------

def section_standardise(train, test):
    heading("3. Standardising by hand, then with StandardScaler")
    mean = train[NUMERIC].mean()
    std = train[NUMERIC].std(ddof=0)
    by_hand = (test[NUMERIC] - mean) / std
    scaler = StandardScaler().fit(train[NUMERIC])
    show("learned means (scaler.mean_)", rounded(scaler.mean_))
    show("learned spreads (scaler.scale_)", rounded(scaler.scale_))
    same = np.allclose(by_hand.to_numpy(), scaler.transform(test[NUMERIC]))
    show("by hand matches StandardScaler?", bool(same))
    scaled_train = scaler.transform(train[NUMERIC])
    show("training means after scaling", rounded(scaled_train.mean(axis=0)))
    show("training spreads after scaling", rounded(scaled_train.std(axis=0)))
    scaled_test = scaler.transform(test[NUMERIC])
    show("test means after scaling (not exactly 0)", rounded(scaled_test.mean(axis=0)))
    print("  fit learns the mean and spread from the training rides only;")
    print("  transform applies those same numbers to any rides you give it.")
    print("  The test rides don't land on exactly 0 and 1, and they shouldn't:")
    print("  they're new data, measured with the training ruler.")


# ---------------------------------------------------------------------------
# 4. Turning categories into columns
# ---------------------------------------------------------------------------

def section_encoding(train, test):
    heading("4. Turning categories into columns")
    stations = sorted(train["start_station"].unique())
    show("start stations", tuple(stations))
    codes = {s: i for i, s in enumerate(stations)}
    show("label codes, 0 to 5", codes)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(train[["start_station"]])
    onehot = pd.DataFrame(enc.transform(test[["start_station"]].head(3)).astype(int),
                          columns=enc.categories_[0], index=test.index[:3])
    onehot.insert(0, "start_station", test["start_station"].head(3))
    show("one-hot, first three test rides", onehot)
    new = pd.DataFrame({"start_station": ["Harbour"]})
    show("a station the encoder never saw", tuple(int(v) for v in enc.transform(new)[0]))
    dummies = pd.get_dummies(test["start_station"].head(3), dtype=int)
    show("pd.get_dummies gives columns", tuple(dummies.columns))
    print("  Numbering stations 0 to 5 invents an order and a distance that")
    print("  aren't there. One-hot gives each station its own 0/1 column.")
    print("  OneHotEncoder remembers the training categories, so the test set")
    print("  gets exactly the same columns, and an unseen station becomes all")
    print("  zeros instead of an error. pd.get_dummies only sees the rows you")
    print("  hand it, which is why it's fine for exploring and risky for models.")


# ---------------------------------------------------------------------------
# 5. Do the new columns help?
# ---------------------------------------------------------------------------

def section_help(train):
    heading("5. Do the new columns help?")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    # NB: fitting the scaler and encoder on all training rides before
    # cross-validation is a small leak. Lesson 036 fixes it properly.
    num = StandardScaler().fit_transform(train[NUMERIC])
    cats = enc.fit_transform(train[["start_station", "bike_type"]])
    y = train[TARGET]
    show("share casual, by bike type", train.groupby("bike_type")[TARGET].mean().round(2))
    rows = {}
    for name, X in [("numbers only", num), ("numbers + station + bike", np.hstack([num, cats]))]:
        for model_name, model in [("logistic", LogisticRegression(max_iter=1000)),
                                  ("k-NN", KNeighborsClassifier(n_neighbors=15))]:
            scores = cross_val_score(model, X, y, cv=folds)
            rows[(name, model_name)] = {"mean": scores.mean(), "spread": scores.std()}
    table = pd.DataFrame(rows).T.round(3)
    show("5-fold CV accuracy on the training rides", table)
    print("  Casual riders take classic and electric bikes at almost the same")
    print("  rate, and stations (exercise 4) are much the same story, so")
    print("  logistic regression gains nothing. k-NN moves up a little, about")
    print("  one fold-spread, which is a hint, not proof. Encoding doesn't")
    print("  make a feature useful; it only lets the model find out.")


# ---------------------------------------------------------------------------
# 6. Which models care?
# ---------------------------------------------------------------------------

def section_who_cares(train, test):
    heading("6. Which models care?")
    scaler = StandardScaler().fit(train[NUMERIC])
    rows = {}
    for name, make in [("logistic regression", lambda: LogisticRegression(max_iter=1000)),
                       ("k-NN", lambda: KNeighborsClassifier(n_neighbors=15))]:
        raw = acc(make(), train[NUMERIC], train[TARGET], test[NUMERIC], test[TARGET])
        scaled = acc(make(), scaler.transform(train[NUMERIC]), train[TARGET],
                     scaler.transform(test[NUMERIC]), test[TARGET])
        rows[name] = {"raw": raw, "standardised": scaled}
    show("test accuracy, raw vs standardised", pd.DataFrame(rows).T)
    print("  k-NN cares, because it measures distances. Here the raw units")
    print("  happened to be workable (duration is a useful column to shout),")
    print("  but section 2 showed one change of units costing it four points.")
    print("  Logistic regression doesn't budge: each weight simply stretches")
    print("  to fit its column. Scaling still helps it train, makes its")
    print("  weights comparable, and regularisation (lesson 041) needs it.")
    print("  Decision trees (lesson 038) split one column at a time and don't")
    print("  care at all.")


def main():
    print("=" * 66)
    print("  Lesson 035: Feature scaling and encoding")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    train, test = split(rides)
    section_scales(train)
    section_knn(train, test)
    section_standardise(train, test)
    section_encoding(train, test)
    section_help(train)
    section_who_cares(train, test)
    print()
    print("Put every number on the same ruler, give every category its own")
    print("column, and learn both from the training rides only. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
