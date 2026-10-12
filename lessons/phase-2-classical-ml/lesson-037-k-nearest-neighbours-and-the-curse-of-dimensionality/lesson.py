"""
Lesson 037 - k-Nearest Neighbours and the curse of dimensionality

Lessons 035 and 036 used k-NN as a canary for scaling and as the last step
of a pipeline. Today it gets the proper look. We classify casual vs member
rides by asking the nearest training neighbours, watch how k changes the
answer, confirm that scaling still matters, and then show why piling on
extra features can make every neighbour look equally far away.

Same cleaned September bike-share rides as lessons 029 to 036, in
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
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
    # identical to lessons 033 to 036, so the numbers line up
    return train_test_split(rides, test_size=0.2, random_state=0, stratify=rides[TARGET])


def r3(x):
    return round(float(x), 3)


def make_preprocess():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("flags", "passthrough", FLAGS),
    ])


def make_knn(k=15):
    return Pipeline([
        ("prep", make_preprocess()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


# ---------------------------------------------------------------------------
# 1. Vote of the nearest rides
# ---------------------------------------------------------------------------

def section_neighbours(train, test):
    heading("1. Vote of the nearest rides")
    model = make_knn(15).fit(train, train[TARGET])
    # one test ride: find its 5 nearest training neighbours after prep
    prep = model.named_steps["prep"]
    knn = model.named_steps["knn"]
    X_train = prep.transform(train)
    X_test = prep.transform(test)
    # refit a 5-neighbour view just to peek at the votes
    peek = KNeighborsClassifier(n_neighbors=5).fit(X_train, train[TARGET])
    distances, indices = peek.kneighbors(X_test[0:1])
    neigh = train.iloc[indices[0]][
        ["rider_type", "duration_min", "distance_km", "weekend", "start_station"]
    ].copy()
    neigh["distance"] = distances[0].round(2)
    show("one test ride", test.iloc[0][
        ["rider_type", "duration_min", "distance_km", "weekend", "start_station"]
    ].to_dict())
    show("its 5 nearest training rides", neigh.reset_index(drop=True))
    votes = neigh["rider_type"].value_counts().to_dict()
    show("votes among the five", votes)
    show("k-NN (k=15) prediction for that ride", int(model.predict(test.iloc[0:1])[0]))
    show("'always member' baseline on the test set", r3((test[TARGET] == 0).mean()))
    show("k-NN (k=15) test accuracy", r3(model.score(test, test[TARGET])))
    print("  k-NN stores the training rides and, for each new one, finds the")
    print("  k closest and takes a majority vote. No formula is fitted; the")
    print("  data itself is the model. That's why scaling mattered in 035:")
    print("  closest is measured in feature space.")


# ---------------------------------------------------------------------------
# 2. How k changes the answer
# ---------------------------------------------------------------------------

def section_k(train, test):
    heading("2. How k changes the answer")
    rows = {}
    for k in [1, 5, 15, 31, 61, 121]:
        model = make_knn(k)
        cv = cross_val_score(model, train, train[TARGET], cv=FOLDS)
        test_acc = model.fit(train, train[TARGET]).score(test, test[TARGET])
        rows[f"k={k}"] = {"CV mean": cv.mean(), "CV spread": cv.std(), "test": test_acc}
    show("accuracy as k grows", pd.DataFrame(rows).T.round(3))
    print("  Tiny k listens to a handful of neighbours and overfits the noise.")
    print("  Huge k averages over half the training set and blurs toward the")
    print("  majority class (member). The sweet spot here sits around 15 to 31.")
    print("  Pick k with cross-validation, not by staring at the test score.")


# ---------------------------------------------------------------------------
# 3. Scaling still matters (a pipeline reminder)
# ---------------------------------------------------------------------------

def section_scaling(train, test):
    heading("3. Scaling still matters (a pipeline reminder)")
    # raw numeric only, no scaling
    raw = KNeighborsClassifier(n_neighbors=15)
    raw_acc = raw.fit(train[NUMERIC], train[TARGET]).score(test[NUMERIC], test[TARGET])
    # scaled numeric only
    scaled = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=15)),
    ])
    scaled_acc = scaled.fit(train[NUMERIC], train[TARGET]).score(test[NUMERIC], test[TARGET])
    # full pipeline from section 1
    full = make_knn(15).fit(train, train[TARGET]).score(test, test[TARGET])
    show("k-NN on raw numbers only", r3(raw_acc))
    show("k-NN on standardised numbers only", r3(scaled_acc))
    show("full pipeline (scale + one-hot + weekend)", r3(full))
    print("  Duration in minutes still shouts over distance in km when the")
    print("  columns are raw; standardising levels the field a little.")
    print("  The full pipeline's test score here is the same bounce you saw")
    print("  in lesson 036 (CV was 0.781). Always keep the scaler inside the")
    print("  pipeline so it re-fits inside every CV fold.")


# ---------------------------------------------------------------------------
# 4. The curse: more features, less meaningful neighbours
# ---------------------------------------------------------------------------

def section_curse(train, test):
    heading("4. The curse: more features, less meaningful neighbours")
    rng = np.random.default_rng(37)
    useful = ["duration_min", "weekend"]
    # baseline: two useful columns, scaled
    base = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=15)),
    ])
    base_cv = cross_val_score(base, train[useful], train[TARGET], cv=FOLDS)
    show("CV accuracy, 2 useful columns", r3(base_cv.mean()))

    rows = {}
    for n_noise in [0, 5, 20, 50, 100]:
        noise_cols = [f"noise_{i}" for i in range(n_noise)]
        X_train = train[useful].copy()
        X_test = test[useful].copy()
        if n_noise:
            X_train = X_train.join(pd.DataFrame(
                rng.normal(size=(len(train), n_noise)),
                index=train.index, columns=noise_cols,
            ))
            X_test = X_test.join(pd.DataFrame(
                rng.normal(size=(len(test), n_noise)),
                index=test.index, columns=noise_cols,
            ))
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=15)),
        ])
        cv = cross_val_score(pipe, X_train, train[TARGET], cv=FOLDS)
        test_acc = pipe.fit(X_train, train[TARGET]).score(X_test, test[TARGET])
        rows[f"+{n_noise} noise"] = {
            "features": 2 + n_noise,
            "CV mean": cv.mean(),
            "test": test_acc,
        }
    show("add pure noise columns", pd.DataFrame(rows).T.round(3))

    # distance concentration sketch on standardised noise
    dims = [2, 10, 50, 200]
    conc = {}
    for d in dims:
        pts = rng.normal(size=(400, d))
        # pairwise distances from first point to the rest
        dist = np.linalg.norm(pts[1:] - pts[0], axis=1)
        # ratio of nearest to farthest among those
        conc[f"d={d}"] = {
            "mean distance": dist.mean(),
            "nearest / farthest": dist.min() / dist.max(),
        }
    show("distances in pure noise (400 points)", pd.DataFrame(conc).T.round(3))
    print("  Useful signal drowned in noise: CV falls as we add random columns.")
    print("  In high dimensions, nearest and farthest neighbours get closer")
    print("  in ratio, so 'closest' stops meaning much. More features are not")
    print("  always better for distance-based models; choose them with care.")


# ---------------------------------------------------------------------------
# 5. When k-NN is a good idea
# ---------------------------------------------------------------------------

def section_when(train, test):
    heading("5. When k-NN is a good idea")
    model = make_knn(15).fit(train, train[TARGET])
    show("test accuracy (k=15 pipeline)", r3(model.score(test, test[TARGET])))
    show("training rides remembered", len(train))
    print("  k-NN shines when similar inputs really do have similar labels,")
    print("  the feature space is low-dimensional and well scaled, and you")
    print("  can afford to store the training set. It struggles with many")
    print("  noisy columns, huge datasets (every prediction scans neighbours),")
    print("  and when you need a compact explanation rather than a vote.")
    print("  Tomorrow's decision trees give you that readable explanation.")


def main():
    print("=" * 66)
    print("  Lesson 037: k-Nearest Neighbours and the curse of dimensionality")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    train, test = split(rides)
    section_neighbours(train, test)
    section_k(train, test)
    section_scaling(train, test)
    section_curse(train, test)
    section_when(train, test)
    print()
    print("Closest only means something when the axes are fair and few.")
    print("Scale, pick k with CV, and be suspicious of every extra column.")
    print()


if __name__ == "__main__":
    main()
