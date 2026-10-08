"""
Lesson 033 - Logistic regression and classification

Until now every model predicted a number: minutes. Today the answer is a
category. Given a finished ride, was the rider a casual rider or a member?
We see why a straight line is the wrong tool, meet the sigmoid that squashes
any number into a probability, fit scikit-learn's LogisticRegression, read
its probabilities and coefficients, and find the decision boundary.

Same cleaned September bike-share rides as lessons 029 to 032, in
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
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

FEATURES = ["distance_km", "duration_min", "electric", "weekend"]
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
    # the thing we want to predict: 1 for a casual rider, 0 for a member
    rides["casual"] = (rides["rider_type"] == "casual").astype(int)
    return rides


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def split(rides):
    # stratify keeps the casual/member mix the same in both halves
    return train_test_split(rides, test_size=0.2, random_state=0, stratify=rides[TARGET])


# ---------------------------------------------------------------------------
# 1. From numbers to categories
# ---------------------------------------------------------------------------

def section_categories(rides, train, test):
    heading("1. From numbers to categories")
    show("rides that were casual riders (share)", round(float(rides[TARGET].mean()), 3))
    show("train rides, test rides", (len(train), len(test)))
    show("casual share in train / test", (round(float(train[TARGET].mean()), 3),
                                          round(float(test[TARGET].mean()), 3)))
    baseline = DummyClassifier(strategy="most_frequent").fit(train[FEATURES], train[TARGET])
    acc = accuracy_score(test[TARGET], baseline.predict(test[FEATURES]))
    show("baseline: always say 'member', accuracy", round(float(acc), 3))
    print("  A model that never thinks at all is right about 68% of the time.")
    print("  That's the bar. Anything we build has to clear it.")


# ---------------------------------------------------------------------------
# 2. Why not a straight line?
# ---------------------------------------------------------------------------

def section_straight_line(train):
    heading("2. Why not a straight line?")
    line = LinearRegression().fit(train[["distance_km"]], train[TARGET])
    km = pd.DataFrame({"distance_km": [0.2, 0.5, 2, 4, 8, 12]})
    out = pd.Series(line.predict(km).round(2), index=km["distance_km"], name="line says")
    show("straight line on 0/1 labels, by distance (km)", out)
    print("  A 12 km ride gets a 'probability' above 1, and the shortest rides")
    print("  dip below 0. A line doesn't know the answer must stay between 0 and 1.")


# ---------------------------------------------------------------------------
# 3. The sigmoid squash
# ---------------------------------------------------------------------------

def section_sigmoid():
    heading("3. The sigmoid squash")
    z = np.array([-6, -2, -1, 0, 1, 2, 6])
    show("sigmoid of", pd.Series(sigmoid(z).round(3), index=z, name="sigmoid(z)"))
    print("  Any number in, something between 0 and 1 out. Zero maps to exactly")
    print("  0.5. Logistic regression is a straight line, z = w*x + b, fed")
    print("  through this squash.")


# ---------------------------------------------------------------------------
# 4. One feature: distance
# ---------------------------------------------------------------------------

def section_one_feature(train):
    heading("4. One feature: distance")
    model = LogisticRegression().fit(train[["distance_km"]], train[TARGET])
    w, b = float(model.coef_[0, 0]), float(model.intercept_[0])
    show("weight w, intercept b", (round(w, 3), round(b, 3)))
    km = pd.DataFrame({"distance_km": [1, 2, 3, 4, 6, 8]})
    probs = model.predict_proba(km)[:, 1]
    by_hand = sigmoid(w * km["distance_km"] + b)
    show("P(casual) by distance (km)", pd.Series(probs.round(3), index=km["distance_km"]))
    show("same as sigmoid(w*km + b) by hand?", bool(np.allclose(probs, by_hand)))
    boundary = -b / w
    show("distance where P(casual) = 0.5 (km)", round(boundary, 2))
    print("  predict_proba gives a column per class; [:, 1] is P(casual).")
    print("  predict just checks whether that is above 0.5, which happens past")
    print("  the boundary distance. Longer ride, more likely a casual rider.")


# ---------------------------------------------------------------------------
# 5. More features, and reading the coefficients
# ---------------------------------------------------------------------------

def section_more_features(train, test):
    heading("5. More features, and reading the coefficients")
    model = LogisticRegression(max_iter=1000).fit(train[FEATURES], train[TARGET])
    coefs = pd.DataFrame({
        "coef": model.coef_[0],
        "odds x": np.exp(model.coef_[0]),
    }, index=FEATURES).round(2)
    show("coefficients (one per feature)", coefs)
    sample = test[FEATURES].head(5)
    peek = sample.assign(
        p_casual=model.predict_proba(sample)[:, 1].round(2),
        predicted=model.predict(sample),
        actual=test[TARGET].head(5),
    )
    show("five test rides", peek)
    acc = accuracy_score(test[TARGET], model.predict(test[FEATURES]))
    show("accuracy on the test rides", round(float(acc), 3))
    print("  Positive coefficient: pushes towards casual. 'odds x' is how much")
    print("  one more unit multiplies the odds. Weekend rides and long rides lean")
    print("  casual. Coefficients only compare fairly when features share a")
    print("  scale, which is lesson 035.")
    return model


# ---------------------------------------------------------------------------
# 6. The decision boundary
# ---------------------------------------------------------------------------

def section_boundary(train):
    heading("6. The decision boundary")
    cols = ["distance_km", "weekend"]
    model = LogisticRegression().fit(train[cols], train[TARGET])
    w_km, w_wk = model.coef_[0]
    b = model.intercept_[0]
    weekday = -b / w_km
    weekend = -(b + w_wk) / w_km
    show("weekday: casual beyond (km)", round(float(weekday), 2))
    show("weekend: casual beyond (km)", round(float(weekend), 2))
    print("  The boundary is where w*x + b = 0, so the probability is exactly 0.5.")
    print("  With two features it's a line through feature space; here it shows")
    print("  up as a shorter cut-off on weekends, when casual riders are about.")


def main():
    print("=" * 66)
    print("  Lesson 033: Logistic regression and classification")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    train, test = split(rides)
    section_categories(rides, train, test)
    section_straight_line(train)
    section_sigmoid()
    section_one_feature(train)
    section_more_features(train, test)
    section_boundary(train)
    print()
    print("A line, squashed into a probability, cut at 0.5. That's logistic")
    print("regression. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
