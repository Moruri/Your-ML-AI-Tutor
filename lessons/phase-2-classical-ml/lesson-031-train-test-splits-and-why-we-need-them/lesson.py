"""
Lesson 031 - Train/test splits and why we need them

Lesson 030 measured every error on the same rides the model learned from.
Today we hold some rides back, train on the rest, and score the model on rides
it has never seen. Then we make a model overfit on purpose, so you can watch
training error fall while real error climbs, and finish with the two things
that make a split trustworthy: luck and time.

Same cleaned September bike-share rides as lessons 029 and 030, in
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

FEATURES = ["distance_km", "electric", "casual", "temp_c", "rain_mm"]
TARGET = "duration_min"

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
    return rides


def mae(model, X, y):
    """Mean absolute error of a fitted model on X, y, in minutes."""
    return round(float(mean_absolute_error(y, model.predict(X))), 2)


# ---------------------------------------------------------------------------
# 1. Marking your own homework
# ---------------------------------------------------------------------------

def section_homework(rides):
    heading("1. Marking your own homework")
    model = LinearRegression().fit(rides[FEATURES], rides[TARGET])
    show("MAE on the rides it learned from (min)", mae(model, rides[FEATURES], rides[TARGET]))
    print("  That's lesson 030's 2.20. It tells you how well the model fits")
    print("  September's rides. It does not tell you how it will do on")
    print("  October's, and that's the only question anyone cares about.")


# ---------------------------------------------------------------------------
# 2. Holding rides back with train_test_split
# ---------------------------------------------------------------------------

def section_split(rides):
    heading("2. Holding rides back with train_test_split")
    X_train, X_test, y_train, y_test = train_test_split(
        rides[FEATURES], rides[TARGET], test_size=0.25, random_state=42
    )
    show("training rows, test rows", (len(X_train), len(X_test)))
    model = LinearRegression().fit(X_train, y_train)    # fit on train ONLY
    show("MAE on training rides (min)", mae(model, X_train, y_train))
    show("MAE on test rides, never seen (min)", mae(model, X_test, y_test))
    print("  A five-feature straight-line model is too simple to memorise")
    print("  anything, so the two numbers are close. The test number is the")
    print("  one you report. Next, a model that isn't so modest.")


# ---------------------------------------------------------------------------
# 3. Overfitting on purpose
# ---------------------------------------------------------------------------

def add_junk_columns(rides, n, seed=0):
    """Add n columns of pure random noise. They carry no information at all."""
    rng = np.random.default_rng(seed)
    junk = pd.DataFrame(rng.normal(size=(len(rides), n)),
                        columns=[f"junk_{i:02d}" for i in range(n)], index=rides.index)
    return pd.concat([rides, junk], axis=1)


TRAIN_RIDES = 30   # try 300 once you've read section 3


def section_overfit(rides):
    heading("3. Overfitting on purpose")
    noisy = add_junk_columns(rides, 30)
    small_train, test = train_test_split(noisy, train_size=TRAIN_RIDES, random_state=1)
    rows = []
    for n_junk in [0, 5, 10, 15, 20, 25, 28]:
        cols = ["distance_km"] + [f"junk_{i:02d}" for i in range(n_junk)]
        model = LinearRegression().fit(small_train[cols], small_train[TARGET])
        rows.append({
            "junk columns": n_junk,
            "train MAE": mae(model, small_train[cols], small_train[TARGET]),
            "test MAE": mae(model, test[cols], test[TARGET]),
        })
    table = pd.DataFrame(rows).set_index("junk columns")
    show(f"{TRAIN_RIDES} training rides, distance plus random junk", table)
    print("  Every junk column is random noise, yet each one lowers the")
    print("  training error: with enough knobs, the model can bend itself to")
    print("  fit the noise in 30 rides. At 28 junk columns it has as many")
    print("  parameters as rides and fits all 30 perfectly. On new rides it")
    print("  is wildly wrong. That gap between train and test is overfitting.")
    if TRAIN_RIDES == 30:
        assert table.loc[28, "train MAE"] < 0.01
        assert table["test MAE"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# 4. The split itself is luck
# ---------------------------------------------------------------------------

def section_luck(rides):
    heading("4. The split itself is luck")
    scores = []
    for seed in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            rides[FEATURES], rides[TARGET], test_size=0.25, random_state=seed
        )
        model = LinearRegression().fit(X_train, y_train)
        scores.append(mae(model, X_test, y_test))
    show("test MAE for random_state 0..9 (min)", scores)
    show("lowest, highest", (min(scores), max(scores)))
    print("  Same model, same data, ten different coin tosses for which rides")
    print("  land in the test set. The score moves by a few tenths of a")
    print("  minute. One split is one opinion; lesson 032 asks several.")
    return scores


# ---------------------------------------------------------------------------
# 5. When order matters: split by time
# ---------------------------------------------------------------------------

def section_time_split(rides):
    heading("5. When order matters: split by time")
    cutoff = pd.Timestamp("2026-09-24")
    past = rides[rides["started_at"] < cutoff]
    future = rides[rides["started_at"] >= cutoff]
    show("rides before / from 24 September", (len(past), len(future)))
    model = LinearRegression().fit(past[FEATURES], past[TARGET])
    show("MAE on the first 23 days (train, min)", mae(model, past[FEATURES], past[TARGET]))
    show("MAE on the last 7 days (test, min)", mae(model, future[FEATURES], future[TARGET]))
    print("  In real life you train on the past and predict the future, so")
    print("  for anything with dates, test on the latest slice instead of a")
    print("  random shuffle. A random split lets the model peek at rides from")
    print("  the same days it's being tested on.")


def main():
    print("=" * 66)
    print("  Lesson 031: Train/test splits and why we need them")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    section_homework(rides)
    section_split(rides)
    section_overfit(rides)
    section_luck(rides)
    section_time_split(rides)
    print()
    print("Fit on train, score on test, and never let the test rides near")
    print("the fitting. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
