"""
Lesson 029 - What a model actually is

Phase 2 starts here. Before any library does it for us, we build the
smallest useful model by hand: a straight line that predicts how long a
bike-share ride takes from how far it goes. Along the way we name the parts
every model has: features, a target, parameters, and a loss that says how
wrong the parameters are.

The data is the cleaned September rides from lesson 028, already tidied and
joined to the weather, in rides_clean.csv next to this file.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

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


def predict(x, slope, intercept):
    """The whole model: a line. Two parameters, one feature."""
    return slope * x + intercept


def mae(y, y_hat):
    """Mean absolute error: the average miss, in the target's own units."""
    return float(np.mean(np.abs(y - y_hat)))


def mse(y, y_hat):
    """Mean squared error: big misses count much more than small ones."""
    return float(np.mean((y - y_hat) ** 2))


# ---------------------------------------------------------------------------
# 1. Features, target, rows
# ---------------------------------------------------------------------------

def section_names(rides):
    heading("1. Features, target, rows")
    show("rides (rows) / columns", rides.shape)
    show("first three rides", rides[["distance_km", "bike_type", "duration_min"]].head(3))
    x = rides["distance_km"].to_numpy()
    y = rides["duration_min"].to_numpy()
    show("feature x = distance_km, mean", round(float(x.mean()), 2))
    show("target  y = duration_min, mean", round(float(y.mean()), 2))
    print("  A feature is something we know before the ride ends (distance).")
    print("  The target is what we want to predict (minutes). Each row is one")
    print("  example the model can learn from.")
    return x, y


# ---------------------------------------------------------------------------
# 2. The laziest model: always guess the average
# ---------------------------------------------------------------------------

def section_baseline(x, y):
    heading("2. The laziest model: always guess the average")
    guess = np.full_like(y, y.mean())
    show("prediction for every ride (min)", round(float(y.mean()), 2))
    show("baseline MAE (min)", round(mae(y, guess), 2))
    print("  This model ignores distance completely. It's useless on its own,")
    print("  but it's the score any real model has to beat. Always compute it.")
    return mae(y, guess)


# ---------------------------------------------------------------------------
# 3. A model is a function with knobs
# ---------------------------------------------------------------------------

def section_knobs(x, y):
    heading("3. A model is a function with knobs")
    guesses = [(3.0, 0.0), (5.0, 0.0), (4.0, 2.0)]
    for slope, intercept in guesses:
        y_hat = predict(x, slope, intercept)
        show(f"slope={slope}, intercept={intercept}: MAE / MSE",
             (round(mae(y, y_hat), 2), round(mse(y, y_hat), 1)))
    print("  Same model shape every time: minutes = slope * km + intercept.")
    print("  Only the parameters change. The loss scores each setting, and")
    print("  'training' just means finding the setting with the lowest loss.")


# ---------------------------------------------------------------------------
# 4. Training by brute force: try lots of knob settings
# ---------------------------------------------------------------------------

def section_grid(x, y):
    heading("4. Training by brute force: try lots of knob settings")
    slopes = np.arange(2.0, 7.01, 0.05)
    intercepts = np.arange(-5.0, 5.01, 0.1)
    best = (None, None, np.inf)
    tried = 0
    for s in slopes:
        for b in intercepts:
            loss = mse(y, predict(x, s, b))
            tried += 1
            if loss < best[2]:
                best = (s, b, loss)
    slope, intercept, loss = best
    show("settings tried", tried)
    show("best slope (min per km)", round(float(slope), 2))
    show("best intercept (min)", round(float(intercept), 2))
    show("best MSE", round(loss, 2))
    print("  This works for two knobs. With 20 features and a grid of 100")
    print("  values each you'd need 100**20 tries. That's why we want maths,")
    print("  or a smarter search, instead of a grid.")
    return slope, intercept


# ---------------------------------------------------------------------------
# 5. The exact answer: least squares in two lines
# ---------------------------------------------------------------------------

def section_least_squares(x, y, grid_slope, grid_intercept):
    heading("5. The exact answer: least squares in two lines")
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    intercept = y.mean() - slope * x.mean()
    show("least-squares slope", round(float(slope), 3))
    show("least-squares intercept", round(float(intercept), 3))
    show("grid search got", (round(float(grid_slope), 2), round(float(grid_intercept), 2)))
    show("np.polyfit(x, y, 1) agrees", tuple(round(float(v), 3) for v in np.polyfit(x, y, 1)))
    print("  For a straight line and squared error there's a formula for the")
    print("  minimum, so no search is needed. Libraries use the same idea.")
    assert abs(slope - grid_slope) < 0.1
    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# 6. Using the model, and what it gets wrong
# ---------------------------------------------------------------------------

def section_use(rides, x, y, slope, intercept, baseline_mae):
    heading("6. Using the model, and what it gets wrong")
    for km in (1.0, 3.0, 8.0):
        show(f"predicted minutes for a {km:g} km ride", round(predict(km, slope, intercept), 1))
    y_hat = predict(x, slope, intercept)
    show("model MAE (min)", round(mae(y, y_hat), 2))
    show("baseline MAE from section 2 (min)", round(baseline_mae, 2))
    rides = rides.assign(error=y - y_hat)
    by_group = rides.groupby(["bike_type", "rider_type"])["error"].mean().round(2)
    show("average error (actual - predicted) by group", by_group)
    print("  The line is much better than guessing the average, but its misses")
    print("  aren't random: it predicts too long for e-bikes and too short for")
    print("  classic bikes. Errors with a pattern mean a feature is missing.")
    print("  Lesson 030 adds those features with scikit-learn.")
    assert mae(y, y_hat) < baseline_mae


def main():
    print("=" * 66)
    print("  Lesson 029: What a model actually is")
    print("=" * 66)
    rides = pd.read_csv(RIDES_CSV, parse_dates=["started_at"])
    x, y = section_names(rides)
    baseline_mae = section_baseline(x, y)
    section_knobs(x, y)
    grid_slope, grid_intercept = section_grid(x, y)
    slope, intercept = section_least_squares(x, y, grid_slope, grid_intercept)
    section_use(rides, x, y, slope, intercept, baseline_mae)
    print()
    print("Features in, prediction out, loss to say how wrong, training to")
    print("turn the knobs. Every model in this course is that. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
