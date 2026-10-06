"""
Lesson 030 - Linear regression with scikit-learn

Lesson 029 fitted a line by hand. Today scikit-learn does the fitting, and we
learn the API that nearly every model in the library shares: build the model,
.fit(X, y), then .predict(X). We check it gives lesson 029's answer, add the
features the one-line model was missing, read the coefficients, and measure
error with MAE and RMSE.

Same cleaned September bike-share rides as lesson 029, in rides_clean.csv
next to this file.

Run it with (inside the venv from lesson 013, after `pip install -r
requirements.txt`, which includes scikit-learn):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

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


def load_rides(path):
    rides = pd.read_csv(path, parse_dates=["started_at"])
    # yes/no columns as 0/1 numbers, so a linear model can use them
    rides["electric"] = (rides["bike_type"] == "electric").astype(int)
    rides["casual"] = (rides["rider_type"] == "casual").astype(int)
    return rides


# ---------------------------------------------------------------------------
# 1. X and y: the shapes scikit-learn expects
# ---------------------------------------------------------------------------

def section_shapes(rides):
    heading("1. X and y: the shapes scikit-learn expects")
    X = rides[["distance_km"]]          # double brackets: a table, 2-D
    y = rides["duration_min"]           # single brackets: a column, 1-D
    show("X shape (rows, features)", X.shape)
    show("y shape (rows,)", y.shape)
    print("  X is always 2-D, even with one feature: one row per example,")
    print("  one column per feature. y is 1-D: one target per row.")
    return X, y


# ---------------------------------------------------------------------------
# 2. fit, then predict
# ---------------------------------------------------------------------------

def section_fit_predict(X, y):
    heading("2. fit, then predict")
    model = LinearRegression()
    model.fit(X, y)
    show("slope, model.coef_[0]", round(float(model.coef_[0]), 3))
    show("intercept, model.intercept_", round(float(model.intercept_), 3))
    new_rides = pd.DataFrame({"distance_km": [1.0, 3.0, 8.0]})
    show("predictions for 1, 3, 8 km",
         [round(float(p), 1) for p in model.predict(new_rides)])
    print("  Same slope and intercept as the least-squares formula in lesson")
    print("  029. Attributes ending in _ (coef_, intercept_) are what fit()")
    print("  learned; they don't exist until you call it.")
    assert abs(model.coef_[0] - 4.41) < 0.01
    return model


# ---------------------------------------------------------------------------
# 3. Measuring error: MAE and RMSE
# ---------------------------------------------------------------------------

def section_metrics(model, X, y):
    heading("3. Measuring error: MAE and RMSE")
    y_hat = model.predict(X)
    baseline = np.full(len(y), y.mean())
    rows = pd.DataFrame({
        "MAE": [mean_absolute_error(y, baseline), mean_absolute_error(y, y_hat)],
        "RMSE": [root_mean_squared_error(y, baseline), root_mean_squared_error(y, y_hat)],
    }, index=["guess the mean", "distance only"]).round(2)
    show("error in minutes", rows)
    errors = np.abs(y - y_hat)
    show("share of rides missed by more than 10 min", round(float(np.mean(errors > 10)), 3))
    print("  MAE is the typical miss. RMSE squares the misses before averaging,")
    print("  so a few big misses push it up. RMSE well above MAE means some")
    print("  rides are being predicted badly, not all rides a little badly.")
    return rows.loc["distance only"]


# ---------------------------------------------------------------------------
# 4. More features, same three lines of code
# ---------------------------------------------------------------------------

FEATURES = ["distance_km", "electric", "casual", "temp_c", "rain_mm"]


def section_more_features(rides, y, one_feature):
    heading("4. More features, same three lines of code")
    X = rides[FEATURES]
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    show("features", FEATURES)
    show("MAE / RMSE, distance only (min)",
         (round(float(one_feature["MAE"]), 2), round(float(one_feature["RMSE"]), 2)))
    show("MAE / RMSE, all five features (min)",
         (round(mean_absolute_error(y, y_hat), 2), round(root_mean_squared_error(y, y_hat), 2)))
    print("  Nothing about the code changed except the columns in X. That's")
    print("  the point of a shared API: the model doesn't care what the")
    print("  columns mean, only that they're numbers.")
    return model, y_hat


# ---------------------------------------------------------------------------
# 5. Reading the coefficients
# ---------------------------------------------------------------------------

def section_coefficients(model):
    heading("5. Reading the coefficients")
    coefs = pd.Series(model.coef_, index=FEATURES).round(2)
    show("coefficient per feature (minutes per unit)", coefs)
    show("intercept (min)", round(float(model.intercept_), 2))
    print("  Each coefficient is 'how many minutes this adds, holding the")
    print("  other features fixed'. An e-bike saves about four and a half")
    print("  minutes; a casual rider adds about one. Weather barely registers.")
    print("  Units matter: temp_c is per degree and rain_mm is per mm, so")
    print("  you can't compare the raw sizes directly.")
    print("  And these describe the data, not cause and effect (lesson 024).")
    return coefs


# ---------------------------------------------------------------------------
# 6. Checking the leftovers
# ---------------------------------------------------------------------------

def section_residuals(rides, y, y_hat):
    heading("6. Checking the leftovers")
    rides = rides.assign(error=y - y_hat)
    by_group = rides.groupby(["bike_type", "rider_type"])["error"].mean().round(2)
    show("average error (actual - predicted) by group", by_group)
    worst = rides.loc[rides["error"].abs().nlargest(3).index,
                      ["distance_km", "bike_type", "rider_type", "duration_min", "error"]]
    show("three biggest misses", worst.round(1))
    print("  The group pattern from lesson 029 is mostly gone. What's left are")
    print("  a few rides the model can't explain (someone stopping for lunch).")
    print("  One honest warning: every score today was measured on the same")
    print("  rides the model learned from. Lesson 031 fixes that.")


def main():
    print("=" * 66)
    print("  Lesson 030: Linear regression with scikit-learn")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    X, y = section_shapes(rides)
    model = section_fit_predict(X, y)
    one_feature = section_metrics(model, X, y)
    model5, y_hat = section_more_features(rides, y, one_feature)
    section_coefficients(model5)
    section_residuals(rides, y, y_hat)
    print()
    print("Build it, fit it, predict with it, then measure how wrong it is.")
    print("That loop is most of scikit-learn. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
