"""
Lesson 041 - Regularisation: Ridge, Lasso and friends

A plain linear regression believes every column it's given and will bend its
coefficients as far as it likes to fit the training rows. With few rows and
many columns, that's a recipe for overfitting. Regularisation adds a penalty
for big coefficients: the model has to earn every bit of weight it puts on a
feature. Ridge shrinks all the weights a little; Lasso pushes the useless ones
all the way to zero, which makes it a feature selector too.

We go back to the regression task from lesson 030 (predict a ride's duration
in minutes), make it deliberately hard with a small training sample and a pile
of pure-noise columns, and watch the penalty rescue it. Then we let
cross-validation pick the penalty, and finish with the same idea inside
logistic regression for the casual-vs-member question.

Same cleaned September bike-share rides as lessons 029 to 040, in
rides_clean.csv next to this file, and the same train/test split.

Run it with (inside the venv from lesson 013, after `pip install -r
requirements.txt`):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files. Every random step has a fixed seed, so your
numbers will match the README.
"""

import warnings
from pathlib import Path

import sklearn

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (ElasticNet, Lasso, LassoCV, LinearRegression,
                                  LogisticRegression, Ridge, RidgeCV)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

# Regression task (sections 1 to 4): predict duration_min.
REAL_NUMERIC = ["distance_km", "temp_c", "rain_mm"]
NOISE = [f"noise_{i:02d}" for i in range(40)]
CATEGORICAL = ["start_station", "bike_type"]
FLAGS = ["weekend", "casual"]
DURATION = "duration_min"
SMALL_N = 60  # rides we pretend to have for training in sections 1 to 4

# Classification task (section 5): casual vs member, as in lessons 033 to 040.
CLS_NUMERIC = ["distance_km", "duration_min", "temp_c", "rain_mm"]
TARGET = "casual"
FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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
    # 40 columns of pure random noise, standing in for "extra data someone
    # handed us". None of them has anything to do with duration.
    rng = np.random.default_rng(41)
    noise = pd.DataFrame(rng.normal(size=(len(rides), len(NOISE))), columns=NOISE,
                         index=rides.index)
    return pd.concat([rides, noise], axis=1)


def split(rides):
    # identical to lessons 033 to 040, so the numbers line up
    return train_test_split(rides, test_size=0.2, random_state=0, stratify=rides[TARGET])


def r2(x):
    return round(float(x), 2)


def r3(x):
    return round(float(x), 3)


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def make_preprocess(numeric, scale=True):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
         CATEGORICAL),
        ("flags", "passthrough", FLAGS),
    ])


def make_reg(model, scale=True):
    return Pipeline([
        ("prep", make_preprocess(REAL_NUMERIC + NOISE, scale=scale)),
        ("model", model),
    ])


def l1_logistic(C):
    # An L1 (Lasso-style) logistic regression. scikit-learn 1.8 renamed the
    # switch from penalty="l1" to l1_ratio=1, so we ask for whichever your
    # version understands.
    major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(C=C, l1_ratio=1, solver="liblinear", max_iter=5000,
                                  random_state=0)
    return LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=5000,
                                  random_state=0)


def fit_report(model, small, test):
    model.fit(small, small[DURATION])
    return (r2(rmse(small[DURATION], model.predict(small))),
            r2(rmse(test[DURATION], model.predict(test))))


def coefs(model):
    names = model.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    return pd.Series(model.named_steps["model"].coef_, index=names)


# ---------------------------------------------------------------------------
# 1. Too many columns, too few rides
# ---------------------------------------------------------------------------

def section_overfit(small, test):
    heading("1. Too many columns, too few rides")
    real = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), REAL_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                  sparse_output=False), CATEGORICAL),
            ("flags", "passthrough", FLAGS),
        ])),
        ("model", LinearRegression()),
    ])
    show("training rides / columns after encoding",
         (len(small), len(make_reg(LinearRegression()).fit(small, small[DURATION])
                         .named_steps["prep"].get_feature_names_out())))
    rows = {
        "real columns only": fit_report(real, small, test),
        "real + 40 noise columns": fit_report(make_reg(LinearRegression()), small, test),
    }
    show("plain linear regression, RMSE in minutes",
         pd.DataFrame(rows, index=["train", "test"]).T)
    big = coefs(make_reg(LinearRegression()).fit(small, small[DURATION]))
    show("largest |coefficient| on a noise column", r2(big[NOISE].abs().max()))
    show("coefficient on distance_km (scaled)", r2(big["distance_km"]))
    print("  With 52 columns and only 60 rides, the model can explain almost")
    print("  anything in the training rows, noise included. Train error drops,")
    print("  test error balloons. Classic overfitting.")


# ---------------------------------------------------------------------------
# 2. Ridge: shrink every coefficient a little
# ---------------------------------------------------------------------------

def section_ridge(small, test):
    heading("2. Ridge: shrink every coefficient a little")
    rows = {}
    for alpha in [0, 0.1, 1, 10, 100, 1000]:
        model = make_reg(LinearRegression() if alpha == 0 else Ridge(alpha=alpha))
        tr, te = fit_report(model, small, test)
        c = coefs(model)
        rows[f"alpha={alpha}"] = {"train RMSE": tr, "test RMSE": te,
                                  "size of weights": r2(np.sqrt((c ** 2).sum())),
                                  "distance coef": r2(c["distance_km"])}
    show("Ridge on 60 rides, real + noise columns", pd.DataFrame(rows).T)
    print("  alpha is the price of a big coefficient. As it rises, the weights")
    print("  shrink, train error creeps up (more bias) and test error falls")
    print("  (less variance), until too much shrinking hurts both.")


# ---------------------------------------------------------------------------
# 3. Lasso: shrink some coefficients all the way to zero
# ---------------------------------------------------------------------------

def section_lasso(small, test):
    heading("3. Lasso: shrink some coefficients all the way to zero")
    rows = {}
    for alpha in [0.01, 0.1, 0.3, 1, 3]:
        model = make_reg(Lasso(alpha=alpha, max_iter=50_000))
        tr, te = fit_report(model, small, test)
        c = coefs(model)
        rows[f"alpha={alpha}"] = {"train RMSE": tr, "test RMSE": te,
                                  "real kept": int((c.drop(NOISE) != 0).sum()),
                                  "noise kept": int((c[NOISE] != 0).sum())}
    show("Lasso on 60 rides (12 real columns, 40 noise)", pd.DataFrame(rows).T.astype(
        {"real kept": int, "noise kept": int}))
    model = make_reg(Lasso(alpha=0.3, max_iter=50_000)).fit(small, small[DURATION])
    c = coefs(model)
    kept = c[c != 0].round(2).sort_values(key=abs, ascending=False)
    show("alpha=0.3, the columns that survived", kept)
    print("  Ridge makes weights small; Lasso makes many of them exactly zero.")
    print("  That's feature selection for free, with one caution: Lasso keeps")
    print("  what helps prediction, which isn't the same as what causes it.")


# ---------------------------------------------------------------------------
# 4. Scale first, then let cross-validation pick alpha
# ---------------------------------------------------------------------------

def section_scale_and_cv(small, test):
    heading("4. Scale first, then let cross-validation pick alpha")
    metres = small.copy()
    metres["distance_km"] = metres["distance_km"] * 1000  # same data, new unit
    rows = {}
    for label, data in [("km", small), ("metres", metres)]:
        for scale in [False, True]:
            model = make_reg(Lasso(alpha=1, max_iter=50_000), scale=scale)
            model.fit(data, data[DURATION])
            c = coefs(model)
            coef = float(c["distance_km"])
            # turn the coefficient back into "minutes per km" so rows compare
            if scale:
                coef /= data["distance_km"].std(ddof=0)
            if label == "metres":
                coef *= 1000
            rows[f"{label}, {'scaled' if scale else 'not scaled'}"] = {
                "minutes per km": r2(coef), "columns kept": int((c != 0).sum())}
    show("Lasso alpha=1, distance in km vs metres", pd.DataFrame(rows).T.astype(
        {"columns kept": int}))
    print("  Without scaling, the penalty depends on the units you happened to")
    print("  pick. Scale inside the pipeline and the units stop mattering.")

    alphas = np.logspace(-2, 3, 30)
    ridge = make_reg(RidgeCV(alphas=alphas)).fit(small, small[DURATION])
    lasso = make_reg(LassoCV(alphas=np.logspace(-2, 1, 30), cv=5, max_iter=50_000,
                             random_state=0)).fit(small, small[DURATION])
    lc = coefs(lasso)
    show("RidgeCV picked alpha", r2(ridge.named_steps["model"].alpha_))
    show("LassoCV picked alpha", r3(lasso.named_steps["model"].alpha_))
    show("test RMSE: RidgeCV / LassoCV",
         (r2(rmse(test[DURATION], ridge.predict(test))),
          r2(rmse(test[DURATION], lasso.predict(test)))))
    show("LassoCV noise columns kept (of 40)", int((lc[NOISE] != 0).sum()))
    print("  RidgeCV and LassoCV try a list of alphas with cross-validation on")
    print("  the training rides and keep the best. The test set stays untouched.")


# ---------------------------------------------------------------------------
# 5. Friends: Elastic Net, and the C in logistic regression
# ---------------------------------------------------------------------------

def section_friends(small, train, test):
    heading("5. Friends: Elastic Net, and the C in logistic regression")
    rows = {}
    for ratio in [0.1, 0.5, 0.9]:
        model = make_reg(ElasticNet(alpha=0.3, l1_ratio=ratio, max_iter=50_000))
        tr, te = fit_report(model, small, test)
        rows[f"l1_ratio={ratio}"] = {"test RMSE": te,
                                     "columns kept": int((coefs(model) != 0).sum())}
    show("ElasticNet alpha=0.3 on 60 rides", pd.DataFrame(rows).T.astype(
        {"columns kept": int}))

    def logit(C, l1=False):
        return Pipeline([
            ("prep", ColumnTransformer([
                ("num", StandardScaler(), CLS_NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                ("flags", "passthrough", ["weekend"]),
            ])),
            ("model", l1_logistic(C) if l1 else LogisticRegression(C=C, max_iter=5000)),
        ])

    rows = {}
    for C in [0.001, 0.01, 0.1, 1, 100]:
        l2 = cross_val_score(logit(C), train, train[TARGET], cv=FOLDS).mean()
        l1_model = logit(C, l1=True).fit(train, train[TARGET])
        l1 = cross_val_score(logit(C, l1=True), train, train[TARGET], cv=FOLDS).mean()
        kept = int((l1_model.named_steps["model"].coef_ != 0).sum())
        rows[f"C={C}"] = {"L2 CV mean": r3(l2), "L1 CV mean": r3(l1), "L1 columns kept": kept}
    show("casual vs member, all 940 training rides", pd.DataFrame(rows).T.astype(
        {"L1 columns kept": int}))
    print("  Logistic regression was regularised all along. C is the inverse")
    print("  of alpha: small C means a strong penalty. With 940 rides and only")
    print("  13 columns there's little to rescue, so the default C=1 is fine.")


def main():
    print("=" * 66)
    print("  Lesson 041: Regularisation: Ridge, Lasso and friends")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    train, test = split(rides)
    small = train.sample(SMALL_N, random_state=0)
    section_overfit(small, test)
    section_ridge(small, test)
    section_lasso(small, test)
    section_scale_and_cv(small, test)
    section_friends(small, train, test)
    print()
    print("Regularisation is a price on confidence. Make the model pay for")
    print("every big weight, let cross-validation set the price, and it stops")
    print("chasing noise.")
    print()


if __name__ == "__main__":
    main()
