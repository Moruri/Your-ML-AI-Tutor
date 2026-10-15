"""
Lesson 043 - Imbalanced data and threshold tuning

Most classifiers quietly assume both classes show up about as often. When
one class is rare, that assumption breaks: a model that always predicts the
common class gets a shiny accuracy and still never finds the rare one.
Today we make a deliberately rare label out of the September rides (long
rides of 30 minutes or more, about one in thirteen), leave distance out of
the features so the rare class is hard to spot, watch accuracy lie, then
fix it three ways: reweighting, resampling, and moving the decision
threshold on purpose.

Same cleaned September bike-share rides as lessons 029 to 042, in
rides_clean.csv next to this file. We keep the same train/test split style
as the classification lessons, stratified on the new rare label.

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

NUMERIC = ["temp_c", "rain_mm"]
CATEGORICAL = ["start_station", "bike_type"]
FLAGS = ["weekend", "casual"]
# Deliberately rare: rides of 30 minutes or more (~8% of the data).
# Story: flag a "long leisure" ride so the app can nudge the rider about
# water, and so ops can spot bikes that need a longer rest.
TARGET = "long"
LONG_MINUTES = 30

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


def r3(x):
    return round(float(x), 3)


def load_rides(path):
    rides = pd.read_csv(path, parse_dates=["started_at"])
    rides["weekend"] = (rides["started_at"].dt.dayofweek >= 5).astype(int)
    rides["casual"] = (rides["rider_type"] == "casual").astype(int)
    rides[TARGET] = (rides["duration_min"] >= LONG_MINUTES).astype(int)
    return rides


def split(rides):
    return train_test_split(
        rides, test_size=0.2, random_state=0, stratify=rides[TARGET],
    )


def make_pipe(class_weight=None):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                  sparse_output=False), CATEGORICAL),
            ("flags", "passthrough", FLAGS),
        ])),
        ("model", LogisticRegression(
            max_iter=5000, random_state=0, class_weight=class_weight,
        )),
    ])


def metrics(y_true, y_pred):
    return {
        "accuracy": r3(accuracy_score(y_true, y_pred)),
        "precision": r3(precision_score(y_true, y_pred, zero_division=0)),
        "recall": r3(recall_score(y_true, y_pred, zero_division=0)),
        "F1": r3(f1_score(y_true, y_pred, zero_division=0)),
        "found / total long": f"{int(y_pred[y_true == 1].sum())}/{int(y_true.sum())}",
    }


def oversample(train, rng):
    """Duplicate rare rows with replacement until classes match (sklearn-only)."""
    majority = train[train[TARGET] == 0]
    minority = train[train[TARGET] == 1]
    n_extra = len(majority) - len(minority)
    extra = minority.sample(n_extra, replace=True, random_state=rng)
    balanced = pd.concat([majority, minority, extra], axis=0)
    return balanced.sample(frac=1, random_state=rng).reset_index(drop=True)


def undersample(train, rng):
    """Keep all rare rows; draw an equal number of common ones."""
    majority = train[train[TARGET] == 0]
    minority = train[train[TARGET] == 1]
    kept = majority.sample(len(minority), random_state=rng)
    balanced = pd.concat([kept, minority], axis=0)
    return balanced.sample(frac=1, random_state=rng).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. A rare class, and an accuracy that lies
# ---------------------------------------------------------------------------

def section_rare(train, test):
    heading("1. A rare class, and an accuracy that lies")
    show("long rides (>= 30 min) in the full data",
         f"{int(train[TARGET].sum() + test[TARGET].sum())}/"
         f"{len(train) + len(test)} "
         f"({r3((train[TARGET].sum() + test[TARGET].sum()) / (len(train) + len(test)))})")
    show("train / test prevalence",
         (r3(train[TARGET].mean()), r3(test[TARGET].mean())))

    always_common = np.zeros(len(test), dtype=int)
    show("always-predict-common on the test set", metrics(test[TARGET], always_common))

    model = make_pipe().fit(train, train[TARGET])
    pred = model.predict(test)
    show("default logistic regression at threshold 0.5", metrics(test[TARGET], pred))
    show("confusion matrix [[TN FP] [FN TP]]",
         confusion_matrix(test[TARGET], pred))
    print("  Accuracy around 0.92 looks fine until you notice the model")
    print("  found none of the long rides. We left distance out on purpose")
    print("  so the rare class is hard to spot — when that is the class you")
    print("  care about, accuracy is the wrong scoreboard (lesson 034).")


# ---------------------------------------------------------------------------
# 2. Reweighting: make the rare class cost more
# ---------------------------------------------------------------------------

def section_reweight(train, test):
    heading("2. Reweighting: make the rare class cost more")
    plain = make_pipe().fit(train, train[TARGET])
    balanced = make_pipe(class_weight="balanced").fit(train, train[TARGET])
    rows = {
        "default (equal weight)": metrics(test[TARGET], plain.predict(test)),
        'class_weight="balanced"': metrics(test[TARGET], balanced.predict(test)),
    }
    show("same model, two weight schemes", pd.DataFrame(rows).T)
    # How balanced sets the weights: n_samples / (n_classes * count)
    n = len(train)
    w0 = n / (2 * (train[TARGET] == 0).sum())
    w1 = n / (2 * (train[TARGET] == 1).sum())
    show("weights it used (common / rare)", (r3(w0), r3(w1)))
    print("  class_weight=\"balanced\" multiplies each class's loss by the")
    print("  inverse of how often it appears. The rare class now hurts about")
    print(f"  {r3(w1 / w0)}x as much when the model gets it wrong, so the")
    print("  decision boundary moves towards catching it.")


# ---------------------------------------------------------------------------
# 3. Resampling: change the training mix
# ---------------------------------------------------------------------------

def section_resample(train, test):
    heading("3. Resampling: change the training mix")
    over = oversample(train, rng=43)
    under = undersample(train, rng=43)
    show("training rows after oversample (common / rare)",
         (int((over[TARGET] == 0).sum()), int((over[TARGET] == 1).sum())))
    show("training rows after undersample (common / rare)",
         (int((under[TARGET] == 0).sum()), int((under[TARGET] == 1).sum())))

    rows = {
        "default on original": metrics(
            test[TARGET], make_pipe().fit(train, train[TARGET]).predict(test)),
        "oversample then fit": metrics(
            test[TARGET], make_pipe().fit(over, over[TARGET]).predict(test)),
        "undersample then fit": metrics(
            test[TARGET], make_pipe().fit(under, under[TARGET]).predict(test)),
        'class_weight="balanced"': metrics(
            test[TARGET],
            make_pipe(class_weight="balanced").fit(train, train[TARGET]).predict(test)),
    }
    show("four ways to train, same test set", pd.DataFrame(rows).T)
    print("  Oversampling duplicates rare rows until the classes match.")
    print("  Undersampling throws away common rows until they match. Both")
    print("  change the *training mix*; the test set stays untouched.")
    print("  Reweighting often gets you most of the way without inventing")
    print("  or discarding rides. Prefer it first; resample when you need to.")


# ---------------------------------------------------------------------------
# 4. Threshold tuning: 0.5 is not sacred
# ---------------------------------------------------------------------------

def section_threshold(train, test):
    heading("4. Threshold tuning: 0.5 is not sacred")
    # Fit on a further split of training rides so the threshold is chosen
    # without peeking at the test set (lesson 034's honesty rule).
    fit_part, tune_part = train_test_split(
        train, test_size=0.25, random_state=0, stratify=train[TARGET],
    )
    model = make_pipe(class_weight="balanced").fit(fit_part, fit_part[TARGET])
    tune_proba = model.predict_proba(tune_part)[:, 1]
    test_proba = model.predict_proba(test)[:, 1]

    rows = {}
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred = (tune_proba >= thr).astype(int)
        rows[f"thr={thr}"] = {
            "precision": r3(precision_score(tune_part[TARGET], pred, zero_division=0)),
            "recall": r3(recall_score(tune_part[TARGET], pred, zero_division=0)),
            "F1": r3(f1_score(tune_part[TARGET], pred, zero_division=0)),
        }
    show("threshold sweep on the tuning slice", pd.DataFrame(rows).T)

    precision, recall, thresholds = precision_recall_curve(
        tune_part[TARGET], tune_proba,
    )
    # precision/recall arrays are one longer than thresholds; F1 for each thr
    f1s = []
    for p, r in zip(precision[:-1], recall[:-1]):
        f1s.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    best_i = int(np.argmax(f1s))
    best_thr = float(thresholds[best_i])
    show("threshold that maximises F1 on the tuning slice", r3(best_thr))
    show("precision / recall / F1 at that threshold (tune)",
         (r3(precision[best_i]), r3(recall[best_i]), r3(f1s[best_i])))

    test_pred = (test_proba >= best_thr).astype(int)
    show("same threshold held out on the test set",
         metrics(test[TARGET], test_pred))
    show("confusion matrix on the test set",
         confusion_matrix(test[TARGET], test_pred))
    print("  predict_proba gives a score, not a decision. The threshold is")
    print("  yours to set: lower it to catch more of the rare class (higher")
    print("  recall, lower precision), raise it when false alarms are costly.")
    print("  Pick it on a tuning slice or with cross-validation — never on")
    print("  the test set — then lock it in and report once.")


def main():
    print("=" * 66)
    print("  Lesson 043: Imbalanced data and threshold tuning")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    train, test = split(rides)
    section_rare(train, test)
    section_reweight(train, test)
    section_resample(train, test)
    section_threshold(train, test)
    print()
    print("When one class is rare, accuracy will flatter you. Reweight or")
    print("resample so the rare class matters, then move the threshold to")
    print("match what a miss actually costs.")
    print()


if __name__ == "__main__":
    main()
