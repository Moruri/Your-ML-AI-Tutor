"""
Lesson 034 - Classification metrics that matter

Lesson 033 ended with a casual-or-member model that is right 79% of the time.
Is that good? It depends entirely on what a mistake costs. Today we open up
the confusion matrix, define precision, recall and F1, move the 0.5 threshold
to match the cost of each kind of mistake (choosing it honestly, without the
test set), and finish with the ROC curve and AUC.

Same cleaned September bike-share rides as lessons 029 to 033, in
rides_clean.csv next to this file, and the same train/test split as 033.

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
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

FEATURES = ["distance_km", "duration_min", "electric", "weekend"]
TARGET = "casual"

# The story: at the end of a casual rider's trip, the app shows a
# "become a member" offer. Showing it to someone who is already a member
# is a small annoyance. Missing a casual rider is a lost sign-up.
COST_FALSE_ALARM = 1   # offer shown to a member (false positive)
COST_MISS = 4          # casual rider never sees the offer (false negative)

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
    # identical to lesson 033, so the numbers line up
    return train_test_split(rides, test_size=0.2, random_state=0, stratify=rides[TARGET])


def cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return COST_FALSE_ALARM * fp + COST_MISS * fn


# ---------------------------------------------------------------------------
# 1. Accuracy hides the mistakes
# ---------------------------------------------------------------------------

def section_accuracy(model, test):
    heading("1. Accuracy hides the mistakes")
    pred = model.predict(test[FEATURES])
    show("model accuracy on test rides", round(float(accuracy_score(test[TARGET], pred)), 3))
    show("'always member' accuracy", round(float((test[TARGET] == 0).mean()), 3))
    show("casual riders in the test set", int(test[TARGET].sum()))
    show("of those, how many the model caught", int(((pred == 1) & (test[TARGET] == 1)).sum()))
    print("  79% sounds fine until you ask about the riders we care about.")
    print("  Accuracy treats every mistake as equal, and they rarely are.")
    return pred


# ---------------------------------------------------------------------------
# 2. The confusion matrix
# ---------------------------------------------------------------------------

def section_confusion(test, pred):
    heading("2. The confusion matrix")
    cm = confusion_matrix(test[TARGET], pred)
    table = pd.DataFrame(cm, index=["actual member", "actual casual"],
                         columns=["said member", "said casual"])
    show("rows are the truth, columns are the guess", table)
    tn, fp, fn, tp = cm.ravel()
    show("TN, FP, FN, TP", (int(tn), int(fp), int(fn), int(tp)))
    print("  Positive means 'casual', the thing we're hunting for. A false")
    print("  positive is a member we wrongly flagged; a false negative is a")
    print("  casual rider we missed.")
    return tn, fp, fn, tp


# ---------------------------------------------------------------------------
# 3. Precision, recall and F1
# ---------------------------------------------------------------------------

def section_precision_recall(test, pred, counts):
    heading("3. Precision, recall and F1")
    tn, fp, fn, tp = counts
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    show("precision = TP / (TP + FP), by hand", round(float(precision), 3))
    show("recall    = TP / (TP + FN), by hand", round(float(recall), 3))
    show("F1, the harmonic mean, by hand", round(float(f1), 3))
    same = (np.isclose(precision, precision_score(test[TARGET], pred))
            and np.isclose(recall, recall_score(test[TARGET], pred))
            and np.isclose(f1, f1_score(test[TARGET], pred)))
    show("same as scikit-learn?", bool(same))
    print("  Precision: when we say casual, how often are we right?")
    print("  Recall: of all the casual riders, how many did we find?")
    print("  This model is fairly precise but finds only about half of them.")


# ---------------------------------------------------------------------------
# 4. Moving the threshold, honestly
# ---------------------------------------------------------------------------

def section_threshold(model, train, test):
    heading("4. Moving the threshold, honestly")
    # out-of-fold probabilities: every training ride scored by a model
    # that never saw it, so we can choose without touching the test set
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = cross_val_predict(LogisticRegression(max_iter=1000), train[FEATURES],
                            train[TARGET], cv=folds, method="predict_proba")[:, 1]
    rows = {}
    for t in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        guess = (oof >= t).astype(int)
        rows[t] = {
            "precision": precision_score(train[TARGET], guess),
            "recall": recall_score(train[TARGET], guess),
            "cost": cost(train[TARGET], guess),
        }
    table = pd.DataFrame(rows).T.round(2)
    table["cost"] = table["cost"].astype(int)
    table.index.name = "threshold"
    show(f"training folds, a miss costs {COST_MISS}, a false alarm {COST_FALSE_ALARM}", table)
    best = float(table["cost"].idxmin())
    show("threshold with the lowest cost", best)
    rule = COST_FALSE_ALARM / (COST_FALSE_ALARM + COST_MISS)
    show(f"rule of thumb: {COST_FALSE_ALARM} / ({COST_FALSE_ALARM} + {COST_MISS})", rule)
    p_test = model.predict_proba(test[FEATURES])[:, 1]
    for t in [0.5, best]:
        guess = (p_test >= t).astype(int)
        show(f"test at {t}: precision, recall, cost", (
            round(float(precision_score(test[TARGET], guess)), 2),
            round(float(recall_score(test[TARGET], guess)), 2),
            int(cost(test[TARGET], guess)),
        ))
    print("  Lowering the threshold trades precision for recall. Which trade is")
    print("  right is a business question, not a maths one: here a miss hurts")
    print("  four times as much, so we accept more false alarms. When the")
    print("  probabilities are trustworthy, the cheapest cut-off lands near the")
    print("  rule of thumb, and here it does.")
    return best


# ---------------------------------------------------------------------------
# 5. ROC curve and AUC
# ---------------------------------------------------------------------------

def section_roc(model, test):
    heading("5. ROC curve and AUC")
    p = model.predict_proba(test[FEATURES])[:, 1]
    fpr, tpr, thresholds = roc_curve(test[TARGET], p)
    picks = [np.argmin(np.abs(thresholds - t)) for t in [0.7, 0.5, 0.3, 0.2, 0.1]]
    curve = pd.DataFrame({
        "threshold": thresholds[picks].round(2),
        "false pos rate": fpr[picks].round(2),
        "true pos rate": tpr[picks].round(2),
    })
    show("a few points on the ROC curve", curve.set_index("threshold"))
    auc = roc_auc_score(test[TARGET], p)
    show("ROC-AUC on the test rides", round(float(auc), 3))
    casual, member = p[test[TARGET] == 1], p[test[TARGET] == 0]
    pairs = (casual[:, None] > member[None, :]).mean() + 0.5 * (casual[:, None] == member[None, :]).mean()
    show("share of (casual, member) pairs ranked right", round(float(pairs), 3))
    print("  The ROC curve shows every threshold at once: how many casual riders")
    print("  you catch against how many members you wrongly flag. AUC sums it")
    print("  up as one number: pick a random casual rider and a random member,")
    print("  and it's the chance the model gives the casual rider the higher")
    print("  probability. 0.5 is a coin toss, 1.0 is perfect.")


# ---------------------------------------------------------------------------
# 6. The one-line report
# ---------------------------------------------------------------------------

def section_report(test, pred):
    heading("6. The one-line report")
    report = classification_report(test[TARGET], pred, target_names=["member", "casual"], digits=2)
    for line in report.rstrip().splitlines():
        print(f"      {line}")
    print("  Precision, recall and F1 for each class, plus support (how many")
    print("  rides of each). Read the row for the class you care about.")


def main():
    print("=" * 66)
    print("  Lesson 034: Classification metrics that matter")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    train, test = split(rides)
    model = LogisticRegression(max_iter=1000).fit(train[FEATURES], train[TARGET])
    pred = section_accuracy(model, test)
    counts = section_confusion(test, pred)
    section_precision_recall(test, pred, counts)
    section_threshold(model, train, test)
    section_roc(model, test)
    section_report(test, pred)
    print()
    print("Decide what a mistake costs, then pick the metric and threshold")
    print("that respect it. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
