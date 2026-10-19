"""
Lesson 048 - Mini-project: end-to-end tabular ML

Phase 2 ends the way Phase 1 did: with one project that uses everything.
We start from the CSV on disk and finish with a model that has been
cross-validated, tuned, tested once, explained, and written up in a page
someone else can read. Every step is something you've already done; the
skill today is doing them in the right order without cutting corners.

The question: when a ride ends, can we tell whether it was a casual rider
or a member? (Picture the app deciding who to show a "become a member"
offer to.) Same September bike-share rides as lessons 029 to 047, in
rides_clean.csv next to this file.

Run it with (inside the venv from lesson 013, after `pip install -r
requirements.txt`):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes one file, output/report.md, next to this file. Every
random step has a fixed seed, so your numbers will match the README. It
takes about ten seconds.
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.base import clone
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"
OUTPUT = HERE / "output"

# Everything we know once a ride has ended. rider_type is the answer, so it
# never goes in, and neither does anything computed from it.
NUMERIC = ["distance_km", "duration_min", "speed_kmh", "temp_c", "rain_mm",
           "hour", "weekend"]
CATEGORICAL = ["start_station", "bike_type"]
FEATURES = NUMERIC + CATEGORICAL

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")


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


# ---------------------------------------------------------------------------
# 1. The question, and the file as it arrives
# ---------------------------------------------------------------------------

def section_load(path):
    heading("1. The question, and the file as it arrives")
    raw = pd.read_csv(path)
    show("rows / columns", raw.shape)
    show("columns", list(raw.columns))
    show("missing values (all columns)", int(raw.isna().sum().sum()))
    show("duplicate ride_ids", int(raw["ride_id"].duplicated().sum()))
    show("rider_type counts", raw["rider_type"].value_counts())
    print("  Check before you trust: shape, gaps, duplicates, and how the")
    print("  label splits. About a third of rides are casual, so a model")
    print("  that always says 'member' is right 68% of the time and useless.")
    return raw


# ---------------------------------------------------------------------------
# 2. Features, and a test set locked away
# ---------------------------------------------------------------------------

def build_features(raw):
    rides = raw.copy()
    started = pd.to_datetime(rides["started_at"])
    rides["hour"] = started.dt.hour
    rides["weekend"] = (started.dt.dayofweek >= 5).astype(int)
    rides["speed_kmh"] = rides["distance_km"] / (rides["duration_min"] / 60)
    rides["casual"] = (rides["rider_type"] == "casual").astype(int)
    return rides


def section_split(raw):
    heading("2. Features, and a test set locked away")
    rides = build_features(raw)
    X, y = rides[FEATURES], rides["casual"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=48)
    show("numeric features", NUMERIC)
    show("categorical features", CATEGORICAL)
    show("train rows / test rows", (len(X_train), len(X_test)))
    show("casual share, train / test", (r3(y_train.mean()), r3(y_test.mean())))
    print("  Three new columns, all from things known when a ride ends. Then")
    print("  25% of rows go in a drawer. Everything until section 5 uses the")
    print("  other 75% only, so the test score stays an honest surprise.")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 3. A baseline and two honest contenders
# ---------------------------------------------------------------------------

def make_logistic():
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))])


def make_boosting():
    prep = ColumnTransformer([
        ("num", "passthrough", NUMERIC),   # trees don't need scaling
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("prep", prep),
                     ("model", HistGradientBoostingClassifier(random_state=0))])


def section_contenders(X_train, y_train):
    heading("3. A baseline and two honest contenders")
    rows = {}
    for name, model in [("always 'member'", DummyClassifier()),
                        ("logistic regression", make_logistic()),
                        ("gradient boosting", make_boosting())]:
        s = cross_validate(model, X_train, y_train, cv=CV,
                           scoring=["roc_auc", "balanced_accuracy"])
        rows[name] = {"auc_mean": r3(s["test_roc_auc"].mean()),
                      "auc_std": r3(s["test_roc_auc"].std()),
                      "bal_acc": r3(s["test_balanced_accuracy"].mean())}
    show("5-fold CV on the training rows, default settings", pd.DataFrame(rows).T)
    print("  The dummy sets the floor: AUC 0.5, balanced accuracy 0.5. Both")
    print("  real models clear it easily. Each is a full pipeline, so the")
    print("  scaling and one-hot encoding are refit inside every fold.")


# ---------------------------------------------------------------------------
# 4. Tune, choose, and set a threshold
# ---------------------------------------------------------------------------

def section_tune(X_train, y_train):
    heading("4. Tune, choose, and set a threshold")
    searches = {
        "logistic regression": GridSearchCV(
            make_logistic(), {"model__C": [0.01, 0.1, 1, 10]},
            cv=CV, scoring="roc_auc"),
        "gradient boosting": GridSearchCV(
            make_boosting(), {"model__learning_rate": [0.03, 0.1],
                              "model__max_depth": [2, 3, None]},
            cv=CV, scoring="roc_auc"),
    }
    rows = {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        best = search.best_index_
        rows[name] = {
            "best_settings": {k.replace("model__", ""): v
                              for k, v in search.best_params_.items()},
            "auc_mean": r3(search.cv_results_["mean_test_score"][best]),
            "auc_std": r3(search.cv_results_["std_test_score"][best]),
        }
    table = pd.DataFrame(rows).T
    show("best of each grid (5-fold CV AUC)", table)

    leader = table["auc_mean"].astype(float).idxmax()
    gap = r3(table.loc[leader, "auc_mean"] - table.loc["logistic regression", "auc_mean"])
    one_std = r3(table.loc[leader, "auc_std"])
    choice = "logistic regression" if gap <= one_std else leader
    show("CV leader / gap to logistic / leader's std", (leader, gap, one_std))
    show("chosen model", choice)
    print("  Rule decided *before* looking: if the simpler model is within one")
    print("  standard deviation of the best, take the simpler one. A gap")
    print("  smaller than the fold-to-fold wobble isn't a real difference.")

    # Out-of-fold probabilities: each training ride scored by a model that
    # never saw it. Pick the threshold on those, still without the test set.
    model = searches[choice].best_estimator_
    proba = cross_val_predict(clone(model), X_train, y_train, cv=CV,
                              method="predict_proba")[:, 1]
    rows = {}
    for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        pred = (proba >= t).astype(int)
        rows[f"{t:.2f}"] = {"bal_acc": r3(balanced_accuracy_score(y_train, pred)),
                            "precision": r3(precision_score(y_train, pred)),
                            "recall": r3(recall_score(y_train, pred))}
    thresholds = pd.DataFrame(rows).T
    threshold = float(thresholds["bal_acc"].idxmax())
    show("out-of-fold scores by threshold", thresholds)
    show("threshold with the best balanced accuracy", threshold)
    print("  Strong regularisation (small C) pulls probabilities towards the")
    print("  base rate, so few rides clear 0.5. Lesson 043's move, done on")
    print("  out-of-fold predictions so the test set stays in the drawer.")
    return model, choice, table, threshold


# ---------------------------------------------------------------------------
# 5. One look at the test set
# ---------------------------------------------------------------------------

def section_test(model, threshold, X_test, y_test):
    heading("5. One look at the test set")
    proba = model.predict_proba(X_test)[:, 1]
    show("test ROC AUC", r3(roc_auc_score(y_test, proba)))
    rows = {}
    for t in [0.5, threshold]:
        pred = (proba >= t).astype(int)
        rows[f"threshold {t:.2f}"] = {
            "bal_acc": r3(balanced_accuracy_score(y_test, pred)),
            "precision": r3(precision_score(y_test, pred)),
            "recall": r3(recall_score(y_test, pred)),
        }
    show("test scores, default vs chosen threshold", pd.DataFrame(rows).T)
    pred = (proba >= threshold).astype(int)
    cm = pd.DataFrame(confusion_matrix(y_test, pred),
                      index=["is member", "is casual"],
                      columns=["said member", "said casual"])
    show(f"confusion matrix at {threshold:.2f}", cm)
    print("  The drawer opens once. The test AUC sits right next to the CV")
    print("  estimate, which is what honest evaluation looks like. At 0.5 the")
    print("  model would miss nearly two in three casual riders; at the")
    print("  chosen threshold it finds two in three, and two in three of its")
    print("  'casual' calls are right. The out-of-fold numbers predicted that.")
    result = dict(rows[f"threshold {threshold:.2f}"])
    result["auc"] = r3(roc_auc_score(y_test, proba))
    result["threshold"] = threshold
    return result


# ---------------------------------------------------------------------------
# 6. Explain the model
# ---------------------------------------------------------------------------

def section_explain(model, X_test, y_test):
    heading("6. Explain the model")
    res = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                 n_repeats=20, random_state=0)
    importance = pd.DataFrame({"mean_drop": res.importances_mean,
                               "std": res.importances_std},
                              index=FEATURES).sort_values("mean_drop", ascending=False)
    show("permutation importance on test (AUC drop)", importance)

    names = model.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(model.named_steps["model"].coef_[0], index=names)
    coefs.index = [n.split("__", 1)[1] for n in coefs.index]
    show("logistic coefficients, numeric (per 1 std)", coefs[NUMERIC].sort_values(ascending=False))
    print("  Shuffling works on whole raw columns, even through the pipeline,")
    print("  so 'start_station' counts as one column, not six. Duration and")
    print("  distance carry the model, weekend is a clear third, and the")
    print("  rest barely move the score.")
    print("  The coefficients agree on direction: longer and weekend push")
    print("  towards casual. Distance and duration are twins (lesson 047),")
    print("  so don't read their individual sizes too literally.")
    return importance, coefs


# ---------------------------------------------------------------------------
# 7. The write-up
# ---------------------------------------------------------------------------

def write_report(n_rows, choice, table, result, importance):
    top = [f"`{name}`" for name in importance.index[:3]]
    paragraphs = [
        "**Question.** When a ride ends, can we tell whether the rider was "
        "casual or a member, to decide who sees a membership offer?",
        f"**Data.** {n_rows:,} September rides from `rides_clean.csv`, about "
        "a third casual, with no missing values or duplicate ride ids. "
        "Features: distance, duration, average speed, temperature, rain, "
        "start hour, weekend flag, start station and bike type.",
        "**Method.** A 75/25 stratified split, with the test set untouched "
        "until the end. Logistic regression and gradient boosting, each a "
        "full pipeline, tuned by 5-fold cross-validated grid search on ROC "
        "AUC. Threshold chosen on out-of-fold predictions.",
    ]
    model_table = ["| Model | Best settings | CV AUC |",
                   "|-------|---------------|--------|"]
    for name, row in table.iterrows():
        settings = ", ".join(f"{k}={v}" for k, v in row["best_settings"].items())
        model_table.append(f"| {name} | {settings} | "
                           f"{row['auc_mean']:.3f} +/- {row['auc_std']:.3f} |")
    after = [
        f"**Choice.** {choice.capitalize()}: within one standard deviation "
        "of the best CV score, and simpler to run and to explain.",
        f"**Test result.** AUC {result['auc']:.3f}. At a threshold of "
        f"{result['threshold']:.2f}, balanced accuracy is "
        f"{result['bal_acc']:.3f}: {result['precision']:.0%} of rides "
        f"flagged casual are casual, and it finds {result['recall']:.0%} "
        "of casual riders.",
        f"**What drives it.** {', '.join(top[:-1])} and {top[-1]} "
        "(permutation importance on test data). Longer rides and weekends "
        "point towards casual riders; station, bike type and weather add "
        "almost nothing.",
        "**Limits.** One month, one city, about 1,200 rides. A random split, "
        "not a time-based one, so it may flatter the model a little. The "
        "importances describe this model, not why people choose a "
        "membership.",
        "**Next steps.** Set the threshold from what an offer costs, test on "
        "October rides, and check the scores station by station.",
    ]
    lines = ["# Casual or member? A one-page model report", ""]
    for text in paragraphs:
        lines += textwrap.wrap(text, width=70) + [""]
    lines += model_table + [""]
    for text in after:
        lines += textwrap.wrap(text, width=70) + [""]
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path, lines


def section_report(n_rows, choice, table, result, importance):
    heading("7. The write-up")
    path, lines = write_report(n_rows, choice, table, result, importance)
    show("report written to", str(path.relative_to(HERE)))
    print()
    for line in lines:
        print(f"  | {line}".rstrip())


def main():
    print("=" * 66)
    print("  Lesson 048: Mini-project: end-to-end tabular ML")
    print("=" * 66)
    raw = section_load(RIDES_CSV)
    X_train, X_test, y_train, y_test = section_split(raw)
    section_contenders(X_train, y_train)
    model, choice, table, threshold = section_tune(X_train, y_train)
    result = section_test(model, threshold, X_test, y_test)
    importance, _ = section_explain(model, X_test, y_test)
    section_report(len(raw), choice, table, result, importance)
    print()
    print("Check the data, lock away a test set, beat a baseline with")
    print("cross-validation, tune a little, choose by a rule you set in")
    print("advance, test once, explain, and write it down for someone else.")
    print()


if __name__ == "__main__":
    main()
