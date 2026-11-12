"""
Lesson 083 - Capstone, part 2: build and evaluate

Lesson 082 chose the project, wrote the brief, sealed a test set and measured
the bar to beat. Today we build: one pipeline from raw text to a team name,
a few honest experiments on the dev set only, an "ask a person" threshold
chosen to fit the brief, a check that the win is bigger than the wobble, and
then, once, the sealed test set. After the score comes the part that teaches
you the most: looking at what it got wrong.

The worked example is still "route each support message to the right team",
using feedback.csv (the same 800 messages, copied next to this script).

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

No downloads and nothing new to install. The script rebuilds the capstone
folder in `output/capstone/` next to itself (ignored by git), adds a run log,
the chosen model and its record. The folder is wiped at the start of every
run. It takes about ten seconds.
"""

import hashlib
import json
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
if OUT.exists():
    shutil.rmtree(OUT)
PROJECT = OUT / "capstone"
(PROJECT / "data" / "raw").mkdir(parents=True)
TOPICS = ["app", "bike", "payment", "station"]
MAX_TO_PERSON = 0.15          # from the brief
SEED = 82


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


# The same helpers as lesson 082, so the split and the baseline are identical.
KEYWORDS = {
    "station": ["dock", "station", "rack", "lit"],
    "payment": ["charge", "charged", "m-pesa", "refund", "pay", "receipt", "fee"],
    "app": ["app", "login", "screen", "crash", "map", "notification", "update"],
    "bike": ["brake", "chain", "tyre", "tire", "seat", "bell", "pedal", "gear", "battery"],
}


def keyword_hits(text):
    words = re.findall(r"[a-z\-]+", text.lower())
    return {t: sum(w.startswith(k) for w in words for k in ks) for t, ks in KEYWORDS.items()}


def rule(text):
    h = keyword_hits(text)
    best = max(h, key=lambda t: (h[t], t == "bike"))
    return best if h[best] else "bike"


def normalise(text):
    return " ".join(re.findall(r"[a-z0-9\-]+", text.lower()))


# ---------------------------------------------------------------------------
heading("1. Pick up exactly where lesson 082 stopped")

raw = PROJECT / "data" / "raw" / "feedback.csv"
shutil.copyfile(HERE / "feedback.csv", raw)
fingerprint = hashlib.sha256(raw.read_bytes()).hexdigest()[:12]
assert fingerprint == "79a3efa5db0e", "feedback.csv is not the snapshot lesson 082 sealed"
df = pd.read_csv(raw)
df["norm"] = df["text"].map(normalise)
df["n_topics_hit"] = [sum(v > 0 for v in keyword_hits(t).values()) for t in df["text"]]
groups = df["norm"].factorize()[0]
dev_idx, test_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df, df["topic"], groups))
dev, test = df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
dev_groups = groups[dev_idx]
show("snapshot fingerprint", fingerprint)
show("dev rows, test rows (test stays shut)", (len(dev), len(test)))
show("first three sealed test ids", sorted(test["feedback_id"])[:3])

# ---------------------------------------------------------------------------
heading("2. One pipeline, from raw text to a team")


def build(kind, C=10.0):
    """Every candidate is one sklearn Pipeline: cleaning and features live inside it."""
    words = TfidfVectorizer(preprocessor=normalise, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    chars = TfidfVectorizer(preprocessor=normalise, analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    if kind == "nb_words":
        return Pipeline([("features", CountVectorizer(preprocessor=normalise, min_df=2)), ("model", MultinomialNB())])
    if kind == "lr_words":
        return Pipeline([("features", words), ("model", LogisticRegression(C=C, max_iter=2000))])
    if kind == "lr_words_chars":
        return Pipeline([("features", FeatureUnion([("words", words), ("chars", chars)])),
                         ("model", LogisticRegression(C=C, max_iter=2000))])
    raise ValueError(kind)


demo = build("lr_words_chars").fit(dev["text"], dev["topic"])
example = "The M-Pesa charge went through twice, can I get a refund?"
show("steps", [name for name, _ in demo.steps])
show("features the model sees", demo.named_steps["features"].transform([example]).shape[1])
show("example message", example)
show("routed to", demo.predict([example])[0])

# ---------------------------------------------------------------------------
heading("3. Experiments on dev only, every run logged")

cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
FOLDS = list(cv.split(dev, dev["topic"], dev_groups))
log_path = PROJECT / "runs.jsonl"


def cross_validate(kind, C=10.0):
    """Out-of-fold predictions on dev, so every dev message is scored by a model that never saw it."""
    pred = np.empty(len(dev), dtype=object)
    proba = np.zeros((len(dev), len(TOPICS)))
    for tr, va in FOLDS:
        if kind == "rules":
            pred[va] = dev["text"].iloc[va].map(rule).to_numpy()
            continue
        m = build(kind, C).fit(dev["text"].iloc[tr], dev["topic"].iloc[tr])
        proba[va] = m.predict_proba(dev["text"].iloc[va])
        pred[va] = np.array(m.classes_)[proba[va].argmax(axis=1)]
    right = pred == dev["topic"].to_numpy()
    per_fold = [float(right[va].mean()) for _, va in FOLDS]
    return pred, proba, right, per_fold


CANDIDATES = [("rules", None), ("nb_words", None), ("lr_words", 10.0),
              ("lr_words_chars", 1.0), ("lr_words_chars", 10.0), ("lr_words_chars", 100.0)]
results = {}
print()
print("  run  candidate              C      dev accuracy  fold spread")
for i, (kind, C) in enumerate(CANDIDATES, 1):
    pred, proba, right, per_fold = cross_validate(kind, C)
    name = kind if C is None else f"{kind} C={C:g}"
    results[name] = {"kind": kind, "C": C, "pred": pred, "proba": proba, "right": right}
    run = {"run": i, "candidate": kind, "C": C, "data": fingerprint, "seed": SEED,
           "dev_accuracy": round(float(right.mean()), 3), "fold_min_max": [round(min(per_fold), 3), round(max(per_fold), 3)]}
    with log_path.open("a") as f:
        f.write(json.dumps(run) + "\n")
    print(f"  {i:>3}  {kind:<20} {'-' if C is None else f'{C:g}':>5}  {right.mean():>12.3f}  "
          f"{min(per_fold):.3f}-{max(per_fold):.3f}")
model_names = [n for n in results if n != "rules"]
best = max(model_names, key=lambda n: results[n]["right"].mean())
print()
show("chosen on dev", best)
show("runs in runs.jsonl", sum(1 for _ in log_path.open()))

# ---------------------------------------------------------------------------
heading("4. The 'ask a person' threshold, chosen to fit the brief")


def route_scores(proba, truth, threshold):
    """Below the threshold, a person routes it (right, but costs a person's time)."""
    conf = proba.max(axis=1)
    pred = np.array(TOPICS)[proba.argmax(axis=1)]
    to_person = conf < threshold
    auto_right = (~to_person) & (pred == truth)
    return {"right_first_time": float((auto_right | to_person).mean()),
            "to_person": float(to_person.mean()),
            "misrouted": float(((~to_person) & (pred != truth)).mean()),
            "auto_accuracy": float(auto_right.sum() / max(1, (~to_person).sum()))}


B = results[best]
truth_dev = dev["topic"].to_numpy()
print()
print("  threshold  right first time  to a person  misrouted")
table = []
for t in np.round(np.arange(0.30, 0.80, 0.05), 2):
    s = route_scores(B["proba"], truth_dev, t)
    table.append((t, s))
    print(f"  {t:>9.2f}  {s['right_first_time']:>16.3f}  {s['to_person']:>11.3f}  {s['misrouted']:>9.3f}")
allowed = [(t, s) for t, s in table if s["to_person"] <= MAX_TO_PERSON]
THRESHOLD, dev_scores = max(allowed, key=lambda ts: ts[1]["right_first_time"])
print()
show("threshold (highest score within 15% to a person)", float(THRESHOLD))
show("dev: right first time, to a person", (round(dev_scores["right_first_time"], 3), round(dev_scores["to_person"], 3)))

# ---------------------------------------------------------------------------
heading("5. Is the win bigger than the wobble?")

rules_right = results["rules"]["right"]
conf_dev = B["proba"].max(axis=1)
pred_dev = np.array(TOPICS)[B["proba"].argmax(axis=1)]
model_rft = (conf_dev < THRESHOLD) | (pred_dev == truth_dev)
rng = np.random.default_rng(SEED)
gains = []
for _ in range(2000):                       # paired bootstrap: same messages for both systems
    i = rng.integers(0, len(dev), len(dev))
    gains.append(model_rft[i].mean() - rules_right[i].mean())
lo, hi = np.percentile(gains, [2.5, 97.5])
show("rules, right first time (dev)", round(float(rules_right.mean()), 3))
show("model + threshold, right first time (dev)", round(float(model_rft.mean()), 3))
show("gain, 95% interval", (round(float(np.mean(gains)), 3), round(float(lo), 3), round(float(hi), 3)))
show("share of resamples where the model loses", round(float(np.mean(np.array(gains) <= 0)), 4))

# ---------------------------------------------------------------------------
heading("6. Open the sealed test set, once")

final = build(B["kind"], B["C"]).fit(dev["text"], dev["topic"])
assert list(final.classes_) == TOPICS
truth_test = test["topic"].to_numpy()
test_scores = route_scores(final.predict_proba(test["text"]), truth_test, THRESHOLD)
rules_test = float((test["text"].map(rule) == test["topic"]).mean())
show("test: rules, right first time", round(rules_test, 3))
for k, v in test_scores.items():
    show(f"test: model, {k.replace('_', ' ')}", round(v, 3))
with log_path.open("a") as f:
    f.write(json.dumps({"run": "test", "candidate": best, "threshold": float(THRESHOLD), "data": fingerprint,
                        **{f"test_{k}": round(v, 3) for k, v in test_scores.items()}}) + "\n")

# ---------------------------------------------------------------------------
heading("7. The second exam: messages the way riders write them")

inbox = pd.read_csv(HERE / "inbox_sample.csv")
proba_in = final.predict_proba(inbox["text"])
truth_in = inbox["topic"].to_numpy()
inbox["pred"] = np.array(TOPICS)[proba_in.argmax(axis=1)]
inbox["conf"] = proba_in.max(axis=1).round(2)
inbox["to_person"] = inbox["conf"] < THRESHOLD
inbox["rule"] = inbox["text"].map(rule)
inbox_scores = route_scores(proba_in, truth_in, THRESHOLD)
rules_in = (inbox["rule"] == inbox["topic"]).to_numpy()
model_in = (inbox["to_person"] | (inbox["pred"] == inbox["topic"])).to_numpy()
gains = []
for _ in range(2000):
    i = rng.integers(0, len(inbox), len(inbox))
    gains.append(model_in[i].mean() - rules_in[i].mean())
show("inbox sample: messages, per topic", (len(inbox), int(inbox["topic"].value_counts().iloc[0])))
show("inbox: rules, right first time", round(float(rules_in.mean()), 3))
for k, v in inbox_scores.items():
    show(f"inbox: model, {k.replace('_', ' ')}", round(v, 3))
lo, hi = np.percentile(gains, [2.5, 97.5])
show("inbox: gain over rules, 95% interval", (round(float(np.mean(gains)), 3), round(float(lo), 3), round(float(hi), 3)))
print()
print("  for understanding only, never for choosing (this is still an exam):")
print("  threshold  right first time  to a person  misrouted")
for t in (0.4, 0.5, 0.6, 0.7):
    s = route_scores(proba_in, truth_in, t)
    print(f"  {t:>9.2f}  {s['right_first_time']:>16.3f}  {s['to_person']:>11.3f}  {s['misrouted']:>9.3f}")
print()
show("meets the brief on test (<= 15% to a person)", test_scores["to_person"] <= MAX_TO_PERSON)
show("meets the brief on the inbox sample", inbox_scores["to_person"] <= MAX_TO_PERSON)
with log_path.open("a") as f:
    f.write(json.dumps({"run": "inbox", "candidate": best, "threshold": float(THRESHOLD),
                        **{f"inbox_{k}": round(v, 3) for k, v in inbox_scores.items()},
                        "inbox_rules_right_first_time": round(float(rules_in.mean()), 3)}) + "\n")

# ---------------------------------------------------------------------------
heading("8. Error analysis: where it goes wrong, and why")

cm = confusion_matrix(truth_in, inbox["pred"], labels=TOPICS)
print()
print("  true \\ top guess  " + "".join(f"{t:>9}" for t in TOPICS))
for t, row in zip(TOPICS, cm):
    print(f"  {t:<17}  " + "".join(f"{n:>9}" for n in row))
print()
inbox["outcome"] = np.where(inbox["to_person"], "to person",
                            np.where(inbox["pred"] == inbox["topic"], "right", "misrouted"))
by_kind = pd.crosstab(inbox["kind"], inbox["outcome"]).reindex(columns=["right", "to person", "misrouted"], fill_value=0)
by_kind = by_kind.reindex(["plain", "typos", "Swahili/Sheng", "no keyword", "two topics", "sarcasm"])
print("  kind of message     right  to person  misrouted   (8 of each)")
for kind, r in by_kind.iterrows():
    print(f"  {kind:<18} {r['right']:>6}  {r['to person']:>9}  {r['misrouted']:>9}")
print()
feats = final.named_steps["features"]
words_step = feats.transformer_list[0][1] if hasattr(feats, "transformer_list") else feats
vocab = set(words_step.vocabulary_)
unknown = lambda t: np.mean([w not in vocab for w in normalise(t).split()])
show("words the model never saw: dev, inbox", (round(float(dev["text"].map(unknown).mean()), 2),
                                                       round(float(inbox["text"].map(unknown).mean()), 2)))
show("average confidence: test, inbox", (round(float(final.predict_proba(test["text"]).max(axis=1).mean()), 2),
                                       round(float(inbox["conf"].mean()), 2)))
wrong = inbox[inbox["outcome"] == "misrouted"].sort_values("conf", ascending=False)
print()
print("  misroutes nobody would catch, most confident first:")
for _, r in wrong.head(5).iterrows():
    print(f"    [{r['topic']} -> {r['pred']}, {r['conf']:.2f}] {r['text']}")

# ---------------------------------------------------------------------------
heading("9. Save the model, its record and its mistakes for lesson 084")

record = {"project": "Route support messages to the right team", "data": fingerprint, "seed": SEED,
          "candidate": best, "threshold": float(THRESHOLD), "topics": TOPICS,
          "dev": {k: round(v, 3) for k, v in dev_scores.items()},
          "test": {k: round(v, 3) for k, v in test_scores.items()},
          "inbox": {k: round(v, 3) for k, v in inbox_scores.items()},
          "rules_right_first_time": {"test": round(rules_test, 3), "inbox": round(float(rules_in.mean()), 3)}}
model_bytes = pickle.dumps(final)
record["model_sha"] = hashlib.sha256(model_bytes).hexdigest()[:12]
(PROJECT / "model.pkl").write_bytes(model_bytes)
(PROJECT / "model_record.json").write_text(json.dumps(record, indent=2))
inbox.to_csv(PROJECT / "inbox_errors.csv", index=False)
show("model file, size in KB", ("model.pkl", round(len(model_bytes) / 1024)))
show("record", {k: record[k] for k in ("candidate", "threshold", "data")})
show("project files", sorted(str(p.relative_to(PROJECT)) for p in PROJECT.rglob("*") if p.is_file()))
