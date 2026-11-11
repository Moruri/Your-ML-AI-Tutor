"""
Lesson 082 - Capstone, part 1: choose and scope

The last three lessons are one project of your own, built the way you'd build
it at work. Today is the part people skip: choosing a project that can
actually be finished, writing down what "done" means, gathering the data,
auditing it, sealing a test set, and measuring the baselines you must beat.
No model is trained today, on purpose.

The worked example is "route each support message to the right team", using
feedback.csv (the same 800 messages as lessons 061-073). The data is familiar
so that the new thing is the process, not the data. Your own capstone can be
anything that passes section 1.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

No downloads and nothing new to install. The script starts a capstone project
folder in `output/capstone/` next to itself (ignored by git), with the brief,
the plan, a data snapshot, an audit and a frozen split. The folder is wiped at
the start of every run. It takes a second or two.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
if OUT.exists():
    shutil.rmtree(OUT)
PROJECT = OUT / "capstone"
for sub in ("data/raw", "src", "tests"):
    (PROJECT / sub).mkdir(parents=True)


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


# ---------------------------------------------------------------------------
heading("1. Score the ideas before you fall in love with one")

CRITERIA = ["someone_acts", "data_today", "can_check", "baseline", "fits_time", "safe_if_wrong"]
GATES = ["data_today", "can_check"]          # a 0 on either rules the idea out
IDEAS = {   # 0 = no, 1 = partly, 2 = yes
    "Route support messages to the right team": [2, 2, 2, 2, 2, 2],
    "Forecast tomorrow's rides for the rebalancing van": [2, 1, 2, 2, 1, 2],
    "Predict which bikes will be stolen": [1, 1, 0, 2, 2, 2],
    "An AI assistant that answers any rider question": [1, 0, 0, 1, 0, 1],
    "Flag fraudulent refund requests": [2, 0, 1, 1, 1, 0],
}
board = pd.DataFrame(IDEAS, index=CRITERIA).T
board["total"] = board.sum(axis=1)
board["passes_gates"] = (board[GATES] > 0).all(axis=1)
board = board.sort_values(["passes_gates", "total"], ascending=False)
print()
print("  idea                                                total  passes gates")
for idea, r in board.iterrows():
    print(f"  {idea:<52} {r['total']:>4}  {r['passes_gates']}")
print()
chosen = board.index[0]
show("chosen", chosen)

# ---------------------------------------------------------------------------
heading("2. Write the brief")

BRIEF = {
    "decision": "Which team's queue a new message lands in, or a person if the system is unsure.",
    "who_acts": "The routing step, automatically. One duty person can hand-route about 40 messages a day.",
    "prediction": "One of station, payment, bike, app, plus a confidence between 0 and 1.",
    "when": "The moment the message arrives, using only its text.",
    "action_cost": "A misrouted message waits about one working day before someone forwards it.",
    "payoff": "Right-first-time messages reach the team that can fix them the same day.",
    "metric": "Share routed right first time, with at most 15% sent to a person.",
    "baseline": "Keyword rules; and 'send everything to a person'.",
}
REQUIRED = ["decision", "who_acts", "prediction", "when", "action_cost", "payoff", "metric", "baseline"]
missing = [k for k in REQUIRED if not BRIEF.get(k)]
show("brief fields filled", f"{len(REQUIRED) - len(missing)} of {len(REQUIRED)}")
show("missing", missing)
(PROJECT / "brief.json").write_text(json.dumps(BRIEF, indent=2))

# ---------------------------------------------------------------------------
heading("3. Scope it: must, nice, not this time")

SCOPE = {
    "must": ["a frozen test set", "a baseline score", "a model that beats it on the dev set",
             "an 'ask a person' threshold", "a script that trains from scratch", "a short write-up"],
    "nice": ["a tiny API (lesson 079)", "a drift check on message length and words (lesson 080)",
             "tests for the cleaning code (lesson 076)"],
    "not this time": ["messages in Swahili or Sheng", "several topics per message", "a large language model"],
}
MILESTONES = [   # (lesson, what is done at the end, hours)
    ("082", "brief, plan, data snapshot, audit, split, baselines", 1.5),
    ("083", "pipeline, experiments logged, best model chosen on dev, one test-set score", 3.0),
    ("084", "minimal service, monitoring check, write-up and reflection", 2.5),
]
hours = sum(h for _, _, h in MILESTONES)
show("must-haves, nice-to-haves, cut", tuple(len(v) for v in SCOPE.values()))
show("time-box, hours over three lessons", hours)
plan = ["# Capstone plan: " + chosen, "", "## Brief", ""]
plan += [f"- **{k.replace('_', ' ').capitalize()}:** {v}" for k, v in BRIEF.items()]
for part, items in SCOPE.items():
    plan += ["", f"## {part.capitalize()}", ""] + [f"- {i}" for i in items]
plan += ["", "## Milestones", "", "| Lesson | Done when | Hours |", "|---|---|---|"]
plan += [f"| {l} | {d} | {h} |" for l, d, h in MILESTONES]
plan += ["", "If a milestone runs over its hours, cut from 'nice' first, never from 'must'."]
(PROJECT / "PLAN.md").write_text("\n".join(plan) + "\n")

# ---------------------------------------------------------------------------
heading("4. Gather the data and snapshot it")

src = HERE / "feedback.csv"
raw = PROJECT / "data" / "raw" / "feedback.csv"
shutil.copyfile(src, raw)
fingerprint = hashlib.sha256(raw.read_bytes()).hexdigest()[:12]
df = pd.read_csv(raw)
show("rows, columns", df.shape)
show("columns", list(df.columns))
show("snapshot fingerprint", fingerprint)

# ---------------------------------------------------------------------------
heading("5. Audit it before you trust it")

KEYWORDS = {   # the same rules as lesson 073
    "station": ["dock", "station", "rack", "lit"],
    "payment": ["charge", "charged", "m-pesa", "refund", "pay", "receipt", "fee"],
    "app": ["app", "login", "screen", "crash", "map", "notification", "update"],
    "bike": ["brake", "chain", "tyre", "tire", "seat", "bell", "pedal", "gear", "battery"],
}


def keyword_hits(text):
    words = re.findall(r"[a-z\-]+", text.lower())
    return {t: sum(w.startswith(k) for w in words for k in ks) for t, ks in KEYWORDS.items()}


def normalise(text):
    return " ".join(re.findall(r"[a-z0-9\-]+", text.lower()))


hits = pd.DataFrame([keyword_hits(t) for t in df["text"]])
topics_hit = (hits > 0).sum(axis=1)
df["norm"] = df["text"].map(normalise)
audit = {
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_ids": int(df["feedback_id"].duplicated().sum()),
    "exact_duplicate_texts": int(df["text"].duplicated().sum()),
    "same_text_after_normalising": int(df["norm"].duplicated().sum()),
    "messages_per_topic": df["topic"].value_counts().sort_index().to_dict(),
    "length_chars_median_max": (int(df["text"].str.len().median()), int(df["text"].str.len().max())),
    "no_keyword_at_all": int((topics_hit == 0).sum()),
    "keywords_from_two_or_more_topics": int((topics_hit >= 2).sum()),
}
for k, v in audit.items():
    show(k.replace("_", " "), v)
two = df[topics_hit >= 2]
show("example with two topics", (two.iloc[0]["text"], two.iloc[0]["topic"]))
(PROJECT / "audit.json").write_text(json.dumps(audit, indent=2))

# ---------------------------------------------------------------------------
heading("6. Seal the test set now")

SEED = 82
groups = df["norm"].factorize()[0]           # identical messages stay on the same side
folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
dev_idx, test_idx = next(folds.split(df, df["topic"], groups))
dev, test = df.iloc[dev_idx], df.iloc[test_idx]
show("dev rows, test rows", (len(dev), len(test)))
show("test messages per topic", test["topic"].value_counts().sort_index().to_dict())
show("same text on both sides", int(test["norm"].isin(set(dev["norm"])).sum()))
split = {"snapshot": fingerprint, "seed": SEED, "test_ids": sorted(test["feedback_id"].tolist())}
(PROJECT / "data" / "split.json").write_text(json.dumps(split, indent=2))

# ---------------------------------------------------------------------------
heading("7. Measure the bar to beat (on dev only)")


def rule(text):
    h = keyword_hits(text)
    best = max(h, key=lambda t: (h[t], t == "bike"))
    return best if h[best] else "bike"


majority = dev["topic"].value_counts().idxmax()
rules = dev["text"].map(rule)
rule_right = rules == dev["topic"]
baselines = {
    "always the biggest topic": round(float((dev["topic"] == majority).mean()), 3),
    "keyword rules": round(float(rule_right.mean()), 3),
}
for name, acc in baselines.items():
    show(f"{name}: right first time", acc)
show("everything to a person: right, hand-routed", (1.0, 1.0))
show("keyword rules, one-topic vs mixed messages", (round(float(rule_right[topics_hit[dev.index] <= 1].mean()), 3),
                                                    round(float(rule_right[topics_hit[dev.index] >= 2].mean()), 3)))
(PROJECT / "baselines.json").write_text(json.dumps(baselines, indent=2))
(PROJECT / "README.md").write_text(f"# {chosen}\n\nCapstone project. Start with PLAN.md, then brief.json.\n"
                                   f"Data snapshot {fingerprint} is in data/raw/; the sealed test ids are in data/split.json.\n")
files = sorted(str(p.relative_to(PROJECT)) for p in PROJECT.rglob("*") if p.is_file())
show("project files so far", files)
