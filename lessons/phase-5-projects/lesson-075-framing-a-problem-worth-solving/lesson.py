"""
Lesson 075 - Framing a problem worth solving

Phase 5 starts here. Until now somebody handed us a dataset and a target. In
real work the first thing you get is a sentence like "can we use AI to keep
riders from leaving?" The hard part is turning that into something a model
can be built, measured and switched on for.

This lesson takes one vague request and works it through: write the decision
down, put money on the outcomes, set the baselines a model has to beat, pick
the threshold from the costs, catch a feature that cheats, and notice when the
honest answer is "don't build a model".

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

No downloads and no data file: the riders are simulated inside the script,
and it runs in a few seconds.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 75


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def r3(x):
    return round(float(x), 3) + 0.0


# ---------------------------------------------------------------------------
heading("1. The request, and the brief behind it")

REQUEST = "Can we use AI to stop riders from leaving?"

REQUIRED = ["decision", "who_acts", "prediction", "when", "action_cost", "payoff", "metric", "baseline"]


def check_brief(brief):
    """Return the parts of a project brief that are still missing."""
    return [k for k in REQUIRED if not str(brief.get(k, "")).strip()]


vague = {"decision": "stop riders leaving", "prediction": "churn"}
BRIEF = {
    "decision": "Which riders get a 100-shilling ride voucher this week",
    "who_acts": "The marketing team, who can message at most 400 riders a week",
    "prediction": "Will this rider take zero rides in the next 60 days?",
    "when": "Every Monday, using only what we know on that Monday",
    "action_cost": 100,
    "payoff": "A kept rider is worth 1,500 shillings; a voucher saves 3 in 10 leavers",
    "metric": "Shillings gained per week, not accuracy",
    "baseline": "Message riders who haven't ridden for 21 days",
}
show("the request", REQUEST)
show("missing from the vague version", check_brief(vague))
show("missing from the written brief", check_brief(BRIEF))

# ---------------------------------------------------------------------------
heading("2. Some riders to practise on")

# 6,000 made-up riders, as of one Monday. The truth is known ("left": no ride
# in the next 60 days) because we invented it. In real life you would wait 60
# days, or use last year's Mondays.
rng = np.random.default_rng(SEED)
n = 6000
riders = pd.DataFrame({
    "rides_last_30": rng.poisson(6, n),
    "days_since_ride": np.minimum(rng.exponential(12, n).astype(int), 90),
    "account_months": rng.integers(1, 49, n),
    "complaints": rng.poisson(0.4, n),
    "pays_by_mpesa": rng.integers(0, 2, n),
})
logit = (
    -1.2
    - 0.18 * riders["rides_last_30"]
    + 0.10 * riders["days_since_ride"]
    - 0.02 * riders["account_months"]
    + 0.60 * riders["complaints"]
    + rng.normal(0, 0.5, n)
)
p_leave = 1 / (1 + np.exp(-logit))
riders["left"] = (rng.random(n) < p_leave).astype(int)
# Recorded after the fact, in the same table. We'll come back to it.
riders["refund_requested"] = ((riders["left"] == 1) & (rng.random(n) < 0.7)).astype(int)

show("riders", len(riders))
show("share who left", r3(riders["left"].mean()))

FEATURES = ["rides_last_30", "days_since_ride", "account_months", "complaints", "pays_by_mpesa"]
train, test = train_test_split(riders, test_size=0.4, random_state=SEED, stratify=riders["left"])

# ---------------------------------------------------------------------------
heading("3. Put money on it, then set the baselines")

COST, VALUE, SAVE_RATE = 100, 1500, 0.30
test_scale = 6000 / len(test)  # report everything per 6,000 riders


def value(sent, frame=test):
    """Expected shillings gained if we send a voucher to the riders in `sent`."""
    sent = np.asarray(sent, dtype=bool)
    saved = frame["left"].to_numpy()[sent].sum() * SAVE_RATE * VALUE
    return round(float((saved - COST * sent.sum()) * test_scale))


break_even = COST / (SAVE_RATE * VALUE)
show("a voucher pays back only if P(leave) is above", r3(break_even))
show("value: do nothing", value(np.zeros(len(test))))
show("value: voucher to everyone", value(np.ones(len(test))))
rule = (test["days_since_ride"] >= 21).to_numpy()
show("value: rule, no ride for 21+ days", value(rule))
show("riders the rule would message", int(rule.sum() * test_scale))
oracle = (test["left"] == 1).to_numpy()
show("ceiling: message exactly the leavers", value(oracle))

# ---------------------------------------------------------------------------
heading("4. Now a model, judged by the decision")

model = GradientBoostingClassifier(random_state=SEED).fit(train[FEATURES], train["left"])
p = model.predict_proba(test[FEATURES])[:, 1]
show("AUC (how well it ranks riders)", r3(roc_auc_score(test["left"], p)))
show("value: send when P(leave) > 0.5", value(p > 0.5))
show("value: send when P(leave) > break-even", value(p > break_even))
show("riders messaged at that threshold", int((p > break_even).sum() * test_scale))

# The marketing team can only message 400 riders a week.
cap = int(400 / test_scale)
top = np.zeros(len(test), dtype=bool)
top[np.argsort(-p)[:cap]] = True
rule_top = np.zeros(len(test), dtype=bool)
rule_top[np.argsort(-test["days_since_ride"].to_numpy(), kind="stable")[:cap]] = True
show("with a cap of 400: model top 400", value(top))
show("with a cap of 400: longest-away 400", value(rule_top))
show("precision of the model's top 400", r3(test["left"].to_numpy()[top].mean()))

# ---------------------------------------------------------------------------
heading("5. A feature that cheats")

# refund_requested sits in the table, so it's tempting. But it's recorded when
# a rider walks away, which is after the Monday we are predicting on.
cheat = FEATURES + ["refund_requested"]
m2 = GradientBoostingClassifier(random_state=SEED).fit(train[cheat], train["left"])
p2 = m2.predict_proba(test[cheat])[:, 1]
show("AUC without the column", r3(roc_auc_score(test["left"], p)))
show("AUC with refund_requested", r3(roc_auc_score(test["left"], p2)))
show("refund_requested, riders who stayed", int(test.loc[test["left"] == 0, "refund_requested"].sum()))
show("value on paper, send when P > break-even", value(p2 > break_even))

# ---------------------------------------------------------------------------
heading("6. When the honest answer is 'don't build a model'")

# Request: "predict which bikes will be stolen". 5,000 bikes, a year of
# history, and 9 thefts. We split the bikes 20 different ways and see how
# much the answer moves.
rng2 = np.random.default_rng(SEED + 1)
bikes = pd.DataFrame({
    "nights_outside": rng2.poisson(3, 5000),
    "station_busy": rng2.integers(0, 2, 5000),
    "age_months": rng2.integers(1, 60, 5000),
})
steal_p = 0.0005 + 0.0016 * bikes["nights_outside"] * (1 - bikes["station_busy"])
bikes["stolen"] = (rng2.random(5000) < steal_p).astype(int)
show("bikes, thefts", (len(bikes), int(bikes["stolen"].sum())))

recalls, found = [], []
for s in range(20):
    tr, te = train_test_split(bikes, test_size=0.4, random_state=s, stratify=bikes["stolen"])
    lr = LogisticRegression(max_iter=500).fit(tr.drop(columns="stolen"), tr["stolen"])
    score = lr.predict_proba(te.drop(columns="stolen"))[:, 1]
    top50 = np.argsort(-score)[:100]  # flag the top 5% of bikes
    hit = te["stolen"].to_numpy()[top50].sum()
    found.append(int(te["stolen"].sum()))
    recalls.append(hit / max(te["stolen"].sum(), 1))
show("thefts in each test set (min, max)", (min(found), max(found)))
show("share of thefts caught in top 5% (min, max)", (r3(min(recalls)), r3(max(recalls))))
show("average share caught", r3(np.mean(recalls)))
by_station = bikes.groupby("station_busy")["stolen"].agg(["sum", "count"])
show("thefts at quiet stations (of bikes there)", (int(by_station.loc[0, "sum"]), int(by_station.loc[0, "count"])))
show("thefts at busy stations (of bikes there)", (int(by_station.loc[1, "sum"]), int(by_station.loc[1, "count"])))
