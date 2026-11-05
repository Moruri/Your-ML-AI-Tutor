"""
Lesson 073 - Evaluating AI systems

Every lesson so far ended with a number: accuracy, loss, precision@5. This
lesson is about trusting those numbers. We take a small text classifier for
rider feedback and ask the questions a careful evaluator asks: is the test
set really unseen? How sure is this number? Where does the system fail? How
do you grade answers that aren't a single label? And did we quietly tune on
the test set?

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

It uses feedback.csv from this folder (the same 800 made-up rider messages as
lesson 061), needs only scikit-learn and pandas, and runs in a few seconds.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
SEED = 73
TOPICS = ["app", "bike", "payment", "station"]


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def r3(x):
    return round(float(x), 3) + 0.0


# A hand-written rule system: the kind of baseline you should always have.
KEYWORDS = {
    "station": ["dock", "station", "rack", "lit"],
    "payment": ["charge", "charged", "m-pesa", "refund", "pay", "receipt", "fee"],
    "app": ["app", "login", "screen", "crash", "map", "notification", "update"],
    "bike": ["brake", "chain", "tyre", "tire", "seat", "bell", "pedal", "gear", "battery"],
}


def keyword_hits(text):
    words = re.findall(r"[a-z\-]+", text.lower())
    return {t: sum(w.startswith(k) for w in words for k in ks) for t, ks in KEYWORDS.items()}


def rule_system(texts):
    out = []
    for text in texts:
        hits = keyword_hits(text)
        best = max(hits, key=lambda t: (hits[t], t == "bike"))
        out.append(best if hits[best] else "bike")
    return np.array(out)


def model_system():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2),
        LogisticRegression(C=3.0, max_iter=2000),
    )


def bootstrap(correct, n=2000, rng=None):
    """Resample the eval set with replacement; return the middle 95% of accuracies."""
    rng = rng or np.random.default_rng(SEED)
    idx = rng.integers(0, len(correct), size=(n, len(correct)))
    scores = correct[idx].mean(axis=1)
    return r3(np.percentile(scores, 2.5)), r3(np.percentile(scores, 97.5))


# ---------------------------------------------------------------------------
heading("1. Two systems, one number each")

df = pd.read_csv(HERE / "feedback.csv")
train, test = train_test_split(df, test_size=200, stratify=df["topic"], random_state=SEED)
show("messages: train, test", (len(train), len(test)))

rules_pred = rule_system(test["text"])
model = model_system().fit(train["text"], train["topic"])
model_pred = model.predict(test["text"])
show("always guess 'app': accuracy", r3((test["topic"] == "app").mean()))
show("keyword rules: accuracy", r3((rules_pred == test["topic"]).mean()))
show("TF-IDF + logistic regression: accuracy", r3((model_pred == test["topic"]).mean()))


# ---------------------------------------------------------------------------
heading("2. Is the test set really unseen?")

vec = TfidfVectorizer().fit(df["text"])
sim = cosine_similarity(vec.transform(test["text"]), vec.transform(train["text"]))
nearest = sim.max(axis=1)
near_dup = nearest >= 0.9
exact = test["text"].str.lower().isin(set(train["text"].str.lower())).to_numpy()
show("test messages also in train (exact)", int(exact.sum()))
show("test messages with a near-twin (cos >= 0.9)", int(near_dup.sum()))
correct = model_pred == test["topic"].to_numpy()
show("model accuracy on near-twins", r3(correct[near_dup].mean()))
show("model accuracy on the rest", r3(correct[~near_dup].mean()))
i = int(np.argmax(nearest))
j = int(np.argmax(sim[i]))
show("example test message", test["text"].iloc[i])
show("  its twin in train", train["text"].iloc[j])

# Real riders don't write like a template. Here are 24 messages written by
# hand to look like what actually arrives: typos, Swahili and Sheng, no
# obvious keyword, two topics at once, sarcasm. Each is labelled with the
# team that would need to fix it.
REAL_WORLD = [
    ("brkes r gone on the green one at uni sq", "bike", "typos"),
    ("pymnt went thru twice smh", "payment", "typos"),
    ("aap keeps loggin me out", "app", "typos"),
    ("the dok wont release the bike", "station", "typos"),
    ("Pesa imekatwa mara mbili, nirudishie please", "payment", "Swahili/Sheng"),
    ("Baiskeli haina breki kabisa", "bike", "Swahili/Sheng"),
    ("App inakataa kufunguka tangu jana", "app", "Swahili/Sheng"),
    ("Hakuna nafasi ya kupark bike pale Park Gate", "station", "Swahili/Sheng"),
    ("Wheel wobbles like crazy downhill", "bike", "no keyword"),
    ("I rode 10 minutes and got billed for 2 hours", "payment", "no keyword"),
    ("Nothing happens when I tap Start Ride", "app", "no keyword"),
    ("Every slot was full so I had to go to the next stop", "station", "no keyword"),
    ("The bike was fine but I was charged twice in the app", "payment", "two topics"),
    ("Love the app, hate the squeaky brakes", "bike", "two topics"),
    ("Paid fine, but the rack outside Uni Square is broken", "station", "two topics"),
    ("Bike was great, shame the map sent me the wrong way", "app", "two topics"),
    ("Great, another flat tyre. Fantastic service.", "bike", "sarcasm"),
    ("10/10 would get charged three times again", "payment", "sarcasm"),
    ("Cool update, now the map is blank", "app", "sarcasm"),
    ("Loving the pitch-dark station, really adds to the adventure", "station", "sarcasm"),
    ("Seat won't go down", "bike", "plain"),
    ("Where's my receipt for yesterday?", "payment", "plain"),
    ("Can't reset my password", "app", "plain"),
    ("Docking point at Market St won't lock", "station", "plain"),
]
real = pd.DataFrame(REAL_WORLD, columns=["text", "topic", "kind"])
real_rules_ok = rule_system(real["text"]) == real["topic"].to_numpy()
real_pred = model.predict(real["text"])
real_ok = real_pred == real["topic"].to_numpy()
show("hand-written real-world messages", len(real))
show("keyword rules on them: accuracy", r3(real_rules_ok.mean()))
show("model on them: accuracy", r3(real_ok.mean()))

clean = ~near_dup
rules_ok = (rules_pred == test["topic"].to_numpy())[clean]
model_ok = correct[clean]


# ---------------------------------------------------------------------------
heading("3. How sure is the number?")

rng = np.random.default_rng(SEED)
show("clean test set: size", int(clean.sum()))
show("model on it: accuracy, 95% interval", (r3(model_ok.mean()), bootstrap(model_ok, rng=rng)))
show("model on real-world set: accuracy, interval", (r3(real_ok.mean()), bootstrap(real_ok, rng=rng)))
show("rules on real-world set: accuracy, interval", (r3(real_rules_ok.mean()), bootstrap(real_rules_ok, rng=rng)))

idx = rng.integers(0, len(real), size=(2000, len(real)))
gap = real_ok[idx].mean(axis=1) - real_rules_ok[idx].mean(axis=1)
show("real-world: model minus rules, average gap", r3(gap.mean()))
show("share of resamples where model wins", r3((gap > 0).mean()))
show("messages only the model gets right", int((real_ok & ~real_rules_ok).sum()))
show("messages only the rules get right", int((~real_ok & real_rules_ok).sum()))


# ---------------------------------------------------------------------------
heading("4. Where does it fail? Slices")

real["pred"] = real_pred
real["ok"] = real_ok
for kind, s in real.groupby("kind", sort=False):
    show(f"{kind:<14}: right out of {len(s)}", int(s["ok"].sum()))
wrong = real[~real["ok"]]
pairs = (wrong["topic"] + " read as " + wrong["pred"]).value_counts().head(3)
show("most common mistakes", {k: int(v) for k, v in pairs.items()})
for text in wrong["text"].head(3):
    show("  missed", text)


# ---------------------------------------------------------------------------
heading("5. Grading answers that aren't a label")

# A help bot answered 12 rider questions. A person marked each answer right
# (1) or wrong (0). Can an automatic grader agree with that person?
REFERENCE = "Refunds for double M-Pesa charges arrive within 3 working days."
ANSWERS = [
    ("Refunds for double M-Pesa charges arrive within 3 working days.", 1),
    ("refunds for double m-pesa charges arrive within 3 working days", 1),
    ("You'll get the double charge refunded to M-Pesa within 3 working days.", 1),
    ("Within three working days the extra M-Pesa charge comes back to you.", 1),
    ("Sorry about that! The refund lands in M-Pesa within 3 working days.", 1),
    ("Refunds for double M-Pesa charges arrive within 5 working days.", 0),
    ("Refunds arrive within 3 working days by bank transfer.", 0),
    ("Please contact support.", 0),
    ("M-Pesa refunds usually take 3 working days, sometimes 30.", 0),
    ("The double charge is refunded to M-Pesa in 3 working days.", 1),
    ("Refunds for double card charges arrive within 3 working days.", 0),
    ("We'll refund the second M-Pesa charge within 3 working days.", 1),
]
human = np.array([h for _, h in ANSWERS])


def norm(s):
    s = s.lower().replace("three", "3")
    return " ".join(re.findall(r"[a-z0-9\-]+", s))


def exact_grader(a):
    return int(a == REFERENCE)


def normalised_grader(a):
    return int(norm(a) == norm(REFERENCE))


def overlap_grader(a, cut=0.5):
    ref, got = set(norm(REFERENCE).split()), set(norm(a).split())
    return int(len(ref & got) / len(ref) >= cut)


def facts_grader(a):
    """A rubric: the facts that must be there, and the ones that must not."""
    n = norm(a)
    must = ["m-pesa" in n, re.search(r"\b3 working days\b", n) is not None]
    must_not = [re.search(r"\b(5|30|card|bank)\b", n) is not None]
    return int(all(must) and not any(must_not))


for name, grader in [("exact match", exact_grader), ("normalised match", normalised_grader),
                     ("word overlap >= 50%", overlap_grader), ("key-facts rubric", facts_grader)]:
    g = np.array([grader(a) for a, _ in ANSWERS])
    show(f"{name}: agreement, kappa", (r3((g == human).mean()), r3(cohen_kappa_score(human, g))))


# ---------------------------------------------------------------------------
heading("6. Don't tune on the test set")

# A column with no signal at all: did the rider leave a tip? It's a coin
# flip, so no model can honestly beat 50%. Watch what tuning does anyway.
coin = np.random.default_rng(SEED).integers(0, 2, len(df))
X_fit, X_rest, y_fit, y_rest = train_test_split(df["text"], coin, train_size=500, random_state=SEED)
X_tune, X_fresh, y_tune, y_fresh = X_rest[:30], X_rest[30:], y_rest[:30], y_rest[30:]
rng = np.random.default_rng(SEED)
tries = []
for k in range(60):
    keep = rng.random(len(X_fit)) < 0.7        # each "idea" trains a little differently
    m = make_pipeline(TfidfVectorizer(ngram_range=(1, int(rng.integers(1, 3)))),
                      LogisticRegression(C=float(10 ** rng.uniform(-1, 1.5)), max_iter=2000))
    m.fit(X_fit[keep], y_fit[keep])
    tries.append((m.score(X_tune, y_tune), m.score(X_fresh, y_fresh)))
tries = np.array(tries)
best = int(np.argmax(tries[:, 0]))
show("ideas tried", len(tries))
show("tuning set (30): average idea, best idea", (r3(tries[:, 0].mean()), r3(tries[best, 0])))
show("best idea on a fresh set (270)", r3(tries[best, 1]))
show("average idea on the fresh set", r3(tries[:, 1].mean()))
