"""
Lesson 084 - Capstone, part 3: ship and reflect

The last lesson of the course. Lesson 083 built a message router that is
perfect on the sealed test set and much less sure of itself on messages
written the way riders really write. Today we ship it anyway, in the mode
that evidence supports: a script that trains it from scratch, a tiny service,
smoke tests, a monitoring plan, a model card, a project README, and an honest
look back at what we'd do differently.

The worked example is still "route each support message to the right team".
Both data files sit next to this script: feedback.csv (the snapshot from
lesson 082) and inbox_sample.csv (the 48 real-style messages from lesson 083).

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

No downloads and nothing new to install. The script builds the finished
project in `output/capstone/` next to itself (ignored by git), starts a small
local web service for a few seconds and shuts it down again. The folder is
wiped at the start of every run. It takes about ten seconds.
"""

import hashlib
import inspect
import json
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
if OUT.exists():
    shutil.rmtree(OUT)
PROJECT = OUT / "capstone"
(PROJECT / "data" / "raw").mkdir(parents=True)
shutil.copyfile(HERE / "feedback.csv", PROJECT / "data" / "raw" / "feedback.csv")
shutil.copyfile(HERE / "inbox_sample.csv", PROJECT / "data" / "inbox_sample.csv")


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


# ---------------------------------------------------------------------------
heading("1. One command trains it from scratch")

# These four pieces become the project's train.py, word for word.
SETTINGS = {"snapshot": "79a3efa5db0e", "seed": 82, "C": 10.0, "threshold": 0.7,
            "topics": ["app", "bike", "payment", "station"]}


def normalise(text):
    return " ".join(re.findall(r"[a-z0-9\-]+", text.lower()))


def build_pipeline(C):
    words = TfidfVectorizer(preprocessor=normalise, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    chars = TfidfVectorizer(preprocessor=normalise, analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    return Pipeline([("features", FeatureUnion([("words", words), ("chars", chars)])),
                     ("model", LogisticRegression(C=C, max_iter=2000))])


def train(folder):
    """Snapshot check, the sealed split from lesson 082, fit on dev, save model and record."""
    folder = Path(folder)
    raw = folder / "data" / "raw" / "feedback.csv"
    fingerprint = hashlib.sha256(raw.read_bytes()).hexdigest()[:12]
    if fingerprint != SETTINGS["snapshot"]:
        raise SystemExit(f"data snapshot changed: {fingerprint}")
    df = pd.read_csv(raw)
    groups = df["text"].map(normalise).factorize()[0]
    split = StratifiedGroupKFold(5, shuffle=True, random_state=SETTINGS["seed"])
    dev_idx, test_idx = next(split.split(df, df["topic"], groups))
    dev, test = df.iloc[dev_idx], df.iloc[test_idx]
    model = build_pipeline(SETTINGS["C"]).fit(dev["text"], dev["topic"])
    proba = model.predict_proba(test["text"])
    to_person = proba.max(axis=1) < SETTINGS["threshold"]
    right = np.array(model.classes_)[proba.argmax(axis=1)] == test["topic"].to_numpy()
    record = {**SETTINGS, "trained_on_rows": len(dev),
              "test_right_first_time": round(float((right | to_person).mean()), 3),
              "test_to_person": round(float(to_person.mean()), 3),
              "test_ids": sorted(test["feedback_id"]),
              "predictions_sha": hashlib.sha256(np.round(proba, 6).tobytes()).hexdigest()[:12]}
    record["model_version"] = "router-" + record["predictions_sha"][:6]
    (folder / "model.pkl").write_bytes(pickle.dumps(model))
    (folder / "model_record.json").write_text(json.dumps(record, indent=2))
    return record


MAIN = '\n\nif __name__ == "__main__":\n    r = train(Path(__file__).resolve().parent)\n    print(r["model_version"], r["test_right_first_time"], r["test_to_person"])\n'
script = ['"""Train the message router from scratch: python train.py"""', "",
          "import hashlib\nimport json\nimport pickle\nimport re\nfrom pathlib import Path", "",
          "import numpy as np\nimport pandas as pd",
          "from sklearn.feature_extraction.text import TfidfVectorizer",
          "from sklearn.linear_model import LogisticRegression",
          "from sklearn.model_selection import StratifiedGroupKFold",
          "from sklearn.pipeline import FeatureUnion, Pipeline", "",
          f"SETTINGS = {SETTINGS!r}", ""]
script += ["\n" + inspect.getsource(f) for f in (normalise, build_pipeline, train)]
(PROJECT / "train.py").write_text("\n".join(script) + MAIN)

runs = []
for attempt in (1, 2):
    done = subprocess.run([sys.executable, "train.py"], cwd=PROJECT, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(done.stderr)
    runs.append(done.stdout.split())
RECORD = json.loads((PROJECT / "model_record.json").read_text())
show("train.py, lines", len((PROJECT / "train.py").read_text().splitlines()))
show("run 1: version, right first time, to person", tuple(runs[0]))
show("run 2: version, right first time, to person", tuple(runs[1]))
show("same model both times", runs[0] == runs[1])

# ---------------------------------------------------------------------------
heading("2. Decide how to ship it, from the evidence")

PER_DAY = 100              # messages a day, the brief's working guess
CAPACITY = 40              # what one duty person can hand-route in a day
RULES = {"test": 0.781, "inbox": 0.625}     # keyword rules, right first time (lesson 083)
model = pickle.loads((PROJECT / "model.pkl").read_bytes())
real = pd.read_csv(HERE / "inbox_sample.csv")
proba = model.predict_proba(real["text"])
real_unsure = proba.max(axis=1) < RECORD["threshold"]
real_right = np.array(model.classes_)[proba.argmax(axis=1)] == real["topic"].to_numpy()
INBOX = {"right_first_time": round(float((real_right | real_unsure).mean()), 3), "to_person": round(float(real_unsure.mean()), 3)}
show("inbox sample: right first time, to a person", (INBOX["right_first_time"], INBOX["to_person"]))
evidence = {"sealed test set (tidy)": RECORD["test_to_person"], "inbox sample (real-style)": INBOX["to_person"]}
for name, share in evidence.items():
    show(f"{name}: to a person per day", round(share * PER_DAY))
show("one duty person can hand-route", CAPACITY)
MODES = {
    "full auto": "Every message routed by the model. Unsafe: real-style misroutes unmeasured.",
    "auto, unsure ones to a person": "What the brief asked for. Fits the test set, not the real-style inbox.",
    "shadow: suggest, person decides": "Person routes as today, sees the suggestion; both are logged.",
}
MODE = "shadow: suggest, person decides"
show("modes considered", list(MODES))
show("shipping as", MODE)
show("switch to the next mode when", "4 weeks of shadow logs show <= 15% unsure and >= 95% agreement")

# ---------------------------------------------------------------------------
heading("3. A tiny routing service")


class Router:
    """Loaded once at start-up: the model, its record, and a request log."""

    def __init__(self, folder):
        self.model = pickle.loads((folder / "model.pkl").read_bytes())
        self.record = json.loads((folder / "model_record.json").read_text())
        self.vocab = set(self.model.named_steps["features"].transformer_list[0][1].vocabulary_)
        self.log_path = folder / "requests.jsonl"
        self.lock = threading.Lock()

    def route(self, text):
        proba = self.model.predict_proba([text])[0]
        words = normalise(text).split()
        return {"suggested_team": self.model.classes_[proba.argmax()],
                "confidence": round(float(proba.max()), 3),
                "unsure": bool(proba.max() < self.record["threshold"]),
                "unknown_words": round(float(np.mean([w not in self.vocab for w in words])) if words else 1.0, 3),
                "mode": "shadow", "model_version": self.record["model_version"]}


class Handler(BaseHTTPRequestHandler):
    router = None

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"status": "ok", "model_version": self.router.record["model_version"]})
        else:
            self.reply(404, {"error": f"no such page: {self.path}"})

    def do_POST(self):
        if self.path != "/route":
            return self.reply(404, {"error": f"no such page: {self.path}"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            text = body["text"].strip() if isinstance(body, dict) and isinstance(body.get("text"), str) else ""
        except ValueError:
            text = ""
        if not text or len(text) > 1000:
            return self.reply(400, {"error": "send JSON like {\"text\": \"...\"}, 1 to 1000 characters"})
        answer = self.router.route(text)
        with self.router.lock:
            with open(self.router.log_path, "a") as f:
                f.write(json.dumps({"text": text, **answer}) + "\n")
        self.reply(200, answer)

    def log_message(self, *args):
        pass


Handler.router = Router(PROJECT)
server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{server.server_address[1]}"


def call(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


show("GET /health", call("/health"))
for text in ["I was charged twice for one ride", "Baiskeli haina breki kabisa"]:
    status, a = call("/route", {"text": text})
    show(text, (status, a["suggested_team"], a["confidence"], a["unsure"], a["unknown_words"]))
show("POST /route with no text", call("/route", {"msg": "hi"}))

# ---------------------------------------------------------------------------
heading("4. Smoke tests before anyone relies on it")

timings = []
for text in pd.read_csv(HERE / "inbox_sample.csv")["text"].tolist() * 4:
    t0 = time.perf_counter()
    call("/route", {"text": text})
    timings.append((time.perf_counter() - t0) * 1000)
CHECKS = {
    "health says ok": call("/health")[1].get("status") == "ok",
    "obvious payment message -> payment": call("/route", {"text": "Refund my M-Pesa charge please"})[1]["suggested_team"] == "payment",
    "obvious station message -> station": call("/route", {"text": "The dock at Market is broken"})[1]["suggested_team"] == "station",
    "empty text is refused with 400": call("/route", {"text": "   "})[0] == 400,
    "confidence is between 0 and 1": 0 <= call("/route", {"text": "hello"})[1]["confidence"] <= 1,
    "version matches the record": call("/health")[1]["model_version"] == RECORD["model_version"],
    "p95 latency under 50 ms": float(np.percentile(timings, 95)) < 50,
}
for name, ok in CHECKS.items():
    show(name, "pass" if ok else "FAIL")
show("all smoke tests pass", all(CHECKS.values()))
show("requests logged so far", sum(1 for _ in open(PROJECT / "requests.jsonl")))
server.shutdown()
server.server_close()

# ---------------------------------------------------------------------------
heading("5. A monitoring plan you can actually run")

router = Router(PROJECT)
df = pd.read_csv(HERE / "feedback.csv")
tidy = df[df["feedback_id"].isin(RECORD["test_ids"])]
ALERTS = {"unsure": 0.15, "unknown_words": 0.25, "agreement": 0.90}
rng = np.random.default_rng(84)
weekly = []
for week, real_share in enumerate([0.0, 0.05, 0.1, 0.2, 0.35, 0.5], start=1):
    n_real = int(PER_DAY * 5 * real_share)
    batch = pd.concat([tidy.sample(PER_DAY * 5 - n_real, replace=True, random_state=rng.integers(1e9)),
                       real.sample(n_real, replace=True, random_state=rng.integers(1e9))])
    answers = pd.DataFrame([router.route(t) for t in batch["text"]])
    agree = (answers["suggested_team"].to_numpy() == batch["topic"].to_numpy()).mean()   # the person's choice, from the shadow log
    row = {"week": week, "real_style": real_share, "unsure": answers["unsure"].mean(),
           "unknown_words": answers["unknown_words"].mean(), "agreement": agree}
    row["alerts"] = [k for k, lim in ALERTS.items() if (row[k] < lim if k == "agreement" else row[k] > lim)]
    weekly.append(row)
print()
print("  week  real-style  unsure  unknown words  agreement  alerts")
for r in weekly:
    print(f"  {r['week']:>4}  {r['real_style']:>10.0%}  {r['unsure']:>6.2f}  {r['unknown_words']:>13.2f}  "
          f"{r['agreement']:>9.2f}  {', '.join(r['alerts']) or '-'}")
print()
first = next((r["week"] for r in weekly if r["alerts"]), None)
show("first week with an alert", first)
show("which monitor fired first", weekly[first - 1]["alerts"] if first else None)
(PROJECT / "MONITORING.md").write_text(
    "# Monitoring\n\nChecked every Monday from requests.jsonl and the shadow log.\n\n"
    "| Signal | Alert when | Why |\n|---|---|---|\n"
    "| Share marked unsure | above 15% | the brief's limit on hand-routing |\n"
    "| Share of unknown words | above 25% | messages the model has no words for (new language, new product) |\n"
    "| Agreement with the duty person | below 90% | the only signal that measures being right |\n\n"
    "On an alert: read 20 of the flagged messages, label them, add them to a fresh evaluation set, then retrain.\n")

# ---------------------------------------------------------------------------
heading("6. The model card")

inbox = real.assign(**pd.DataFrame([router.route(t) for t in real["text"]]))
inbox["right"] = inbox["suggested_team"] == inbox["topic"]
by_kind = inbox.groupby("kind", sort=False)[["right", "unsure"]].mean().round(2)
card = [
    "# Model card: support message router", "",
    f"**Version:** {RECORD['model_version']}  |  **Data snapshot:** {RECORD['snapshot']}  |  **Mode:** shadow (suggests, a person decides)", "",
    "## What it is for", "",
    "Suggesting which team (app, bike, payment, station) should handle a rider's message, in English, at the moment it arrives.", "",
    "## What it is not for", "",
    "Messages in Swahili or Sheng, messages about several problems at once, or anything about a rider's safety: those go to a person.", "",
    "## Data", "",
    f"800 labelled messages from feedback.csv, 200 per topic, written from templates. Trained on {RECORD['trained_on_rows']}; 160 sealed as a test set; 48 hand-written real-style messages as a second check.", "",
    "## How well it works", "",
    "| Evaluation | Right first time | Sent to a person | Keyword rules |", "|---|---|---|---|",
    f"| Sealed test set (160, tidy) | {RECORD['test_right_first_time']:.3f} | {RECORD['test_to_person']:.3f} | {RULES['test']:.3f} |",
    f"| Inbox sample (48, real-style) | {INBOX['right_first_time']:.3f} | {INBOX['to_person']:.3f} | {RULES['inbox']:.3f} |", "",
    "Top-guess accuracy on the inbox sample by kind of message:", "",
    "| Kind | Top guess right | Unsure |", "|---|---|---|",
] + [f"| {k} | {r['right']:.2f} | {r['unsure']:.2f} |" for k, r in by_kind.iterrows()] + [
    "", "## Limitations", "",
    "- The training data is balanced and tidy; real inboxes are neither. Scores on the test set are an upper bound.",
    "- About half the words in real-style messages were never seen in training, so it marks itself unsure often.",
    "- The 48-message sample is small: treat its numbers as plus or minus 10 points.", "",
    "## Monitoring and retraining", "",
    "See MONITORING.md. Retrain with `python train.py` after adding newly labelled messages; the snapshot check stops it until you update the fingerprint in SETTINGS on purpose.", "",
]
(PROJECT / "MODEL_CARD.md").write_text("\n".join(card) + "\n")
show("model card sections", [l[3:] for l in card if l.startswith("## ")])
show("weakest kind of message (top guess right)", (by_kind["right"].idxmin(), float(by_kind["right"].min())))

# ---------------------------------------------------------------------------
heading("7. The project README and the look back")

MUST = {
    "a frozen test set": bool(RECORD["test_ids"]),
    "a baseline score": True,
    "a model that beats it": RECORD["test_right_first_time"] > RULES["test"],
    "an 'ask a person' threshold": "threshold" in RECORD,
    "a script that trains from scratch": runs[0] == runs[1],
    "a short write-up": True,
}
NICE = {"a tiny API": True, "a drift check": True, "tests": all(CHECKS.values())}
RETRO = [
    ("Went well", "Sealing the test set first and logging every run: no number in the card is tuned on its own exam."),
    ("Went badly", "The real-style sample arrived in lesson 083. Had it existed in lesson 082, the brief's 15% limit would have looked unrealistic on day one."),
    ("Surprised us", "A model that is perfect on tidy data was unsure about half of real messages. Its unsureness was the most useful thing about it."),
    ("Next time", "Collect 100 real messages before choosing a model; set the threshold on them; report scores on the real mix of topics."),
]
readme = ["# Route support messages to the right team", "",
          "Suggests which team should handle a rider's message. Runs in shadow mode: a person still decides.", "",
          "## Run it", "", "```bash", "python train.py          # trains from data/raw/feedback.csv, checks the snapshot", "```", "",
          "Then load `model.pkl` and `model_record.json`, as the service in lesson 084 does.", "",
          "## Results", "", f"Right first time on the sealed test set: {RECORD['test_right_first_time']:.3f} (keyword rules: {RULES['test']:.3f}).",
          f"On 48 real-style messages it misroutes few but is unsure about {INBOX['to_person']:.0%} of them. Details in MODEL_CARD.md.", "",
          "## Checklist", ""]
readme += [f"- [{'x' if ok else ' '}] {k}" for k, ok in {**MUST, **NICE}.items()]
readme += ["", "## Looking back", ""] + [f"- **{k}:** {v}" for k, v in RETRO]
(PROJECT / "README.md").write_text("\n".join(readme) + "\n")
show("must-haves done", f"{sum(MUST.values())} of {len(MUST)}")
show("nice-to-haves done", f"{sum(NICE.values())} of {len(NICE)}")
show("project files", sorted(str(p.relative_to(PROJECT)) for p in PROJECT.rglob("*") if p.is_file()))

# ---------------------------------------------------------------------------
heading("8. Eighty-four lessons later")

PHASES = [("0", "Python foundations for data work", 12), ("1", "Data science basics", 16),
          ("2", "Classical machine learning", 20), ("3", "Deep learning foundations", 12),
          ("4", "Modern AI", 14), ("5", "Projects and production intuition", 10)]
for p, name, n in PHASES:
    show(f"Phase {p}: {name}", f"{n} lessons")
show("lessons finished", sum(n for _, _, n in PHASES))
