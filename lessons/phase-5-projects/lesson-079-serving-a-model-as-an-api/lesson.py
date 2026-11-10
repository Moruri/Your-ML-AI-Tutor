"""
Lesson 079 - Serving a model as an API

A model in a notebook helps one person. A model behind a small web service
helps any program that can send a request. This lesson trains the ride model
from lesson 078, saves it with its version record, loads it in a tiny HTTP
server built from the standard library, and calls it from a client the way
another program would: health check, predictions, bad input, many callers at
once, and a log of every request.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

No downloads, no data file and nothing new to install: the server is Python's
own `http.server`, the client is `urllib`. The server runs in a background
thread on a free port on your own machine (127.0.0.1) and is shut down at the
end. Everything is written into an `output/` folder next to this script
(ignored by git), and it is wiped at the start of every run.
"""

import hashlib
import inspect
import json
import pickle
import shutil
import statistics
import threading
import time
from datetime import date, timedelta
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


# ---------------------------------------------------------------------------
# The data and the feature function, the same shape as lesson 078.

def make_rides(seed=79, days=90):
    rng = np.random.default_rng(seed)
    rows = []
    for station, base in zip(("Uni Square", "Market", "Lakeside"), (60, 45, 30)):
        dates = pd.date_range("2026-08-01", periods=days)
        temp = 22 + 5 * np.sin(np.arange(days) / 15) + rng.normal(0, 1.5, days)
        rain = np.where(rng.random(days) < 0.25, rng.exponential(6, days), 0.0)
        weekend = (dates.dayofweek >= 5) * 12
        rides = base + 1.2 * (temp - 22) - 0.9 * rain + weekend + rng.normal(0, 4, days)
        rows.append(pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "station": station,
                                  "temp_c": temp.round(1), "rain_mm": rain.round(1),
                                  "rides": rides.round().clip(0).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def build_features(df):
    """Raw rides in, model-ready table out. The only place features are made."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    g = df.groupby("station")["rides"]
    df["lag_1"] = g.shift(1)
    df["avg_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["weekday"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df.dropna().reset_index(drop=True)


FEATURES = ["temp_c", "rain_mm", "lag_1", "avg_7", "is_weekend"]

# ---------------------------------------------------------------------------
heading("1. Train once, save the model and its record")

rides = make_rides()
feats = build_features(rides)
train, test = feats[feats["date"] < "2026-10-15"], feats[feats["date"] >= "2026-10-15"]
model = Ridge().fit(train[FEATURES], train["rides"])
mae = round(float(mean_absolute_error(test["rides"], model.predict(test[FEATURES]))), 3)

blob = pickle.dumps(model)
(OUT / "model.pkl").write_bytes(blob)
card = {
    "model_version": hashlib.sha256(blob).hexdigest()[:8],
    "feature_version": hashlib.sha256((inspect.getsource(build_features) + ",".join(FEATURES)).encode()).hexdigest()[:8],
    "features": FEATURES,
    "test_mae": mae,
    "trained_up_to": "2026-10-14",
}
(OUT / "model_card.json").write_text(json.dumps(card, indent=2))
rides.to_csv(OUT / "history.csv", index=False)       # the server's view of "what happened so far"
show("training rows, test rows", (len(train), len(test)))
show("test MAE (rides per day)", mae)
show("files the server will load", sorted(p.name for p in OUT.iterdir()))

# ---------------------------------------------------------------------------
heading("2. A tiny prediction service")

STATIONS = sorted(rides["station"].unique())


class Service:
    """Everything the server needs, loaded from disk once at start-up."""

    def __init__(self, folder):
        self.model = pickle.loads((folder / "model.pkl").read_bytes())
        self.card = json.loads((folder / "model_card.json").read_text())
        self.history = pd.read_csv(folder / "history.csv")
        self.log_path = folder / "requests.jsonl"
        self.lock = threading.Lock()

    def check(self, body):
        """Return a list of plain-English problems with a request (empty means fine)."""
        problems = []
        for field, kind in [("station", str), ("date", str), ("temp_c", (int, float)), ("rain_mm", (int, float))]:
            if field not in body:
                problems.append(f"missing field '{field}'")
            elif not isinstance(body[field], kind) or isinstance(body[field], bool):
                problems.append(f"'{field}' has the wrong type")
        if problems:
            return problems
        if body["station"] not in STATIONS:
            problems.append(f"unknown station '{body['station']}'")
        if not -10 <= body["temp_c"] <= 45:
            problems.append("temp_c should be in degrees Celsius, between -10 and 45")
        if not 0 <= body["rain_mm"] <= 200:
            problems.append("rain_mm should be between 0 and 200")
        try:
            date.fromisoformat(body["date"])
        except ValueError:
            problems.append("date should look like 2026-10-30")
        return problems

    def predict(self, body):
        day = date.fromisoformat(body["date"])
        day_s, yesterday = day.isoformat(), (day - timedelta(days=1)).isoformat()
        h = self.history
        past = h[(h["station"] == body["station"]) & (h["date"] < day_s)].tail(7)
        if len(past) < 7 or past["date"].iloc[-1] != yesterday:
            return None                                      # we don't know the last 7 days before this one
        today = pd.DataFrame([{**{k: body[k] for k in ("date", "station", "temp_c", "rain_mm")}, "rides": 0}])
        row = build_features(pd.concat([past, today], ignore_index=True)).iloc[[-1]]
        return round(float(self.model.predict(row[FEATURES])[0]), 1)

    def log(self, record):
        with self.lock:                                      # many threads, one file
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")


class Handler(BaseHTTPRequestHandler):
    service = None                                           # set before the server starts

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            c = self.service.card
            self.reply(200, {"status": "ok", "model_version": c["model_version"],
                             "feature_version": c["feature_version"]})
        else:
            self.reply(404, {"error": f"no such page: {self.path}"})

    def do_POST(self):
        start = time.perf_counter()
        status, payload, body = self.answer()
        if self.path == "/predict":                          # every request, good or bad, goes in the log
            self.service.log({"request": body, "status": status, **payload,
                              "ms": round((time.perf_counter() - start) * 1000, 2)})
        self.reply(status, payload)

    def answer(self):
        if self.path != "/predict":
            return 404, {"error": f"no such page: {self.path}"}, None
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return 400, {"error": ["the body should be a JSON object"]}, None
        problems = self.service.check(body)
        if problems:
            return 400, {"error": problems}, body
        try:
            pred = self.service.predict(body)
        except Exception as e:                               # a bug in *our* code: say so, don't hang up
            return 500, {"error": [f"server error: {type(e).__name__}"]}, body
        if pred is None:
            return 422, {"error": ["no history for the 7 days before that date"]}, body
        return 200, {"station": body["station"], "date": body["date"], "predicted_rides": pred,
                     "model_version": self.service.card["model_version"]}, body

    def log_message(self, *args):                            # keep the terminal quiet
        pass


Handler.service = Service(OUT)
server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)     # port 0: "any free port"
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
BASE = f"http://127.0.0.1:{server.server_address[1]}"
show("server running in a background thread", thread.is_alive())
show("endpoints", ["GET /health", "POST /predict"])

# ---------------------------------------------------------------------------
heading("3. Calling it from another program")


def call(method, path, payload=None):
    """What any client does: send JSON, get a status code and JSON back."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


status, health = call("GET", "/health")
show("GET /health", (status, health["status"]))
show("versions match the model card", health["model_version"] == card["model_version"])

ask = {"station": "Market", "date": "2026-10-30", "temp_c": 24.0, "rain_mm": 0.0}
status, answer = call("POST", "/predict", ask)
show("POST /predict status", status)
show("answer", answer)

# The same prediction, made directly in this process with the same function.
past = rides[(rides["station"] == "Market") & (rides["date"] < "2026-10-30")].tail(7)
local_row = build_features(pd.concat([past, pd.DataFrame([{**ask, "rides": 0}])], ignore_index=True)).iloc[[-1]]
local = round(float(model.predict(local_row[FEATURES])[0]), 1)
show("same as predicting in-process", answer["predicted_rides"] == local)
status, wet = call("POST", "/predict", {**ask, "rain_mm": 20.0})
show("same day with 20 mm of rain", wet["predicted_rides"])

# ---------------------------------------------------------------------------
heading("4. Bad requests get clear answers, not crashes")

cases = [
    ("missing a field", {"station": "Market", "date": "2026-10-30", "temp_c": 24.0}),
    ("Fahrenheit by mistake", {**ask, "temp_c": 75.2}),
    ("a station we've never seen", {**ask, "station": "Airport"}),
    ("temperature sent as text", {**ask, "temp_c": "24"}),
    ("a date with no history", {**ask, "date": "2027-01-15"}),
]
for label, body in cases:
    status, reply = call("POST", "/predict", body)
    print(f"  {label:<28} -> {status} {reply['error']}")
status, reply = call("GET", "/predictions")
print(f"  {'a page that does not exist':<28} -> {status} {reply['error']!r}")
show("server still up afterwards", call("GET", "/health")[0] == 200)

# ---------------------------------------------------------------------------
heading("5. Many callers at once")

days = pd.date_range("2026-10-08", "2026-10-30").strftime("%Y-%m-%d")
batch = [{"station": s, "date": d, "temp_c": 22.0, "rain_mm": 0.0} for s in STATIONS for d in days]
one_by_one = [call("POST", "/predict", r)[1]["predicted_rides"] for r in batch]
with ThreadPoolExecutor(max_workers=8) as pool:
    together = [r[1]["predicted_rides"] for r in pool.map(lambda r: call("POST", "/predict", r), batch)]
show("requests sent, twice over", len(batch))
show("8 callers at once got the same answers", together == one_by_one)
logged = [json.loads(line) for line in (OUT / "requests.jsonl").read_text().splitlines()]
ms = [r["ms"] for r in logged if r["status"] == 200]
show("median ms inside the server (yours varies)", round(statistics.median(ms), 1))

# ---------------------------------------------------------------------------
heading("6. The request log: the start of monitoring")

log = pd.DataFrame(logged)
show("requests logged", len(log))
show("by status code", log["status"].value_counts().sort_index().to_dict())
first = logged[0]
show("one log line", {"request": first["request"], "status": first["status"],
                      "predicted_rides": first["predicted_rides"], "model_version": first["model_version"]})
ok = log[log["status"] == 200]
show("mean prediction by station", ok.groupby("station")["predicted_rides"].mean().round(1).to_dict())
show("share of requests rejected", round(float((log["status"] != 200).mean()), 3))

# ---------------------------------------------------------------------------
heading("7. Shutting down cleanly")

server.shutdown()
server.server_close()
thread.join(timeout=5)
show("server thread stopped", not thread.is_alive())
try:
    urllib.request.urlopen(BASE + "/health", timeout=2)
    still = True
except (urllib.error.URLError, ConnectionError, OSError):
    still = False
show("anyone still answering?", still)
