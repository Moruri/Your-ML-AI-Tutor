"""
Lesson 077 - Experiment tracking

After a week of trying things, "which run was that?" is the question that eats
your afternoon. This lesson builds a tiny experiment tracker from the standard
library (one folder per run, with settings, scores and files), uses it for a
small sweep, then shows the habits that make the log worth trusting: compare
runs fairly, measure the noise, reload an artifact, skip repeats, and record
failures.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

No downloads and no data file: the bike-share demand data is simulated. The
tracker writes into an `output/` folder next to this script (ignored by git).
It takes a few seconds.
"""

import hashlib
import json
import pickle
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
if OUT.exists():
    shutil.rmtree(OUT)
RUNS = OUT / "runs"


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


# ---------------------------------------------------------------------------
# Data: the same simulated bike-share table as lessons 075 and 076.

def make_data(seed=77, days=150):
    rng = np.random.default_rng(seed)
    rows = []
    for station, base in zip(("Uni Square", "Market", "Lakeside"), (60, 45, 30)):
        dates = pd.date_range("2026-01-01", periods=days)
        temp = 22 + 5 * np.sin(np.arange(days) / 18) + rng.normal(0, 1.5, days)
        rain = np.where(rng.random(days) < 0.25, rng.exponential(6, days), 0.0)
        weekend = (dates.dayofweek >= 5) * 12
        rides = base + 1.2 * (temp - 22) - 0.9 * rain + weekend + rng.normal(0, 4, days)
        rows.append(pd.DataFrame({"date": dates, "station": station, "temp_c": temp.round(1),
                                  "rain_mm": rain.round(1), "weekday": dates.dayofweek,
                                  "rides": rides.round().clip(0).astype(int)}))
    df = pd.concat(rows, ignore_index=True).sort_values(["station", "date"]).reset_index(drop=True)
    g = df.groupby("station")["rides"]
    df["lag_1"] = g.shift(1)
    df["avg_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    return df.dropna().reset_index(drop=True)


DATA = make_data()
FEATURES = ["temp_c", "rain_mm", "weekday", "lag_1", "avg_7"]
dates = sorted(DATA["date"].unique())
TRAIN = DATA[DATA["date"] <= dates[-61]]            # everything up to 60 days before the end
VALID = DATA[(DATA["date"] > dates[-61]) & (DATA["date"] <= dates[-31])]   # 30 days to choose with
TEST = DATA[DATA["date"] > dates[-31]]              # last 30 days: looked at once, at the end
DATA_HASH = hashlib.sha256(pd.util.hash_pandas_object(DATA, index=False).values.tobytes()).hexdigest()[:12]


def fit_and_score(model_name, params, seed):
    if model_name == "ridge":
        model = Ridge(**params)
    else:
        model = RandomForestRegressor(random_state=seed, n_jobs=1, **params)
    model.fit(TRAIN[FEATURES], TRAIN["rides"])
    scores = {
        "val_mae": float(mean_absolute_error(VALID["rides"], model.predict(VALID[FEATURES]))),
        "train_mae": float(mean_absolute_error(TRAIN["rides"], model.predict(TRAIN[FEATURES]))),
    }
    return model, scores


# ---------------------------------------------------------------------------
heading("1. The problem: ten runs, no notes")

# Ten quick tries, printed and forgotten, the way a notebook usually goes.
tries = [("ridge", {"alpha": a}) for a in (0.1, 1, 10)] + \
        [("forest", {"n_estimators": n, "max_depth": d}) for n, d in ((20, 3), (20, 6), (100, 3), (100, 6), (100, None), (200, 6), (200, None))]
quick = []
for name, params in tries:
    _, s = fit_and_score(name, params, seed=1)
    quick.append(round(s["val_mae"], 3))
show("runs made", len(quick))
show("scores that scrolled past", quick)
show("what the notebook remembers", "only the last one")
show("which settings gave the best score?", "...it was one of those")

# ---------------------------------------------------------------------------
heading("2. A tiny tracker")


class Run:
    """One folder per run: params.json, metrics.json, status.json and any files."""

    def __init__(self, name, params, seed, tags=()):
        self.config = {"model": name, "params": params, "seed": seed, "data_hash": DATA_HASH}
        text = json.dumps(self.config, sort_keys=True)
        self.id = hashlib.sha256(text.encode()).hexdigest()[:8]
        self.tags = list(tags)
        self.dir = RUNS / self.id
        self.metrics = {}

    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        (self.dir / "params.json").write_text(json.dumps(self.config, indent=2, sort_keys=True))
        self._status("running")
        return self

    def log(self, **metrics):
        self.metrics.update(metrics)

    def save(self, filename, content):
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(self.dir / filename, mode) as f:
            f.write(content)

    def _status(self, status, error=None):
        info = {"status": status, "tags": self.tags, "seconds": round(time.time() - self.started, 3)}
        if error:
            info["error"] = error
        (self.dir / "status.json").write_text(json.dumps(info))

    def __exit__(self, kind, value, tb):
        (self.dir / "metrics.json").write_text(json.dumps(self.metrics, indent=2, sort_keys=True))
        if kind:
            self._status("failed", error=f"{kind.__name__}: {value}")
            return True            # we recorded it; don't stop the whole sweep
        self._status("finished")


def load_runs():
    rows = []
    for d in sorted(RUNS.iterdir()):
        cfg = json.loads((d / "params.json").read_text())
        met = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
        st = json.loads((d / "status.json").read_text())
        rows.append({"id": d.name, "model": cfg["model"], "params": cfg["params"], "seed": cfg["seed"],
                     "status": st["status"], "tags": st["tags"], **met})
    return pd.DataFrame(rows)


with Run("ridge", {"alpha": 1}, seed=1) as r:
    model, scores = fit_and_score("ridge", {"alpha": 1}, seed=1)
    r.log(**scores)
    r.save("model.pkl", pickle.dumps(model))
first_id = r.id
show("run id (hash of the settings)", first_id)
show("files in the run folder", sorted(p.name for p in r.dir.iterdir()))
show("same settings give the same id", Run("ridge", {"alpha": 1}, seed=1).id == first_id)
show("different settings give a new id", Run("ridge", {"alpha": 2}, seed=1).id != first_id)
show("data hash is part of the settings", r.config["data_hash"] == DATA_HASH)

# ---------------------------------------------------------------------------
heading("3. A sweep you can question")

for name, params in tries:
    with Run(name, params, seed=1, tags=["sweep-1"]) as r:
        model, scores = fit_and_score(name, params, seed=1)
        r.log(**scores)
        r.save("model.pkl", pickle.dumps(model))
table = load_runs()
show("runs on disk", len(table))
best = table.sort_values("val_mae").iloc[0]
show("best run id", best["id"])
show("best settings", (best["model"], best["params"]))
show("best validation MAE", round(float(best["val_mae"]), 3))
show("worst validation MAE", round(float(table["val_mae"].max()), 3))
top3 = table.sort_values("val_mae").head(3)
show("top 3 (model, val MAE)", [(m, round(float(v), 3)) for m, v in zip(top3["model"], top3["val_mae"])])
deep = table[table["params"].apply(lambda p: p.get("max_depth", 0) is None)]
show("deepest forest: train MAE vs val MAE", (round(float(deep["train_mae"].min()), 3), round(float(deep["val_mae"].min()), 3)))

# ---------------------------------------------------------------------------
heading("4. Comparing fairly: how big is the noise?")

# Is the best forest really better than the best ridge? One seed can't say.
best_forest = table[table["model"] == "forest"].sort_values("val_mae").iloc[0]
best_ridge = table[table["model"] == "ridge"].sort_values("val_mae").iloc[0]
show("best ridge settings", best_ridge["params"])
show("best forest settings", best_forest["params"])

forest_scores = []
for seed in range(2, 7):
    with Run("forest", best_forest["params"], seed=seed, tags=["seeds"]) as r:
        model, scores = fit_and_score("forest", best_forest["params"], seed=seed)
        r.log(**scores)
    forest_scores.append(scores["val_mae"])
show("forest val MAE over 5 new seeds", [round(s, 3) for s in forest_scores])
show("forest spread (max - min)", round(max(forest_scores) - min(forest_scores), 3))
spread = max(forest_scores) - min(forest_scores)
worse_by = float(np.mean(forest_scores)) - float(best_ridge["val_mae"])
show("forest is worse than ridge by (mean MAE)", round(worse_by, 3))
show("gap larger than the seed spread?", bool(worse_by > spread))
show("verdict", "keep the simple model" if worse_by > spread else "too close to call")

# Diff two runs: which settings changed, and how far did the score move?
a, b = table[table["id"] == best_ridge["id"]].iloc[0], best_forest
show("model in each run", (a["model"], b["model"]))
show("val MAE change, ridge -> forest (1 seed)", round(float(b["val_mae"] - a["val_mae"]), 3))

# ---------------------------------------------------------------------------
heading("5. An artifact you can reload")

row = load_runs().sort_values("val_mae").iloc[0]
loaded = pickle.loads((RUNS / row["id"] / "model.pkl").read_bytes())
again = float(mean_absolute_error(VALID["rides"], loaded.predict(VALID[FEATURES])))
show("reloaded model's validation MAE", round(again, 3))
show("matches the logged number", bool(abs(again - row["val_mae"]) < 1e-9))
test_mae = float(mean_absolute_error(TEST["rides"], loaded.predict(TEST[FEATURES])))
show("test MAE (looked at once, at the end)", round(test_mae, 3))
show("baseline: same as yesterday, test MAE", round(float(mean_absolute_error(TEST["rides"], TEST["lag_1"])), 3))
(RUNS / row["id"] / "notes.txt").write_text("Chosen on validation. Test score recorded once: %.3f\n" % test_mae)
show("files in the chosen run", sorted(p.name for p in (RUNS / row["id"]).iterdir()))

# ---------------------------------------------------------------------------
heading("6. Skipping repeats, and recording failures")


def already_done(run):
    status = RUNS / run.id / "status.json"
    return status.exists() and json.loads(status.read_text())["status"] == "finished"


before = len(list(RUNS.iterdir()))
skipped = 0
for name, params in tries[:4]:
    candidate = Run(name, params, seed=1)
    if already_done(candidate):
        skipped += 1
show("sweep repeated, runs skipped", skipped)
show("runs on disk, before and after", (before, len(list(RUNS.iterdir()))))

with Run("forest", {"n_estimators": -5}, seed=1, tags=["oops"]) as r:
    fit_and_score("forest", {"n_estimators": -5}, seed=1)
failed = load_runs()
failed = failed[failed["status"] == "failed"]
show("failed runs recorded", len(failed))
err = json.loads((RUNS / failed.iloc[0]["id"] / "status.json").read_text())["error"]
show("error text kept", err[:60])
show("finished vs failed", {k: int(v) for k, v in load_runs()["status"].value_counts().items()})
