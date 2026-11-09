"""
Lesson 078 - Data versioning and feature pipelines

Code has git. Data usually has a file called `rides_final_v2_REAL.csv`. This
lesson builds a small data-versioning habit from the standard library and
pandas: fingerprint every dataset, store snapshots under their hash, check each
new delivery against the last one, and make features with one versioned
function whose results are cached by (data version, feature version).

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

No downloads and no data file: the ride data is simulated. Everything is
written into an `output/` folder next to this script (ignored by git), and it
is wiped at the start of every run. It takes a few seconds.
"""

import hashlib
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
if OUT.exists():
    shutil.rmtree(OUT)
STORE = OUT / "store"
STORE.mkdir(parents=True)


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


# ---------------------------------------------------------------------------
# The data: three stations, 90 days. Delivery 1 is the first export; the others
# are what the "data team" sends later, each with a different surprise.

def make_rides(seed=78, days=90):
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


d1 = make_rides()
d2 = d1.copy()                                   # a few corrected rows
d2.loc[[5, 40, 100], "rides"] += 7
d3 = d1.copy()                                   # looks the same, but the temperature is in Fahrenheit
d3["temp_c"] = (d3["temp_c"] * 9 / 5 + 32).round(1)
d3 = d3.rename(columns={"rain_mm": "rainfall"})  # and a column got renamed
d3.loc[d3.index[-12:], "station"] = "Airport"    # and a new station appeared

# ---------------------------------------------------------------------------
heading("1. The problem: same name, different data")

OUT.joinpath("rides.csv").write_text(d1.to_csv(index=False))
score_before = None


def quick_score(csv_path):
    df = pd.read_csv(csv_path).sort_values(["station", "date"])
    df["lag_1"] = df.groupby("station")["rides"].shift(1)
    df = df.dropna()
    train, test = df.iloc[:-60], df.iloc[-60:]
    cols = ["temp_c", "rain_mm", "lag_1"]
    model = Ridge().fit(train[cols], train["rides"])
    return round(float(mean_absolute_error(test["rides"], model.predict(test[cols]))), 3)


score_before = quick_score(OUT / "rides.csv")
OUT.joinpath("rides.csv").write_text(d2.to_csv(index=False))      # someone "just fixes a few rows"
score_after = quick_score(OUT / "rides.csv")
show("file name", "rides.csv")
show("MAE on Monday", score_before)
show("MAE after somebody fixed three rows", score_after)
show("file name changed?", False)
show("can you get Monday's file back?", "no")

# ---------------------------------------------------------------------------
heading("2. Fingerprints: a hash of the contents")


def table_hash(df):
    """Hash the values, not the file layout: row order and CSV quirks don't matter."""
    ordered = df.sort_values(list(df.columns)).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).values.tobytes()).hexdigest()[:12]


h1, h2 = table_hash(d1), table_hash(d2)
show("hash of delivery 1", h1)
show("hash of delivery 2", h2)
show("three rows changed -> hash changed", h1 != h2)
show("same data, shuffled rows -> same hash", table_hash(d1.sample(frac=1, random_state=0)) == h1)
tweaked = d1.copy()
tweaked.loc[0, "temp_c"] += 0.1
show("one value off by 0.1 -> hash changed", table_hash(tweaked) != h1)

# ---------------------------------------------------------------------------
heading("3. Snapshots, stored under their hash")


def save_snapshot(df, note):
    h = table_hash(df)
    folder = STORE / h
    if folder.exists():
        return h, False                             # already have exactly this data
    folder.mkdir()
    df.to_csv(folder / "data.csv", index=False)
    schema = {c: str(t) for c, t in df.dtypes.items()}
    manifest = {"hash": h, "rows": len(df), "columns": schema, "note": note}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return h, True


def load_snapshot(h):
    return pd.read_csv(STORE / h / "data.csv")


s1, new1 = save_snapshot(d1, "first export")
s2, new2 = save_snapshot(d2, "corrected rows")
s1b, new1b = save_snapshot(d1.sample(frac=1, random_state=3), "same export, sent again")
show("snapshots in the store", sorted(p.name for p in STORE.iterdir()) == sorted([s1, s2]))
show("number of snapshots", len(list(STORE.iterdir())))
show("re-sent delivery 1 stored again?", new1b)
show("files in a snapshot", sorted(p.name for p in (STORE / s1).iterdir()))
manifest = json.loads((STORE / s1 / "manifest.json").read_text())
show("manifest rows, columns", (manifest["rows"], list(manifest["columns"])))
show("score on snapshot 1 (Monday's number)", quick_score(STORE / s1 / "data.csv"))
show("score on snapshot 2", quick_score(STORE / s2 / "data.csv"))
show("a model can now say which data it used", {"data_version": s1})

# ---------------------------------------------------------------------------
heading("4. Checking a new delivery before you trust it")


def compare(old, new):
    """Return a list of plain-English problems between two deliveries."""
    problems = []
    if list(old.columns) != list(new.columns):
        gone, added = set(old.columns) - set(new.columns), set(new.columns) - set(old.columns)
        problems.append(f"columns changed: missing {sorted(gone)}, new {sorted(added)}")
    for col in old.columns:
        if col not in new.columns:
            continue
        if old[col].dtype.kind in "if":
            lo, hi = old[col].mean(), new[col].mean()
            if lo != 0 and abs(hi - lo) / abs(lo) > 0.25:
                problems.append(f"{col}: mean moved from {lo:.1f} to {hi:.1f}")
        else:
            fresh = set(new[col]) - set(old[col])
            if col == "station" and fresh:
                problems.append(f"{col}: new values {sorted(fresh)}")
    if new.isna().mean().max() > 0.05:
        problems.append("more than 5% missing in some column")
    return problems


p2, p3 = compare(d1, d2), compare(d1, d3)
show("delivery 2 problems", p2)
show("delivery 3 problems found", len(p3))
for p in p3:
    print("    -", p)

# ---------------------------------------------------------------------------
heading("5. A feature pipeline with a version")


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


def feature_version():
    """A hash of the feature code and the column list. Edit either and it changes."""
    text = inspect.getsource(build_features) + ",".join(FEATURES)
    return hashlib.sha256(text.encode()).hexdigest()[:8]


FV = feature_version()
show("feature version", FV)
show("same code again -> same version", feature_version() == FV)
feats = build_features(d1)
show("rows in, rows out", (len(d1), len(feats)))
show("rows lost to warm-up (7 days x 3 stations)", len(d1) - len(feats))
show("first row has no unknown past", bool(feats[["lag_1", "avg_7"]].notna().all().all()))

# ---------------------------------------------------------------------------
heading("6. Caching by (data version, feature version)")

CACHE = OUT / "features"
CACHE.mkdir()
calls = {"built": 0, "reused": 0}


def get_features(data_hash, df):
    path = CACHE / f"{data_hash}-{feature_version()}.csv"
    if path.exists():
        calls["reused"] += 1
        return pd.read_csv(path, parse_dates=["date"])
    calls["built"] += 1
    out = build_features(df)
    out.to_csv(path, index=False)
    return out


get_features(s1, d1)
get_features(s1, d1)
get_features(s2, d2)
show("calls: built, reused", (calls["built"], calls["reused"]))
show("cached files", sorted(p.name for p in CACHE.iterdir()) == sorted([f"{s1}-{FV}.csv", f"{s2}-{FV}.csv"]))

# Someone "improves" the feature code: the version changes, so old caches are not reused.
source = inspect.getsource(build_features).replace("rolling(7)", "rolling(14)")
old_version, new_version = FV, hashlib.sha256((source + ",".join(FEATURES)).encode()).hexdigest()[:8]
show("edited feature code -> new version", new_version != old_version)
show("old cache file would be hit by new code", f"{s1}-{new_version}.csv" in {p.name for p in CACHE.iterdir()})

# ---------------------------------------------------------------------------
heading("7. Training and serving use the same function")

hist = d1[d1["station"] == "Market"].tail(10)
today = pd.DataFrame({"date": ["2026-10-30"], "station": ["Market"], "temp_c": [24.0], "rain_mm": [0.0], "rides": [0]})
batch = build_features(pd.concat([hist, today], ignore_index=True))
one = batch.iloc[-1]
show("serving row date", str(one["date"].date()))
show("lag_1 is yesterday's rides", bool(one["lag_1"] == hist["rides"].iloc[-1]))
show("avg_7 is the mean of the last 7 days", bool(abs(one["avg_7"] - hist["rides"].tail(7).mean()) < 1e-9))
show("features used by the model", FEATURES)
show("columns in the serving row", sorted(batch.columns) == sorted(feats.columns))

model = Ridge().fit(feats[FEATURES], feats["rides"])
record = {"data_version": s1, "feature_version": FV, "features": FEATURES}
(OUT / "model_card.json").write_text(json.dumps(record, indent=2))
show("what we'd store with the model", record)
