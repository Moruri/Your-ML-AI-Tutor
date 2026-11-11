"""
Lesson 081 - Performance: making it fast enough

"Make it faster" is not a goal. "95% of requests answered in under 5 ms, and
the nightly batch of 100,000 forecasts done in under a second" is a goal. This
lesson starts from a deliberately naive version of the ride service from lesson
079, measures it against a budget like that, profiles it to find where the time
really goes, then fixes it one step at a time: load once, skip needless
overhead, batch, cache, and finally shrink a bigger model by quantising its
weights to 8-bit integers. After each step we measure again and check that the
answers did not change.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

No downloads, no data file and nothing new to install. Timings depend on your
computer, so your milliseconds will differ from the README; the *ratios* and
every non-timing number should be close. A small `output/` folder is written
next to this script (ignored by git) and wiped at the start of every run. It
takes about ten seconds, most of it training the bigger model in section 6.
"""

import cProfile
import io
import pickle
import pstats
import shutil
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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


def timed(fn, repeats):
    """Run fn `repeats` times; return (median, p95) in milliseconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def ms(pair):
    """Round a (median, p95) pair for printing, keeping small numbers readable."""
    return tuple(float(f"{v:.2g}") if v < 0.1 else round(v, 2) for v in pair)


# ---------------------------------------------------------------------------
# The same data and feature function as lesson 079.

def make_rides(seed=81, days=90):
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
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    return df.dropna().reset_index(drop=True)


FEATURES = ["temp_c", "rain_mm", "lag_1", "avg_7", "is_weekend"]

rides = make_rides()
feats = build_features(rides)
model = Ridge().fit(feats[FEATURES], feats["rides"])
(OUT / "model.pkl").write_bytes(pickle.dumps(model))
rides.to_csv(OUT / "history.csv", index=False)

BUDGET_MS = 2.0          # p95 for one request
BATCH_BUDGET_S = 1.0     # nightly batch of 100,000 rows

# ---------------------------------------------------------------------------
heading("1. Set a budget, then measure the naive version")


def predict_naive(station, temp_c, rain_mm):
    """What a first version often looks like: do everything, every time."""
    loaded = pickle.loads((OUT / "model.pkl").read_bytes())
    history = pd.read_csv(OUT / "history.csv")
    f = build_features(history)
    last = f[f["station"] == station].iloc[-1]
    row = pd.DataFrame([{"temp_c": temp_c, "rain_mm": rain_mm, "lag_1": last["rides"],
                         "avg_7": f[f["station"] == station]["rides"].tail(7).mean(), "is_weekend": 0}])
    return float(loaded.predict(row[FEATURES])[0])


answer_naive = predict_naive("Market", 21.0, 0.0)
med, p95 = timed(lambda: predict_naive("Market", 21.0, 0.0), 60)
show("budget: p95 per request (ms)", BUDGET_MS)
show("naive: median, p95 (ms)", ms((med, p95)))
show("within budget?", p95 <= BUDGET_MS)
show("answer for Market, 21 C, dry", round(answer_naive, 2))

# ---------------------------------------------------------------------------
heading("2. Profile before you touch anything")

prof = cProfile.Profile()
prof.enable()
for _ in range(30):
    predict_naive("Market", 21.0, 0.0)
prof.disable()
stats = pstats.Stats(prof, stream=io.StringIO())
total = stats.total_tt
shares = {"read_csv": 0.0, "build_features": 0.0, "pickle.loads": 0.0, "model.predict": 0.0}
for (file, line, name), (cc, nc, tt, ct, callers) in stats.stats.items():
    if name == "read_csv":
        shares["read_csv"] = max(shares["read_csv"], ct)
    elif name == "build_features":
        shares["build_features"] = max(shares["build_features"], ct)
    elif "_pickle.loads" in name:
        shares["pickle.loads"] = max(shares["pickle.loads"], ct)
    elif name == "predict" and "linear_model" in file:
        shares["model.predict"] = max(shares["model.predict"], ct)
for label, ct in shares.items():
    show(f"share of time, {label} (%)", round(100 * ct / total))
show("share of time, everything else (%)", round(100 * (total - sum(shares.values())) / total))
buf = io.StringIO()
pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(12)
(OUT / "profile_naive.txt").write_text(buf.getvalue())

# ---------------------------------------------------------------------------
heading("3. Load once, keep what you need in memory")

LOADED = pickle.loads((OUT / "model.pkl").read_bytes())
f_all = build_features(pd.read_csv(OUT / "history.csv"))
LATEST = {s: (float(g["rides"].iloc[-1]), float(g["rides"].tail(7).mean())) for s, g in f_all.groupby("station")}


def predict_warm(station, temp_c, rain_mm):
    lag_1, avg_7 = LATEST[station]
    row = pd.DataFrame([{"temp_c": temp_c, "rain_mm": rain_mm, "lag_1": lag_1, "avg_7": avg_7, "is_weekend": 0}])
    return float(LOADED.predict(row[FEATURES])[0])


COEF, INTERCEPT = LOADED.coef_.copy(), float(LOADED.intercept_)


def predict_lean(station, temp_c, rain_mm):
    lag_1, avg_7 = LATEST[station]
    return float(np.dot(COEF, (temp_c, rain_mm, lag_1, avg_7, 0)) + INTERCEPT)


warm = timed(lambda: predict_warm("Market", 21.0, 0.0), 300)
lean = timed(lambda: predict_lean("Market", 21.0, 0.0), 3000)
show("load once: median, p95 (ms)", ms(warm))
show("plain numpy maths: median, p95 (ms)", ms(lean))
show("same answers?", (round(predict_warm("Market", 21.0, 0.0), 6) == round(answer_naive, 6),
                       round(predict_lean("Market", 21.0, 0.0), 6) == round(answer_naive, 6)))
show("speed-up of numpy over naive, about", f"{round(med / lean[0]):,}x")

# ---------------------------------------------------------------------------
heading("4. Batch: one call for many rows")

rng = np.random.default_rng(1)
N = 100_000
night = pd.DataFrame({"temp_c": rng.normal(20, 4, N).round(1), "rain_mm": np.where(rng.random(N) < 0.25, 5.0, 0.0),
                      "lag_1": rng.integers(20, 80, N).astype(float), "avg_7": rng.uniform(25, 75, N).round(1),
                      "is_weekend": rng.integers(0, 2, N)})[FEATURES]
sample = 2_000
t0 = time.perf_counter()
one_by_one = [LOADED.predict(night.iloc[[i]])[0] for i in range(sample)]
per_row = (time.perf_counter() - t0) / sample
t0 = time.perf_counter()
all_at_once = LOADED.predict(night)
batch_s = time.perf_counter() - t0
show("rows in the nightly batch", N)
show("one row at a time, estimated (s)", round(per_row * N, 1))
show("one call for all rows (s)", round(batch_s, 3))
show("within the batch budget?", batch_s <= BATCH_BUDGET_S)
show("largest difference between the two", float(np.max(np.abs(np.array(one_by_one) - all_at_once[:sample]))))

# ---------------------------------------------------------------------------
heading("5. Cache answers people ask for again and again")

DATA_VERSION = "2026-10-29"     # bump this whenever the history changes


@lru_cache(maxsize=1024)
def forecast(station, day, data_version):
    """Pretend this is expensive: a weather lookup plus the model."""
    time.sleep(0.002)
    return predict_lean(station, 21.0, 0.0)


stations = ["Uni Square", "Market", "Lakeside"]
days = [f"2026-10-{d}" for d in range(30, 32)] + [f"2026-11-0{d}" for d in range(1, 6)]
asks = [(stations[i % 3], days[j]) for i, j in zip(rng.integers(0, 3, 2000), rng.zipf(1.6, 2000) % 7)]
t0 = time.perf_counter()
for s, d in asks:
    forecast(s, d, DATA_VERSION)
cached_s = time.perf_counter() - t0
info = forecast.cache_info()
show("requests, distinct questions", (len(asks), len(set(asks))))
show("cache hits, misses", (info.hits, info.misses))
show("time with cache vs without, about (s)", (round(cached_s, 2), round(len(asks) * 0.002, 2)))
DATA_VERSION = "2026-10-30"     # yesterday's rides arrived
forecast("Market", "2026-11-01", DATA_VERSION)
show("after new data: a miss, not a stale answer", forecast.cache_info().misses == info.misses + 1)

# ---------------------------------------------------------------------------
heading("6. Quantise a bigger model: smaller, almost as good")

big = build_features(make_rides(seed=810, days=730))            # two years this time
X, y = big[FEATURES], big["rides"]
train_mask = big["date"] < "2028-01-01"                           # train on the past, test on the future
scaler = StandardScaler().fit(X[train_mask])
mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=300, early_stopping=True, random_state=0)
mlp.fit(scaler.transform(X[train_mask]), y[train_mask])
Xs_test = scaler.transform(X[~train_mask])
y_test = y[~train_mask].to_numpy()


def forward(weights, biases, x):
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = x @ w + b
        if i < len(weights) - 1:
            x = np.maximum(x, 0)
    return x.ravel()


def quantise(w):
    """Symmetric 8-bit: store int8 values plus one float scale per layer."""
    scale = np.abs(w).max() / 127
    return np.round(w / scale).astype(np.int8), np.float32(scale)


w64 = [w.astype(np.float64) for w in mlp.coefs_]
b64 = [b.astype(np.float64) for b in mlp.intercepts_]
w32 = [w.astype(np.float32) for w in w64]
b32 = [b.astype(np.float32) for b in b64]
q = [quantise(w) for w in w64]
w8 = [qi.astype(np.float32) * s for qi, s in q]          # dequantise to use
x64, x32 = Xs_test.astype(np.float64), Xs_test.astype(np.float32)

size = {"float64": sum(w.nbytes for w in w64), "float32": sum(w.nbytes for w in w32),
        "int8": sum(qi.nbytes + 4 for qi, _ in q)}
mae = {"float64": mean_absolute_error(y_test, forward(w64, b64, x64)),
       "float32": mean_absolute_error(y_test, forward(w32, b32, x32)),
       "int8": mean_absolute_error(y_test, forward(w8, b32, x32))}
show("weights in the network", sum(w.size for w in w64))
show("weight size in KB: float64, float32, int8", tuple(round(v / 1024) for v in size.values()))
show("test MAE: float64, float32, int8", tuple(round(v, 3) for v in mae.values()))
show("largest change in one prediction, int8", round(float(np.max(np.abs(forward(w8, b32, x32) - forward(w64, b64, x64)))), 3))
ridge = Ridge().fit(X[train_mask], y[train_mask])
show("ridge test MAE, for comparison", round(mean_absolute_error(y_test, ridge.predict(X[~train_mask])), 3))
t64 = timed(lambda: forward(w64, b64, x64), 50)[0]
t32 = timed(lambda: forward(w32, b32, x32), 50)[0]
show("test rows", len(y_test))
show("forward pass, float64 vs float32 (ms)", ms((t64, t32)))
np.savez(OUT / "mlp_int8.npz", *[qi for qi, _ in q], scales=np.array([s for _, s in q]))
show("files written", sorted(p.name for p in OUT.iterdir()))

# ---------------------------------------------------------------------------
heading("7. Stop when it's fast enough")

final = timed(lambda: predict_lean("Market", 21.0, 0.0), 3000)
show("request p95: naive vs now (ms)", ms((p95, final[1])))
show("request budget met?", final[1] <= BUDGET_MS)
show("batch budget met?", batch_s <= BATCH_BUDGET_S)
