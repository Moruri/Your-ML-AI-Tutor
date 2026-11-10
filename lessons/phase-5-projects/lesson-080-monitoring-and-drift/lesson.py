"""
Lesson 080 - Monitoring and drift

A deployed model never throws an error when the world changes. It just keeps
answering, a little more wrongly each week. This lesson trains a ride model on
the spring and early summer, then watches twelve weeks of live traffic in which
three things happen: autumn arrives, a rival scheme opens near one station, and
another station's weather feed quietly switches to Fahrenheit. We build the
three monitors that catch them (input drift, prediction drift and live error),
see which monitor catches which change, and fix what we find.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

No downloads and no data file: the ride data is simulated. A small weekly
report is written to an `output/` folder next to this script (ignored by git),
which is wiped at the start of every run. It takes a second or two.
"""

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
OUT.mkdir()


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


# ---------------------------------------------------------------------------
# The world. Training: 13 weeks from 2026-05-04. Live: 12 weeks from 2026-08-03.
# Three things change in the live period, and nobody tells the model:
#   - from week 5, autumn: temperatures slide about 12 degrees by week 12;
#   - from week 3, a rival scheme near Market takes 30% of its riders;
#   - from week 9, Lakeside's weather feed sends Fahrenheit in the temp_c column.

STATIONS = ("Uni Square", "Market", "Lakeside")
BASE = {"Uni Square": 60, "Market": 45, "Lakeside": 30}
LIVE_START = pd.Timestamp("2026-08-03")
RIVAL_FROM, FAHRENHEIT_FROM, AUTUMN_FROM = 3, 9, 5


def make_world(seed=80):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-05-04", periods=25 * 7)
    rows = []
    for station in STATIONS:
        n = len(dates)
        t = np.arange(n)
        temp = 22 + 2.5 * np.sin(t / 6) - 12 * np.clip((t - 91 - 7 * (AUTUMN_FROM - 1)) / 56, 0, 1) + rng.normal(0, 1.5, n)
        rain = np.where(rng.random(n) < 0.25, rng.exponential(6, n), 0.0)
        weekend = (dates.dayofweek >= 5) * 12
        rides = BASE[station] + 1.2 * (temp - 22) - 0.9 * rain + weekend + rng.normal(0, 4, n)
        week = np.where(dates >= LIVE_START, (dates - LIVE_START).days // 7 + 1, 0)
        if station == "Market":
            rides = np.where(week >= RIVAL_FROM, rides * 0.7, rides)
        sent_temp = temp.copy()
        if station == "Lakeside":
            sent_temp = np.where(week >= FAHRENHEIT_FROM, temp * 9 / 5 + 32, temp)
        rows.append(pd.DataFrame({"date": dates, "week": week, "station": station,
                                  "temp_c": sent_temp.round(1), "rain_mm": rain.round(1),
                                  "rides": rides.round().clip(0).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def features(df):
    X = df[["temp_c", "rain_mm"]].copy()
    X["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    for s in STATIONS:
        X[f"at_{s}"] = (df["station"] == s).astype(int)
    return X


world = make_world()
ref = world[world["week"] == 0]
live = world[world["week"] > 0].copy()
model = Ridge().fit(features(ref), ref["rides"])
ref_pred = model.predict(features(ref))
live["pred"] = model.predict(features(live))
live["abs_err"] = (live["rides"] - live["pred"]).abs()

# ---------------------------------------------------------------------------
heading("1. A model that never complains")

baseline = round(float(mean_absolute_error(ref["rides"], ref_pred)), 2)
weekly_mae = live.groupby("week")["abs_err"].mean().round(2)
show("training days, live days", (ref["date"].nunique(), live["date"].nunique()))
show("MAE on the training period", baseline)
show("live MAE, week by week", weekly_mae.tolist())
show("errors or warnings raised by the model", 0)

# ---------------------------------------------------------------------------
heading("2. Input drift: compare this week with the reference")


def psi(reference, current, bins=10):
    """Population stability index: how far a distribution has moved from the reference."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


rng = np.random.default_rng(0)
temps = ref["temp_c"].to_numpy()
for n in (7, 300):
    noise = np.mean([psi(temps, rng.choice(temps, n)) for _ in range(200)])
    show(f"PSI of the reference vs itself, {n} rows", round(float(noise), 2))
show("PSI vs itself shifted up 3 degrees, 300 rows", round(psi(temps, rng.choice(temps, 300) + 3), 2))


def shift_score(reference, current):
    """How many reference standard deviations the mean has moved. Works on small windows."""
    return float((np.mean(current) - np.mean(reference)) / np.std(reference))


rows = []
for w, g in live.groupby("week"):
    row = {"week": w}
    for s in STATIONS:
        row[s] = shift_score(ref.loc[ref["station"] == s, "temp_c"], g.loc[g["station"] == s, "temp_c"])
    row["rain"] = shift_score(ref["rain_mm"], g["rain_mm"])
    rows.append(row)
drift = pd.DataFrame(rows).set_index("week").round(1)
flags = drift.abs() > 2
print()
print("  week  temp shift: Uni Sq  Market  Lakeside   rain shift")
for w, r in drift.iterrows():
    marks = [c for c in drift.columns if flags.loc[w, c]]
    print(f"  {w:>4}  {r.iloc[0]:>17.1f} {r.iloc[1]:>7.1f} {r.iloc[2]:>9.1f} {r['rain']:>12.1f}"
          + (f"  <- drift: {', '.join(marks)}" if marks else ""))
print()
first_flag = {c: (int(flags[c].idxmax()) if flags[c].any() else None) for c in drift.columns}
show("first week each input drifted", first_flag)
show("Lakeside temp_c mean, weeks 8 and 9", tuple(live[(live["station"] == "Lakeside") & live["week"].isin([8, 9])]
                                                .groupby("week")["temp_c"].mean().round(1)))

# ---------------------------------------------------------------------------
heading("3. Prediction drift: watch what the model says")

ref_mean = pd.Series(ref_pred, index=ref.index).groupby(ref["station"]).mean()
pred_week = live.groupby(["week", "station"])["pred"].mean().unstack()[list(STATIONS)]
moved = (pred_week - ref_mean[list(STATIONS)]).round(1)
show("mean prediction in training, by station", ref_mean[list(STATIONS)].round(1).to_dict())
show("change in mean prediction, week 2", moved.loc[2].to_dict())
show("change in mean prediction, week 12", moved.loc[12].to_dict())
show("Market, mean prediction weeks 1-2 vs 3-4", (round(float(pred_week.loc[1:2, "Market"].mean()), 1),
                                                  round(float(pred_week.loc[3:4, "Market"].mean()), 1)))

# ---------------------------------------------------------------------------
heading("4. Live error, once the real numbers arrive")

ref_err = pd.Series(np.abs(ref["rides"] - ref_pred), index=ref.index).groupby(ref["station"]).mean()
err_week = live.groupby(["week", "station"])["abs_err"].mean().unstack()[list(STATIONS)]
alarm = err_week > 1.5 * ref_err[list(STATIONS)]
two_in_a_row = alarm & alarm.shift(1, fill_value=False)
print()
print("  week   MAE: Uni Sq  Market  Lakeside")
for w, r in err_week.iterrows():
    marks = [s for s in STATIONS if two_in_a_row.loc[w, s]]
    print(f"  {w:>4}  {r.iloc[0]:>11.1f} {r.iloc[1]:>7.1f} {r.iloc[2]:>9.1f}" + (f"  <- alert: {', '.join(marks)}" if marks else ""))
print()
show("training MAE by station", ref_err[list(STATIONS)].round(1).to_dict())
first_alert = {s: (int(two_in_a_row[s].idxmax()) if two_in_a_row[s].any() else None) for s in STATIONS}
show("first alert week by station", first_alert)
m38 = live[(live["station"] == "Market") & live["week"].between(3, 8)]
show("Market, weeks 3-8: mean actual vs predicted", (round(float(m38["rides"].mean()), 1), round(float(m38["pred"].mean()), 1)))

# ---------------------------------------------------------------------------
heading("5. Which monitor caught what")

summary = {   # inputs: a drift flag for that station by the time it mattered
    "autumn (cooler days)": {"inputs": bool(flags["Uni Square"].any()),
                             "predictions": bool(moved.loc[12, "Uni Square"] < -5),
                             "error": bool(two_in_a_row["Uni Square"].any())},
    "rival scheme at Market": {"inputs": bool(flags.loc[RIVAL_FROM:RIVAL_FROM + 1, ["Market", "rain"]].any().any()),
                               "predictions": bool(abs(pred_week.loc[3:4, "Market"].mean() - pred_week.loc[1:2, "Market"].mean()) > 5),
                               "error": bool(two_in_a_row["Market"].any())},
    "Fahrenheit at Lakeside": {"inputs": bool(flags.loc[FAHRENHEIT_FROM, "Lakeside"]),
                               "predictions": bool(moved.loc[FAHRENHEIT_FROM, "Lakeside"] > 10),
                               "error": bool(two_in_a_row["Lakeside"].any())},
}
for event, seen in summary.items():
    print(f"  {event:<24} inputs {str(seen['inputs']):<6} predictions {str(seen['predictions']):<6} error {seen['error']}")

# ---------------------------------------------------------------------------
heading("6. Responding: fix the feed, then retrain on recent data")

later = live[live["week"] >= 10].copy()
fixed = later.copy()
lake = fixed["station"] == "Lakeside"
fixed.loc[lake, "temp_c"] = ((fixed.loc[lake, "temp_c"] - 32) * 5 / 9).round(1)   # convert at the door


def mae_by_station(m, df):
    err = np.abs(df["rides"] - m.predict(features(df)))
    return err.groupby(df["station"]).mean()[list(STATIONS)].round(1).to_dict()


show("weeks 10-12, old model, raw feed", mae_by_station(model, later))
show("weeks 10-12, old model, feed fixed", mae_by_station(model, fixed))
recent = live[live["week"].between(3, 8)]                       # after the rival opened, before the bad feed
window = pd.concat([ref, recent])
weights = np.where(window["week"] >= 3, 3.0, 1.0)                # recent weeks count more
retrained = Ridge().fit(features(window), window["rides"], sample_weight=weights)
show("weeks 10-12, retrained, feed fixed", mae_by_station(retrained, fixed))
only_recent = Ridge().fit(features(recent), recent["rides"])
show("weeks 10-12, trained on weeks 3-8 only", mae_by_station(only_recent, fixed))

report = {"baseline_mae": baseline, "weekly_mae": weekly_mae.to_dict(), "first_alert": first_alert,
          "input_shift": drift.to_dict(orient="index")}
(OUT / "monitoring_report.json").write_text(json.dumps(report, indent=2, default=float))
show("report written", sorted(p.name for p in OUT.iterdir()))
