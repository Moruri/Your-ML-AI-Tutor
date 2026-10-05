"""
Lesson 028 - Mini-project: exploratory data analysis

A fresh dataset, start to finish: one month of rides from a small city
bike-share scheme, plus the daily weather. We go from a messy raw CSV to a
short written analysis with three charts you'd be happy to show a colleague,
using every Phase 1 tool along the way: cleaning (018), groupby (019),
joins (020), dates (021), honest plots (022), summaries (023), correlation
(024), the bootstrap (026) and a permutation test (027).

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes three PNG charts and a short report.md to output/ (next to
this file). Delete the folder and re-run at any time.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_september.csv"
WEATHER_CSV = HERE / "weather_september.csv"
OUT = HERE / "output"

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)
RNG = np.random.default_rng(28)   # reproducible resampling (lesson 016)
MAX_MINUTES = 180                  # longer than this = bike not docked properly


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        print(f"  {label}:")
        for line in value.to_string().splitlines():
            print(f"      {line}")
    else:
        print(f"  {label:<46} -> {value!r}")


def save(fig, name):
    """Save a figure to output/, close it, and report the file."""
    path = OUT / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    show(f"saved output/{name}", f"{path.stat().st_size:,} bytes")


# ---------------------------------------------------------------------------
# 1. First look: what did we actually get?
# ---------------------------------------------------------------------------

def section_first_look():
    heading("1. First look: what did we actually get?")
    raw = pd.read_csv(RIDES_CSV)
    show("rows, columns", raw.shape)
    show("first rows", raw.head(4))
    show("missing values per column", raw.isna().sum())
    show("station spellings", sorted(raw["start_station"].unique()))
    show("bike_type spellings", sorted(raw["bike_type"].unique()))
    show("duration_min summary", raw["duration_min"].describe().round(1))
    print("  Before any analysis, write down what looks wrong. Here: station")
    print("  names with stray spaces and lower case, two spellings each for")
    print("  bike types, missing durations and ratings, and a max duration")
    print("  of over a day, which no one actually rides.")
    return raw


# ---------------------------------------------------------------------------
# 2. Clean it, and keep a log of what you changed
# ---------------------------------------------------------------------------

def section_clean(raw):
    heading("2. Clean it, and keep a log of what you changed")
    log = []
    rides = raw.copy()

    rides["start_station"] = rides["start_station"].str.strip().str.title()
    rides["bike_type"] = (rides["bike_type"].str.lower()
                          .replace({"e-bike": "electric"}))
    log.append("tidied station names and bike types")

    before = len(rides)
    rides = rides.drop_duplicates()
    log.append(f"dropped {before - len(rides)} exact duplicate rows")

    before = len(rides)
    rides = rides.dropna(subset=["duration_min"])
    log.append(f"dropped {before - len(rides)} rides with no duration")

    before = len(rides)
    rides = rides[rides["duration_min"] <= MAX_MINUTES]
    log.append(f"dropped {before - len(rides)} rides over {MAX_MINUTES} minutes "
               "(bikes not docked)")

    rides["started_at"] = pd.to_datetime(rides["started_at"])
    rides["date"] = rides["started_at"].dt.normalize()
    rides["hour"] = rides["started_at"].dt.hour
    rides["weekend"] = rides["started_at"].dt.dayofweek >= 5
    rides["speed_kmh"] = rides["distance_km"] / (rides["duration_min"] / 60)
    log.append("added date, hour, weekend and speed columns")
    log.append(f"kept missing ratings as NaN ({rides['rating'].isna().sum()} rides)")

    for line in log:
        print(f"  - {line}")
    show("clean rows", len(rides))
    show("stations now", sorted(rides["start_station"].unique()))
    show("bike types now", sorted(rides["bike_type"].unique()))
    assert rides["ride_id"].is_unique
    assert rides["duration_min"].between(0, MAX_MINUTES).all()
    return rides, log


# ---------------------------------------------------------------------------
# 3. Who rides, and when?
# ---------------------------------------------------------------------------

def section_when(rides):
    heading("3. Who rides, and when?")
    per_day = rides.groupby(["weekend"])["date"].nunique()
    counts = rides.groupby("weekend").size()
    avg_per_day = (counts / per_day).round(1).rename({False: "weekday", True: "weekend"})
    show("average rides per day", avg_per_day)

    mix = (pd.crosstab(rides["weekend"], rides["rider_type"], normalize="index")
           .round(2).rename(index={False: "weekday", True: "weekend"}))
    show("rider mix (share of rides)", mix)

    by_hour = (rides.groupby(["weekend", "hour"]).size()
               .unstack("weekend", fill_value=0)
               .rename(columns={False: "weekday", True: "weekend"}))
    by_hour = by_hour.div(per_day.values, axis=1)   # rides per day, not totals
    peak_wd = int(by_hour["weekday"].idxmax())
    peak_we = int(by_hour["weekend"].idxmax())
    show("busiest weekday hour", f"{peak_wd}:00")
    show("busiest weekend hour", f"{peak_we}:00")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(by_hour.index, by_hour["weekday"], marker="o", label="Weekdays")
    ax.plot(by_hour.index, by_hour["weekend"], marker="o", label="Weekends")
    ax.set_title("Weekdays peak at commuting times; weekends peak after lunch")
    ax.set_xlabel("Hour the ride started")
    ax.set_ylabel("Average rides per day")
    ax.set_xticks(range(7, 22, 2))
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.text(0.99, -0.2, "Source: city bike-share export, September 2026",
            transform=ax.transAxes, ha="right", fontsize=8, color="grey")
    save(fig, "rides_by_hour.png")
    print("  Weekdays look like commuting: morning and early-evening spikes,")
    print("  mostly members. Weekends are a gentle hill after lunch, with")
    print("  far more casual riders.")
    return {"avg_per_day": avg_per_day, "mix": mix, "peak_wd": peak_wd, "peak_we": peak_we}


# ---------------------------------------------------------------------------
# 4. Members vs casual riders
# ---------------------------------------------------------------------------

def section_members(rides):
    heading("4. Members vs casual riders")
    summary = (rides.groupby("rider_type")["duration_min"]
               .agg(["count", "median", "mean"]).round(1))
    show("ride duration (minutes)", summary)

    member = rides.loc[rides["rider_type"] == "member", "duration_min"].to_numpy()
    casual = rides.loc[rides["rider_type"] == "casual", "duration_min"].to_numpy()
    boot = np.array([
        np.median(RNG.choice(casual, len(casual))) - np.median(RNG.choice(member, len(member)))
        for _ in range(2000)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    gap = float(np.median(casual) - np.median(member))
    show("median gap, casual - member (min)", round(gap, 1))
    show("95% bootstrap CI for that gap", (round(float(lo), 1), round(float(hi), 1)))

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 75, 2.5)
    ax.hist(member, bins=bins, alpha=0.6, label=f"Members (median {np.median(member):.0f} min)")
    ax.hist(casual, bins=bins, alpha=0.6, label=f"Casual riders (median {np.median(casual):.0f} min)")
    ax.set_title("Casual riders take rides about twice as long as members")
    ax.set_xlabel("Ride duration (minutes, rides over 75 min not shown)")
    ax.set_ylabel("Number of rides")
    ax.legend()
    save(fig, "duration_by_rider_type.png")
    print("  Durations are right-skewed (lesson 023), so we compare medians,")
    print("  and the interval (lesson 026) shows the gap is nowhere near zero.")
    return {"summary": summary, "gap": gap, "ci": (float(lo), float(hi))}


# ---------------------------------------------------------------------------
# 5. Does rain keep people off the bikes?
# ---------------------------------------------------------------------------

def section_weather(rides):
    heading("5. Does rain keep people off the bikes?")
    weather = pd.read_csv(WEATHER_CSV, parse_dates=["date"])
    daily = rides.groupby("date").agg(rides=("ride_id", "size"),
                                      weekend=("weekend", "first"))
    daily = daily.reset_index().merge(weather, on="date", how="left", validate="1:1")
    assert daily["rain_mm"].notna().all()
    daily["wet"] = daily["rain_mm"] > 0

    show("days, wet days", (len(daily), int(daily["wet"].sum())))
    wd = daily[~daily["weekend"]]
    by_wet = wd.groupby("wet")["rides"].mean().round(1).rename({False: "dry", True: "wet"})
    show("weekday rides per day, dry vs wet", by_wet)
    show("correlation, rain_mm vs rides (weekdays)",
         round(float(wd["rain_mm"].corr(wd["rides"])), 2))

    # permutation test (lesson 027): is the wet-day drop more than chance?
    dry = wd.loc[~wd["wet"], "rides"].to_numpy(dtype=float)
    wet = wd.loc[wd["wet"], "rides"].to_numpy(dtype=float)
    observed = dry.mean() - wet.mean()
    pooled = np.concatenate([dry, wet])
    null = np.empty(5000)
    for i in range(len(null)):
        s = RNG.permutation(pooled)
        null[i] = s[:len(dry)].mean() - s[len(dry):].mean()
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)
    show("dry - wet weekday gap (rides per day)", round(float(observed), 1))
    show("permutation p-value", round(float(p), 4))

    fig, ax = plt.subplots(figsize=(8, 4))
    for is_weekend, marker, label in [(False, "o", "Weekday"), (True, "s", "Weekend")]:
        part = daily[daily["weekend"] == is_weekend]
        ax.scatter(part["rain_mm"], part["rides"], marker=marker, label=label)
    ax.set_title("Rainier days had fewer rides, on weekdays and weekends")
    ax.set_xlabel("Rainfall that day (mm)")
    ax.set_ylabel("Rides that day")
    ax.set_ylim(bottom=0)
    ax.legend()
    save(fig, "rides_vs_rain.png")
    print("  We join on date (lesson 020) and split weekdays from weekends")
    print("  so the weekly rhythm doesn't muddy the weather effect (the")
    print("  confounding trap from lesson 024). The drop is big and the")
    print("  p-value tiny, but it's still one month: check it holds next month.")
    assert p < 0.001
    return {"by_wet": by_wet, "gap": float(observed), "p": float(p),
            "r": float(wd["rain_mm"].corr(wd["rides"])), "wet_days": int(daily["wet"].sum())}


# ---------------------------------------------------------------------------
# 6. Write it up
# ---------------------------------------------------------------------------

def section_report(rides, log, when, members, weather):
    heading("6. Write it up")
    avg = when["avg_per_day"]
    mix = when["mix"]
    m = members["summary"]
    lo, hi = members["ci"]
    lines = [
        "# Bike-share in September 2026: a first look",
        "",
        f"Based on {len(rides):,} cleaned rides over 30 days, plus daily weather.",
        "",
        "## What we found",
        "",
        f"1. **Two different kinds of use.** Weekdays averaged {avg['weekday']} rides a day, "
        f"peaking at {when['peak_wd']}:00, and {mix.loc['weekday', 'member']:.0%} were by members. "
        f"Weekends averaged {avg['weekend']} a day, peaking at {when['peak_we']}:00, and only "
        f"{mix.loc['weekend', 'member']:.0%} were by members. See `rides_by_hour.png`.",
        f"2. **Casual riders ride longer.** Median ride: {m.loc['casual', 'median']:.0f} min "
        f"for casual riders vs {m.loc['member', 'median']:.0f} min for members. The gap is "
        f"probably between {lo:.0f} and {hi:.0f} minutes (95% bootstrap interval). "
        "See `duration_by_rider_type.png`.",
        f"3. **Rain costs rides.** On weekdays, dry days averaged {weather['by_wet']['dry']} "
        f"rides and wet days {weather['by_wet']['wet']}, a drop of about "
        f"{weather['gap']:.0f} a day (permutation p < 0.001, "
        f"{weather['wet_days']} of 30 days had rain). See `rides_vs_rain.png`.",
        "",
        "## Caveats",
        "",
        "- One month of data; September weather and term dates may not be typical.",
        f"- Ratings are missing for {rides['rating'].isna().mean():.0%} of rides, "
        "so we didn't lean on them.",
        "- Rain and rides move together, but other things (cold, events) could also play a part.",
        "",
        "## What we cleaned",
        "",
        *[f"- {line[0].upper()}{line[1:]}." for line in log],
        "",
        "## Suggested next step",
        "",
        "Plan bike rebalancing around the weekday commuting peaks, and check next month "
        "whether the wet-day drop holds before using weather forecasts for staffing.",
    ]
    path = OUT / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    show("saved output/report.md", f"{path.stat().st_size:,} bytes")
    print()
    for line in lines[4:14]:
        print(f"  {line}")
    print("  ...")


def main():
    print("=" * 66)
    print("  Lesson 028: Mini-project: exploratory data analysis")
    print("=" * 66)
    OUT.mkdir(exist_ok=True)
    raw = section_first_look()
    rides, log = section_clean(raw)
    when = section_when(rides)
    members = section_members(rides)
    weather = section_weather(rides)
    section_report(rides, log, when, members, weather)
    print()
    print("Raw CSV in, three honest charts and a one-page write-up out.")
    print("That's Phase 1. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
