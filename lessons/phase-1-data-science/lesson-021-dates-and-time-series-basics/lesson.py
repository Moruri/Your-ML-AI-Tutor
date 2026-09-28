"""
Lesson 021 - Dates and time series basics

Work with time without getting lost. Parse text into real datetimes, pull
parts out with the .dt accessor, slice by date, resample to daily, weekly and
hourly totals (and notice the empty weekends), smooth with rolling averages,
and attach a timezone so that times mean the same thing everywhere.
Finishes with the owner's "how did September go, day by day?" questions.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders_september.csv"      # the clean September from lessons 017 and 018

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)


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


def load_orders(path):
    """The clean month, timestamps parsed, with a revenue column."""
    orders = pd.read_csv(path, parse_dates=["timestamp"])
    orders["revenue"] = orders["price"] * orders["quantity"]
    return orders


# ---------------------------------------------------------------------------
# 1. Text that looks like a date isn't one
# ---------------------------------------------------------------------------

def section_parsing():
    heading("1. Text that looks like a date isn't one")
    raw = pd.read_csv(ORDERS_CSV)
    show("read_csv without parse_dates: dtype", str(raw["timestamp"].dtype))
    show("raw['timestamp'].max()  (just the 'biggest' text)", raw["timestamp"].max())

    ts = pd.to_datetime(raw["timestamp"], format="%Y-%m-%d %H:%M")
    show("pd.to_datetime(..., format=...): dtype", str(ts.dtype))
    show("ts.min(), ts.max()", (str(ts.min()), str(ts.max())))
    show("ts.max() - ts.min()  (a Timedelta)", str(ts.max() - ts.min()))
    show("ts[0] + pd.to_timedelta(90, unit='min')", str(ts[0] + pd.to_timedelta(90, unit="min")))

    messy = pd.Series(["2026-09-01 07:01", "01/09/2026 07:05", "not a date", None])
    show("messy text, errors='coerce', format='mixed', dayfirst=True",
         pd.to_datetime(messy, errors="coerce", format="mixed", dayfirst=True))
    print("  Anything that can't be read becomes NaT ('not a time'), the datetime NaN.")
    print("  03/04/2026 is 3 April in London and March 4 in New York: say which with")
    print("  format= (best) or dayfirst=. Never let pandas guess silently.")


# ---------------------------------------------------------------------------
# 2. The .dt accessor, and slicing by date
# ---------------------------------------------------------------------------

def section_dt(orders):
    heading("2. The .dt accessor, and slicing by date")
    ts = orders["timestamp"]
    parts = pd.DataFrame({
        "timestamp": ts,
        "date": ts.dt.date,
        "day_name": ts.dt.day_name(),
        "hour": ts.dt.hour,
        "week": ts.dt.isocalendar().week,
    }).head(3)
    show("parts of a timestamp", parts)
    print("  .dt is to datetimes what .str is to text: a toolbox for the whole column.")

    days = ts.dt.normalize()                           # midnight of each timestamp's day
    show("trading days in September", days.nunique())
    show("day names that appear", sorted(ts.dt.day_name().unique()))

    by_time = orders.set_index("timestamp").sort_index()
    show("rows on 2026-09-16  (by_time.loc['2026-09-16'])", len(by_time.loc["2026-09-16"]))
    show("rows 14-18 Sept  (by_time.loc['2026-09-14':'2026-09-18'])",
         len(by_time.loc["2026-09-14":"2026-09-18"]))
    mask = orders["timestamp"].between("2026-09-14", "2026-09-19")   # the same, as a mask
    show("same with between() on the column", int(mask.sum()))
    print("  With a DatetimeIndex, .loc takes date strings, and a slice includes BOTH ends,")
    print("  whole days included: '2026-09-18' means all of the 18th.")
    return by_time


# ---------------------------------------------------------------------------
# 3. resample: totals per day, week, hour
# ---------------------------------------------------------------------------

def section_resample(by_time):
    heading("3. resample: totals per day, week, hour")
    daily = by_time["revenue"].resample("D").sum()
    show("by_time['revenue'].resample('D').sum().head(7)", daily.head(7))
    show("len(daily)  (every calendar day, weekends too)", len(daily))
    print("  resample makes a row for EVERY day in the range. The shop is shut at weekends,")
    print("  so those rows are 0. An average over them would be wrong.")
    show("mean of daily incl. weekends", round(float(daily.mean()), 2))
    trading = daily[daily > 0]
    show("mean over trading days only", round(float(trading.mean()), 2))

    weekly = by_time["revenue"].resample("W-SUN").sum()
    show("resample('W-SUN').sum()  (weeks ending Sunday)", weekly)
    print("  Weekly bins are labelled by their LAST day. The first and last weeks of the")
    print("  month are partial (Tue-Fri and Mon-Wed), so they look low. Not a slump.")

    hourly = by_time["revenue"].resample("h").sum()
    show("resample('h') rows  (every hour of the month)", len(hourly))
    show("... of which had any sales", int((hourly > 0).sum()))
    typical = by_time.groupby(by_time.index.hour)["revenue"].sum()
    show("groupby(index.hour): revenue by hour of day", typical)
    print("  resample = a timeline (this Tuesday 8am, then 9am...). groupby(hour) = a")
    print("  typical day (all the 8ams together). Different questions, different tools.")
    return daily, trading


# ---------------------------------------------------------------------------
# 4. Rolling averages: seeing the trend through the wiggle
# ---------------------------------------------------------------------------

def section_rolling(trading):
    heading("4. Rolling averages: seeing the trend through the wiggle")
    table = pd.DataFrame({
        "revenue": trading,
        "rolling_5": trading.rolling(5).mean().round(2),
        "rolling_5_min3": trading.rolling(5, min_periods=3).mean().round(2),
    })
    table.index = table.index.strftime("%a %d")
    show("trading-day revenue with 5-day rolling means", table)
    print("  rolling(5) averages each day with the 4 before it: one trading week. The")
    print("  first 4 rows are NaN because there isn't a full window yet (min_periods")
    print("  relaxes that). Each value only uses the past, so it lags the raw numbers.")
    compare = pd.DataFrame({"rolling(5)": trading.rolling(5).mean(),
                            "rolling('7D')": trading.rolling("7D").mean()}).round(2).head(7)
    compare.index = compare.index.strftime("%a %d")
    show("5 rows vs 7 calendar days", compare)
    print("  A number counts ROWS; a string like '7D' counts TIME. Here a calendar week")
    print("  always holds 5 trading days, so they agree once the window fills. The time")
    print("  window just starts straight away, with whatever days it has.")


# ---------------------------------------------------------------------------
# 5. Timezones
# ---------------------------------------------------------------------------

def section_timezones(orders):
    heading("5. Timezones")
    first = orders["timestamp"].iloc[0]
    show("first timestamp  (naive: no timezone)", str(first))
    show("its .tz", first.tz)
    london = orders["timestamp"].dt.tz_localize("Europe/London")
    show("tz_localize('Europe/London')", str(london.iloc[0]))
    show("tz_convert('UTC')", str(london.dt.tz_convert("UTC").iloc[0]))
    show("tz_convert('Africa/Nairobi')", str(london.dt.tz_convert("Africa/Nairobi").iloc[0]))
    print("  localize = 'these times were London times' (adds a label, same clock).")
    print("  convert  = 'show me the same moment somewhere else' (the clock changes).")

    clocks_change = pd.date_range("2026-10-25 00:00", periods=4, freq="h", tz="UTC")
    show("UTC hours on 25 Oct 2026, as London time",
         pd.Series(clocks_change.tz_convert("Europe/London").strftime("%H:%M %Z"),
                   index=clocks_change.strftime("%H:%M UTC")))
    print("  London's clocks go back that night: 01:00 happens twice. Naive local times")
    print("  can't tell those apart. Store and compute in UTC; convert to local for people.")


# ---------------------------------------------------------------------------
# 6. Putting it together: September, day by day
# ---------------------------------------------------------------------------

def section_together(orders, daily, trading):
    heading("6. Putting it together: September, day by day")
    worst = trading.nsmallest(3)
    worst.index = worst.index.strftime("%a %d %b")
    show("three quietest trading days", worst.round(2))
    print("  16 and 17 September were the two rainy days, back to back; that's the dip")
    print("  in the rolling mean. Tuesday 29th is low too, but on its own.")

    by_weekday = trading.groupby(trading.index.day_name()).mean()
    by_weekday = by_weekday.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    show("average revenue per trading day, by weekday", by_weekday.round(2))

    by_time = orders.set_index("timestamp")
    card = (by_time["payment"] == "card").resample("W-SUN").mean()
    card.index = card.index.strftime("w/e %d %b")
    show("card share of orders, by week", card.round(3))

    assert len(orders) == 1214
    assert round(daily.sum(), 2) == 6627.50
    assert len(trading) == 22
    assert set(orders["timestamp"].dt.dayofweek) <= {0, 1, 2, 3, 4}
    assert by_weekday.idxmax() == "Friday"
    assert set(worst.index.str[4:6]) >= {"16", "17"}
    assert card.iloc[-1] > card.iloc[0]
    print(f"  22 trading days, £{daily.sum():,.2f}, Fridays busiest, card share up in the")
    print("  second half of the month.")


def main():
    print("=" * 66)
    print("  Lesson 021: Dates and time series basics")
    print("=" * 66)
    orders = load_orders(ORDERS_CSV)
    section_parsing()
    by_time = section_dt(orders)
    daily, trading = section_resample(by_time)
    section_rolling(trading)
    section_timezones(orders)
    section_together(orders, daily, trading)
    print()
    print("Parse on the way in, resample for timelines, groupby for typical days, and")
    print("keep a timezone on anything that crosses a border. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
