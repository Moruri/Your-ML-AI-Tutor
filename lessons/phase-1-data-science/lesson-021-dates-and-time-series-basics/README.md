# Lesson 021 - Dates and time series basics

**Phase 1 - Data science basics** | Week 3, Day 5 | Friday 2026-09-25

> **Goal:** parse dates, resample, compute rolling windows and handle
> timezones without getting lost, so that "per day", "per week" and
> "the trend" become one-liners instead of headaches.

Time: about 50 minutes. Needs the venv from lesson 013 (`pandas`).

---

Every order the shop takes has a timestamp. So far we've used it lightly:
lesson 019 pulled out the weekday and the hour. But the owner's questions
are really about **time**:

- "What did we take each day? Each week?"
- "Is business going up or down, once you ignore the day-to-day wiggle?"
- "Our card reader company reports in UTC. Why don't their numbers match
  ours?"

Time is the most common kind of data there is (sales, sensor readings,
website visits, heart rates), and it has traps that numbers and text
don't: text that only *looks* like a date, days with no data, weeks that
start on different days, and clocks that jump an hour twice a year.
pandas has good tools for all of them, and this lesson walks through them
in order.

## How to follow along

Venv active, `python` in the repo root. The data is `orders_september.csv`,
the clean month from lessons 017-019. Full script:

```bash
python lessons/phase-1-data-science/lesson-021-dates-and-time-series-basics/lesson.py
```

It writes no files.

## 1. Text that looks like a date isn't one

Read the CSV without telling pandas about dates, and the timestamp column
is plain text (`object`):

```
read_csv without parse_dates: dtype            -> 'object'
```

Text *sorts* fine here only because it's written year-month-day. You
can't subtract it, ask its weekday, or group it by week. Turn it into real
datetimes, either while reading (`parse_dates=["timestamp"]`, as lessons
017-019 did) or afterwards:

```python
ts = pd.to_datetime(raw["timestamp"], format="%Y-%m-%d %H:%M")
```

```
ts.max() - ts.min()  (a Timedelta)             -> '29 days 09:56:00'
ts[0] + pd.to_timedelta(90, unit='min')        -> '2026-09-01 08:31:00'
```

Now the column has dtype `datetime64[ns]`. Subtracting two datetimes gives a
**Timedelta** (a length of time), and adding a Timedelta to a datetime
moves it along.

`format=` uses the same codes as lesson 007's `strftime`: `%Y` year, `%m`
month, `%d` day, `%H:%M` hours and minutes. Giving it is faster, and it's
*safer*: `03/04/2026` is 3 April in London and 4 March in New York. If you
can't give a format (mixed styles, messy exports), say `dayfirst=True` or
not, and use `errors="coerce"` so anything unreadable becomes **`NaT`**
("not a time"), the datetime version of `NaN`. Then count the `NaT`s, as
lesson 018 did with `NaN`s.

## 2. The `.dt` accessor, and slicing by date

`.dt` is to a datetime column what `.str` is to a text column: a toolbox
that works on every row at once.

```
            timestamp        date day_name  hour  week
0 2026-09-01 07:01:00  2026-09-01  Tuesday     7    36
```

| Code | Gives |
|------|-------|
| `ts.dt.date` | The date part (as Python dates). |
| `ts.dt.normalize()` | Midnight of the same day (stays a datetime, groups nicely). |
| `ts.dt.day_name()`, `ts.dt.dayofweek` | `"Tuesday"`, or 0 = Monday ... 6 = Sunday. |
| `ts.dt.hour`, `.minute`, `.month`, `.year` | The obvious parts. |
| `ts.dt.isocalendar().week` | ISO week number (weeks start Monday). |
| `ts.dt.strftime("%a %d %b")` | Formatted text, e.g. `"Tue 01 Sep"`. |

Every September order falls on a weekday: 22 trading days, Monday to Friday.

To pick out a period, make the timestamp the **index**. A
`DatetimeIndex` lets `.loc` take date strings:

```python
by_time = orders.set_index("timestamp").sort_index()
by_time.loc["2026-09-16"]                  # all of the 16th: 48 orders
by_time.loc["2026-09-14":"2026-09-18"]     # Mon-Fri of week 38: 274 orders
```

A partial date means "all of it", so `"2026-09-18"` at the end of a
slice includes the whole of the 18th. Unlike Python slices, both ends are
included. Without an index, `orders["timestamp"].between("2026-09-14",
"2026-09-19")` gives the same 274 rows, but note the end has to be the 19th
there, since a bare date means midnight.

## 3. `resample`: totals per day, week, hour

`resample` is `groupby` for time: it cuts a `DatetimeIndex` into regular
bins and summarises each one.

```python
daily = by_time["revenue"].resample("D").sum()
```

```
2026-09-04    406.9
2026-09-05      0.0
2026-09-06      0.0
2026-09-07    261.5
```

Look at 5 and 6 September. `resample` makes a row for **every** day in the
range, including the weekends when the shop is shut, and a sum of nothing
is 0. That's useful (it shows the gaps), but it's a trap for averages:

```
mean of daily incl. weekends                   -> 220.92
mean over trading days only                    -> 301.25
```

An average day is £301, not £221. Decide which days *count* before you
average. Here, `daily[daily > 0]` keeps the trading days.

Common frequencies: `"D"` day, `"W-SUN"` weeks ending Sunday (`"W-MON"`
etc. for other days), `"MS"` month start, `"h"` hour, `"15min"`.

```
resample('W-SUN').sum()  (weeks ending Sunday):
2026-09-06    1290.2
...
2026-10-04     770.2
```

Two things to notice. Weekly bins are **labelled by their last day**, so
"2026-10-04" means the week of 28 September to 4 October. And the first
and last weeks are partial (four days and three days), so they look low.
That's the calendar, not a slump.

`resample("h")` makes all 706 hours of the month, only 214 of which had a
sale. For "what does a typical day look like?", group by the hour instead:

```python
by_time.groupby(by_time.index.hour)["revenue"].sum()
```

`resample` gives a **timeline** (Tuesday 8am, then Tuesday 9am ...).
`groupby(hour)` gives a **typical day** (all the 8ams together). Different
questions.

## 4. Rolling averages: seeing the trend through the wiggle

Daily revenue jumps around: Fridays are high, rainy days are low. To see the
underlying trend, average each day with the few before it:

```python
trading.rolling(5).mean()
```

```
           revenue  rolling_5
Tue 01       342.3        NaN
...
Mon 07       261.5     310.34
...
Wed 16       234.7     310.58
Thu 17       234.2     285.00
Fri 18       350.3     284.44
```

With five trading days a week, `rolling(5)` is "the last trading week". The
first four values are `NaN` because the window isn't full yet;
`min_periods=3` would allow partial windows. A rolling mean only looks
backwards, so it **lags**: the rain on the 16th and 17th pulls the average
down for the next few days, even after business recovers.

A number counts **rows**; a string counts **time**. `rolling("7D")` means
"the last seven calendar days". With data only on weekdays, a calendar week
always holds five trading days, so here the two agree once the window
fills. With irregular data (some days missing, several readings on others)
they don't, so say which you mean.

## 5. Timezones

The timestamps in the file are **naive**: `07:01` with no timezone. The shop
knows that means London. A computer doesn't.

```python
london = orders["timestamp"].dt.tz_localize("Europe/London")
london.dt.tz_convert("UTC")
```

```
tz_localize('Europe/London')                   -> '2026-09-01 07:01:00+01:00'
tz_convert('UTC')                              -> '2026-09-01 06:01:00+00:00'
tz_convert('Africa/Nairobi')                   -> '2026-09-01 09:01:00+03:00'
```

- **`tz_localize`** says "these times *were* London times". It adds a
  label; the clock reading stays 07:01. In September London is on British
  Summer Time, one hour ahead of UTC (`+01:00`).
- **`tz_convert`** says "show me the same *moment* somewhere else". The
  clock reading changes: 06:01 in UTC, 09:01 in Nairobi.

That's the card-reader mystery: their report, in UTC, puts the 07:xx rush
in the 06:00 hour, and any late order near midnight would land on a
different *date*. Convert one side before comparing.

And once a year, local time is ambiguous. On 25 October 2026 London's
clocks go back:

```
00:00 UTC    01:00 BST
01:00 UTC    01:00 GMT
```

01:00 happens twice. A naive "01:30" can't say which one. The rule
professionals follow is: **store and compute in UTC, convert to local time
for people to read.**

## 6. Putting it together: September, day by day

```
three quietest trading days:
Tue 29 Sep    227.2
Thu 17 Sep    234.2
Wed 16 Sep    234.7

average revenue per trading day, by weekday:
Monday       277.18
...
Friday       367.18

card share of orders, by week:
w/e 06 Sep    0.755
w/e 13 Sep    0.747
w/e 20 Sep    0.712
w/e 27 Sep    0.857
w/e 04 Oct    0.862
```

The story for the owner: 16 and 17 September were the two rainy days,
back to back, which is why the rolling mean dipped mid-month. Fridays take
about £90 more than a typical Monday. And card payments jumped from about
72% to 86% of orders in the second half of the month. Worth asking what
changed (a new card reader? a "cashless" sign?), which is a question the
data can prompt but can't answer.

## What you can do now

- Parse text into datetimes with `pd.to_datetime`, using `format=`, and
  catch bad values with `errors="coerce"` and `NaT`.
- Do arithmetic with Timedeltas.
- Pull parts out with `.dt`, and slice with a `DatetimeIndex` and `.loc`.
- `resample` to days, weeks and hours, and handle the empty bins honestly.
- Tell a timeline (`resample`) from a typical day (`groupby` on the hour).
- Smooth with `rolling`, and know that it lags and that numbers count rows
  while strings count time.
- `tz_localize` and `tz_convert`, and why UTC is the safe place to store
  time.

## What to do now

1. Run `lesson.py`. Try `resample("W-MON")` and see how the weekly labels
   and totals change.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   finds the busiest 15 minutes of the month.
3. Next, lesson 022: plotting that tells the truth. The numbers from today
   make much better pictures than tables. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Time looks simple until it isn't. Parse it properly, decide which days
count, and put a timezone on anything that might travel.
