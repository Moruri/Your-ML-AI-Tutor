# Lesson 021 - Exercises

Three quick checks, a hands-on task about the shop's rush hour, and an
optional question about a misleading weekly chart. Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-021-dates-and-time-series-basics/lesson.py my_lesson_021.py
python my_lesson_021.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-021-dates-and-time-series-basics").resolve()
```

`load_orders` and `ORDERS_CSV` are there to reuse.

---

## 1. What comes out?

```python
a = pd.to_datetime("2026-09-25 17:30")
b = pd.to_datetime("2026-09-28 07:00")
print(b - a)
print((b - a).total_seconds() / 3600)
print(a.day_name(), b.day_name())
```

<details>
<summary>Check yourself</summary>

```
2 days 13:30:00
61.5
Friday Monday
```

Subtracting gives a Timedelta; `.total_seconds()` turns it into a plain
number you can divide. That's the weekend: 61.5 hours between Friday's
closing and Monday's opening.

</details>

## 2. localize or convert?

For each, would you use `tz_localize` or `tz_convert`?

- (a) A CSV of naive times, which you know were recorded in Nairobi.
- (b) Timestamps already marked UTC, which you want to show in London time.
- (c) Naive times from a server whose documentation says "all times UTC".

<details>
<summary>Check yourself</summary>

- **(a) `tz_localize("Africa/Nairobi")`**: the times have no zone, and you
  are *telling* pandas which one they're in.
- **(b) `tz_convert("Europe/London")`**: they already have a zone; you want
  the same moments on a different clock.
- **(c) `tz_localize("UTC")`**, and then `tz_convert` if you need local
  time. You can't convert something that has no zone yet: pandas raises an
  error if you try, which is exactly what you want.

</details>

## 3. Why the difference?

`daily = by_time["revenue"].resample("D").sum()` has 30 rows. The shop
traded on 22 days. What are the other 8, and what would
`by_time["revenue"].resample("D").mean()` show for them instead of 0?

<details>
<summary>Check yourself</summary>

The eight weekend days (5-6, 12-13, 19-20, 26-27 September). `resample`
makes every bin in the range. The *sum* of no values is 0, but the *mean*
of no values is undefined, so `.mean()` gives `NaN` for those days.
That's actually more honest: "no data", not "zero". Whichever you get,
decide deliberately whether those days belong in your averages.

</details>

## 4. Hands-on: when is the rush?

The owner wants to know whether a second person behind the counter at the
morning peak is worth it.

**a) Cups per day.** Using `resample`, total cups (`quantity`) per
*trading* day. Which day sold the most cups, and which the fewest?

**b) The busiest 15 minutes.** Count orders in every 15-minute slot of the
month. Which slot was busiest, and how many orders did it have?

**c) Gaps between orders.** Within each day, how long passes between one
order and the next? Find the median gap, and the longest gap in any single
day (and when it ended).

**d) The morning share.** What fraction of the month's revenue came
between 07:00 and 08:59?

Hints, if you want them:

- For (b), `resample("15min")` then `.count()` on any column, then
  `.idxmax()` and `.max()`.
- For (c), sort by time, then
  `orders.groupby(orders["timestamp"].dt.date)["timestamp"].diff()`.
  Grouping by day stops the overnight gap from counting. `idxmax()` on the
  gaps gives a row label you can look up.
- For (d), a `DatetimeIndex` has `.between_time("07:00", "08:59")`.

<details>
<summary>Expected results</summary>

```
a) most:   Fri 04 Sep, 113 cups
   fewest: Thu 17 Sep, 68 cups
b) 2026-09-04 08:15 (08:15-08:29), 9 orders
c) median gap 6 minutes; longest 2 hours 20 minutes, ending 2026-09-07 16:57
d) 35.9% of revenue, in 2 of the shop's 10 hours
```

Nine orders in fifteen minutes is one every 100 seconds, with milk to
steam. And more than a third of the money arrives in the first two hours.
That's a reasonable case for a second pair of hands from 7 to 9, and not
much of one for the afternoon.

</details>

<details>
<summary>One way to write it</summary>

```python
orders = load_orders(ORDERS_CSV).sort_values("timestamp")
by_time = orders.set_index("timestamp")

# a)
cups = by_time["quantity"].resample("D").sum()
cups = cups[cups > 0]
print("most:  ", cups.idxmax().strftime("%a %d %b"), cups.max())
print("fewest:", cups.idxmin().strftime("%a %d %b"), cups.min())

# b)
per_slot = by_time["order_id"].resample("15min").count()
print(per_slot.idxmax(), per_slot.max())

# c)
gaps = orders.groupby(orders["timestamp"].dt.date)["timestamp"].diff().dropna()
print("median:", gaps.median())
print("longest:", gaps.max(), "ending", orders.loc[gaps.idxmax(), "timestamp"])

# d)
morning = by_time.between_time("07:00", "08:59")["revenue"].sum()
print(f"{morning / by_time['revenue'].sum():.1%}")
```

</details>

## 5. (Optional) The weekly chart that lies

Someone plots `resample("W-SUN").sum()` of revenue as a line and says
"Look, business fell off a cliff at the end of September." What's wrong,
and name two ways to fix the chart.

<details>
<summary>Check yourself</summary>

The last bin (labelled 4 October) only contains 28-30 September: three
trading days, not five. It's low because it's short. (The first bin, Tue-Fri,
is short too.)

Fixes: plot revenue *per trading day* in each week instead of the total
(divide by the number of trading days in the bin); drop or clearly mark the
partial weeks; or use a daily series with a 5-day rolling mean, which has
no partial-bin problem. Tomorrow's lesson is all about charts that don't
mislead like this.

</details>

---

That's the dates lesson done. If one thing sticks, let it be: *decide which
days count before you average, and store time in UTC*.
