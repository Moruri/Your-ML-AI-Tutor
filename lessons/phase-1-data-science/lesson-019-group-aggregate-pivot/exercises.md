# Lesson 019 - Exercises

Three quick checks, a hands-on task following oat milk through September,
and an optional question about averages of averages. Predict first, then
run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-019-group-aggregate-pivot/lesson.py my_lesson_019.py
python my_lesson_019.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-019-group-aggregate-pivot").resolve()
```

`load_orders`, `WEEKDAYS` and `SIZES` are there to reuse.

---

## 1. Which question does each line answer?

Write each one as a plain-English question.

```python
orders.groupby("size")["revenue"].sum()
orders.groupby("size")["revenue"].mean()
orders.groupby("size").size()
orders.groupby(["payment", "size"])["quantity"].sum().unstack()
orders.pivot_table(index="hour", values="order_id", aggfunc="count")
```

<details>
<summary>Check yourself</summary>

- Total revenue from each size of cup, over the month.
- The average order value, for orders of each size.
- How many orders were for each size.
- Cups sold, for each payment method and size, as a payment-by-size
  table.
- How many orders came in during each hour of the day (all days
  together).

If you found yourself writing "the average... of what, exactly?", good.
That's the habit.

</details>

## 2. agg or transform?

For each, would you use `agg` (one row per group) or `transform` (a
value on every row)?

- (a) A table of total revenue per drink, for a report.
- (b) Flag every order that's more than twice its drink's average.
- (c) Each order's share of its hour's revenue.
- (d) The number of distinct drinks sold each day.

<details>
<summary>Check yourself</summary>

- **(a) `agg`** (or just `.sum()`): one row per drink is the answer.
- **(b) `transform`**: you need the drink average *next to each order*
  to compare: `orders["revenue"] > 2 * orders.groupby("drink")["revenue"].transform("mean")`.
- **(c) `transform`**: divide by `groupby(hour_key)["revenue"].transform("sum")`.
  (Group by day *and* hour if you mean "that particular hour on that
  day".)
- **(d) `agg`**: `orders.groupby("day")["drink"].nunique()`, one row per
  day.

Rule of thumb: if the answer is a *table about the groups*, `agg`. If it's
a *new column on the original table*, `transform`.

</details>

## 3. Spot the trap

A colleague reports "Tuesday and Wednesday are our best days" using:

```python
orders.groupby("weekday")["revenue"].sum()
```

What's wrong, and what would you run instead?

<details>
<summary>Check yourself</summary>

September 2026 has five Tuesdays and five Wednesdays but only four
Mondays, Thursdays and Fridays. Summing per weekday rewards Tuesday and
Wednesday for simply occurring more often. Total per day, then average the
days per weekday (section 7):

```python
daily = orders.groupby(["day", "weekday"], as_index=False)["revenue"].sum()
daily.groupby("weekday")["revenue"].mean()
```

That shows Friday is the best day by some way, and Wednesday is actually
the *quietest* on average. Sums compare fairly only when every group had
the same chance to add up.

</details>

## 4. Hands-on: oat milk's September

The owner thinks oat milk is growing and wants to know whether to order
more. Answer with groupby, pivot_table and crosstab.

**a) Oat share by week.** Among drinks with milk, what share were oat in
each ISO week? (`orders["timestamp"].dt.isocalendar().week` gives the
week number.) Include how many milky orders each week had.

**b) Is the last week really lower?** Week 40 looks like a dip. How many
days of September fall in week 40, and what does that do to how much you
trust its number?

**c) Oat by drink and size.** A pivot table of oat *share* with drinks
down the side and sizes across the top. (Hint: make a True/False column
`is_oat`, then `aggfunc="mean"`.)

**d) Card vs cash by weekday.** The share of orders paid by card on each
weekday, in calendar order.

**e) Each drink's best day.** For each drink, the date it took the most
revenue, and how much.

Hints, if you want them:

- For (a), filter to milky rows first, then
  `.groupby(week)["is_oat"].agg(orders="count", oat_share="mean")`.
- For (d), `pd.crosstab(orders["weekday"], orders["payment"], normalize="index").reindex(WEEKDAYS)`.
- For (e), group by `["drink", "day"]`, sum revenue, then within each
  drink use `.idxmax()` and `.max()`:
  `per_day.groupby("drink").idxmax()` gives `(drink, day)` pairs.

<details>
<summary>Expected results</summary>

```
a) week  orders  oat_share
     36     171      0.187
     37     216      0.241
     38     200      0.265
     39     212      0.283
     40     114      0.237

b) Week 40 is only 28-30 September: 3 days, 114 milky orders.
   Its share is based on about half as many orders as the others,
   so it wobbles more (lesson 016). The trend from 36 to 39 is the
   stronger evidence.

d) Monday 76%, Tuesday 75%, Wednesday 80%, Thursday 76%, Friday 82% card

e) cappuccino  2026-09-25  £130.60
   espresso    2026-09-09   £28.90
   flat white  2026-09-04  £136.20
   latte       2026-09-23  £147.50
   tea         2026-09-11   £47.50
```

(a) is a real trend: from under a fifth to over a quarter in four weeks.
But notice how (b) changes what you'd say. A careful answer to the owner:
"oat has been rising steadily through September, from about 19% to 28%
of milky drinks; the last few days look lower, but that's only three days
of data." The number of rows behind each figure is part of the answer.
Always put the count next to the share.

(c)'s exact numbers are yours to find; look for whether any drink-size
combination stands out, and check how many orders it's based on before
you believe it.

</details>

<details>
<summary>One way to write it</summary>

```python
orders = load_orders(ORDERS_CSV)
orders["day"] = orders["timestamp"].dt.date
orders["week"] = orders["timestamp"].dt.isocalendar().week
milky = orders[orders["milk"] != "none"].copy()
milky["is_oat"] = milky["milk"] == "oat"

# a)
print(milky.groupby("week")["is_oat"].agg(orders="count", oat_share="mean").round(3))

# b)
print(orders.loc[orders["week"] == 40, "day"].nunique(), "days in week 40")

# c)
print(milky.pivot_table(index="drink", columns="size", values="is_oat", aggfunc="mean")[SIZES].round(3))

# d)
print(pd.crosstab(orders["weekday"], orders["payment"], normalize="index").reindex(WEEKDAYS).round(3))

# e)
per_day = orders.groupby(["drink", "day"])["revenue"].sum()
best = pd.DataFrame({"day": per_day.groupby("drink").idxmax().str[1],
                     "revenue": per_day.groupby("drink").max()})
print(best)
```

In (e), `idxmax()` on a MultiIndex Series returns the *whole* label,
`(drink, day)`, and `.str[1]` picks the second part of each tuple. A
little awkward; `per_day.reset_index().sort_values("revenue").groupby("drink").tail(1)`
is another way, which some people find easier to read.

</details>

## 5. (Optional) The average of averages

The owner wants "the average order value for September" and a colleague
offers:

```python
orders.groupby("drink")["revenue"].mean().mean()
```

It gives about £5.09. `orders["revenue"].mean()` gives £5.46. Which is
right, and why do they differ?

<details>
<summary>Check yourself</summary>

`orders["revenue"].mean()`, £5.46, is the average order value.

The other is an *average of five averages*, one per drink, and it treats
each drink as equally important: espresso's £2.69 average, from 156
orders, counts exactly as much as latte's £6.11, from 377. That drags the
answer down towards the drinks that are cheap *and* less popular.

Averages of averages are only equal to the overall average when every
group is the same size. When groups differ in size (they almost always
do), average the raw rows, or weight each group's average by its count.
This is a classic way reports go subtly wrong, and now you'll spot it.

</details>

---

That's the morning of Week 3, Day 4 done. This afternoon, lesson 020 joins
the orders to the menu. If one thing sticks, let it be: *say the question
with its units before you groupby: per order, per day, or per drink?*
