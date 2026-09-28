# Lesson 018 - Exercises

Three quick checks, a hands-on task that throws new mess at `clean()` and
teaches it to reject what it can't fix, and an optional question about
filling. Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-018-cleaning-data-missing-values-types-and-duplicates/lesson.py my_lesson_018.py
python my_lesson_018.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-018-cleaning-data-missing-values-types-and-duplicates").resolve()
```

`RAW_CSV`, `clean`, `check`, `expected_price`, `MEDIUM_PRICE`,
`DRINK_ALIASES` and `NUMBER_WORDS` are there to reuse.

---

## 1. What prints?

```python
import numpy as np
import pandas as pd

s = pd.Series([4.0, np.nan, 2.0, np.nan])
print(s.sum())
print(s.mean())
print(s.count())
print(len(s))
print(s.isna().mean())
print((s == np.nan).sum())
```

<details>
<summary>Check yourself</summary>

```
6.0      NaN skipped
3.0      (4 + 2) / 2: skipped in the count too
2        count() counts non-missing values
4        len() counts everything
0.5      half the values are missing: isna().mean() is the missing SHARE
0        NaN never equals anything, so == np.nan finds nothing. Use isna().
```

The gap between `count()` (2) and `len()` (4) is a quick way to see how
much is missing. `s.isna().mean()` is the same thing as a share, and
`df.isna().mean()` gives it for every column at once.

</details>

## 2. Fill, drop, or label?

For each gap, what would you do, and what's your one-sentence reason?

- (a) A `size` column is blank for 3 of 1,200 espresso orders. Espresso
  only comes in one size.
- (b) A `customer_age` column is blank for 40% of rows, because the
  loyalty app only asks some customers.
- (c) A `temperature` column (the day's weather) is blank for one day in
  the middle of a month of daily readings.
- (d) A `refund_amount` column is blank for most rows.

<details>
<summary>Check yourself</summary>

- **(a) Fill** with the one size. "Espresso only comes in one size" is a
  true sentence.
- **(b) Label/leave missing.** 40% is far too many to invent, and the
  people who *are* asked may differ from those who aren't. Filling with
  the average age would make the data look much more certain than it is.
  Analyse age only where it's known, and say so.
- **(c) It depends, and saying so is the right answer.** For a chart, you
  might fill from the neighbouring days (lesson 021 shows how) and note it.
  For a precise calculation, leave it missing. Either way, one gap in a
  smooth series is a very different thing from 40% missing.
- **(d) Probably fill with 0**, but only after checking. If blank means
  "no refund", 0 is true. If blank means "refund not recorded yet", 0 is a
  lie. Find out which before you fill.

</details>

## 3. Coerce, or not?

A colleague writes:

```python
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])
```

What happens to a row whose price is `£3.80`? Why is this worse than it
looks?

<details>
<summary>Check yourself</summary>

`£3.80` can't be read as a number, so `errors="coerce"` turns it into
`NaN`, and `dropna` deletes the row. A perfectly good sale disappears,
with no message. On this month's raw file, that would silently remove
every `£` price *and* every blank price: 40 real orders, about £200.

The fix is lesson 018's order: coerce to *find* the problems, look at
them, fix them (`str.replace("£", "")`), then convert strictly so any
problem you didn't expect raises an error. Coerce-then-drop in one go
turns "I don't understand this value" into "this row never existed".

</details>

## 4. Hands-on: new mess, and a rejects pile

Next month's file will have problems `clean()` has never seen. Simulate
four of them by adding rows to the raw data:

```python
raw = pd.read_csv(RAW_CSV)
extra = pd.DataFrame([
    {"order_id": "S9001", "timestamp": "2026-09-30 17:05", "drink": "Mocha", "size": "medium",
     "milk": "whole", "price": "4.00", "quantity": "1", "payment": "card"},
    {"order_id": "S9002", "timestamp": "2026-09-30 17:10", "drink": "latte", "size": "medium",
     "milk": "whole", "price": "3.80", "quantity": "-1", "payment": "card"},
    {"order_id": "S9003", "timestamp": "2026-09-30 17:15", "drink": "tea", "size": "large",
     "milk": "none", "price": "free", "quantity": "1", "payment": "cash"},
    {"order_id": "S0100", "timestamp": "2026-09-30 17:20", "drink": "latte", "size": "small",
     "milk": "whole", "price": "3.30", "quantity": "2", "payment": "card"},
])
nastier = pd.concat([raw, extra], ignore_index=True)
```

**a) What breaks?** Run `clean(nastier)`. Which of the four rows makes it
fail, and with what error? (Predict first.) Which of the other three would
have got through `clean` without the failure, and would `check` have
caught them?

**b) A rejects pile.** Write `clean_with_rejects(raw)` that returns
`(orders, rejects)`:

- `rejects` is a DataFrame of the raw rows that can't be cleaned, with a
  `reason` column: drink not on the menu, price that isn't a number
  (and isn't blank), quantity that isn't a whole number of at least 1,
  or an order ID that appears on two rows with *different* details.
- `orders` is `clean(...)` applied to everything else, and must pass
  `check`.

**c) Report.** How many orders, how much revenue, how many rejects, and
the reason for each.

Hints, if you want them:

- Work on a deduplicated copy (`raw.drop_duplicates()`), so exact
  duplicates are still quietly removed and only *conflicting* IDs are
  rejected.
- Use `errors="coerce"` to make helper columns (`price_num`,
  `quantity_num`) for testing, while keeping the originals untouched.
- Build a `reasons` Series of empty strings with the same index, and add
  text with masks: `reasons[mask] += "drink not on menu; "`. Rows whose
  reason is still `""` are good.
- For conflicting IDs, `df["order_id"].duplicated(keep=False)` *after*
  dropping exact duplicates marks both rows of each conflicting pair.

<details>
<summary>Expected results</summary>

```
a) ValueError: Unable to parse string "free" at position 1216
   (the strict to_numeric on price)
   Without it: Mocha would reach check() and fail "only drinks on the menu";
   quantity -1 would fail "quantity >= 1"; the second S0100 would fail
   "one row per order". clean() + check() catch all four, one at a time.

c) 1,213 orders, £6,623.70, 5 rejects
   S0100  order_id used twice      (the original, from 2 September)
   S9001  drink not on menu
   S9002  bad quantity
   S9003  price not a number
   S0100  order_id used twice      (the new one)
```

Both copies of S0100 go to the rejects pile, including the original
September order. That's deliberate: when two rows claim the same ID with
different details, you can't know which is right, so a human decides.
That's why revenue is £3.80 lower than £6,627.50: the original S0100 was
a medium latte. It'll come back once someone checks it.

</details>

<details>
<summary>One way to write it</summary>

```python
def clean_with_rejects(raw):
    df = raw.drop_duplicates().copy()
    df["drink"] = df["drink"].str.strip().str.lower().replace(DRINK_ALIASES)
    price_num = pd.to_numeric(df["price"].str.replace("£", "", regex=False), errors="coerce")
    quantity_num = pd.to_numeric(df["quantity"].str.strip().replace(NUMBER_WORDS), errors="coerce")

    reasons = pd.Series("", index=df.index)
    reasons[~df["drink"].isin(list(MEDIUM_PRICE))] += "drink not on menu; "
    reasons[df["price"].notna() & price_num.isna()] += "price not a number; "
    reasons[quantity_num.isna() | (quantity_num < 1)] += "bad quantity; "
    reasons[df["order_id"].duplicated(keep=False)] += "order_id used twice; "

    bad = reasons != ""
    rejects = raw.loc[df.index[bad]].assign(reason=reasons[bad].str.rstrip("; "))
    orders = clean(raw.loc[df.index[~bad]])
    return orders, rejects


orders, rejects = clean_with_rejects(nastier)
check(orders)
revenue = (orders["price"] * orders["quantity"]).sum()
print(f"{len(orders):,} orders, £{revenue:,.2f}, {len(rejects)} rejects")
print(rejects[["order_id", "drink", "price", "quantity", "reason"]])
```

This is lesson 008's `load_orders_carefully` (keep the good, record the
bad) rebuilt for tables. Masks replace the per-row `try`/`except`, and the
rejects are a DataFrame you can save with `.to_csv` for someone to fix.

</details>

## 5. (Optional) Why not fill price with the drink's average?

Someone suggests a cleverer fill for missing prices: the average price
*of that drink* (`df.groupby("drink")["price"].transform("mean")`), not the
overall mean. Better than the overall mean? Better than the menu rule?

<details>
<summary>Check yourself</summary>

Better than the overall mean, yes: a missing latte gets a latte-ish price.
But still worse than the menu rule, for two reasons. The average latte
price (about £3.91, a blend of small, medium, large, whole and oat) is
*still* a price nothing on the menu costs, and it ignores information we
have (the row's size and milk). The menu rule uses everything we know and
gives the exact right answer, and we *proved* it with the "0 known prices
disagree" check.

The general lesson: when there's a real rule that generated the data, use
the rule. Statistical fills (means, medians, groups) are for when there
isn't one, and they should always be reported as estimates. (And
`groupby(...).transform` is tomorrow's lesson; you'll see it again.)

</details>

---

That's Week 3, Day 3 done. Lessons 019 and 020 arrive tomorrow
(Thursday). If one thing sticks from today, let it be: *find it,
understand why, fix it on purpose (or label it honestly), then check*.
