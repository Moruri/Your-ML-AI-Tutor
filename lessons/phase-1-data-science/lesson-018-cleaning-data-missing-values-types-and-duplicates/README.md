# Lesson 018 - Cleaning data: missing values, types and duplicates

**Phase 1 - Data science basics** | Week 3, Day 3 | Wednesday 2026-09-23

> **Goal:** find and fix the boring problems that ruin analyses before
> they start (gaps, wrong types, inconsistent text and duplicate rows),
> and fix them *honestly*, so the clean table is right and you can say
> why.

Time: about 55 minutes. Needs the venv from lesson 013 (`pandas`).

---

A confession: this morning's file was too good to be true. Lesson 017's
`orders_september.csv` was the month *after* cleaning. This afternoon you
get the month as the till actually exported it,
`orders_september_raw.csv`, and it's the usual story. Blank prices.
`Latte`, `LATTE` and ` latte`. Prices with `£` in front. A quantity
written as `two`. Rows exported twice.

People say data scientists spend most of their time cleaning data. It's
roughly true, and it's not wasted time: a model or a chart built on dirty
data is wrong in ways that are very hard to spot afterwards. You did this
by hand in lessons 007, 008 and 012, row by row. Today the same thinking,
with pandas doing a whole column at a time.

The thinking is the important part. Every fix today follows the same
three steps:

1. **Find it.** Use pandas to *see* the problem before touching anything.
2. **Understand it.** Work out *why* it's there, because that decides the
   fix.
3. **Fix it on purpose**, or decide honestly that it can't be fixed.

And at the end, one more: **check**, in code, that the result is right.

## How to follow along

Venv active, `python` in the repo root, `import pandas as pd` and
`import numpy as np`. Full script:

```bash
python lessons/phase-1-data-science/lesson-018-cleaning-data-missing-values-types-and-duplicates/lesson.py
```

It writes one file, `output/orders_september_clean.csv`, next to itself.

## 1. First look

```python
raw = pd.read_csv("orders_september_raw.csv")
raw.shape      # (1229, 8)
raw.dtypes     # everything is 'object', including price and quantity
```

Two warning lights before you've read a single row. First, `price` and
`quantity` are `object`. When `read_csv` meets a column where *every*
value looks like a number, it makes it a number column; one `£3.60` or
`two` anywhere and the whole column stays text. So "a number column
showing `object`" always means "something in here isn't a number".

Second, 1,229 rows, when the month had 1,214 orders. Fifteen too many.

And `raw["drink"].unique()` shows sixteen spellings of five drinks. The
shape of the problem is clear before we fix anything, which is the point
of looking first.

## 2. Missing values

```python
raw.isna().sum()
```

```
order_id      0
timestamp     0
drink         0
size          0
milk         20
price        25
quantity      0
payment      31
```

`isna()` gives a True/False table the same shape as the data (True where
a value is missing), and `.sum()` counts the Trues in each column:
lesson 015's reductions, with labels. `read_csv` turns empty cells into
`NaN` automatically, which is why this works.

Now the important step: *why* is each one missing? Look at the rows:

```python
raw.loc[raw["milk"].isna(), "drink"].value_counts()
# tea 10, espresso 10
```

Every missing milk is on a tea or an espresso, drinks that never have
milk. The till left the cell blank instead of writing `none`. So these
gaps *mean something*: "none". That's a very different situation from
the missing payments, where there's no pattern and simply no record of
how the person paid. Same `NaN`, different meanings, different fixes.
Section 6 handles each.

Three facts about `NaN` worth having in your head:

- **pandas skips it** in `sum`, `mean`, `count` and friends, so
  `pd.Series([2, NaN, 4]).mean()` is `3.0`. Usually helpful; occasionally
  it hides how much is missing, which is why you count first.
- **NumPy doesn't**: `np.array([2, np.nan, 4]).mean()` is `nan`. One gap
  poisons the whole answer.
- **`NaN` is not equal to anything, even itself.** `np.nan == np.nan` is
  `False`. So `df[df["price"] == np.nan]` finds nothing, ever. Always use
  `.isna()` and `.notna()`.

## 3. Duplicates

```python
raw.duplicated().sum()                  # 15   rows identical to an earlier row
raw["order_id"].duplicated().sum()      # 15   order IDs seen before
```

`duplicated()` marks each row that's an exact copy of an earlier one.
Checking the ID column separately matters: if an order ID appeared twice
with *different* details, `duplicated()` on whole rows would miss it,
and that would be a more worrying problem (which copy is right?). Here
both counts are 15, and `duplicated(keep=False)` (which marks *every*
copy, first included) lets you eyeball the pairs: identical in every
column. The till exported them twice.

```python
deduped = raw.drop_duplicates()         # (1214, 8)
```

Keeping them would count those orders' money twice. Removing exact
duplicates of a row with a unique ID is one of the few fixes that's
almost always safe.

## 4. Tidying text

Every string method from lesson 003 exists for a whole column, behind
`.str`:

```python
df["drink"] = df["drink"].str.strip().str.lower().replace(DRINK_ALIASES)
```

`.str.strip()` removes the stray spaces, `.str.lower()` the capitals, and
`.replace({"capp": "cappuccino", "flat-white": "flat white"})` swaps
known aliases. Sixteen spellings become five.

Then *check*, don't assume:

```python
set(df["drink"]) - set(MEDIUM_PRICE)     # set()  nothing left that isn't on the menu
```

That one line is lesson 012's "an unknown alias is caught, not guessed
at". If next month's file says `flatwhite`, this is where you'll find out.

## 5. Fixing types

Price first. Before converting, find out *what* won't convert:

```python
as_number = pd.to_numeric(df["price"], errors="coerce")
df.loc[as_number.isna() & df["price"].notna(), "price"].value_counts()
# £3.60  3,  £1.70  3,  £3.70  2, ...
```

`pd.to_numeric(..., errors="coerce")` means "convert what you can, and
turn anything else into `NaN`". That makes it a detective: comparing
where it produced `NaN` with where the value *wasn't* already blank shows
exactly the values that aren't numbers. Here, only `£` signs. So:

```python
df["price"] = pd.to_numeric(df["price"].str.replace("£", "", regex=False))
```

This time *without* `errors="coerce"`. Once you know what the problems
are and have fixed them, convert strictly, so that any problem you
*didn't* anticipate raises an error instead of quietly becoming `NaN`.
Coerce to investigate; convert strictly to fix. (`regex=False` says "the
£ is just a character, not a pattern". It's a good habit for plain text
replacements.)

Quantity is the same idea: strip the spaces, then find what isn't digits
(`one`, `two`, `three`), map the words with a dict, and `.astype(int)`.

And timestamps become real datetimes:

```python
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M")
```

Giving the `format` means pandas doesn't have to guess (lesson 021 is all
about dates, and why guessing is dangerous).

## 6. Filling gaps honestly

Three columns still have gaps, and each gets a different treatment,
because each gap means something different:

**Milk: fill, because we know what it means.**

```python
df["milk"] = df["milk"].fillna("none")
```

We showed in section 2 that every blank milk is a tea or an espresso, so
"none" is not a guess. It's what the blank meant.

**Price: fill, because we can prove the rule.**

The menu fixes every price: medium price, plus 50p for large or minus 50p
for small, plus 40p for oat milk. `expected_price(df)` works that out for
every row. Before using it to fill anything, the script checks it against
the prices we *do* have:

```python
known prices that disagree with the menu  -> 0
```

All 1,189 known prices match the rule exactly. *That* check is what
earns us the right to fill the 25 blanks from the menu. If even a handful
had disagreed (a price rise mid-month, say), filling from the menu would
be a guess, and we'd need to think again.

Compare with the popular lazy fix, filling with the column mean. It
"works" (no more `NaN`), and it prices a missing latte at whatever the
average of all drinks happens to be, a price that doesn't exist on the
menu. Filling with the mean, median or zero without a reason doesn't clean
data; it quietly invents it.

**Payment: don't fill, because we can't know.**

```python
df["payment"] = df["payment"].fillna("unknown")
```

Thirty orders have no payment method, and nothing in the data tells us
which. Guessing "card" because most people pay by card would push the
card share up by a made-up amount. Dropping the rows would throw away
thirty real sales. So we keep them, labelled honestly: their money counts
towards revenue, and they're excluded from any card-vs-cash calculation.
Leaving a value missing (or labelled unknown) is a perfectly good
decision, when it's a *decision*.

The general rule: **fill a gap only when you can say, in one sentence,
why the filled value is true.** "Espresso never has milk" passes. "Most
people pay by card" doesn't.

## 7. Putting it together: a cleaning function you can trust

All of that, as one function, each step one line:

```python
def clean(raw):
    df = raw.drop_duplicates().copy()
    df["drink"] = df["drink"].str.strip().str.lower().replace(DRINK_ALIASES)
    df["price"] = pd.to_numeric(df["price"].str.replace("£", "", regex=False))
    df["quantity"] = df["quantity"].str.strip().replace(NUMBER_WORDS).astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M")
    df["milk"] = df["milk"].fillna("none")
    df["price"] = df["price"].fillna(expected_price(df))
    df["payment"] = df["payment"].fillna("unknown")
    return df.reset_index(drop=True)
```

And a second function that *checks* the result against every rule the
rest of the analysis relies on:

```python
def check(df):
    assert df["order_id"].is_unique, "one row per order"
    assert not df[[...key columns...]].isna().any().any()
    assert set(df["drink"]) <= set(MEDIUM_PRICE), "only drinks on the menu"
    assert (df["quantity"] >= 1).all()
    assert np.isclose(df["price"], expected_price(df)).all(), "every price matches the menu"
```

```
  raw rows 1,229 -> clean rows 1,214  (15 duplicates removed)
  revenue £6,627.50 from 1,922 cups
```

1,214 orders and £6,627.50: exactly lesson 017's month, to the penny.

Why two functions? `clean` handles the problems you *know about*. `check`
catches the ones you don't. Next month, when the till writes `flatwhite`
or a price of `free`, `clean` won't know what to do, but `check` (or one
of `clean`'s strict conversions) will stop with an error that says
what's wrong. That's lesson 008's "crash on the unexpected", for tables.

The `.copy()` after `drop_duplicates()` is there so that later
assignments change our own table and not some view of `raw` (lesson 015's
view surprise; pandas warns about it with the famous
`SettingWithCopyWarning`). When you filter a DataFrame and then plan to
change the result, `.copy()` it.

## What you can do now

- Spot trouble from `dtypes` (a number column showing `object`) and row
  counts, before looking at a single row.
- Count missing values with `isna().sum()`, find the rows, and work out
  *why* they're missing.
- Explain how `NaN` behaves: skipped by pandas, spread by NumPy, never
  equal to itself.
- Find and remove duplicates with `duplicated()` and `drop_duplicates()`,
  checking the ID column as well as whole rows.
- Tidy text columns with `.str` methods and `.replace`, then check nothing
  unknown remains.
- Investigate with `pd.to_numeric(errors="coerce")`, then convert strictly.
- Decide, for each gap, whether to fill, drop or label it unknown, with a
  one-sentence reason.
- Separate cleaning from checking, and assert the rules your analysis
  relies on.

## What to do now

1. Run `lesson.py`, then open `output/orders_september_clean.csv` and the
   raw file side by side. Find one row that was fixed in each way.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   throws four new kinds of mess at `clean()` and teaches it to reject
   rows it can't fix.
3. That's Week 3, Day 3. Lessons 019 (group, aggregate, pivot) and 020
   (joining tables and reshaping) arrive tomorrow. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Cleaning is where you get to know a dataset properly: its habits, its
quirks, the ways it was made. Most of the interesting questions you'll
ask later start with something you noticed here.
