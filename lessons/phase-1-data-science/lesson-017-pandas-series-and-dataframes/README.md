# Lesson 017 - pandas: Series and DataFrames

**Phase 1 - Data science basics** | Week 3, Day 3 | Wednesday 2026-09-23

> **Goal:** load a CSV into a DataFrame, inspect it with
> `head`/`info`/`describe`, and select rows and columns, so that a table
> of a thousand orders is something you can question in a line.

Time: about 50 minutes. Needs the venv from lesson 013 (`pandas`).

---

The coffee shop has sent a whole month: every order from September 2026,
1,214 of them, with the time, the drink, the size, the milk, the price,
how many cups and how they paid. In Phase 0 you'd have reached for
`csv.DictReader`, a list of dicts, and a page of loops. With NumPy
(lessons 014 to 016) you'd have fast maths but one array per column, all
kept in line by hand.

**pandas** gives you the table itself. It's built on NumPy (every column
is an array underneath), and adds the two things NumPy lacks for this kind
of data: **columns of different types** side by side, and **labels** on
rows and columns, so you can say `orders["price"]` instead of "column 5".
It's the most used data library in Python, and from today it's your main
tool for the rest of Phase 1.

pandas is big. You don't need most of it, and nobody remembers all of it.
Today is the core: the two objects, loading a file, looking at it, and
picking out the bits you want.

## How to follow along

Venv active, `python` in the repo root, and the conventional import:

```python
import pandas as pd
```

Full script:

```bash
python lessons/phase-1-data-science/lesson-017-pandas-series-and-dataframes/lesson.py
```

It writes no files. The data is `orders_september.csv`, next to the
script.

## 1. A Series is an array with labels

```python
prices = pd.Series([3.80, 2.20, 3.70, 3.90, 2.50],
                   index=["latte", "espresso", "cappuccino", "flat white", "tea"])
```

```
latte         3.8
espresso      2.2
cappuccino    3.7
flat white    3.9
tea           2.5
```

A **Series** is a one-dimensional array (`.values` gives you the NumPy
array underneath) plus an **index**: a label for each element. It behaves
like both of the things you know:

- Like a **dict**: `prices["latte"]` is `3.8`.
- Like an **array**: `prices * 1.05` raises every price, and
  `prices[prices > 3]` masks, exactly as in lesson 015. The labels come
  along for the ride.

The labels earn their keep when two Series meet:

```python
cups = pd.Series({"tea": 2, "latte": 8, "espresso": 4})
prices * cups
```

```
cappuccino     NaN
espresso       8.8
flat white     NaN
latte         30.4
tea            5.0
```

pandas lined the two up **by label**, not by position: tea with tea, latte
with latte, whatever order they were in. That's called **alignment**, and
it's why you can't accidentally multiply the latte price by the tea
count. Labels that only one side has give **`NaN`** ("not a number"),
pandas' marker for a missing value. You'll meet `NaN` a lot tomorrow.

## 2. A DataFrame is a table of Series

```python
week = pd.DataFrame({
    "drink": ["latte", "espresso", "cappuccino", "latte", "flat white"],
    "size": ["medium", "small", "large", "large", "medium"],
    "price": [3.80, 2.20, 4.20, 4.30, 3.90],
    "quantity": [2, 1, 1, 1, 3],
})
```

```
        drink    size  price  quantity
0       latte  medium    3.8         2
1    espresso   small    2.2         1
2  cappuccino   large    4.2         1
3       latte   large    4.3         1
4  flat white  medium    3.9         3
```

A **DataFrame** is a table: a collection of columns, each one a Series,
all sharing one index (the row labels down the left, `0` to `4` here
because we didn't choose any). Each column has its own dtype, so text,
floats and ints live happily side by side. That's the thing a 2D NumPy
array couldn't do.

The attributes you'll check constantly:

```python
week.shape       # (5, 4)          rows, columns: same as NumPy
week.columns     # the column names
week.index       # the row labels
week.dtypes      # each column's type
```

Text columns show as `object`. That's pandas for "Python objects, usually
strings". Numbers show as `int64` and `float64`, the NumPy dtypes from
lesson 014.

## 3. Loading a CSV and getting to know it

```python
orders = pd.read_csv("orders_september.csv")
```

One line, and the whole file is a DataFrame, header row as column names,
numbers already turned into numbers. Compare that with lesson 007's
`DictReader` loop and lesson 008's type conversions.

Then, **before anything else**, four calls. They're lesson 012's "look
before you trust" for tables:

```python
orders.shape        # (1214, 8)
orders.head()       # the first 5 rows (tail() for the last)
orders.info()       # every column: its dtype and how many values are NOT missing
orders.describe()   # count, mean, spread, min, quartiles and max of each number column
```

`info()` is the most underrated of the four:

```
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   order_id   1214 non-null   object
 ...
 5   price      1214 non-null   float64
 6   quantity   1214 non-null   int64
```

It tells you, for every column, whether pandas read it as numbers or as
text (a price column showing `object` means something in it isn't a
number) and whether anything's missing (a count lower than the number of
rows). This file is clean, so everything is 1214 and the types are
right. Tomorrow's file won't be.

`describe()` gives the shape of every number column in one table:

```
         price  quantity
count  1214.00   1214.00
mean      3.46      1.58
std       0.89      0.82
min       1.70      1.00
25%       3.20      1.00
50%       3.70      1.00
75%       4.20      2.00
max       4.80      4.00
```

Glance at min and max first: they're where impossible values show up
(a negative quantity, a £380 latte). Here, prices from £1.70 to £4.80 and
quantities from 1 to 4 look right. `describe(include="object")` does text
columns: how many distinct values each has, and the most common one.

## 4. Choosing columns

```python
orders["drink"]                 # one column: a Series
orders[["drink", "price"]]      # a list of columns: a DataFrame
orders.price                    # attribute style: same as orders["price"]
```

Single name in brackets gives a Series. A *list* of names (hence the
double brackets) gives a smaller table. The attribute style is handy for
typing at `>>>`, but it doesn't work for names with spaces or names that
clash with DataFrame methods (a column called `count` or `size`, for
instance: `orders.size` is the number of cells, not your column), and you
can never use it to *create* a column. `orders["name"]` always works; use
that in scripts.

## 5. Choosing rows: masks, `.loc` and `.iloc`

Filtering rows is lesson 015's boolean masking, with column names:

```python
orders[orders["drink"] == "latte"]                           # 377 rows
orders[(orders["milk"] == "oat") & (orders["quantity"] >= 3)]  # & and brackets, as before
orders[orders["drink"].isin(["tea", "espresso"])]            # "is one of these"
```

`.isin` is the neat way to write "drink is tea or drink is espresso".

For more control, two indexers, both used as `[rows, columns]`:

- **`.loc`** selects by **label** (and by mask):

  ```python
  orders.loc[0]                                          # the row labelled 0
  orders.loc[orders["quantity"] == 4, ["order_id", "drink", "quantity"]]
  ```

  The second line is the everyday form: "rows where this is true, and
  just these columns".

- **`.iloc`** selects by **position**, exactly like a NumPy array:

  ```python
  orders.iloc[-1]          # the last row
  orders.iloc[:3, :4]      # first three rows, first four columns
  ```

Right now the labels are `0, 1, 2, ...`, so labels and positions agree
and the difference seems academic. It stops being academic the moment you
filter or sort. After `lattes = orders[orders["drink"] == "latte"]`, the
first row of `lattes` might be *labelled* 5. `lattes.loc[0]` is then an
error (no row labelled 0), and `lattes.iloc[0]` is the first row. Rule:
**`.loc` for labels and conditions, `.iloc` for "the first", "the last",
"the tenth".**

The index doesn't have to be numbers. `orders.set_index("order_id")`
makes the order IDs the row labels, so `.loc["S0100"]` finds that order
directly, just like a dict lookup.

## 6. New columns and quick summaries

```python
orders["revenue"] = orders["price"] * orders["quantity"]
```

A new column is vectorised arithmetic on existing ones, assigned to a new
name. Every row at once, no loop. Then:

```python
orders["revenue"].sum()                       # 6627.5
orders["drink"].value_counts()                # how many orders of each drink, biggest first
orders["milk"].value_counts(normalize=True)   # as shares instead of counts
orders.sort_values("revenue", ascending=False).head(3)   # the three biggest orders
```

`value_counts()` is lesson 005's `Counter`, sorted, as a Series you can
keep working with. `sort_values` sorts the entire table by one column
(or several: pass a list), keeping each row together.

Series have all the summary methods you'd expect: `.sum()`, `.mean()`,
`.median()`, `.min()`, `.max()`, `.std()`, `.nunique()` (number of
distinct values). And unlike NumPy's, they skip missing values by
default, which matters tomorrow.

## 7. Putting it together

```
  Orders:                 1,214
  Cups:                   1,922
  Revenue:                £6,627.50
  Average order:          £5.46  (median £4.20)
  Oat share of milky:     25%
  Paid by card:           77% of revenue
  Top drink by revenue:   latte
```

Each line is one expression. "Oat share of milky" is a nice one to read
slowly:

```python
(orders.loc[orders["milk"] != "none", "milk"] == "oat").mean()
```

Take the milk column for rows that have milk at all (tea and espresso say
`none`), ask which are oat, and average the True/False values. Mask,
select, compare, mean: every piece is something you've done before.

The last line sneaks in `groupby`, which is lesson 019's whole topic. It
does lesson 005's "total per drink" in one call.

## What you can do now

- Explain what a Series is (an array with labels) and what alignment does
  when two Series meet.
- Explain what a DataFrame is (a table of Series sharing an index) and read
  its `shape`, `columns`, `index` and `dtypes`.
- Load a CSV with `pd.read_csv`, and get to know it with `head`, `tail`,
  `info` and `describe`.
- Select columns with `df["col"]` and `df[["a", "b"]]`, and rows with
  masks, `.isin`, `.loc` and `.iloc`, knowing when labels and positions
  differ.
- Add a computed column, and summarise with `sum`, `mean`, `median`,
  `value_counts` and `sort_values`.

## What to do now

1. Run `lesson.py`, then load the file yourself at `>>>` and try the four
   "get to know it" calls without looking.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   answers the owner's questions about milk and sizes.
3. Lesson 018 (cleaning data) is this afternoon, with a much messier
   version of this file. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

pandas can feel like a huge API at first. It's really a handful of ideas
(labelled columns, masks, vectorised maths) used over and over. You
already knew most of them this morning.
