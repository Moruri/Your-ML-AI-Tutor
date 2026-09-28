# Lesson 020 - Joining tables and reshaping

**Phase 1 - Data science basics** | Week 3, Day 4 | Thursday 2026-09-24

> **Goal:** merge datasets on keys, and move between wide and long
> formats with `melt`/`pivot`, so that questions needing two tables
> (or a table the "wrong" shape) are no harder than questions needing one.

Time: about 50 minutes. Needs the venv from lesson 013 (`pandas`).

---

The owner has two new questions:

1. "How much *profit* did each drink make in September?"
2. "Did we hit our weekly cup targets?"

Neither can be answered from the orders alone. Profit needs the **cost**
of each cup, which lives in `menu.csv`, a small table with one row per
drink. Targets live in `targets_wide.csv`, which the owner typed into a
spreadsheet with one *column* per week. The information exists; it's just
spread across tables, in different shapes.

That's normal. Real data almost never arrives as one tidy table. Sales in
one system, products in another, targets in someone's spreadsheet. Two
skills bring it together:

- **Joining** (`merge`): lining up rows from two tables that share a key,
  like the drink name.
- **Reshaping** (`melt` and `pivot`): turning a wide table (one column per
  week) into a long one (one row per drink per week) and back again.

## How to follow along

Venv active, `python` in the repo root. This lesson has three data files:
`orders_september.csv` (the clean month), `menu.csv` (six drinks,
including a mocha that wasn't sold in September) and `targets_wide.csv`.
Full script:

```bash
python lessons/phase-1-data-science/lesson-020-joining-tables-and-reshaping/lesson.py
```

It writes no files.

## 1. Two tables, one question

```
        drink category  medium_price  cost_per_cup
0       latte    milky           3.8          1.05
1  cappuccino    milky           3.7          1.00
...
5       mocha    milky           4.0          1.30
```

To know what an order cost to make, you need its `quantity` (from
`orders`) times its drink's `cost_per_cup` (from `menu`). The column both
tables share, `drink`, is the **key** that connects them.

For one column, you already know a way: a lookup, like lesson 005's dict.

```python
cost_lookup = menu.set_index("drink")["cost_per_cup"]    # a Series: drink -> cost
orders["drink"].map(cost_lookup)                          # each order's cost per cup
```

`.map` with a Series looks up each value in its index. That's fine for
one column. When you want several columns from the other table, or
you want to *see* what matched, you want `merge`.

## 2. `merge`

```python
joined = orders.merge(menu, on="drink", how="left", validate="many_to_one")
```

For every row of `orders`, pandas finds the row of `menu` with the same
`drink` and glues its columns on:

```
  order_id       drink  quantity category  cost_per_cup
0    S0001  cappuccino         2    milky          1.00
1    S0002    espresso         1    black          0.45
...
```

Three arguments to understand:

- **`on="drink"`**: the key column, which must exist in both tables.
- **`how="left"`**: keep every row of the left table (`orders`), whether
  or not it finds a partner. Section 3 has the alternatives.
- **`validate="many_to_one"`**: a promise, checked by pandas, that the
  key is *unique* in the right-hand table. Many orders per drink; one menu
  row per drink. Section 4 shows why this matters so much.

And one habit: **check the row count after every join.** 1,214 orders in,
1,214 rows out. If the number changed, something about the keys isn't
what you thought.

## 3. The four joins, and seeing what didn't match

What happens to rows that *don't* find a partner depends on `how`. To see
all four, `lesson.py` joins a small `sold` table (cups per drink, with tea
missing and a new "chai" the menu doesn't know about yet) to the menu
(which has mocha, never sold):

```
how='inner'  -> 4 rows: cappuccino, espresso, flat white, latte
how='left'   -> 5 rows: ... plus chai
how='right'  -> 6 rows: ... plus tea and mocha
how='outer'  -> 7 rows: everything from either side
```

| `how` | Keeps | Use when |
|-------|-------|----------|
| `"inner"` | Only keys found in **both**. | You only care about rows that match. (The default, and a dangerous one: unmatched rows vanish silently.) |
| `"left"` | Every row of the **left** table. | Adding information *to* a table you care about, like cost onto orders. The most common choice. |
| `"right"` | Every row of the **right** table. | Rarely; swap the tables and use left. |
| `"outer"` | **Everything** from either side. | Comparing two lists: what's in one and not the other? |

Where a row has no partner, the other table's columns are `NaN`.

The best tool for checking a join is `indicator=True`, which adds a
`_merge` column saying where each row came from:

```
        drink   cups  cost_per_cup      _merge
0  cappuccino  483.0          1.00        both
1        chai   12.0           NaN   left_only
...
5       mocha    NaN          1.30  right_only
6         tea    NaN          0.35  right_only
```

`left_only`: chai sold, but has no cost, so any profit figure would be
missing it. `right_only`: tea's sales are missing, and mocha genuinely
didn't sell. `value_counts()` on `_merge` after an important join tells
you in one line whether everything matched. Make it a habit.

## 4. Different key names, and the join that multiplies rows

When the key has different names in the two tables, name each side:

```python
orders.merge(prices, left_on="drink", right_on="item", how="left")
```

Now the big trap. Suppose someone accidentally lists latte *twice* in the
menu file. Join, and every latte order matches *both* menu rows, so every
latte order appears twice:

```
rows before, rows after joining a menu with latte twice -> (1214, 1591)
revenue before -> after (!)                              -> (6627.5, 8929.1)
```

No error. No warning. 377 extra rows, and revenue inflated by more than
£2,000. This is one of the most common serious bugs in data analysis,
because the output *looks* perfectly normal.

`validate=` catches it:

```python
orders.merge(oops_menu, on="drink", how="left", validate="many_to_one")
# MergeError: Merge keys are not unique in right dataset; not a many-to-one merge
```

The options are `"one_to_one"`, `"one_to_many"`, `"many_to_one"` and
`"many_to_many"`. Say which relationship you *expect*, and pandas checks
it for you. It costs one argument and turns a silent disaster into a loud,
obvious error: lesson 008's philosophy, applied to joins.

## 5. Stacking tables: `concat`

Joining puts tables *side by side*, matched on a key. Sometimes you want
to put them *one on top of another*: two exports, two shops, two months,
all with the same columns.

```python
whole = pd.concat([first_half, second_half], ignore_index=True)
```

`ignore_index=True` renumbers the rows `0, 1, 2, ...`, instead of
keeping each piece's original labels (which would repeat). Passing
`keys=["early", "late"]` instead labels which piece each row came from, so
you can group by it afterwards.

`concat` lines columns up by *name*. If one table has a column the other
doesn't, you get `NaN` for the rows that lack it. Check `.columns` of each
piece before stacking.

## 6. Wide and long: `melt` and `pivot`

The owner's targets file is **wide**: one row per drink, one column per
week.

```
        drink  week_36  week_37  week_38  week_39  week_40
0       latte       95      120      120      120       75
1  cappuccino       80      100      100      100       60
...
```

Easy to read and type in a spreadsheet. Awkward to compute with: there's
no "week" column to group or join on, and the week numbers are buried in
column *names*. The **long** form has one row per drink per week:

```python
targets_long = targets_wide.melt(id_vars="drink", var_name="week", value_name="target")
targets_long["week"] = targets_long["week"].str.removeprefix("week_").astype(int)
```

```
        drink  week  target
0       latte    36      95
1  cappuccino    36      80
...
```

`melt` keeps the `id_vars` columns as they are, and turns every other
column into two: one holding the old column *name* (`var_name`), one
holding the *value* (`value_name`). Same 25 numbers, 25 rows instead of 5.
Then a little text cleaning turns `"week_36"` into the number `36`, so it
can match the orders' week numbers.

`pivot` goes the other way, long to wide:

```python
targets_long.pivot(index="drink", columns="week", values="target")
```

It's `pivot_table` from lesson 019, minus the aggregating. It needs each
(index, column) pair to appear only once; if pairs repeat, you need
`pivot_table` with an `aggfunc`.

A good rule: **keep data long while you work, and go wide at the end for
people to read.** Long tables are what `groupby`, `merge`, and the plotting
tools in lesson 022 want. Wide tables are what humans want in a report.

## 7. Putting it together

**Profit per drink**: join the menu's cost onto every order, then group:

```python
joined = orders.merge(menu[["drink", "category", "cost_per_cup"]], on="drink",
                      how="left", validate="many_to_one")
assert len(joined) == len(orders) and joined["cost_per_cup"].notna().all()
joined["profit"] = joined["revenue"] - joined["cost_per_cup"] * joined["quantity"]
```

```
            revenue    cost   profit  margin
drink
latte        2301.6  617.40  1684.20    0.73
cappuccino   1859.4  483.00  1376.40    0.74
flat white   1428.6  393.80  1034.80    0.72
tea           618.0   86.10   531.90    0.86
espresso      419.9  111.15   308.75    0.74
```

The `assert` is the row-count habit plus "every order found a cost",
written down so it's checked every time. Tea has by far the best margin
(86p of every pound is profit), which revenue alone would never have
shown.

**Cups against target**: total the cups per drink per week (a long table),
join to the melted targets on *two* keys, and pivot the result wide for
reading:

```python
actual = orders.groupby(["drink", "week"], as_index=False)["quantity"].sum()
vs = actual.merge(targets_long, on=["drink", "week"], how="outer",
                  validate="one_to_one", indicator=True)
vs["pct_of_target"] = vs["quantity"] / vs["target"]
vs.pivot(index="drink", columns="week", values="pct_of_target")
```

```
week          36    37    38    39    40
drink
cappuccino  121%  106%  111%  118%   85%
espresso     82%  126%  130%  106%  110%
flat white  162%  140%   89%   90%  102%
latte       100%  114%  108%  129%   96%
tea         154%  129%  109%  140%   73%
```

An outer join with `indicator=True` and an assert that everything is
`both` guarantees no drink-week was silently dropped from either side.
Four drink-weeks came in more than 10% under target, and two of them are
in week 40, which (lesson 019's exercises) is only three days long.
Flat white's mid-month dip is the one worth asking about.

## What you can do now

- Explain what a key is, and use `.map` for a single-column lookup.
- Join tables with `merge`, choosing `how` (inner, left, right, outer) on
  purpose, and use `left_on`/`right_on` when key names differ.
- Check every join: row counts, `indicator=True` and `_merge`.
- Protect against duplicated keys with `validate=`.
- Stack tables with `pd.concat`, with `ignore_index` or `keys`.
- Turn wide tables long with `melt` and long tables wide with `pivot`, and
  know which shape suits which job.

## What to do now

1. Run `lesson.py`. Then delete the `validate=` from the "oops" merge in
   section 4 and notice that nothing complains.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one adds
   the cost of milk to the profit calculation.
3. That's Week 3, Day 4. Lessons 021 (dates and time series) and 022
   (plotting that tells the truth) arrive tomorrow. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Joins are where a lot of analyses go quietly wrong, and where careful
people stand out. Count your rows, look at `_merge`, and say what
relationship you expect with `validate`. That's most of it.
