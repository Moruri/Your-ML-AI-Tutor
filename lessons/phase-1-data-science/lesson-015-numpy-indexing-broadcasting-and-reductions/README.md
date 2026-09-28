# Lesson 015 - NumPy indexing, broadcasting and reductions

**Phase 1 - Data science basics** | Week 3, Day 2 | Tuesday 2026-09-22

> **Goal:** slice, mask and aggregate arrays, and predict what
> broadcasting will do before running it, so you can ask an array any
> question ("which days", "how many", "per drink") in one line.

Time: about 50 minutes. Needs the venv from lesson 013 (`numpy`).

---

Yesterday you met the array and did maths on all of it at once. But real
questions are rarely about *all* of it. They're about *some* of it: the
orders over £5, Friday's numbers, the latte column. Or they're about
*summarising along a direction*: cups per day, revenue per drink.

Those three skills are today's lesson, and between them they cover most of
what people actually do with NumPy:

- **Indexing**: picking out the elements you want, by position, by slice,
  or with a True/False **mask**.
- **Broadcasting**: the rule NumPy uses when arrays of *different* shapes
  meet, like a 5-by-5 table times a list of five prices.
- **Reductions**: collapsing an array along one direction (`axis`) with
  `sum`, `mean`, `max` and friends.

Most of today uses one small table: the week's cups, a row per day and a
column per drink. It's the same `week` you met in lesson 014, now called
`WEEK`:

```
            latte  espresso  cappuccino  flat white  tea
Mon           2       1          1           0        0
Tue           1       2          0           3        0
Wed           1       0          2           0        1
Thu           2       1          0           1        0
Fri           2       0          2           0        1
```

## How to follow along

Venv active, `python` in the repo root, and:

```python
import numpy as np
WEEK = np.array([[2, 1, 1, 0, 0], [1, 2, 0, 3, 0], [1, 0, 2, 0, 1],
                 [2, 1, 0, 1, 0], [2, 0, 2, 0, 1]])
```

Full script:

```bash
python lessons/phase-1-data-science/lesson-015-numpy-indexing-broadcasting-and-reductions/lesson.py
```

It writes no files.

## 1. Indexing and slicing, and the view surprise

One-dimensional arrays index and slice exactly like lists (lesson 004):

```python
cups = np.array([4, 2, 3, 1, 2, 5, 0])
cups[0], cups[-1]       # 4, 0
cups[2:5]               # [3, 1, 2]     stop is excluded
cups[::2]               # [4, 3, 2, 0]  every second one
cups[::-1]              # reversed
```

With one big difference. Slicing a *list* makes a copy. Slicing an
*array* makes a **view**: a window onto the same memory.

```python
middle = cups[2:5]
middle[0] = 99
cups                    # [ 4,  2, 99,  1,  2,  5,  0]    the original changed!
```

NumPy does this on purpose: slicing a huge array is instant because
nothing is copied. But it means changing a slice changes the original,
which is lesson 006's "functions that change their inputs" problem in
disguise. When you want an independent piece, say so:

```python
safe = cups[2:5].copy()
```

Rule of thumb: slices are views; if you're going to *change* one, `.copy()`
it first. (Masks and position lists, below, always make copies, so they're
safe.)

## 2. Rows and columns

In two dimensions, you index with `[row, column]`:

```python
WEEK[1, 3]        # 3      Tuesday (row 1), flat white (column 3)
WEEK[0]           # [2, 1, 1, 0, 0]    Monday: a whole row
WEEK[:, 0]        # [2, 1, 1, 2, 2]    latte: a whole column
WEEK[:2, :3]      # the top-left corner: Mon-Tue, first three drinks
```

A colon on its own means "every one along this dimension", so `WEEK[:,
0]` reads "every row, column 0". Each part of `[rows, columns]` can be a
single number, a slice, or (next section) a mask. Picking a whole column
is the everyday way to get at one variable in a table of data, and pandas
columns (lesson 017) are the friendlier version of exactly this.

## 3. Boolean masks

Yesterday you saw that `quantity > 1` gives an array of `True`/`False`.
Put that array *inside the square brackets* and you get only the elements
where it's `True`:

```python
quantity = np.array([2, 1, 1, 1, 3, ...])     # one per order
price    = np.array([380, 220, 420, 430, 390, ...])

big = quantity > 1
quantity[big]          # [2, 3, 2, 2, 2, 2, 2]
price[big]             # the prices of those same orders
```

This is **boolean masking**, and it's how you filter in NumPy (and in
pandas). The mask doesn't have to come from the array you're indexing:
`price[big]` uses a mask built from `quantity`. As long as the arrays
line up (one entry per order), a condition on one picks rows from all of
them.

Combine conditions with `&` (and), `|` (or) and `~` (not):

```python
price[(quantity > 1) & (price > 380)]      # multi-cup orders of pricier drinks
price[(price < 250) | (price > 400)]       # the cheap and the dear
price[~big]                                # single-cup orders
```

Two traps, both of which you *will* hit:

- **Brackets round every comparison are required.** `&` binds more
  tightly than `>`, so `quantity > 1 & price > 380` is read as `quantity >
  (1 & price) > 380`, which is nonsense (and usually an error). Just
  always bracket.
- **Use `&`, `|`, `~`, not `and`, `or`, `not`.** The words are for single
  True/False values; on an array they raise `ValueError: The truth value
  of an array with more than one element is ambiguous`. When you see that
  message, you've used `and` where you meant `&`.

Two more mask tools:

```python
np.where(quantity > 1, "multi", "single")   # pick one of two values, per element
capped = price.copy()
capped[capped > 400] = 400                  # change only the selected elements
```

## 4. Choosing by position: integer arrays and `argsort`

You can also index with a list (or array) of positions:

```python
drinks = np.array(["latte", "espresso", "cappuccino", "flat white", "tea"])
drinks[[0, 2, 4]]            # ['latte', 'cappuccino', 'tea']
```

The most useful source of such positions is `np.argsort`, which returns
the positions that *would* sort an array:

```python
totals = WEEK.sum(axis=0)             # [8, 4, 5, 4, 2]   cups per drink
order = np.argsort(totals)[::-1]      # [0, 2, 3, 1, 4]   biggest first
drinks[order]                         # ['latte', 'cappuccino', 'flat white', 'espresso', 'tea']
totals[order]                         # [8, 5, 4, 4, 2]
```

Why not just `np.sort(totals)`? Because then you'd have sorted numbers
and no idea which drink each belongs to. Sorting by *positions* lets you
put the names and the numbers in the same order, so they stay lined up.
It's the NumPy version of lesson 004's `sorted(..., key=...)`. (Espresso
and flat white tie on 4; ties come out in whichever order the sort
happens to leave them. If ties matter, decide how to break them.)

`argmax` and `argmin` are the one-answer versions: the position of the
biggest or smallest value.

## 5. Broadcasting

What should `WEEK * MEDIUM_CENTS` do, when `WEEK` is 5-by-5 and
`MEDIUM_CENTS` is five prices, one per drink?

```python
MEDIUM_CENTS = np.array([380, 220, 370, 390, 250])
WEEK * MEDIUM_CENTS
# [[ 760,  220,  370,    0,    0],
#  [ 380,  440,    0, 1170,    0],
#  ...
```

Each column is multiplied by its own price: every row of cups times the
row of prices. NumPy "stretched" the five prices down all five rows
without copying anything. That's **broadcasting**, and you used the
simplest case yesterday: `arr + 0.10` stretches one number over the whole
array.

**The rule**: line the two shapes up *from the right*. Going leftwards,
each pair of sizes must be **equal**, or **one of them is 1** (or
missing). Sizes of 1 and missing sizes get stretched to match.

```
WEEK           (5, 5)
MEDIUM_CENTS      (5,)     right-hand 5s match; the missing one stretches -> works, per COLUMN

WEEK           (5, 5)
busy_col       (5, 1)      5 vs 1: the 1 stretches; 5 vs 5 matches -> works, per ROW

WEEK           (5, 5)
something         (3,)     5 vs 3: not equal, neither is 1 -> ValueError
```

The middle case is the one to really understand. Suppose Friday was busy,
and you want to scale each *day* (row) by a factor:

```python
busy = np.array([1.0, 1.0, 1.0, 1.0, 1.5])     # shape (5,)
WEEK * busy                    # runs, but scales the COLUMNS (tea x 1.5), not Friday!
WEEK * busy.reshape(5, 1)      # shape (5, 1): scales each ROW. Friday x 1.5. Correct.
```

Because `WEEK` happens to be square, the wrong version *runs without an
error* and gives a wrong answer. That's the most dangerous kind of bug.
A (5,) array always lines up with the **last** dimension (columns). To
line something up with rows, make it a column: `.reshape(5, 1)`, or
`busy[:, np.newaxis]`, which means the same thing.

The habit that saves you: **before running a broadcast, write the two
shapes down right-aligned and say out loud which way the stretch goes.**
It takes five seconds, and the goal of today's lesson is that you can do
it.

## 6. Reductions along an axis

A **reduction** turns many numbers into fewer: `sum`, `mean`, `max`,
`min`, `std`, `argmax`. With no arguments, they reduce everything to one
number. With `axis`, they reduce along one direction:

```python
WEEK.sum()           # 23                    everything
WEEK.sum(axis=0)     # [8, 4, 5, 4, 2]       per drink (one per column)
WEEK.sum(axis=1)     # [4, 6, 4, 4, 5]       per day (one per row)
WEEK.argmax(axis=1)  # [0, 3, 2, 0, 0]       which drink topped each day
```

Which axis is which trips everyone up. The rule that works: **the axis you
name is the one that disappears.** `WEEK` has shape `(5 days, 5
drinks)`. `axis=0` removes the days, leaving one number per drink. `axis=1`
removes the drinks, leaving one number per day. (Friday's `argmax` is 0,
latte, because latte and cappuccino tie on 2 and argmax takes the first.)

And the partner to broadcasting: `keepdims=True` keeps the collapsed axis
as size 1, so the result broadcasts straight back:

```python
day_totals = WEEK.sum(axis=1, keepdims=True)    # shape (5, 1), not (5,)
share = WEEK / day_totals                       # each cell as a share of its day
share.sum(axis=1)                               # [1., 1., 1., 1., 1.]
```

Without `keepdims`, `day_totals` would be `(5,)`, line up with columns,
and give a wrong answer: exactly the section 5 trap. "Share of the row" and
"difference from the column mean" are everyday calculations, and
`keepdims=True` is how you write them safely.

## 7. Putting it together: the week as a table, no loops

`lesson.py` builds a *revenue* table from `orders.csv`, a row per day and a
column per drink, using the day of the week and the drink as positions.
(`np.add.at(table, (day_idx, drink_idx), cents * qty)` means "for each
order, add its revenue into cell [day, drink]". It's the one slightly
exotic line, and it's worth reading twice.)

Then five questions, each a line or two:

```python
per_day = table.sum(axis=1)
per_drink = table.sum(axis=0)
DAYS[per_day.argmax()]                    # 1. best day
drinks[per_drink.argmax()]                # 2. best drink
drinks[table.argmax(axis=1)]              # 3. top drink each day (argmax gives positions!)
table[:, 0] / per_day                     # 4. latte's share of each day
(table[:, 4] == 0).sum()                  # 5. days with no tea
```

```
  1. Best day:                Tue (£20.40)
  2. Best drink:              latte (£29.90)
  3. Top drink each day:      latte, flat white, cappuccino, latte, cappuccino
  4. Latte's share each day:  54%, 21%, 28%, 52%, 40%
  5. Days with no tea sold:   3 (first: Mon)
```

The whole table adds up to 7970 pence, and the latte column to 2990.
Question 3 uses the positions from `argmax(axis=1)` to index the array of
drink names: a reduction feeding straight into fancy indexing. That's
what fluent NumPy looks like, and you just read it.

## What you can do now

- Index and slice arrays in one and two dimensions with `[row, column]`,
  and know that slices are views (use `.copy()` before changing one).
- Filter with boolean masks, combine them with `&`, `|` and `~` (with
  brackets), and use `np.where`.
- Use integer arrays and `argsort` to reorder several arrays together.
- State the broadcasting rule (right-align shapes; sizes equal or 1) and
  predict whether a broadcast works and along which direction.
- Reduce along an axis, knowing that the named axis disappears, and use
  `keepdims=True` to broadcast the result back.

## What to do now

1. Run `lesson.py`. In section 5, notice that `WEEK * busy` ran without
   an error. Change `WEEK` to have six rows (add a Saturday) and see what
   happens to that line.
2. Do the exercises in [`exercises.md`](exercises.md). The prediction
   questions are the real test.
3. Lesson 016 (randomness you can reproduce) is this afternoon. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Broadcasting is the one NumPy idea that takes a while to settle. If it's
still wobbly, that's normal: write shapes down, right-aligned, every time,
and in a week you'll find you've stopped needing to.
