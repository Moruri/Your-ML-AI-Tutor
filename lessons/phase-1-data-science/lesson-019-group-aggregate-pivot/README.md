# Lesson 019 - Group, aggregate, pivot

**Phase 1 - Data science basics** | Week 3, Day 4 | Thursday 2026-09-24

> **Goal:** answer "average X per Y" questions with `groupby`, `agg` and
> `pivot_table`, and be precise about what X and Y are, because that's
> where the real mistakes happen.

Time: about 50 minutes. Needs the venv from lesson 013 (`pandas`).

---

Nearly every question an owner, a manager or a colleague asks about data
has the same shape:

- What's the **revenue per drink**?
- What's the **average order per weekday**?
- How many **cups per hour**?
- What **share of milky drinks** are oat, **for each drink**?

"Something, per something." In lesson 005 you answered these with a dict
and a loop; in lesson 012 with a `defaultdict` per question. Today they
each become one line, using an idea called **split-apply-combine**:

1. **Split** the rows into groups (all the lattes, all the teas, ...).
2. **Apply** a summary to each group (sum, mean, count, ...).
3. **Combine** the answers into a new, smaller table.

pandas does all three with `groupby`. Once you've got it, a huge range of
questions stop being programming problems and become "what exactly am I
grouping by, and what am I summarising?", which is where the thinking
belongs.

## How to follow along

Venv active, `python` in the repo root. The script loads September with a
`revenue` column and two date parts added (`weekday`, `hour`; lesson 021
explains `.dt` properly):

```python
orders = pd.read_csv("orders_september.csv", parse_dates=["timestamp"])
orders["revenue"] = orders["price"] * orders["quantity"]
orders["weekday"] = orders["timestamp"].dt.day_name()
orders["hour"] = orders["timestamp"].dt.hour
```

Full script:

```bash
python lessons/phase-1-data-science/lesson-019-group-aggregate-pivot/lesson.py
```

It writes no files.

## 1. Split, apply, combine

```python
orders.groupby("drink")["revenue"].sum()
```

```
drink
cappuccino    1859.4
espresso       419.9
flat white    1428.6
latte         2301.6
tea            618.0
```

Read it left to right as a sentence: *group the orders by drink, take the
revenue column, sum it within each group.* The result is a Series indexed
by drink, sorted alphabetically by the group labels. Add
`.sort_values(ascending=False)` for biggest first.

Swap the column or the function and you've asked a different question:

```python
orders.groupby("drink")["quantity"].mean()     # average cups per order, per drink
orders.groupby("drink").size()                 # how many orders (rows) per drink
```

`.size()` counts rows in each group. (`.count()` also exists; it counts
non-missing values per column, which is the same thing on clean data and
different on dirty data.)

## 2. Several summaries at once: `agg`

Pass a list to `.agg` for several functions on one column:

```python
orders.groupby("drink")["revenue"].agg(["count", "sum", "mean", "median"])
```

Or, the form to use in real code, **named aggregation**, which lets you
summarise different columns in different ways and name every result:

```python
summary = orders.groupby("drink").agg(
    orders=("order_id", "count"),
    cups=("quantity", "sum"),
    revenue=("revenue", "sum"),
    avg_order=("revenue", "mean"),
    biggest=("revenue", "max"),
)
```

```
            orders  cups  revenue  avg_order  biggest
drink
latte          377   588   2301.6       6.11     18.8
cappuccino     317   483   1859.4       5.87     18.4
flat white     219   358   1428.6       6.52     19.2
tea            145   246    618.0       4.26     12.0
espresso       156   247    419.9       2.69      6.8
```

Each line is `new_name=(column, function)`. The result is a clean
one-row-per-drink table with columns named exactly as you'll refer to
them. Look at it for a moment: flat whites are fewer than cappuccinos but
have the biggest average order. That's the kind of thing a good summary
table shows you without being asked.

## 3. Groups of groups

Group by a list of columns and you get one group per *combination*:

```python
two_keys = orders.groupby(["drink", "size"])["quantity"].sum()
```

```
drink       size
cappuccino  large     159
            medium    212
            small     112
espresso    small     247
...
```

The result has a two-level index (pandas calls it a **MultiIndex**): drink
on the outside, size inside. It's correct, but long and awkward to read.
Two ways to make it friendlier:

```python
two_keys.unstack()        # move the inner level (size) into columns: a drink-by-size table
two_keys.reset_index()    # turn both index levels into ordinary columns: a tidy long table
```

`unstack()` gives a table for people to read. `reset_index()` gives a
table for further processing, or for saving to CSV. You'll use both
constantly. Notice the `NaN`s in the unstacked table: espresso only comes
small, so there are simply no medium or large espresso rows. `NaN` there
means "no such combination", not "missing data".

## 4. `pivot_table`: the summary table

`groupby` then `unstack` is so common that there's a function for it,
with a name spreadsheet users will recognise:

```python
orders.pivot_table(index="drink", columns="size", values="revenue",
                   aggfunc="sum", margins=True, margins_name="total")
```

```
size         small  medium   large   total
drink
cappuccino   370.0   805.6   683.8  1859.4
espresso     419.9     NaN     NaN   419.9
flat white   332.6   685.6   410.4  1428.6
latte        480.6  1091.2   729.8  2301.6
tea          112.0   320.0   186.0   618.0
total       1715.1  2902.4  2010.0  6627.5
```

Four arguments, and each answers a question:

- `index`: what goes down the side? (drink)
- `columns`: what goes across the top? (size)
- `values`: what are we summarising? (revenue)
- `aggfunc`: how? (`"sum"`, `"mean"`, `"count"`, `"median"`, ...)

`margins=True` adds a total row and column. (Careful with it on means:
the margin is the mean over all the underlying rows, not the mean of the
cells.)

One practical snag: pandas sorts labels alphabetically, so weekdays come
out Friday, Monday, Thursday... `.reindex(WEEKDAYS)` puts rows in the
order you give. The same trick works for sizes (`[["small", "medium",
"large"]]` on the columns).

## 5. Counting combinations: `crosstab`

When the question is "how often does each pair turn up?", `pd.crosstab`
is the shortcut:

```python
pd.crosstab(orders["drink"], orders["milk"])
```

It counts rows for every combination of the two columns. Add
`normalize="index"` and each row becomes shares that add up to 1:

```python
milky = orders[orders["milk"] != "none"]
pd.crosstab(milky["drink"], milky["milk"], normalize="index")
```

```
milk          oat  whole
drink
cappuccino  0.262  0.738
flat white  0.242  0.758
latte       0.233  0.767
```

About a quarter of every milky drink is oat. Shares are how you compare
groups of different sizes: 88 oat lattes sounds like more than 83 oat
cappuccinos, but as a *share*, cappuccino drinkers choose oat slightly
more often. (`normalize="columns"` makes columns add to 1;
`normalize="all"` makes the whole table add to 1.)

## 6. `transform`: a group's answer on every row

`agg` shrinks the table to one row per group. Sometimes you want the
group's answer *next to each original row*, to compare a row with its
group. That's `transform`:

```python
orders["drink_avg"] = orders.groupby("drink")["revenue"].transform("mean")
orders["vs_drink_avg"] = orders["revenue"] - orders["drink_avg"]
```

```
        drink  revenue  drink_avg  vs_drink_avg
0  cappuccino      8.4       5.87          2.53
1    espresso      1.7       2.69         -0.99
2  flat white     13.6       6.52          7.08
```

`transform("mean")` returns a Series *the same length as `orders`*, where
every latte row holds the latte average, every tea row the tea average,
and so on, lined up by index. So "how does this order compare with its
drink's average?" is a subtraction.

The same trick gives shares within a group:

```python
orders["share_of_day"] = orders["revenue"] / orders.groupby("day")["revenue"].transform("sum")
```

Each order's revenue divided by its own day's total. (It's lesson 015's
`keepdims=True` idea, "reduce, then broadcast back", done with labels
instead of shapes.)

## 7. Putting it together: weekdays and hours

The owner asks: "Which weekday is best?" This is where precision about X
and Y matters, because there are two reasonable-sounding answers and only
one is right.

**Wrong:** `orders.groupby("weekday")["revenue"].mean()`. That's the
average *order* per weekday: how much a typical order is worth on a
Monday. Useful, but not what was asked.

**Right:** total each *day* first, then average the days per weekday:

```python
daily = orders.groupby(["day", "weekday"], as_index=False)["revenue"].sum()
daily.groupby("weekday")["revenue"].agg(days="count", avg_day="mean")
```

```
           days  avg_day
weekday
Monday        4   277.18
Tuesday       5   293.84
Wednesday     5   274.48
Thursday      4   302.12
Friday        4   367.18
```

Fridays take about £367 on an average day, a good £65 more than any other
weekday. (The `days` column matters too: September 2026 has five Tuesdays
and Wednesdays but only four of the others. Summing per weekday instead of
averaging would have flattered Tuesdays and Wednesdays just for turning
up more often.) `as_index=False` keeps `day` and `weekday` as ordinary
columns instead of an index, ready for the second groupby.

Then cups per hour, per weekday, as a pivot table divided by the number of
each weekday in the month (broadcasting over columns, lesson 015):

```
weekday  Monday  Tuesday  Wednesday  Thursday  Friday
hour
7           9.0     13.0       13.2       8.8    13.5
8          20.8     17.2       19.2      18.5    24.0
9          14.8     12.6       10.8      17.0    18.2
...
```

8am is the peak every day, and Friday's 8am rush is the busiest hour of
the week: about 24 cups. That's a staffing decision, in a table.

The lesson inside the lesson: **before you type `groupby`, say the
question out loud with its units.** "Average revenue per *day*, for each
weekday" and "average revenue per *order*, for each weekday" are both one
line, and they give different answers.

## What you can do now

- Explain split-apply-combine, and write `df.groupby(key)[column].func()`.
- Use `.agg` with a list of functions and with named aggregation.
- Group by several columns, and use `unstack()` and `reset_index()` to
  reshape the result.
- Build summary tables with `pivot_table` (index, columns, values,
  aggfunc, margins), and put labels in a sensible order with `reindex`.
- Count and compare combinations with `crosstab` and `normalize`.
- Use `transform` to put a group's answer on every row, for comparisons and
  shares.
- Tell "average per order" from "average per day" and choose on purpose.

## What to do now

1. Run `lesson.py`, then try section 7's *wrong* version at `>>>` and see
   how different its answer is.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   tracks oat milk's rise through the month.
3. Lesson 020 (joining tables and reshaping) is this afternoon. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

`groupby` is the single most useful thing in pandas. Most of the
analyses you'll ever write are a clean table, a couple of groupbys, and a
careful sentence about what the numbers mean.
