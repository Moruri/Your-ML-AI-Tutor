# Lesson 017 - Exercises

Three quick checks, a hands-on task answering the owner's questions about
September, and an optional question about alignment. Predict first, then
run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-017-pandas-series-and-dataframes/lesson.py my_lesson_017.py
python my_lesson_017.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-017-pandas-series-and-dataframes").resolve()
```

Or just start fresh at `>>>` with
`orders = pd.read_csv("lessons/phase-1-data-science/lesson-017-pandas-series-and-dataframes/orders_september.csv")`.

---

## 1. Series or DataFrame?

For each, is the result a Series, a DataFrame, or a single value?

```python
orders["price"]
orders[["price"]]
orders["price"].mean()
orders[orders["quantity"] > 2]
orders.loc[5]
orders.loc[5, "drink"]
orders.loc[:, ["drink", "size"]]
orders["drink"].value_counts()
```

<details>
<summary>Check yourself</summary>

```
Series           one column by name
DataFrame        a LIST with one name: a one-column table
single value     a float
DataFrame        a mask keeps whole rows
Series           one row; its index is the column names
single value     one row, one column: a cell
DataFrame        all rows, two columns
Series           counts, indexed by drink
```

The `["price"]` vs `[["price"]]` difference catches everyone. When a
function complains it wanted a DataFrame and got a Series (or the other
way round), check the brackets. `type(x)` settles it.

</details>

## 2. `.loc` or `.iloc`?

```python
lattes = orders[orders["drink"] == "latte"]
print(lattes.index[:3])
```

This prints `Index([5, 9, 12], dtype='int64')`. What does each of these
give?

```python
lattes.iloc[0]
lattes.loc[5]
lattes.loc[0]
lattes.iloc[0:2]
```

<details>
<summary>Check yourself</summary>

- `lattes.iloc[0]`: the first latte, whose label is 5.
- `lattes.loc[5]`: the same row, by its label.
- `lattes.loc[0]`: `KeyError: 0`. There's no row *labelled* 0 in
  `lattes`; order 0 was a cappuccino and got filtered out.
- `lattes.iloc[0:2]`: the first two lattes (labels 5 and 9).

The filter kept the original labels, which is usually what you want: you
can always find each row in the original table. If you'd rather have
fresh `0, 1, 2...` labels, `lattes.reset_index(drop=True)` renumbers.

</details>

## 3. Spot the bugs

Two lines, two different problems. What goes wrong with each, and what's
the fix?

```python
large_lattes = orders[orders["drink"] == "latte" & orders["size"] == "large"]

orders.revenue = orders.price * orders.quantity
```

<details>
<summary>Check yourself</summary>

**Line 1** raises an error (a `TypeError` about `&` on strings, or the
"truth value is ambiguous" `ValueError`). `&` binds tighter than `==`, so
Python tries `"latte" & orders["size"]` first. Bracket each comparison:

```python
orders[(orders["drink"] == "latte") & (orders["size"] == "large")]
```

**Line 2** doesn't raise an error, which is worse. pandas sets a plain
Python attribute called `revenue` on the DataFrame object, and does *not*
create a column (recent versions print a `UserWarning` saying so).
`orders["revenue"]` then gives a `KeyError`. Always create columns with
brackets: `orders["revenue"] = ...`.

</details>

## 4. Hands-on: the owner's September questions

Load the file and add the `revenue` column. Then answer each question
with pandas (masks, `.loc`, summaries). No loops.

**a) Large cups.** How many orders were for large drinks, and what share
of September's revenue did they bring in?

**b) Oat vs whole.** Among drinks with milk, what's the average *price*
of an oat order and of a whole-milk order? The menu charges 40p extra for
oat. Is the gap you see about 40p?

**c) Before nine.** How many orders came in before 9am, and what share of
all *cups* is that? (The hour is characters 11 and 12 of the timestamp
text: `orders["timestamp"].str[11:13].astype(int)`. Lesson 021 has a
proper way.)

**d) The priciest cup.** What's the highest price on any order, and which
drink, size and milk does it belong to?

**e) One order.** Make `order_id` the index and look up order `S0777`.
What was it?

Hints, if you want them:

- For (a), `orders.loc[orders["size"] == "large", "revenue"].sum()` over
  `orders["revenue"].sum()`.
- For (b), `orders.loc[orders["milk"] == "oat", "price"].mean()`.
- For (d), `orders["price"].max()` gives the value; a mask with
  `== orders["price"].max()` finds the rows; `.drop_duplicates()` on the
  columns you care about shows each combination once.

<details>
<summary>Expected results</summary>

```
a) 308 large orders, 30% of revenue
b) oat £4.21, whole £3.81: a gap of 39p, close to the 40p surcharge
c) 422 orders before 9am, 36% of cups
d) £4.80: large, oat, flat white
e) S0777: 2026-09-18 14:23, 3 medium cappuccinos with whole milk, £11.10, card
```

(b) is worth a second look. The gap isn't *exactly* 40p because oat and
whole orders don't have the same mix of drinks and sizes: if oat drinkers
happen to buy slightly fewer large cups, the average oat price comes out
a bit lower than "whole + 40p". Comparing averages across groups always
has this catch. Lesson 024 is about when a difference between groups
really means what it seems to.

</details>

<details>
<summary>One way to write it</summary>

```python
orders = pd.read_csv(ORDERS_CSV)
orders["revenue"] = orders["price"] * orders["quantity"]

# a)
large = orders["size"] == "large"
print(f"{large.sum()} large orders, "
      f"{orders.loc[large, 'revenue'].sum() / orders['revenue'].sum():.0%} of revenue")

# b)
oat = orders.loc[orders["milk"] == "oat", "price"].mean()
whole = orders.loc[orders["milk"] == "whole", "price"].mean()
print(f"oat £{oat:.2f}, whole £{whole:.2f}: a gap of {(oat - whole) * 100:.0f}p")

# c)
early = orders["timestamp"].str[11:13].astype(int) < 9
print(f"{early.sum()} orders before 9am, "
      f"{orders.loc[early, 'quantity'].sum() / orders['quantity'].sum():.0%} of cups")

# d)
top = orders["price"].max()
print(top, orders.loc[orders["price"] == top, ["drink", "size", "milk"]].drop_duplicates())

# e)
print(orders.set_index("order_id").loc["S0777"])
```

Every answer is a mask, a selection and a summary. That pattern covers a
remarkable share of real data questions.

</details>

## 5. (Optional) Alignment, helpful and surprising

```python
a = pd.Series([1, 2, 3])
b = pd.Series([10, 20, 30], index=[1, 2, 3])
print(a + b)
```

Predict the output. Then: when is this behaviour a lifesaver, and when is
it a trap?

<details>
<summary>Check yourself</summary>

```
0     NaN
1    12.0
2    23.0
3     NaN
```

`a` has labels 0, 1, 2; `b` has 1, 2, 3. pandas adds matching *labels*:
1 with 1 (2 + 10), 2 with 2 (3 + 20). Labels 0 and 3 have no partner:
`NaN`. (The values became floats, because `NaN` is a float.)

**Lifesaver:** combining data from different sources, like prices from
the menu and counts from the till, in different orders, with some drinks
missing from one. Alignment matches them correctly without you sorting
anything.

**Trap:** after filtering or sorting, two Series you *think* line up by
position don't, because their labels differ. You get `NaN`s or quietly
mismatched rows. If you really mean "by position", say so:
`a.values + b.values`, or `reset_index(drop=True)` on both first.

</details>

---

That's the morning of Week 3, Day 3 done. This afternoon's lesson 018
cleans a messy version of this same month. If one thing sticks, let it be:
*shape, head, info, describe before anything else; brackets for columns;
`.loc` for labels, `.iloc` for positions*.
