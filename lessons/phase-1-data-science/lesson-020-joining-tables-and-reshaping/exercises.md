# Lesson 020 - Exercises

Three quick checks, a hands-on task that adds milk costs to the profit
picture, and an optional question about joining on more than one column.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-020-joining-tables-and-reshaping/lesson.py my_lesson_020.py
python my_lesson_020.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-020-joining-tables-and-reshaping").resolve()
```

`load_orders`, `ORDERS_CSV`, `MENU_CSV` and `TARGETS_CSV` are there to
reuse.

---

## 1. How many rows?

```python
left = pd.DataFrame({"drink": ["latte", "latte", "tea", "chai"], "cups": [2, 1, 1, 3]})
right = pd.DataFrame({"drink": ["latte", "tea", "mocha"], "cost": [1.05, 0.35, 1.30]})
```

How many rows does each give?

```python
left.merge(right, on="drink", how="inner")
left.merge(right, on="drink", how="left")
left.merge(right, on="drink", how="right")
left.merge(right, on="drink", how="outer")
```

<details>
<summary>Check yourself</summary>

```
inner  3   latte, latte, tea           (chai and mocha have no partner)
left   4   latte, latte, tea, chai     (chai with cost NaN)
right  4   latte, latte, tea, mocha    (both lattes match the one latte cost; mocha with cups NaN)
outer  5   latte, latte, tea, chai, mocha
```

The `right` result has *four* rows although `right` has three: each of
its rows appears once per match, and latte matches twice. "Keep every row
of the right table" doesn't mean "have as many rows as the right table".

</details>

## 2. Which validate?

Which `validate=` would you write for each join?

- (a) Orders onto the menu, on `drink`.
- (b) Each day's total revenue onto each day's weather, on `date`.
- (c) The menu onto orders (menu on the left), on `drink`.
- (d) Customers onto orders, on `customer_id`, where a customer can have
  many orders and an order has one customer.

<details>
<summary>Check yourself</summary>

- **(a) `"many_to_one"`**: many orders per drink, one menu row per drink.
- **(b) `"one_to_one"`**: one total per day, one weather reading per day.
- **(c) `"one_to_many"`**: the same relationship as (a), seen from the
  other side.
- **(d) `"one_to_many"`**, with customers on the left.

You'll rarely want `"many_to_many"`. If a join genuinely is many-to-many,
the output can be enormous and it's worth stopping to ask whether that's
really what you mean.

</details>

## 3. Wide or long?

For each, would you rather have the data wide or long?

- (a) Computing total cups per drink across all weeks.
- (b) Showing the owner a table of targets to edit.
- (c) Joining targets to actual sales.
- (d) Printing a drink-by-week report for a meeting.

<details>
<summary>Check yourself</summary>

- **(a) Long**: `groupby("drink")["target"].sum()` works directly. (Wide
  works too, with `.sum(axis=1)` over the week columns, but only if you
  pick the right columns.)
- **(b) Wide**: that's the shape people type into.
- **(c) Long**: you need a `week` column to join on.
- **(d) Wide**: that's the shape people read.

Work long, present wide. `melt` on the way in, `pivot` on the way out.

</details>

## 4. Hands-on: the cost of milk

The owner points out that the menu's `cost_per_cup` doesn't include milk,
and oat milk costs more. Here's what milk costs per cup:

```python
milk_costs = pd.DataFrame({"milk": ["whole", "oat", "none"],
                           "milk_cost": [0.10, 0.25, 0.00]})
```

**a) Two joins.** Join the menu (on `drink`) *and* `milk_costs` (on
`milk`) onto the orders, both as `many_to_one` left joins. Check the row
count and that no cost is missing.

**b) Real profit.** Cost per order is `(cost_per_cup + milk_cost) *
quantity`. What's September's total profit now? How much did milk take
off lesson 020's £4,936.05?

**c) By category.** Profit per menu `category` (milky, black, tea).

**d) Is oat worth it?** For each milk type: total cups, total profit, and
profit *per cup*. The shop charges 40p extra for oat and it costs 15p more
than whole. Does oat earn more per cup than whole?

**e) Never sold.** Which drinks on the menu sold nothing in September? Use
a join with `indicator=True` (an "anti-join").

Hints, if you want them:

- Chain them: `orders.merge(menu, ...).merge(milk_costs, ...)`.
- For (d), `groupby("milk").agg(cups=("quantity", "sum"), profit=("profit", "sum"))`,
  then divide.
- For (e), join the menu (left) to the *distinct* drinks sold
  (`orders[["drink"]].drop_duplicates()`), and keep rows whose `_merge` is
  `"left_only"`.

<details>
<summary>Expected results</summary>

```
a) 1,214 rows, no missing costs
b) profit £4,741.25; milk cost £194.80
c) black £308.75, milky £3,900.60, tea £531.90
d) milk   cups   profit  per_cup
   none    493   840.65    1.705
   oat     346  1012.60    2.927
   whole  1083  2888.00    2.667
e) ['mocha']
```

(d): yes, about 26p more profit per cup for oat than whole. The 40p
surcharge more than covers the 15p extra cost. That's a useful thing to
tell an owner who's wondering whether to drop the surcharge to encourage
oat: it would cost them about 26p a cup on a growing share of sales
(lesson 019's exercises).

</details>

<details>
<summary>One way to write it</summary>

```python
orders = load_orders(ORDERS_CSV)
menu = pd.read_csv(MENU_CSV)
milk_costs = pd.DataFrame({"milk": ["whole", "oat", "none"], "milk_cost": [0.10, 0.25, 0.00]})

# a)
joined = (orders
          .merge(menu, on="drink", how="left", validate="many_to_one")
          .merge(milk_costs, on="milk", how="left", validate="many_to_one"))
assert len(joined) == len(orders)
assert joined[["cost_per_cup", "milk_cost"]].notna().all().all()

# b)
joined["cost"] = (joined["cost_per_cup"] + joined["milk_cost"]) * joined["quantity"]
joined["profit"] = joined["revenue"] - joined["cost"]
total = joined["profit"].sum()
print(f"profit £{total:,.2f}; milk cost £{4936.05 - total:,.2f}")

# c)
print(joined.groupby("category")["profit"].sum().round(2))

# d)
by_milk = joined.groupby("milk").agg(cups=("quantity", "sum"), profit=("profit", "sum"))
by_milk["per_cup"] = by_milk["profit"] / by_milk["cups"]
print(by_milk.round(3))

# e)
sold = orders[["drink"]].drop_duplicates()
check = menu.merge(sold, on="drink", how="left", indicator=True)
print(check.loc[check["_merge"] == "left_only", "drink"].tolist())
```

Wrapping a chain of methods in brackets, one per line, is a common pandas
style: it reads top to bottom like a recipe.

</details>

## 5. (Optional) Joining on two keys

In section 7, actual cups were joined to targets `on=["drink", "week"]`.
What would go wrong if you joined `on="drink"` only?

<details>
<summary>Check yourself</summary>

Each drink has five rows in `actual` (one per week) and five in
`targets_long`. Joining on drink alone matches *every* week's actual with
*every* week's target: 5 x 5 = 25 rows per drink, 125 in total, with
week 36's sales set against week 39's target and so on. It's a
many-to-many join, and every "percent of target" in it is nonsense.

`validate="one_to_one"` would have caught it immediately. The key for a
join is *whatever uniquely identifies a row* on the side you're matching:
here, the pair (drink, week). When one column isn't unique, ask what
the other part of the key is.

</details>

---

That's Week 3, Day 4 done. Lessons 021 and 022 arrive tomorrow (Friday).
If one thing sticks from today, let it be: *count rows before and after,
look at `_merge`, and tell pandas what relationship you expect with
`validate`*.
