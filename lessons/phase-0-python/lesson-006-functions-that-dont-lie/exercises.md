# Lesson 006 - Exercises

Three quick checks, one hands-on refactor, and an optional design question.
Predict first, then run.

For the hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-0-python/lesson-006-functions-that-dont-lie/lesson.py my_lesson_006.py
python my_lesson_006.py
```

(No data file today, so the copy runs from anywhere. `ORDERS`, `pounds` and
the section 2 helpers are all there for you to reuse.)

---

## 1. What did the caller get?

```python
def describe(drink, cents):
    print(f"{drink}: £{cents / 100:.2f}")

label = describe("latte", 380)
print(label)
print(f"{label!r}")
```

Three lines are printed. What are they?

<details>
<summary>Check yourself</summary>

```
latte: £3.80     <- printed by describe itself, while it ran
None             <- label is None, because describe has no return
None             <- !r of None is still None
```

If you wanted `label` to be `'latte: £3.80'`, `describe` needed `return`
instead of `print`. Then the *caller* prints it, if printing is what the
caller wants.

</details>

## 2. The shared list

```python
def add_cup(drink, cups=[]):
    cups.append(drink)
    return cups

morning = add_cup("latte")
afternoon = add_cup("tea")
print(afternoon)
print(morning is afternoon)

evening = add_cup("mocha", [])
print(evening)
print(morning)
```

Four lines. Predict them, then explain the last two.

<details>
<summary>Check yourself</summary>

```
['latte', 'tea']     <- the default list was created once and shared
True                 <- morning and afternoon are the SAME list
['mocha']            <- we passed our own fresh list, so the default wasn't used
['latte', 'tea']     <- ...and the shared default is still sitting there, unchanged
```

The `[]` in the signature is built when Python reads the `def`, not on each
call. Passing a list explicitly sidesteps it, but the trap is still there for
the next caller. Fix it properly:

```python
def add_cup(drink, cups=None):
    if cups is None:
        cups = []
    cups.append(drink)
    return cups
```

</details>

## 3. Spot the lie

Four functions. For each one: does it keep its promise? If not, what's the
lie, and what's the one-line fix?

```python
# (a)
def is_large(order):
    if order[2] == "large":
        return True

# (b)
def cheapest(prices):
    prices.sort()
    return prices[0]

# (c)
def total_pounds(orders):
    return f"£{sum(p * q for *_, p, q in orders) / 100:.2f}"

# (d)
def load_clean_and_print(path):
    ...   # opens the file, fixes the prices, prints a table
```

<details>
<summary>Check yourself</summary>

**(a) Half a lie.** For a medium order it returns `None`, not `False`. That
works inside `if is_large(o):` but breaks `is_large(o) == False`, and it
shows up as `None` in any list you build. An `is_` function should always
return a real boolean: `return order[2] == "large"`.

**(b) A lie.** The name says "tell me the cheapest". The body rearranges the
caller's list on the way. If `prices` was in date order, that's gone.
`return min(prices)` does the job without touching anything. (If you really
did need it sorted, `sorted(prices)[0]` still leaves the original alone.)

**(c) Honest, with a quibble.** It returns a string, and the name
`total_pounds` sort of says so. But most people reading `total_pounds(orders)`
would expect a *number* they could add to something. `format_total(orders)`
or `total_as_pounds_str` would make the string obvious. Not a bug; a name
that could be clearer. Better still: `total_cents(orders)` returning an int,
and format it at the print.

**(d) Three jobs.** The name has two "and"s in it, which is the tell. Split
into `load_orders(path)`, `clean_prices(orders)` (returning a new list) and
`print_table(orders)`. Then you can test the first two and reuse them in a
report that doesn't print.

</details>

## 4. Hands-on: the blob

Someone wrote this on a Friday afternoon. It works. Nobody wants to touch it.

```python
ORDERS.sort(key=lambda o: o[3] * o[4], reverse=True)
totals = {}
for o in ORDERS:
    totals[o[1]] = totals.get(o[1], 0) + o[3] * o[4]
print("drink        revenue")
for d in totals:
    if totals[d] >= 2000:
        flag = " *"
    else:
        flag = ""
    print(f"{d:<12} £{totals[d] / 100:>6.2f}{flag}")
grand = 0
for d in totals:
    grand = grand + totals[d]
print(f"total        £{grand / 100:>6.2f}")
print("* = £20 or more")
```

Paste it at the bottom of your copy (below `main()`, or replace `main()`'s
body with it) and run it once to see what it produces. Then:

**a) Name the jobs.** Before writing any code, list what this blob does, in
plain English, one item per line. You should find at least four separate
jobs, and one thing it does that it shouldn't.

**b) Split it into functions.** Write:

- `line_total(order)` - cents for one order (you've seen this one).
- `revenue_by_drink(orders)` - returns a dict, drink to cents. Must not
  change `orders`.
- `format_row(drink, cents, star_from_cents)` - returns one line of the
  table as a string, with `" *"` on the end if `cents >= star_from_cents`.
- `print_report(orders, star_from_cents=2000)` - the only function that
  prints. It should show the drinks **biggest revenue first**, then the
  total, then the legend.

Then call `print_report(ORDERS)`.

**c) Prove nothing was harmed.** After calling `print_report(ORDERS)`, add
`assert ORDERS[0] == ("2026-09-07", "latte", "medium", 380, 2)`. The blob
would fail this. Yours shouldn't.

**d) Use the default.** Call `print_report(ORDERS, star_from_cents=1000)`.
One more drink should get a star, and the legend should say `£10.00`.

**e) Five asserts.** Two for `line_total`, two for `revenue_by_drink` (one
of them on an empty list), one for `format_row`.

Hints, if you want them:

- The blob's first line sorts `ORDERS` in place by line total. Look at the
  order the drinks come out in: it isn't even revenue order. The sort is
  both harmful (it rearranges shared data) and useless (it doesn't do what
  the author hoped). Your `print_report` should sort the *totals*, with
  `sorted(revenue.items(), key=lambda item: item[1], reverse=True)`.
- The grand total is `sum(revenue.values())`. No loop needed.
- `" *" if cents >= star_from_cents else ""` is the one-line form of the
  blob's four-line `if`/`else`. Either is fine.

<details>
<summary>Expected results</summary>

The blob prints (notice the odd order and the first-order line at the end):

```
drink        revenue
flat white   £ 15.60
cappuccino   £ 20.00 *
latte        £ 29.90 *
espresso     £  8.80
tea          £  5.40
total        £ 79.70
* = £20 or more
```

Your `print_report(ORDERS)` should print:

```
drink        revenue
latte        £ 29.90 *
cappuccino   £ 20.00 *
flat white   £ 15.60
espresso     £  8.80
tea          £  5.40
total        £ 79.70
* = £20.00 or more
```

And `print_report(ORDERS, star_from_cents=1000)` puts a star on flat white
too, with the legend `* = £10.00 or more`. The assert in (c) passes.

</details>

<details>
<summary>One way to write it</summary>

```python
# a) The jobs:
#    1. work out the value of each order line
#    2. add those up per drink
#    3. decide which drinks get a star
#    4. print a table, a total and a legend
#    ...and it sorts ORDERS in place, which is nobody's business but the caller's.

# b)
def line_total(order):
    """Cents for one order line."""
    _, _, _, price, qty = order
    return price * qty


def revenue_by_drink(orders):
    """Total cents per drink, as a plain dict. Leaves orders alone."""
    totals = {}
    for order in orders:
        totals[order[1]] = totals.get(order[1], 0) + line_total(order)
    return totals


def format_row(drink, cents, star_from_cents):
    """One table line, starred if it's a big earner."""
    star = " *" if cents >= star_from_cents else ""
    return f"{drink:<12} £{cents / 100:>6.2f}{star}"


def print_report(orders, star_from_cents=2000):
    """Print the revenue table. This is the only function that prints."""
    revenue = revenue_by_drink(orders)
    print(f"{'drink':<12} revenue")
    for drink, cents in sorted(revenue.items(), key=lambda item: item[1], reverse=True):
        print(format_row(drink, cents, star_from_cents))
    print(f"{'total':<12} £{sum(revenue.values()) / 100:>6.2f}")
    print(f"* = £{star_from_cents / 100:.2f} or more")


print_report(ORDERS)

# c)
assert ORDERS[0] == ("2026-09-07", "latte", "medium", 380, 2)

# d)
print_report(ORDERS, star_from_cents=1000)

# e)
assert line_total(("d", "tea", "large", 290, 1)) == 290
assert line_total(("d", "latte", "medium", 380, 2)) == 760
assert revenue_by_drink([]) == {}
assert revenue_by_drink(ORDERS[:2]) == {"latte": 760, "espresso": 220}
assert format_row("tea", 540, 2000) == "tea          £  5.40"
```

Count the lines: the refactor is *longer* than the blob. That's normal and
it's fine. You've traded fifteen lines nobody could safely change for five
functions that each have one job, four of which can be checked with an
`assert`. The blob couldn't be tested at all without reading a screen.

If you used `defaultdict(int)` in `revenue_by_drink` instead of `.get`,
good; just remember to `return dict(totals)` so callers get a plain dict
that doesn't grow when they look at it (yesterday's trap).

</details>

## 5. (Optional) Tuple or dict?

Here are two ways to summarise a list of prices:

```python
def summarise(prices):
    return len(prices), sum(prices) / len(prices), min(prices), max(prices)

def summarise_dict(prices):
    return {"count": len(prices), "mean": sum(prices) / len(prices),
            "min": min(prices), "max": max(prices)}
```

Both are honest: consistent shape, nothing mutated, no printing. Which one
would you rather *call*, and why? What happens to each when you add a fifth
statistic next week?

<details>
<summary>Check yourself</summary>

With four values, the dict is kinder. `s["mean"]` tells the reader what it
is; `s[1]` makes them go and look. And when you add `"median"` next week, the
dict grows without breaking anyone: `count, mean, lo, hi = summarise(prices)`
would crash ("too many values to unpack") the moment a fifth item appeared.

Tuples are the right call for two or three values with an obvious order
(`lo, hi = cheapest_and_dearest(...)`). Past that, name the fields. Lesson
010 gives you a third option, `dataclasses`, that reads like a dict and
behaves like a fixed record. For now: two or three things, tuple; more than
that, dict.

</details>

---

That's Week 2, Day 2 done. Lessons 007 and 008 arrive tomorrow (Wednesday).
If one thing sticks from today, let it be: *return, don't print, and let the
name tell the truth*.
