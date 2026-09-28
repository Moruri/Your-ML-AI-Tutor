# Lesson 010 - Exercises

Three quick checks, a hands-on task that sharpens `Order` and builds a
model you can score, and an optional question about when *not* to write a
class. Predict first, then run.

For the hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-0-python/lesson-010-a-little-bit-of-classes/lesson.py my_lesson_010.py
python my_lesson_010.py
```

As in lesson 008, change the `HERE = ...` line near the top so the copy
can find the data:

```python
HERE = Path("lessons/phase-0-python/lesson-010-a-little-bit-of-classes").resolve()
```

`Order`, `load_orders`, `AverageCupsModel` and `pounds` are all there for
you to reuse.

---

## 1. What prints?

```python
from dataclasses import dataclass

@dataclass
class Cup:
    drink: str
    size: str = "medium"

a = Cup("latte")
b = Cup("latte", "medium")
c = Cup(size="large", drink="tea")

print(a)
print(a == b)
print(a is b)
print(c.size)
```

<details>
<summary>Check yourself</summary>

```
Cup(drink='latte', size='medium')
True
False
large
```

`==` compares the *values*, field by field, which dataclasses write for
you. `is` asks "are these the very same object?", and they aren't: two
separate `Cup(...)` calls make two objects that happen to hold the same
values. Keyword arguments work in any order, just like any function.

</details>

## 2. Spot the bug

```python
class Counter:
    def __init__(self):
        count = 0

    def add(self):
        self.count += 1

tally = Counter()
tally.add()
```

This raises an error on the last line. Which one, and what's the one-word
fix?

<details>
<summary>Check yourself</summary>

`AttributeError: 'Counter' object has no attribute 'count'`.

`count = 0` inside `__init__` makes an ordinary local variable that
vanishes when `__init__` finishes. To store it *on the object* you need
`self.count = 0`. The one-word fix is `self.`. Forgetting `self.` is the
single most common class bug; when an attribute "should be there" and
isn't, check `__init__` first.

</details>

## 3. Dataclass or dict or plain class?

Pick one for each, and say why in a sentence:

- (a) A row you've just read with `DictReader`, which you'll clean and
  throw away in the next line.
- (b) A validated sale that the rest of the program will rely on.
- (c) Something that learns from data in one method and uses it in
  another.

<details>
<summary>Check yourself</summary>

**(a) Dict.** It's what `DictReader` gives you, it lives for one line, and
wrapping it in a class buys nothing.

**(b) Frozen dataclass.** Named fields, validation in `__post_init__`, and
nobody can change it after the fact. This is `Order`.

**(c) Plain class.** Its job is behaviour, and its state (what it learned)
changes when you call `fit()`. This is `AverageCupsModel`, and every
scikit-learn model.

</details>

## 4. Hands-on: a sharper Order, and a model you can score

**a) Better messages.** Right now `Order.from_row` with a quantity of
`"two"` raises `invalid literal for int() with base 10: 'two'`, which
doesn't name the column. Change `from_row` so the price and quantity
conversions each have their own `try`/`except ValueError`, and raise
lesson 008 style messages:

```
quantity 'two' is not a whole number
price 'free' is not a number
```

(Remember `raise ... from err`.)

**b) A method.** Give `Order` a `weekday()` method that returns the day
name, e.g. `'Monday'`. Check: the first order in `orders.csv` is a
Monday.

**c) A new model.** Write a plain class `AverageSpendModel` with:

- `fit(orders)`: learns the average `total_cents()` per order, for each
  drink, and the overall average. Returns `self`.
- `predict(drink)`: the learned average for that drink, or the overall
  average for a drink it hasn't seen. Raises `RuntimeError` if called
  before `fit`.
- `score(orders)`: the **mean absolute error**: for each order, the gap
  between `predict(order.drink)` and the real `order.total_cents()`,
  ignoring the sign (`abs`), averaged over all orders.

**d) Compare with a lazy baseline.** How good is "always predict the
overall average"? Work out its mean absolute error too. Which is lower?

Hints, if you want them:

- For (a), keep the `try` around one conversion each (lesson 008,
  section 4): convert price in one `try`, quantity in another, then call
  `cls(...)`.
- For (c), `AverageCupsModel.fit` is 90% of what you need. Swap
  `order.quantity` for `order.total_cents()`.
- For (d), the baseline is one line once you have the model:
  `sum(abs(o.total_cents() - model.overall_average) for o in orders) / len(orders)`.

<details>
<summary>Expected results</summary>

```
a) quantity 'two' is not a whole number
   price 'free' is not a number
b) Monday
c) predict('latte')      -> 598.0     (£5.98 per latte order, on average)
   predict('flat white') -> 780.0
   predict('mocha')      -> 531.33    (overall average)
   score(orders)         -> 158.58    (about £1.59 off per order)
d) baseline error        -> 232.27    (about £2.32 off per order)
```

Knowing the drink makes the prediction about 73p better per order.
That's the whole game of machine learning in one sentence: *does knowing
this thing help me guess better than not knowing it?* You'll ask that
question, with better tools, about every model in Phase 2. And we scored
on the same orders we trained on, which flatters the model. Lesson 031 is
entirely about why that's a problem.

</details>

<details>
<summary>One way to write it</summary>

```python
# a) inside Order
    @classmethod
    def from_row(cls, raw):
        row = {key: (value or "").strip() for key, value in raw.items() if key is not None}
        try:
            price_cents = round(float(row["price"].replace("£", "")) * 100)
        except ValueError as err:
            raise ValueError(f"price {row['price']!r} is not a number") from err
        try:
            quantity = int(row["quantity"])
        except ValueError as err:
            raise ValueError(f"quantity {row['quantity']!r} is not a whole number") from err
        return cls(date=date.fromisoformat(row["date"]), drink=row["drink"].lower(),
                   size=row["size"].lower(), price_cents=price_cents, quantity=quantity)

# b) inside Order
    def weekday(self):
        return self.date.strftime("%A")


# c)
class AverageSpendModel:
    def __init__(self):
        self.average_by_drink = {}
        self.overall_average = None

    def fit(self, orders):
        totals, counts = {}, {}
        for order in orders:
            totals[order.drink] = totals.get(order.drink, 0) + order.total_cents()
            counts[order.drink] = counts.get(order.drink, 0) + 1
        self.average_by_drink = {d: totals[d] / counts[d] for d in totals}
        self.overall_average = sum(totals.values()) / sum(counts.values())
        return self

    def predict(self, drink):
        if self.overall_average is None:
            raise RuntimeError("call fit() before predict()")
        return self.average_by_drink.get(drink, self.overall_average)

    def score(self, orders):
        errors = [abs(self.predict(o.drink) - o.total_cents()) for o in orders]
        return sum(errors) / len(errors)


orders = load_orders(ORDERS_CSV)
print(orders[0].weekday())
model = AverageSpendModel().fit(orders)
for drink in ["latte", "flat white", "mocha"]:
    print(drink, round(model.predict(drink), 2))
print("score", round(model.score(orders), 2))
baseline = sum(abs(o.total_cents() - model.overall_average) for o in orders) / len(orders)
print("baseline", round(baseline, 2))
```

`fit` returning `self` is why `AverageSpendModel().fit(orders)` works in
one line. scikit-learn does exactly the same, and you'll see
`model = SomeModel().fit(X, y)` all over Phase 2.

</details>

## 5. (Optional) When is a class overkill?

A colleague writes this:

```python
class PoundsFormatter:
    def __init__(self):
        pass

    def format(self, cents):
        return f"£{cents / 100:.2f}"

print(PoundsFormatter().format(7970))
```

What would you suggest, and why?

<details>
<summary>Check yourself</summary>

A plain function: `def pounds(cents): return f"£{cents / 100:.2f}"`. The
class stores nothing (`__init__` does nothing), so every instance is
identical and the object is just a longer way to reach a function.

A good test: *does it hold data that belongs together, or remember
something between calls?* If neither, it's a function. Python is happy
with plain functions; you don't need a class to be "proper". The best
Python codebases have a few well-chosen classes and lots of small
functions.

</details>

---

That's Week 2, Day 4 done. Lessons 011 and 012 arrive tomorrow (Friday),
and 012 is the Phase 0 mini-project. If one thing sticks from today, let
it be: *a dataclass is a dict that knows its own keys, checks its own
values, and carries its own methods*.
