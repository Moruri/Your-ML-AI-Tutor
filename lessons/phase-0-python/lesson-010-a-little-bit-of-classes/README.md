# Lesson 010 - A little bit of classes

**Phase 0 - Python foundations for data work** | Week 2, Day 4 | Thursday 2026-09-17

> **Goal:** bundle related data and behaviour with `dataclasses`, so that
> an order is a thing with named fields and its own methods, and so that
> the ML code you'll read later (`model.fit(...)`, `model.predict(...)`)
> looks familiar rather than magical.

Time: about 45 minutes. No installs. Python 3.10+.

---

Every order this week has been a dict: `{"date": ..., "drink": ...,
"price_cents": ..., "quantity": ...}`. Dicts have served us well. They're
quick to make, easy to print, and `csv.DictReader` hands them over for
free.

But they have two blind spots. A dict has no idea which keys it *should*
have, so `order["quantiy"] = 3` (spot the typo) quietly adds a sixth key
instead of complaining. And the things you *do* with an order, like
working out what it cost, live in separate functions scattered around
the file, so every script ends up writing `row["price_cents"] *
row["quantity"]` again.

A **class** fixes both. It's a blueprint that says "an order has exactly
these fields, and here's what you can ask one". This lesson is
deliberately *a little bit* of classes: enough to write clean data
objects with `@dataclass` and read other people's classes without
flinching. We'll skip inheritance and the rest of the zoo. You won't
need them for a long while, and some people never do.

## How to follow along

Terminal in the repo root, `python`, type along at `>>>`. Full script:

```bash
python lessons/phase-0-python/lesson-010-a-little-bit-of-classes/lesson.py
```

The script writes no files. It reads `orders.csv`, the same clean week
as lesson 009.

## 1. Where dicts start to creak

```python
order = {"date": "2026-09-07", "drink": "latte", "size": "medium",
         "price_cents": 380, "quantity": 2}

order["quantiy"] = 3     # typo
order["quantity"]        # still 2
order.get("qty")         # None, no error
```

No error, anywhere. The dict now has a stray key, the real quantity is
unchanged, and the bug surfaces later as a wrong number in a report,
which (lesson 008) is the most expensive place for a bug to surface.

This isn't a reason to stop using dicts. For a row straight out of a CSV,
a dict is perfect. It's a reason to have something stricter for the data
your program *relies on*.

## 2. A class, the long way, then the dataclass way

Here's a class written the traditional way:

```python
class OrderLongHand:
    def __init__(self, date, drink, size, price_cents, quantity):
        self.date = date
        self.drink = drink
        self.size = size
        self.price_cents = price_cents
        self.quantity = quantity

order = OrderLongHand("2026-09-07", "latte", "medium", 380, 2)
order.quantity           # 2
```

Some vocabulary, since you'll see it everywhere:

- `class OrderLongHand:` defines a new **type**. By convention, class names
  are `CapitalisedWords`.
- `OrderLongHand(...)` makes an **instance** (or **object**): one
  particular order. You can make as many as you like from one class.
- `__init__` is the function that runs when an instance is made. Its job
  is to store the values.
- `self` is the instance being built. `self.quantity = quantity` means
  "this order's quantity is the value passed in". Things stored on `self`
  are called **attributes**, and you read them with a dot: `order.quantity`.

It works, but look at the repetition: every field name typed three times.
And printing one gives `<__main__.OrderLongHand object at 0x7f...>`, and
two orders with identical values aren't `==`. You'd need to write more
special methods to fix that.

Python's `dataclasses` module writes all of it for you:

```python
from dataclasses import dataclass

@dataclass
class OrderDraft:
    date: str
    drink: str
    size: str
    price_cents: int
    quantity: int
```

Each line under the class is a **field**: a name and a type hint. The
`@dataclass` line above the class (a *decorator*; think of it as "and
please add the usual machinery") reads those fields and generates
`__init__`, a readable `repr`, and `==`:

```python
draft = OrderDraft("2026-09-07", "latte", "medium", 380, 2)
print(draft)    # OrderDraft(date='2026-09-07', drink='latte', size='medium', price_cents=380, quantity=2)
draft == OrderDraft("2026-09-07", "latte", "medium", 380, 2)    # True
asdict(draft)   # back to a plain dict, when you need one (to write a CSV, say)
```

And the typo?

```python
OrderDraft("2026-09-07", "latte", "medium", 380, quantiy=2)
# TypeError: OrderDraft.__init__() got an unexpected keyword argument 'quantiy'. Did you mean 'quantity'?
```

The class knows its fields, so a misspelling fails *when you build the
object*, with a message that even suggests the fix.

One honest caveat: type hints like `quantity: int` are notes for humans
and tools, not checks. `OrderDraft("x", "y", "z", "free", None)` works
without complaint. Section 4 is how you add real checks.

## 3. Methods: behaviour that lives with the data

A function defined inside a class is a **method**:

```python
@dataclass
class OrderWithMethods:
    date: str
    drink: str
    size: str
    price_cents: int
    quantity: int

    def total_cents(self):
        return self.price_cents * self.quantity

    def describe(self):
        return f"{self.quantity} x {self.size} {self.drink} on {self.date} = {pounds(self.total_cents())}"

order = OrderWithMethods("2026-09-08", "flat white", "medium", 390, 3)
order.total_cents()     # 1170
order.describe()        # '3 x medium flat white on 2026-09-08 = £11.70'
```

The first parameter of every method is `self`, and you never pass it
yourself: `order.total_cents()` is Python's shorthand for
`OrderWithMethods.total_cents(order)`. The order before the dot becomes
`self`. `lesson.py` calls it both ways to prove it.

This is the real point of classes. The knowledge "an order's cost is price
times quantity" now lives *on the order*, in one place, and anyone holding
an order can ask for it.

## 4. Building objects safely: `from_row` and `__post_init__`

Here's the `Order` we'll actually use:

```python
@dataclass(frozen=True)
class Order:
    date: date
    drink: str
    size: str
    price_cents: int
    quantity: int = 1

    def __post_init__(self):
        if self.drink not in MENU:
            raise ValueError(f"drink {self.drink!r} is not on the menu")
        ...
        if self.quantity < 1:
            raise ValueError(f"quantity {self.quantity} must be at least 1")

    @classmethod
    def from_row(cls, raw):
        row = {key: (value or "").strip() for key, value in raw.items() if key is not None}
        return cls(
            date=date.fromisoformat(row["date"]),
            drink=row["drink"].lower(),
            size=row["size"].lower(),
            price_cents=round(float(row["price"].replace("£", "")) * 100),
            quantity=int(row["quantity"]),
        )
```

Three new things:

- **`__post_init__`** runs automatically straight after the generated
  `__init__`. It's the place for rules: lesson 008's validation, living
  on the class itself. Now there is *no way* to make an `Order` with a
  quantity of `-1`, whether it comes from a CSV or from your own code.
- **`@classmethod`** makes a method you call on the *class*, not on an
  instance: `Order.from_row(raw)`. It gets the class as `cls` and returns
  a new instance. It's the conventional way to say "another way to build
  one of these", here from a CSV row of strings. Conversion (strings to
  numbers and dates) happens in `from_row`; rules happen in
  `__post_init__`.
- **`quantity: int = 1`** is a default. Fields with defaults must come
  after fields without, same as function arguments (lesson 006).

Notice that `date` is now a real `datetime.date`, not a string, so
`order.date.strftime("%A")` tells you it was a Monday. When the data is an
object, it can carry proper types.

## 5. Frozen, and the mutable-default trap

`@dataclass(frozen=True)` makes instances read-only:

```python
order.quantity = 20
# FrozenInstanceError: cannot assign to field 'quantity'
```

An order is a record of something that happened. It shouldn't change after
the fact, and freezing it means no function can quietly change it behind
your back (lesson 006's "don't mutate your inputs", enforced for free).
Freeze data objects by default; unfreeze the ones that genuinely need to
change.

Some objects *do* need to change. A customer's tab grows as they order:

```python
@dataclass
class Tab:
    name: str
    orders: list = field(default_factory=list)

    def add(self, order):
        self.orders.append(order)
```

`field(default_factory=list)` means "call `list()` to make a *fresh*
empty list for each new `Tab`". The obvious-looking `orders: list = []`
would give every tab the *same* list, which is lesson 006's mutable
default bug, and dataclasses refuse it outright with a `ValueError`.
Good. That's one bug you can't write.

## 6. A tiny model with `fit()` and `predict()`

Here's why we're learning this now, in a course about ML:

```python
class AverageCupsModel:
    def __init__(self):
        self.average_by_drink = {}
        self.overall_average = None

    def fit(self, orders):
        ...                              # learn average cups per order, per drink
        return self

    def predict(self, drink):
        if self.overall_average is None:
            raise RuntimeError("call fit() before predict()")
        return self.average_by_drink.get(drink, self.overall_average)

model = AverageCupsModel()
model.fit(orders)
model.predict("latte")      # 1.6
model.predict("mocha")      # 1.53, the overall average, for a drink it's never seen
```

As a model it's barely a model. As a *shape*, it's exactly what you'll
use for the rest of this course. Every scikit-learn model in Phase 2 is
an object you create, then `.fit(data)` so it learns, then
`.predict(new_data)` so it uses what it learned. What it learned is
stored on `self` between the two calls. That's lesson 001's `data ->
pattern -> prediction`, with the pattern living inside an object.

This one is a plain class, not a dataclass, because its job is behaviour
rather than holding a fixed set of fields. Rough rule: **mostly data?
dataclass. Mostly doing things, with state that changes? Plain class.**

## 7. Putting it together

```python
def load_orders(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [Order.from_row(raw) for raw in csv.DictReader(f)]

orders = load_orders(ORDERS_CSV)
total = sum(order.total_cents() for order in orders)
cups = sum(order.quantity for order in orders)
```

```
    latte          £29.90
    cappuccino     £20.00
    flat white     £15.60
    espresso        £8.80
    tea             £5.40
    total          £79.70   (23 cups, 15 orders)
```

Same week, same £79.70. But `order.total_cents()` and `order.quantity`
read like English, a typo in a field name is an immediate error, a bad
row can't become an `Order`, and the arithmetic lives in exactly one
place.

## What you can do now

- Explain what a class, an instance, an attribute, a method and `self`
  are, in plain words.
- Write a `@dataclass` with typed fields and defaults, and say what it
  generates for you (`__init__`, `repr`, `==`).
- Add methods that use `self`, and a `@classmethod` that builds an
  instance from raw data.
- Validate in `__post_init__`, so invalid objects can't exist.
- Use `frozen=True` for records that shouldn't change, and
  `field(default_factory=list)` for a mutable default.
- Recognise the `fit()` / `predict()` shape, and why the model is an
  object.

## What to do now

1. Run `lesson.py`. In section 2, compare the two printed orders; that
   difference alone is worth the decorator.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   builds a menu item class and a better model.
3. That's Week 2, Day 4. Lessons 011 (iterators, generators and lazy data)
   and 012 (the Phase 0 mini-project) arrive tomorrow. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

You now have the two tools that keep a growing program tidy: modules
this morning, to put code in the right *file*, and classes this afternoon,
to put data and behaviour in the right *place*.
