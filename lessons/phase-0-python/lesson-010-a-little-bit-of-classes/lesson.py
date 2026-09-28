"""
Lesson 010 - A little bit of classes

Bundle related data and behaviour together. We start with the plain class
Python has always had, see how much typing @dataclass saves, give an Order
some methods, build Orders safely from CSV rows, freeze them so they can't be
changed by accident, and finish with a tiny "model" class with fit() and
predict() - the exact shape every scikit-learn model has in Phase 2.

Run it with:

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

import csv
from dataclasses import FrozenInstanceError, asdict, dataclass, field
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders.csv"          # the clean week from lessons 005-009

MENU = {"espresso": 220, "tea": 250, "cappuccino": 370, "latte": 380, "flat white": 390}
SIZES = ("small", "medium", "large")


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def pounds(cents):
    return f"£{cents / 100:.2f}"


# ---------------------------------------------------------------------------
# 1. Where dicts start to creak
# ---------------------------------------------------------------------------


def section_dicts_creak():
    heading("1. Where dicts start to creak")

    order = {"date": "2026-09-07", "drink": "latte", "size": "medium", "price_cents": 380, "quantity": 2}
    show("order['price_cents'] * order['quantity']", order["price_cents"] * order["quantity"])

    order["quantiy"] = 3                      # typo: a NEW key, silently
    show("after order['quantiy'] = 3, quantity is", order["quantity"])
    show("order.get('qty')  (wrong name, no error)", order.get("qty"))
    print("  A dict will hold any key you like, including typos. Nothing says which keys")
    print("  an order SHOULD have, and 'cost of this order' lives in some other function.")


# ---------------------------------------------------------------------------
# 2. A class, the long way, then the dataclass way
# ---------------------------------------------------------------------------


class OrderLongHand:
    """What we'd write without dataclasses. Works, but look at all that repetition."""

    def __init__(self, date, drink, size, price_cents, quantity):
        self.date = date
        self.drink = drink
        self.size = size
        self.price_cents = price_cents
        self.quantity = quantity


@dataclass
class OrderDraft:
    """The same thing. @dataclass writes __init__, a readable repr, and == for us."""

    date: str
    drink: str
    size: str
    price_cents: int
    quantity: int


def section_dataclass():
    heading("2. A class, the long way, then the dataclass way")

    long_hand = OrderLongHand("2026-09-07", "latte", "medium", 380, 2)
    show("long_hand.quantity", long_hand.quantity)
    print(f"  print(long_hand)  -> {long_hand}")
    print("  It works, but printing it tells you nothing, and two identical ones aren't ==.")
    show("OrderLongHand(...) == OrderLongHand(...)",
         OrderLongHand("2026-09-07", "latte", "medium", 380, 2) == long_hand)

    print()
    draft = OrderDraft("2026-09-07", "latte", "medium", 380, 2)
    print(f"  print(draft)      -> {draft}")
    show("draft == OrderDraft(same values)", draft == OrderDraft("2026-09-07", "latte", "medium", 380, 2))
    show("draft.quantity", draft.quantity)
    show("asdict(draft)", asdict(draft))

    print()
    print("  And the typo from section 1?")
    try:
        OrderDraft("2026-09-07", "latte", "medium", 380, quantiy=2)
    except TypeError as err:
        print(f"    OrderDraft(..., quantiy=2)  -> TypeError: {err}")
    print("  A class knows its own fields. Misspell one when you build it and Python says so.")


# ---------------------------------------------------------------------------
# 3. Methods: behaviour that lives with the data
# ---------------------------------------------------------------------------


@dataclass
class OrderWithMethods:
    date: str
    drink: str
    size: str
    price_cents: int
    quantity: int

    def total_cents(self):
        """What this order cost, in cents."""
        return self.price_cents * self.quantity

    def describe(self):
        """One human-readable line."""
        return f"{self.quantity} x {self.size} {self.drink} on {self.date} = {pounds(self.total_cents())}"


def section_methods():
    heading("3. Methods: behaviour that lives with the data")

    order = OrderWithMethods("2026-09-08", "flat white", "medium", 390, 3)
    show("order.total_cents()", order.total_cents())
    show("order.describe()", order.describe())
    print("  `self` is the order the method was called on. order.total_cents() is really")
    print("  OrderWithMethods.total_cents(order): Python passes the object in for you.")
    show("OrderWithMethods.total_cents(order)", OrderWithMethods.total_cents(order))


# ---------------------------------------------------------------------------
# 4. Building objects safely: from_row and __post_init__
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Order:
    """One validated order. Frozen: once built, it can't be changed by accident."""

    date: date
    drink: str
    size: str
    price_cents: int
    quantity: int = 1

    def __post_init__(self):
        # Runs straight after the generated __init__. The last line of defence.
        if self.drink not in MENU:
            raise ValueError(f"drink {self.drink!r} is not on the menu")
        if self.size not in SIZES:
            raise ValueError(f"size {self.size!r} is not one of {', '.join(SIZES)}")
        if self.price_cents <= 0:
            raise ValueError(f"price {self.price_cents} must be more than zero")
        if self.quantity < 1:
            raise ValueError(f"quantity {self.quantity} must be at least 1")

    @classmethod
    def from_row(cls, raw):
        """A raw CSV row (all strings) -> an Order. Conversion here, rules in __post_init__."""
        row = {key: (value or "").strip() for key, value in raw.items() if key is not None}
        try:
            return cls(
                date=date.fromisoformat(row["date"]),
                drink=row["drink"].lower(),
                size=row["size"].lower(),
                price_cents=round(float(row["price"].replace("£", "")) * 100),
                quantity=int(row["quantity"]),
            )
        except KeyError as err:
            raise ValueError(f"missing column {err}") from err

    def total_cents(self):
        return self.price_cents * self.quantity


def section_building():
    heading("4. Building objects safely: from_row and __post_init__")

    raw = {"date": "2026-09-07", "drink": "Latte", "size": "medium", "price": "3.80", "quantity": "2"}
    order = Order.from_row(raw)
    print(f"  Order.from_row(raw) -> {order}")
    show("order.date.strftime('%A')", order.date.strftime("%A"))
    show("type(order.date).__name__  (a real date)", type(order.date).__name__)

    print()
    print("  Bad data is refused at the door, whichever way the Order is built:")
    attempts = [
        ("Order.from_row(quantity '-1')", lambda: Order.from_row({**raw, "quantity": "-1"})),
        ("Order.from_row(drink 'mocha')", lambda: Order.from_row({**raw, "drink": "mocha"})),
        ("Order.from_row(quantity 'two')", lambda: Order.from_row({**raw, "quantity": "two"})),
        ("Order(date(...), 'tea', 'venti', 250)", lambda: Order(date(2026, 9, 7), "tea", "venti", 250)),
    ]
    for label, attempt in attempts:
        try:
            attempt()
        except ValueError as err:
            print(f"    {label:<40} ValueError: {err}")

    print()
    show("Order(date(2026, 9, 9), 'tea', 'medium', 250)", Order(date(2026, 9, 9), "tea", "medium", 250))
    print("  quantity has a default of 1, so you can leave it out.")


# ---------------------------------------------------------------------------
# 5. Frozen, and the mutable-default trap
# ---------------------------------------------------------------------------


@dataclass
class Tab:
    """A customer's running tab. Each Tab needs its OWN list of orders."""

    name: str
    orders: list = field(default_factory=list)

    def add(self, order):
        self.orders.append(order)

    def total_cents(self):
        return sum(order.total_cents() for order in self.orders)


def section_frozen_and_defaults():
    heading("5. Frozen, and the mutable-default trap")

    order = Order(date(2026, 9, 7), "latte", "medium", 380, 2)
    try:
        order.quantity = 20
    except FrozenInstanceError as err:
        print(f"  order.quantity = 20  -> FrozenInstanceError: {err}")
    print("  A frozen order is a fact about what happened. Facts shouldn't change.")

    print()
    print("  Tab has a list field. field(default_factory=list) gives each Tab a fresh list:")
    sam, priya = Tab("Sam"), Tab("Priya")
    sam.add(order)
    show("sam.total_cents()", sam.total_cents())
    show("len(priya.orders)  (should be 0)", len(priya.orders))
    print("  Writing `orders: list = []` is refused by dataclasses with a ValueError,")
    print("  because one shared list for every Tab is lesson 006's mutable-default bug.")
    try:
        @dataclass
        class BadTab:
            name: str
            orders: list = []   # noqa: this is the mistake on purpose
    except ValueError as err:
        print(f"    -> ValueError: {str(err)[:60]}...")


# ---------------------------------------------------------------------------
# 6. A tiny model with fit() and predict()
# ---------------------------------------------------------------------------


class AverageCupsModel:
    """
    Predicts cups sold for a drink as the average quantity seen per order.

    Silly as a model, perfect as a shape: every scikit-learn model you'll meet
    in Phase 2 is an object with .fit(data) that learns, and .predict(x) that
    uses what it learned.
    """

    def __init__(self):
        self.average_by_drink = {}     # learned in fit(); empty until then
        self.overall_average = None

    def fit(self, orders):
        totals, counts = {}, {}
        for order in orders:
            totals[order.drink] = totals.get(order.drink, 0) + order.quantity
            counts[order.drink] = counts.get(order.drink, 0) + 1
        self.average_by_drink = {drink: totals[drink] / counts[drink] for drink in totals}
        self.overall_average = sum(totals.values()) / sum(counts.values())
        return self                    # so you can write model = AverageCupsModel().fit(orders)

    def predict(self, drink):
        if self.overall_average is None:
            raise RuntimeError("call fit() before predict()")
        return self.average_by_drink.get(drink, self.overall_average)


def section_model():
    heading("6. A tiny model with fit() and predict()")

    model = AverageCupsModel()
    try:
        model.predict("latte")
    except RuntimeError as err:
        print(f"  model.predict('latte') before fit -> RuntimeError: {err}")

    orders = load_orders(ORDERS_CSV)
    model.fit(orders)
    for drink in ["latte", "espresso", "tea", "mocha"]:
        show(f"model.predict({drink!r})", round(model.predict(drink), 2))
    print("  data -> pattern -> prediction, lesson 001's idea, now wearing a class.")
    print("  The object remembers what it learned between fit() and predict().")


# ---------------------------------------------------------------------------
# 7. Putting it together
# ---------------------------------------------------------------------------


def load_orders(path):
    """Read a clean till export into a list of Order objects."""
    with open(path, newline="", encoding="utf-8") as f:
        return [Order.from_row(raw) for raw in csv.DictReader(f)]


def section_together():
    heading("7. Putting it together")

    orders = load_orders(ORDERS_CSV)
    show("len(orders)", len(orders))
    print(f"  orders[0] -> {orders[0]}")

    revenue = {}
    for order in orders:
        revenue[order.drink] = revenue.get(order.drink, 0) + order.total_cents()
    for drink, cents in sorted(revenue.items(), key=lambda item: item[1], reverse=True):
        print(f"    {drink:<12} {pounds(cents):>8}")
    total = sum(order.total_cents() for order in orders)
    cups = sum(order.quantity for order in orders)
    print(f"    {'total':<12} {pounds(total):>8}   ({cups} cups, {len(orders)} orders)")

    assert total == 7970, "the clean week should total £79.70"
    assert cups == 23, "the clean week should be 23 cups"
    print("  order.total_cents() instead of row['price_cents'] * row['quantity']:")
    print("  the arithmetic lives in one place, with the data it belongs to.")


def main():
    print("=" * 66)
    print("  Lesson 010: A little bit of classes")
    print("=" * 66)
    section_dicts_creak()
    section_dataclass()
    section_methods()
    section_building()
    section_frozen_and_defaults()
    section_model()
    section_together()
    print()
    print("Data plus the behaviour that belongs to it, in one named place. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
