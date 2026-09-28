"""
coffee_tools - the coffee shop helpers we've built since lesson 005, in one place.

This file is a MODULE: other Python files can `import coffee_tools` and use
everything defined here. It's also a SCRIPT: run it directly and the block at
the bottom does a quick self-check.

    python coffee_tools.py

Nothing in this file prints anything when it's imported. That's on purpose.
"""

import csv
from datetime import date

# The menu board, medium size, in cents (lesson 005).
MENU = {
    "espresso": 220,
    "tea": 250,
    "cappuccino": 370,
    "latte": 380,
    "flat white": 390,
}

RAW_FIELDS = ["date", "drink", "size", "price", "quantity"]
SIZES = ("small", "medium", "large")


def pounds(cents):
    """Format whole cents as pounds for printing. Maths in cents, display in pounds."""
    return f"£{cents / 100:.2f}"


def validate_row(raw):
    """One raw CSV row -> one clean dict. Raises ValueError naming the column and the value (lesson 008)."""
    row = {key: (value or "").strip() for key, value in raw.items() if key is not None}

    missing = [field for field in RAW_FIELDS if not row.get(field)]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")

    try:
        when = date.fromisoformat(row["date"])
    except ValueError as err:
        raise ValueError(f"date {row['date']!r} is not YYYY-MM-DD") from err

    drink = row["drink"].lower()
    if drink not in MENU:
        raise ValueError(f"drink {row['drink']!r} is not on the menu")

    size = row["size"].lower()
    if size not in SIZES:
        raise ValueError(f"size {row['size']!r} is not one of {', '.join(SIZES)}")

    try:
        price_cents = round(float(row["price"].replace("£", "")) * 100)
    except ValueError as err:
        raise ValueError(f"price {row['price']!r} is not a number") from err
    if price_cents <= 0:
        raise ValueError(f"price {row['price']!r} must be more than zero")

    try:
        quantity = int(row["quantity"])
    except ValueError as err:
        raise ValueError(f"quantity {row['quantity']!r} is not a whole number") from err
    if quantity < 1:
        raise ValueError(f"quantity {quantity} must be at least 1")

    return {"date": when.isoformat(), "drink": drink, "size": size,
            "price_cents": price_cents, "quantity": quantity}


def load_orders_carefully(path):
    """Read a till export. Returns (orders, rejects). A missing file is NOT caught."""
    orders, rejects = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for line_number, raw in enumerate(csv.DictReader(f), start=2):
            try:
                orders.append(validate_row(raw))
            except ValueError as err:
                rejects.append({"line": line_number, "reason": str(err)})
    return orders, rejects


def revenue_by_drink(orders):
    """Total cents per drink, as a plain dict. Does not change orders."""
    totals = {}
    for row in orders:
        totals[row["drink"]] = totals.get(row["drink"], 0) + row["price_cents"] * row["quantity"]
    return totals


if __name__ == "__main__":
    # Only runs for `python coffee_tools.py`, never on `import coffee_tools`.
    print(f"coffee_tools self-check (__name__ is {__name__!r})")
    assert pounds(7970) == "£79.70"
    sample = {"date": "2026-09-07", "drink": "Latte", "size": "medium", "price": "3.80", "quantity": "2"}
    assert validate_row(sample)["price_cents"] == 380
    assert revenue_by_drink([validate_row(sample)]) == {"latte": 760}
    print("  all good")
