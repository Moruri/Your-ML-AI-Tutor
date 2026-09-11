"""
Lesson 002 - Python warm-up for data people

The handful of Python pieces you'll use every day in data work, then a real
(tiny) CSV read with the standard library.

Run it with:

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.
"""

import csv
from pathlib import Path

# The CSV lives next to this script. Building the path this way means the
# script works no matter which folder you run it from.
CSV_PATH = Path(__file__).parent / "coffee_orders.csv"


def heading(title):
    print()
    print(title)
    print("-" * len(title))


# ---------------------------------------------------------------------------
# 1. Variables
# ---------------------------------------------------------------------------


def section_variables():
    heading("1. Variables: giving things names")

    drink = "latte"          # str
    price = 3.80             # float
    quantity = 2             # int
    total = price * quantity

    print(f"{quantity} x {drink} at {price:.2f} = {total:.2f}")

    # Data files hand you TEXT. "3.80" is not a number until you convert it.
    price_text = "3.80"
    print(f"price_text is a {type(price_text).__name__}: {price_text!r}")
    print(f"float(price_text) * 2 = {float(price_text) * 2}")


# ---------------------------------------------------------------------------
# 2. Lists
# ---------------------------------------------------------------------------


def section_lists():
    heading("2. Lists: many things, in order")

    prices = [3.80, 2.20, 4.20, 4.30]
    print(f"prices        = {prices}")
    print(f"len(prices)   = {len(prices)}")
    print(f"prices[0]     = {prices[0]}   (first; Python counts from 0)")
    print(f"prices[-1]    = {prices[-1]}   (last)")
    print(f"sum(prices)   = {sum(prices)}")
    print(f"max(prices)   = {max(prices)}")

    prices.append(2.50)
    print(f"after append  = {prices}")


# ---------------------------------------------------------------------------
# 3. Dicts
# ---------------------------------------------------------------------------


def section_dicts():
    heading("3. Dicts: looking things up by name")

    # One row of a CSV is a dict: column name -> value.
    order = {"drink": "latte", "size": "medium", "price": 3.80}
    print(f"order          = {order}")
    print(f"order['drink'] = {order['drink']}")
    order["quantity"] = 2
    print(f"after adding   = {order}")

    # The "running total per category" pattern. You'll use this constantly.
    sales = {}
    for drink, money in [("latte", 7.60), ("latte", 3.80), ("tea", 2.50)]:
        sales[drink] = sales.get(drink, 0) + money
    print(f"running totals = {sales}")
    # 11.399999999999999 is not a bug in your code. Floats can't store most
    # decimals exactly, so adding them leaves crumbs. Lesson 003 covers it.
    print("(that 11.3999... is how floats work; :.2f hides it when printing)")


# ---------------------------------------------------------------------------
# 4. Loops
# ---------------------------------------------------------------------------


def section_loops():
    heading("4. Loops: doing something to every item")

    prices = [3.80, 2.20, 4.20, 4.30]
    print("each price, formatted:")
    for p in prices:
        print(f"  {p:.2f}")

    print("filter while looping (prices over 4.00):")
    big = []
    for p in prices:
        if p > 4:
            big.append(p)
    print(f"  {big}")

    print("looping over a dict with .items():")
    sales = {"latte": 11.4, "tea": 2.5}
    for drink, total in sales.items():
        print(f"  {drink}: {total:.2f}")


# ---------------------------------------------------------------------------
# 5. Functions
# ---------------------------------------------------------------------------


def line_total(price, quantity):
    """Money for one line of an order."""
    return price * quantity


def section_functions():
    heading("5. Functions: naming a piece of logic")
    print(f"line_total(3.80, 2) = {line_total(3.80, 2):.2f}")
    print(f"line_total(2.20, 3) = {line_total(2.20, 3):.2f}")
    print("Same logic, called twice, named once.")


# ---------------------------------------------------------------------------
# 6. Reading a CSV with the standard library
# ---------------------------------------------------------------------------


def load_orders(path):
    """
    Read the CSV into a list of dicts and convert the numeric columns.

    Doing the conversion here, once, means nothing downstream ever has to
    remember that CSV values arrive as strings.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["price"] = float(row["price"])
        row["quantity"] = int(row["quantity"])
    return rows


def revenue_by(rows, column):
    """Total revenue grouped by any column: 'drink', 'date', 'size', ..."""
    totals = {}
    for row in rows:
        money = line_total(row["price"], row["quantity"])
        totals[row[column]] = totals.get(row[column], 0) + money
    return totals


def print_ranked(totals, label):
    """Print a dict of totals, biggest first."""
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    for key, total in ranked:
        print(f"  {key:<12} {total:>7.2f}")
    print(f"  {'total':<12} {sum(totals.values()):>7.2f}   ({label})")


def section_csv():
    heading("6. Reading a CSV with the standard library")

    orders = load_orders(CSV_PATH)
    print(f"Loaded {len(orders)} rows from {CSV_PATH.name}")
    print(f"First row: {orders[0]}")
    print("(price and quantity are numbers now, not strings)")

    print()
    print("Which drink made the most money this week?")
    print_ranked(revenue_by(orders, "drink"), "by drink")

    print()
    print("Same function, different column. Revenue per day:")
    print_ranked(revenue_by(orders, "date"), "by date")


def main():
    print("=" * 66)
    print("  Lesson 002: Python warm-up for data people")
    print("=" * 66)
    section_variables()
    section_lists()
    section_dicts()
    section_loops()
    section_functions()
    section_csv()
    print()
    print("That's the toolkit. Now do the exercises in exercises.md.")
    print()


if __name__ == "__main__":
    main()
