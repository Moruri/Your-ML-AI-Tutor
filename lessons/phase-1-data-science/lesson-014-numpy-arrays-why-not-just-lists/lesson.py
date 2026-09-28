"""
Lesson 014 - NumPy arrays: why not just lists?

Meet the NumPy array: see where it behaves differently from a list, make
arrays a handful of ways, read their shape and dtype, understand why every
element has the same type (and what happens when that goes wrong), do maths on
whole arrays at once, time it against a plain Python loop, and redo the
coffee shop's weekly totals in a few lines.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

import csv
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders.csv"          # the clean week from Phase 0: £79.70, 23 cups


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    text = repr(value) if not isinstance(value, np.ndarray) else np.array2string(value, separator=", ")
    text = text.replace("\n", "\n" + " " * 52)
    print(f"  {label:<46} -> {text}")


# ---------------------------------------------------------------------------
# 1. Lists and arrays look alike, and behave very differently
# ---------------------------------------------------------------------------


def section_lists_vs_arrays():
    heading("1. Lists and arrays look alike, and behave very differently")

    prices = [3.80, 2.20, 4.20]
    show("prices * 2           (a list)", prices * 2)
    show("prices + [0.10] * 3  (a list)", prices + [0.10] * 3)
    show("[p * 2 for p in prices]  (what we meant)", [round(p * 2, 2) for p in prices])

    arr = np.array(prices)
    show("np.array(prices)", arr)
    show("arr * 2", arr * 2)
    show("arr + 0.10", arr + 0.10)
    print("  To a list, * and + mean 'repeat' and 'join'. To an array they mean arithmetic,")
    print("  on every element at once. No loop, no comprehension.")


# ---------------------------------------------------------------------------
# 2. Making arrays
# ---------------------------------------------------------------------------


def section_making():
    heading("2. Making arrays")
    show("np.array([2, 1, 1, 1])", np.array([2, 1, 1, 1]))
    show("np.arange(7, 17)  (opening hours)", np.arange(7, 17))
    show("np.arange(0, 1, 0.25)", np.arange(0, 1, 0.25))
    show("np.linspace(0, 1, 5)  (5 points, ends included)", np.linspace(0, 1, 5))
    show("np.zeros(3)", np.zeros(3))
    show("np.ones(3, dtype=int)", np.ones(3, dtype=int))
    show("np.full(5, 380)", np.full(5, 380))


# ---------------------------------------------------------------------------
# 3. Shape, size and dimensions
# ---------------------------------------------------------------------------


def section_shape():
    heading("3. Shape, size and dimensions")

    cups = np.array([4, 2, 3, 1, 2])
    show("cups", cups)
    show("cups.shape", cups.shape)
    show("cups.ndim, cups.size", (cups.ndim, cups.size))

    print()
    # Rows are days (Mon-Fri), columns are drinks (latte, espresso, cappuccino, flat white, tea).
    week = np.array([
        [2, 1, 1, 0, 0],
        [1, 2, 0, 3, 0],
        [1, 0, 2, 0, 1],
        [2, 1, 0, 1, 0],
        [2, 0, 2, 0, 1],
    ])
    show("week  (5 days x 5 drinks)", week)
    show("week.shape", week.shape)
    show("week.ndim, week.size", (week.ndim, week.size))
    show("week.sum()", int(week.sum()))
    show("np.arange(12).reshape(3, 4)", np.arange(12).reshape(3, 4))
    print("  shape is (rows, columns). A list of lists can be ragged; an array can't.")
    try:
        np.array([[1, 2], [3]], dtype=int)
    except ValueError as err:
        print(f"    np.array([[1, 2], [3]], dtype=int) -> ValueError: {str(err)[:55]}...")
    return week


# ---------------------------------------------------------------------------
# 4. dtype: one type for every element
# ---------------------------------------------------------------------------


def section_dtype():
    heading("4. dtype: one type for every element")

    show("np.array([1, 2, 3]).dtype", str(np.array([1, 2, 3]).dtype))
    show("np.array([1.5, 2, 3]).dtype", str(np.array([1.5, 2, 3]).dtype))
    show("np.array([1, 2, 3.0])", np.array([1, 2, 3.0]))
    show("np.array([1, 'two', 3])", np.array([1, "two", 3]))
    print("  One dtype for the whole array. Mix ints and floats: everything becomes float.")
    print("  Mix in a string: everything becomes a string, and maths stops working.")

    print()
    prices = np.array(["3.80", "2.20", "4.20"])          # what csv hands you: text
    show("prices_text.astype(float)", prices.astype(float))
    show("np.array([3.7, 2.2]).astype(int)  (truncates!)", np.array([3.7, 2.2]).astype(int))

    print()
    small = np.array([100, 120], dtype=np.int8)           # int8 holds -128..127
    show("np.array([100, 120], dtype=int8) + 10", small + np.int8(10))
    print("  Fixed-size numbers can overflow and wrap around, silently. Python ints never do.")
    print("  The default int64 goes past nine quintillion, so you'll rarely meet this, but")
    print("  now you'll recognise it if a count ever goes negative for no reason.")
    show("np.array([0.1, 0.2]).sum() == 0.3", bool(np.array([0.1, 0.2]).sum() == 0.3))
    show("np.isclose(np.array([0.1, 0.2]).sum(), 0.3)", bool(np.isclose(np.array([0.1, 0.2]).sum(), 0.3)))
    print("  And floats are still floats (lesson 003). Compare with np.isclose, not ==.")


# ---------------------------------------------------------------------------
# 5. Vectorised maths and comparisons
# ---------------------------------------------------------------------------


def section_vectorised():
    heading("5. Vectorised maths and comparisons")

    price_cents = np.array([380, 220, 420, 430, 390])
    quantity = np.array([2, 1, 1, 1, 3])
    show("price_cents * quantity", price_cents * quantity)
    show("(price_cents * quantity).sum()", int((price_cents * quantity).sum()))
    show("price_cents / 100", price_cents / 100)
    show("np.round(price_cents * 1.05)  (a 5% rise)", np.round(price_cents * 1.05))
    print("  Two arrays of the same shape combine element by element: the first with the")
    print("  first, the second with the second, and so on.")

    print()
    show("quantity > 1", quantity > 1)
    show("(quantity > 1).sum()  (True counts as 1)", int((quantity > 1).sum()))
    show("price_cents.mean(), .min(), .max()",
         (float(price_cents.mean()), int(price_cents.min()), int(price_cents.max())))
    try:
        np.array([1, 2, 3]) + np.array([1, 2])
    except ValueError as err:
        print(f"  np.array([1, 2, 3]) + np.array([1, 2])  -> ValueError: {err}")
    print("  Shapes that don't line up are refused, loudly. Lesson 015 covers the rules.")


# ---------------------------------------------------------------------------
# 6. Why it's faster
# ---------------------------------------------------------------------------


def best_of(func, repeat=5):
    """The fastest of a few runs, in milliseconds. The fastest run has the least noise."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times) * 1000


def section_speed():
    heading("6. Why it's faster")

    n = 1_000_000
    prices_list = [380 + (i % 5) * 10 for i in range(n)]
    qty_list = [1 + i % 3 for i in range(n)]
    prices_arr = np.array(prices_list)
    qty_arr = np.array(qty_list)

    def with_lists():
        return sum(p * q for p, q in zip(prices_list, qty_list))

    def with_arrays():
        return int((prices_arr * qty_arr).sum())

    assert with_lists() == with_arrays()
    list_ms, array_ms = best_of(with_lists), best_of(with_arrays)
    print(f"  Revenue from {n:,} orders, plain Python: {list_ms:7.1f} ms")
    print(f"  Revenue from {n:,} orders, NumPy:        {array_ms:7.1f} ms")
    print(f"  About {list_ms / array_ms:.0f}x faster, same answer.")
    print("  An array is one block of raw numbers, all the same type, side by side in memory.")
    print("  The loop runs in compiled C, not in Python, one element after another.")
    show("prices_arr.nbytes  (8 bytes per int64)", f"{prices_arr.nbytes:,} bytes")


# ---------------------------------------------------------------------------
# 7. Putting it together: the week, as arrays
# ---------------------------------------------------------------------------


def load_columns(path):
    """Read the clean CSV into a dict of NumPy arrays, one per column."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "drink": np.array([row["drink"] for row in rows]),
        "price_cents": np.round(np.array([row["price"] for row in rows]).astype(float) * 100).astype(int),
        "quantity": np.array([row["quantity"] for row in rows]).astype(int),
    }


def section_together():
    heading("7. Putting it together: the week, as arrays")

    cols = load_columns(ORDERS_CSV)
    revenue = cols["price_cents"] * cols["quantity"]
    show("cols['quantity']", cols["quantity"])
    show("revenue per order (cents)", revenue)
    total, cups = int(revenue.sum()), int(cols["quantity"].sum())
    print(f"  total £{total / 100:.2f} from {cups} cups; average order £{revenue.mean() / 100:.2f}")

    print()
    for drink in np.unique(cols["drink"]):
        is_drink = cols["drink"] == drink                     # an array of True/False
        money = f"£{revenue[is_drink].sum() / 100:.2f}"
        print(f"    {drink:<12} {money:>7}   ({int(is_drink.sum())} orders)")

    assert total == 7970 and cups == 23, "same week, same numbers"
    print("  £79.70 and 23 cups, one more time. The loop over drinks is the only loop left,")
    print("  and lesson 015 (masks) and lesson 019 (groupby) will take care of that one too.")


def main():
    print("=" * 66)
    print("  Lesson 014: NumPy arrays: why not just lists?")
    print("=" * 66)
    show("np.__version__", np.__version__)
    section_lists_vs_arrays()
    section_making()
    section_shape()
    section_dtype()
    section_vectorised()
    section_speed()
    section_together()
    print()
    print("One type, one block of memory, maths on the whole thing at once. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
