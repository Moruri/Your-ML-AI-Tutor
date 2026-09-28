"""
Lesson 015 - NumPy indexing, broadcasting and reductions

Pick out exactly the numbers you want from an array (by position, by slice, by
a True/False mask, by a list of positions), learn the one rule that decides
what happens when arrays of different shapes meet (broadcasting), and add
things up along rows or columns with `axis`. Finishes by building the coffee
shop's week as a days-by-drinks table and answering five questions with no
loops at all.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

import csv
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders.csv"          # the clean week: £79.70, 23 cups

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DRINKS = ["latte", "espresso", "cappuccino", "flat white", "tea"]

# Cups sold: a row per day, a column per drink (the same table as lesson 014).
WEEK = np.array([
    [2, 1, 1, 0, 0],
    [1, 2, 0, 3, 0],
    [1, 0, 2, 0, 1],
    [2, 1, 0, 1, 0],
    [2, 0, 2, 0, 1],
])
MEDIUM_CENTS = np.array([380, 220, 370, 390, 250])   # medium price of each drink, same order


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    if isinstance(value, np.ndarray):
        text = np.array2string(value, separator=", ").replace("\n", "\n" + " " * 52)
    else:
        text = repr(value)
    print(f"  {label:<46} -> {text}")


# ---------------------------------------------------------------------------
# 1. Indexing and slicing, and the view surprise
# ---------------------------------------------------------------------------


def section_indexing():
    heading("1. Indexing and slicing, and the view surprise")

    cups = np.array([4, 2, 3, 1, 2, 5, 0])
    show("cups", cups)
    show("cups[0], cups[-1]", (int(cups[0]), int(cups[-1])))
    show("cups[2:5]", cups[2:5])
    show("cups[::2]", cups[::2])
    show("cups[::-1]", cups[::-1])
    print("  Same rules as list slicing (lesson 004): start, stop (excluded), step.")

    print()
    middle = cups[2:5]
    middle[0] = 99
    show("middle = cups[2:5]; middle[0] = 99; cups", cups)
    print("  A slice of an array is a VIEW of the same memory, not a copy (lists copy).")
    print("  Change the slice and you change the original. Use .copy() when you mean it:")
    cups[2] = 3
    safe = cups[2:5].copy()
    safe[0] = 99
    show("safe = cups[2:5].copy(); safe[0] = 99; cups", cups)


# ---------------------------------------------------------------------------
# 2. Rows and columns
# ---------------------------------------------------------------------------


def section_2d():
    heading("2. Rows and columns")

    show("WEEK", WEEK)
    show("WEEK[1, 3]  (Tuesday, flat white)", int(WEEK[1, 3]))
    show("WEEK[0]     (Monday: a whole row)", WEEK[0])
    show("WEEK[:, 0]  (latte: a whole column)", WEEK[:, 0])
    show("WEEK[:2, :3]  (Mon-Tue, first three drinks)", WEEK[:2, :3])
    show("WEEK[-1, -1]  (Friday, tea)", int(WEEK[-1, -1]))
    print("  [row, column]. A colon on its own means 'all of them'.")


# ---------------------------------------------------------------------------
# 3. Boolean masks
# ---------------------------------------------------------------------------


def section_masks():
    heading("3. Boolean masks")

    quantity = np.array([2, 1, 1, 1, 3, 2, 2, 1, 1, 1, 2, 1, 2, 2, 1])
    price = np.array([380, 220, 420, 430, 390, 220, 370, 380, 250, 390, 330, 220, 420, 380, 290])
    big = quantity > 1
    show("big = quantity > 1", big.astype(int))
    show("quantity[big]", quantity[big])
    show("price[big]  (a mask works on ANY array this long)", price[big])
    show("price[(quantity > 1) & (price > 380)]", price[(quantity > 1) & (price > 380)])
    show("price[(price < 250) | (price > 400)]", price[(price < 250) | (price > 400)])
    show("price[~big]  (~ means 'not')", price[~big])
    print("  Combine masks with & (and), | (or), ~ (not). Brackets round each comparison")
    print("  are required: `quantity > 1 & price > 380` means something else entirely.")

    print()
    show("np.where(quantity > 1, 'multi', 'single')[:5]", np.where(quantity > 1, "multi", "single")[:5])
    capped = price.copy()
    capped[capped > 400] = 400
    show("capped[capped > 400] = 400", capped)
    print("  np.where picks between two values per element. Assigning through a mask")
    print("  changes only the selected elements.")


# ---------------------------------------------------------------------------
# 4. Choosing by position: integer arrays and argsort
# ---------------------------------------------------------------------------


def section_fancy():
    heading("4. Choosing by position: integer arrays and argsort")

    drinks = np.array(DRINKS)
    totals = WEEK.sum(axis=0)                        # cups per drink (section 6 explains axis)
    show("drinks[[0, 2, 4]]", drinks[[0, 2, 4]])
    show("totals  (cups per drink)", totals)
    order = np.argsort(totals)[::-1]                 # positions, biggest first
    show("np.argsort(totals)[::-1]  (positions)", order)
    show("drinks[order]", drinks[order])
    show("totals[order]", totals[order])
    print("  argsort gives the positions that WOULD sort the array. Use them to put")
    print("  several arrays in the same order, so names and numbers stay lined up.")


# ---------------------------------------------------------------------------
# 5. Broadcasting
# ---------------------------------------------------------------------------


def section_broadcasting():
    heading("5. Broadcasting")

    show("WEEK.shape, MEDIUM_CENTS.shape", (WEEK.shape, MEDIUM_CENTS.shape))
    revenue = WEEK * MEDIUM_CENTS
    show("WEEK * MEDIUM_CENTS  (each column x its price)", revenue)
    print("  (5, 5) and (5,) line up from the RIGHT: 5 matches 5, and the missing")
    print("  dimension is stretched, so every row is multiplied by the same prices.")

    print()
    busy = np.array([1.0, 1.0, 1.0, 1.0, 1.5])                     # a per-DAY factor
    try:
        WEEK * busy
        print("  WEEK * busy works... but multiplies COLUMNS (drinks), not days!")
    except ValueError as err:
        print(f"  {err}")
    per_day = busy.reshape(5, 1)                                    # or busy[:, np.newaxis]
    show("busy.reshape(5, 1).shape", per_day.shape)
    show("WEEK * busy.reshape(5, 1)  (Friday x 1.5)", WEEK * per_day)
    print("  (5, 5) and (5, 1): the 1 stretches across the columns, so each ROW gets")
    print("  its own factor. A (5,) array always lines up with columns; to line up with")
    print("  rows, make it a column: shape (5, 1).")

    print()
    try:
        WEEK * np.array([1, 2, 3])
    except ValueError as err:
        print(f"  WEEK * np.array([1, 2, 3]) -> ValueError: {err}")
    print("  (5, 5) and (3,): 5 against 3, neither is 1, so NumPy refuses. Good.")


# ---------------------------------------------------------------------------
# 6. Reductions along an axis
# ---------------------------------------------------------------------------


def section_reductions():
    heading("6. Reductions along an axis")

    show("WEEK.sum()           (everything)", int(WEEK.sum()))
    show("WEEK.sum(axis=0)     (down the rows: per drink)", WEEK.sum(axis=0))
    show("WEEK.sum(axis=1)     (across columns: per day)", WEEK.sum(axis=1))
    show("WEEK.max(axis=1)", WEEK.max(axis=1))
    show("WEEK.argmax(axis=1)  (which drink topped each day)", WEEK.argmax(axis=1))
    show("WEEK.mean(axis=0).round(2)", WEEK.mean(axis=0).round(2))
    print("  axis=0 collapses the rows (the answer has one value per column).")
    print("  axis=1 collapses the columns (one value per row). The axis you name disappears.")

    print()
    day_totals = WEEK.sum(axis=1, keepdims=True)
    show("WEEK.sum(axis=1, keepdims=True).shape", day_totals.shape)
    share = WEEK / day_totals
    show("(WEEK / day_totals).round(2)  (share of each day)", share.round(2))
    show("each row of the share sums to", share.sum(axis=1).round(6))
    print("  keepdims=True leaves the answer as (5, 1), ready to broadcast back over rows.")


# ---------------------------------------------------------------------------
# 7. Putting it together: the week as a table, no loops
# ---------------------------------------------------------------------------


def load_week_revenue(path):
    """Return a (days, drinks) table of revenue in cents, built from the CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    day_idx = np.array([date.fromisoformat(r["date"]).weekday() for r in rows])   # Mon=0 .. Fri=4
    drink_idx = np.array([DRINKS.index(r["drink"]) for r in rows])
    cents = np.round(np.array([r["price"] for r in rows]).astype(float) * 100).astype(int)
    qty = np.array([r["quantity"] for r in rows]).astype(int)
    table = np.zeros((len(DAYS), len(DRINKS)), dtype=int)
    np.add.at(table, (day_idx, drink_idx), cents * qty)       # add each order into its cell
    return table


def section_together():
    heading("7. Putting it together: the week as a table, no loops")

    table = load_week_revenue(ORDERS_CSV)
    print("  Revenue in pence, a row per day, a column per drink:")
    print("          " + "".join(f"{d[:6]:>8}" for d in DRINKS) + "   total")
    for day, row in zip(DAYS, table):
        print(f"    {day:<6}" + "".join(f"{v:>8}" for v in row) + f"{row.sum():>8}")

    drinks = np.array(DRINKS)
    per_day = table.sum(axis=1)
    per_drink = table.sum(axis=0)
    print()
    print(f"  1. Best day:                {DAYS[per_day.argmax()]} (£{per_day.max() / 100:.2f})")
    print(f"  2. Best drink:              {drinks[per_drink.argmax()]} (£{per_drink.max() / 100:.2f})")
    print(f"  3. Top drink each day:      {', '.join(drinks[table.argmax(axis=1)])}")
    latte_share = table[:, 0] / per_day
    print(f"  4. Latte's share each day:  {', '.join(f'{s:.0%}' for s in latte_share)}")
    no_tea = DAYS[int(np.flatnonzero(table[:, 4] == 0)[0])]
    print(f"  5. Days with no tea sold:   {int((table[:, 4] == 0).sum())} (first: {no_tea})")

    assert table.sum() == 7970, "the whole table is the week's £79.70"
    assert per_drink[0] == 2990, "latte took £29.90"
    print("  Every cell adds up to £79.70, and latte's column to £29.90. Masks, broadcasting")
    print("  and reductions: a whole report without a single for loop over the numbers.")


def main():
    print("=" * 66)
    print("  Lesson 015: NumPy indexing, broadcasting and reductions")
    print("=" * 66)
    section_indexing()
    section_2d()
    section_masks()
    section_fancy()
    section_broadcasting()
    section_reductions()
    section_together()
    print()
    print("Select with masks, line shapes up from the right, name the axis to collapse. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
