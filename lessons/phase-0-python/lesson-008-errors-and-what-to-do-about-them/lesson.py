"""
Lesson 008 - Errors, and what to do about them

Read a traceback from the bottom up, meet the eight or so errors you'll see
most, catch exactly the one you expect with try/except, validate rows at the
boundary so bad data can't get further in, raise errors with messages worth
reading, and learn the three honest things to do in an except block (none of
which is `pass`). Finishes with a loader that keeps the good rows, records
the bad ones, and never crashes for a reason it could have explained.

Run it with:

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script WRITES one file, output/rejected_rows.csv, next to itself.
"""

import csv
import traceback
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
MESSY_CSV = HERE / "orders_messy.csv"          # the till export, with rows that can't be fixed
OUTPUT_DIR = HERE / "output"
REJECTS_CSV = OUTPUT_DIR / "rejected_rows.csv"

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


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def pounds(cents):
    """Format whole cents as pounds for printing. Maths in cents, display in pounds."""
    return f"£{cents / 100:.2f}"


# ---------------------------------------------------------------------------
# 1. Reading a traceback, bottom up
# ---------------------------------------------------------------------------


def quantity_of(row):
    """The quantity column as an int. Trusts the row completely (that's the point)."""
    return int(row["quantity"])


def cups_in(rows):
    """Total cups across some rows."""
    total = 0
    for row in rows:
        total += quantity_of(row)
    return total


def section_traceback():
    heading("1. Reading a traceback, bottom up")

    rows = [
        {"drink": "latte", "quantity": "2"},
        {"drink": "tea", "quantity": "two"},
    ]
    print("  cups_in(rows), where the second row says 'two'. Here's the traceback,")
    print("  caught so the script can carry on, but printed exactly as Python would:")
    print()
    try:
        cups_in(rows)
    except ValueError:
        for line in traceback.format_exc().rstrip().splitlines():
            print(f"    {line}")

    print()
    print("  Read it from the BOTTOM:")
    print("    1. Last line: WHAT went wrong.  ValueError: invalid literal for int() ... 'two'")
    print("    2. Line above it: WHERE, the exact line that blew up, in quantity_of.")
    print("    3. Then upward: WHO called it. cups_in called quantity_of; the section called cups_in.")
    print("  The top of the traceback is the outside of your program. The bottom is the bang.")


# ---------------------------------------------------------------------------
# 2. A field guide to the errors you'll meet
# ---------------------------------------------------------------------------


def section_field_guide():
    heading("2. A field guide to the errors you'll meet")

    orders = [("latte", 2), ("tea", 1)]
    row = {"drink": "latte", "quantity": None}       # what a short CSV row looks like

    attempts = [
        ("int('3.5')",                      lambda: int("3.5")),
        ("int('two')",                      lambda: int("two")),
        ("3 + ' lattes'",                   lambda: 3 + " lattes"),
        ("MENU['mocha']",                   lambda: MENU["mocha"]),
        ("orders[5]",                       lambda: orders[5]),
        ("row['quantity'].strip()  (None!)", lambda: row["quantity"].strip()),
        ("'latte'.push('x')",               lambda: "latte".push("x")),
        ("open(HERE / 'missing.csv')",      lambda: open(HERE / "missing.csv")),
        ("7970 / 0",                        lambda: 7970 / 0),
        ("pirce  (a typo)",                 lambda: pirce),   # deliberately undefined
    ]
    print("  Each line is run, and what Python throws is written next to it.")
    print("  (This demo uses `except Exception` to catch anything. Section 3 is why YOU shouldn't.)")
    print()
    for label, attempt in attempts:
        try:
            attempt()
        except Exception as err:
            print(f"  {label:<38} {type(err).__name__}: {err}")

    print()
    print("  SyntaxError is different: Python finds it before running a single line.")
    try:
        compile("if cups = 3:\n    pass\n", "<your file>", "exec")
    except SyntaxError as err:
        print(f"  if cups = 3:                           SyntaxError: {err.msg}")
    print("  Nothing to catch there. Read the message, fix the line, run again.")


# ---------------------------------------------------------------------------
# 3. try/except on purpose
# ---------------------------------------------------------------------------


def parse_quantity(text):
    """Whole-number text -> int. Anything else -> None. Never a string, never an error."""
    try:
        return int(text)
    except ValueError:
        return None


def cups_from_bad(text):
    """DON'T do this. The bare except hides the typo on the next line."""
    try:
        return int(txet)              # 'txet' is a typo for 'text'
    except:                           # catches EVERYTHING, including the typo
        return 0


def cups_from(text):
    """The honest version: catches only the error we expect, so the typo would show."""
    try:
        return int(text)
    except ValueError:
        return 0


def section_try_except():
    heading("3. try/except on purpose")

    print("  A narrow except, for the one error you expect and know how to handle:")
    for raw in ["2", " 3 ", "two", "", "3.5", "-1"]:
        show(f"parse_quantity({raw!r})", parse_quantity(raw))
    print("  int() copes with spaces and minus signs; everything else is a ValueError -> None.")

    print()
    print("  Now the bare except. cups_from_bad has a TYPO in it (txet for text):")
    show("cups_from_bad('2')  (should be 2)", cups_from_bad("2"))
    show("cups_from_bad('two')", cups_from_bad("two"))
    print("  Every answer is 0, and there's no error to tell you why. The NameError from the")
    print("  typo was swallowed along with the ValueError you meant to catch.")

    print()
    print("  Same function, narrow except, no typo:")
    show("cups_from('2')", cups_from("2"))
    show("cups_from('two')", cups_from("two"))
    print("  Had the typo been here, `except ValueError` would have let the NameError through,")
    print("  and you'd have fixed it in ten seconds instead of finding it in a report next month.")

    print()
    print("  Two more shapes you'll use:")
    try:
        int(None)
    except (TypeError, ValueError) as err:
        print(f"    except (TypeError, ValueError) as err:   -> {type(err).__name__}: {err}")
    try:
        MENU["mocha"]
    except KeyError as err:
        print(f"    except KeyError as err:                  -> not on the menu: {err}")
    print("  A tuple catches several kinds at once. `as err` gives you the message to reuse.")


# ---------------------------------------------------------------------------
# 4. else, finally, and why `with` exists
# ---------------------------------------------------------------------------


def first_line_of(path):
    """The header line of a file, or None if the file isn't there. Closes the file either way."""
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        print(f"    no file at {path.name}")
        return None
    else:
        print("    opened fine, reading the header")
        header = f.readline().rstrip("\n")
    finally:
        print("    finally: this line runs whether or not the open worked")
    f.close()
    return header


def section_else_finally_with():
    heading("4. else, finally, and why `with` exists")

    print("  first_line_of(MESSY_CSV):")
    show("returned", first_line_of(MESSY_CSV))
    print("  first_line_of(HERE / 'nope.csv'):")
    show("returned", first_line_of(HERE / "nope.csv"))

    print()
    print("  `else` runs only if the try succeeded. `finally` runs no matter what.")
    print("  Keep the try block SMALL: one line that can fail, not the whole function.")

    print()
    print("  Closing a file in `finally` is such a common need that `with` does it for you:")
    with open(MESSY_CSV, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
    show("f.closed  (after the with-block)", f.closed)
    show("header", header)
    print("  `with` = 'open this, and close it when the block ends, even if the block raises'.")


# ---------------------------------------------------------------------------
# 5. Validate before it hurts
# ---------------------------------------------------------------------------


def validate_row(raw):
    """
    One raw CSV row -> one clean dict with real types.

    Raises ValueError with a message that names the column and the bad value.
    Checks are in order of "cheapest first", and the first failure wins.
    """
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


def section_validate():
    heading("5. Validate before it hurts")

    good = {"date": "2026-09-07", "drink": "Latte", "size": "medium", "price": "3.80", "quantity": "2"}
    show("validate_row(good)", validate_row(good))

    print()
    print("  Now every kind of bad row, and what validate_row says about each:")
    bad_rows = [
        {"date": "2026-09-08", "drink": "mocha", "size": "medium", "price": "", "quantity": "1"},
        {"date": "09/09/2026", "drink": "latte", "size": "large", "price": "4.30", "quantity": "two"},
        {"date": "2026-09-10", "drink": "espresso", "size": "small", "price": None, "quantity": None},
        {"date": "2026-09-11", "drink": "tea", "size": "large", "price": "2.90", "quantity": "-1"},
        {"date": "2026-09-11", "drink": "latte", "size": "venti", "price": "3.80", "quantity": "1"},
        {"date": "2026-09-11", "drink": "latte", "size": "large", "price": "free", "quantity": "1"},
    ]
    for raw in bad_rows:
        try:
            validate_row(raw)
        except ValueError as err:
            print(f"    {str(err):<48} <- {raw}")

    print()
    print("  Each message names the column and quotes the value. Whoever reads it (you, at")
    print("  11pm) knows which row to look at and what to fix, without opening the code.")

    print()
    print("  `raise ... from err` keeps the original error attached:")
    try:
        validate_row(bad_rows[1])
    except ValueError as err:
        show("str(err)", str(err))
        show("type(err.__cause__).__name__", type(err.__cause__).__name__)
        show("str(err.__cause__)", str(err.__cause__))
    print("  Python shows both in a traceback: 'The above exception was the direct cause...'")


# ---------------------------------------------------------------------------
# 6. When not to swallow an error
# ---------------------------------------------------------------------------


def load_export(path):
    """
    Read the raw text of a till export.

    If the file is missing, re-raise with a message that says what we were
    looking for and what to do about it. We don't return '' or None: a missing
    export is not an empty export, and pretending it is would hide the problem.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"No till export at {path}. Copy this week's export into {path.parent.name}/ "
            f"and name it {path.name}."
        ) from err


def section_dont_swallow():
    heading("6. When not to swallow an error")

    print("  Three honest things to do in an except block:")
    print("    1. Recover, with a fallback you could defend out loud.  (parse_quantity -> None)")
    print("    2. Record it and move on.                               (section 7's rejects file)")
    print("    3. Add context and re-raise.                            (load_export, below)")
    print("  And one dishonest thing: `except: pass`. The error happened. You just can't see it.")

    print()
    print("  load_export with a missing file. The error is still an error, but a useful one:")
    try:
        load_export(HERE / "orders_week_38.csv")
    except FileNotFoundError as err:
        print(f"    FileNotFoundError: {err}")
        print(f"    (caused by: {err.__cause__})")

    print()
    print("  Compare with the quiet version, which 'works':")

    def load_export_quietly(path):
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    text = load_export_quietly(HERE / "orders_week_38.csv")
    show("load_export_quietly(missing)", text)
    show("len(text.splitlines())  rows this week", len(text.splitlines()))
    print("  Zero rows, zero revenue, no error. The report goes out saying the shop sold")
    print("  nothing. That's what swallowing an error costs: it moves the bang somewhere")
    print("  else, later, where nobody's looking.")

    print()
    print("  Rule of thumb: catch an error where you can DO something about it. If you can't,")
    print("  don't catch it. A loud crash with a good traceback is a gift to whoever's next.")


# ---------------------------------------------------------------------------
# 7. Putting it together: good rows, rejected rows, nothing hidden
# ---------------------------------------------------------------------------


def load_orders_carefully(path):
    """
    Read a till export. Returns (orders, rejects).

    orders  - the rows that passed validate_row, with real types.
    rejects - one dict per bad row: the line number, the reason, and the raw values.
    A missing file is NOT caught here: that's a problem for the caller to hear about.
    """
    orders = []
    rejects = []
    with open(path, newline="", encoding="utf-8") as f:
        # start=2: line 1 of the file is the header, so the first data row is line 2.
        for line_number, raw in enumerate(csv.DictReader(f), start=2):
            try:
                orders.append(validate_row(raw))
            except ValueError as err:
                rejects.append({"line": line_number, "reason": str(err),
                                **{field: raw.get(field) or "" for field in RAW_FIELDS}})
    return orders, rejects


def write_rejects(rejects, path):
    """Write the rejected rows and their reasons as a CSV so a human can fix them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["line", "reason", *RAW_FIELDS])
        writer.writeheader()
        writer.writerows(rejects)
    return len(rejects)


def revenue_by_drink(orders):
    """Total cents per drink, as a plain dict. Does not change orders."""
    totals = {}
    for row in orders:
        totals[row["drink"]] = totals.get(row["drink"], 0) + row["price_cents"] * row["quantity"]
    return totals


def section_together():
    heading("7. Putting it together: good rows, rejected rows, nothing hidden")

    orders, rejects = load_orders_carefully(MESSY_CSV)
    print(f"  {MESSY_CSV.name}: {len(orders) + len(rejects)} rows read, "
          f"{len(orders)} good, {len(rejects)} rejected.")

    print()
    print("  Rejected, with the line number in the file and the reason:")
    for reject in rejects:
        print(f"    line {reject['line']:>2}  {reject['reason']}")

    written = write_rejects(rejects, REJECTS_CSV)
    print(f"  {written} rows written to {REJECTS_CSV.relative_to(HERE)} for a human to fix.")

    print()
    print("  The report, from the good rows only:")
    revenue = revenue_by_drink(orders)
    for drink, cents in sorted(revenue.items(), key=lambda item: item[1], reverse=True):
        print(f"    {drink:<12} {pounds(cents):>8}")
    cups = sum(row["quantity"] for row in orders)
    print(f"    {'total':<12} {pounds(sum(revenue.values())):>8}   ({cups} cups, {len(orders)} orders)")

    assert sum(revenue.values()) == 7970, "the 15 good rows should total £79.70"
    assert cups == 23, "the 15 good rows should be 23 cups"
    assert len(rejects) == 4, "four rows in the export can't be fixed automatically"
    print()
    print("  £79.70 and 23 cups: the same week as lessons 005 to 007. The four bad rows")
    print("  didn't crash the report and didn't quietly change the numbers. Both matter.")

    print()
    print("  And the missing-file case is still LOUD, because we chose not to catch it:")
    try:
        load_orders_carefully(HERE / "orders_week_38.csv")
    except FileNotFoundError as err:
        print(f"    FileNotFoundError: {err}")


def main():
    print("=" * 66)
    print("  Lesson 008: Errors, and what to do about them")
    print("=" * 66)
    section_traceback()
    section_field_guide()
    section_try_except()
    section_else_finally_with()
    section_validate()
    section_dont_swallow()
    section_together()
    print()
    print("Read it bottom up, catch what you expect, validate at the door, never `pass`. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
