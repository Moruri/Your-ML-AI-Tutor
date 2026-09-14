"""
Lesson 003 - Numbers, strings and the things that bite

Ints, floats and strings: how to move between them, where floats surprise
you, how to keep money exact, how to print numbers tidily, and the string
methods that turn messy text into clean values.

Run it with:

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.
"""

import math
from decimal import Decimal


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    """Print a label and a value, using repr so strings show their quotes."""
    print(f"  {label:<34} -> {value!r}")


# ---------------------------------------------------------------------------
# 1. Three kinds of value, and moving between them
# ---------------------------------------------------------------------------


def section_three_kinds():
    heading("1. Three kinds of value, and moving between them")

    for value in (2, 3.80, "latte"):
        print(f"  {value!r:<10} is a {type(value).__name__}")

    print()
    show('int("2")', int("2"))
    show('float("3.80")', float("3.80"))
    show("str(3.8)", str(3.8))
    show("int(3.99)   (truncates!)", int(3.99))
    show("float(2)", float(2))
    show('int(float("3.80"))', int(float("3.80")))

    print()
    print('  int("3.80") on its own is an error. Let\'s prove it safely:')
    try:
        int("3.80")
    except ValueError as err:
        print(f"    ValueError: {err}")
    print("  (try/except is lesson 008; for now just see that it fails.)")


# ---------------------------------------------------------------------------
# 2. Floats: the surprise, and why
# ---------------------------------------------------------------------------


def section_floats():
    heading("2. Floats: the surprise, and why")

    show("0.1 + 0.2", 0.1 + 0.2)
    show("0.1 + 0.2 == 0.3", 0.1 + 0.2 == 0.3)
    show("math.isclose(0.1 + 0.2, 0.3)", math.isclose(0.1 + 0.2, 0.3))
    show("round(0.1 + 0.2, 2) == 0.3", round(0.1 + 0.2, 2) == 0.3)

    print()
    print("  Watch the crumbs pile up. Adding 3.90 to itself ten times:")
    total = 0.0
    for _ in range(10):
        total += 3.90
    show("float total", total)
    show("what you meant", 39.0)
    print("  Harmless for a sensor reading. Not harmless for a till.")


# ---------------------------------------------------------------------------
# 3. Money: use cents, or use Decimal
# ---------------------------------------------------------------------------


def section_money():
    heading("3. Money: use cents, or use Decimal")

    print("  Option A: whole cents in an int. Integers are always exact.")
    price_cents = 390
    total_cents = 0
    for _ in range(10):
        total_cents += price_cents
    show("total_cents", total_cents)
    print(f"  printed as money: £{total_cents / 100:.2f}")

    print()
    print("  Option B: the decimal module, built from STRINGS.")
    show('Decimal("0.1") + Decimal("0.2")', Decimal("0.1") + Decimal("0.2"))
    show('Decimal("3.90") * 10', Decimal("3.90") * 10)
    print()
    print("  Why strings? Because a float already carries the error with it:")
    show("Decimal(0.1)", Decimal(0.1))
    show('Decimal("0.1")', Decimal("0.1"))


# ---------------------------------------------------------------------------
# 4. Rounding, and telling Python how to print numbers
# ---------------------------------------------------------------------------


def section_rounding_and_formats():
    heading("4. Rounding, and telling Python how to print numbers")

    show("round(3.14159, 2)", round(3.14159, 2))
    show("round(2.5)   (banker's rounding)", round(2.5))
    show("round(3.5)", round(3.5))
    show("round(4.5)", round(4.5))

    print()
    print("  Format specs: change how a number LOOKS, not what it IS.")
    n = 1234.5678
    show("f'{n:.2f}'", f"{n:.2f}")
    show("f'{n:,.2f}'", f"{n:,.2f}")
    show("f'{n:.0f}'", f"{n:.0f}")
    show("f'{0.256:.1%}'", f"{0.256:.1%}")
    show("f'{n:>12.2f}'", f"{n:>12.2f}")
    show("f'{\"latte\":<10}|'", f"{'latte':<10}|")
    show("f'{7:03d}'", f"{7:03d}")

    print()
    print("  Put alignment specs together and you get a table for free:")
    rows = [("latte", 3.80, 7), ("flat white", 3.90, 4), ("tea", 2.50, 12)]
    print(f"  {'drink':<12}{'price':>8}{'cups':>6}{'revenue':>10}")
    for drink, price, cups in rows:
        print(f"  {drink:<12}{price:>8.2f}{cups:>6}{price * cups:>10.2f}")


# ---------------------------------------------------------------------------
# 5. f-strings, properly
# ---------------------------------------------------------------------------


def section_fstrings():
    heading("5. f-strings, properly")

    price, qty = 3.80, 3
    print(f"  Any expression works inside the braces: {qty} cups = £{price * qty:.2f}")

    raw = " latte "
    print(f"  Without !r: got {raw}   <- looks fine")
    print(f"  With !r:    got {raw!r}   <- ah. Spaces.")

    total = price * qty
    print(f"  The = spec for quick debugging: {total=}")


# ---------------------------------------------------------------------------
# 6. String methods for messy data
# ---------------------------------------------------------------------------


def section_string_methods():
    heading("6. String methods for messy data")

    raw = "  Flat White  "
    show("raw", raw)
    show("raw.strip()", raw.strip())
    show("raw.lower()", raw.lower())
    show("raw.strip().lower()", raw.strip().lower())
    show("raw   (unchanged!)", raw)

    print()
    variants = ["Latte", "latte ", " LATTE", "latte"]
    normalised = []
    for v in variants:
        normalised.append(v.strip().lower())
    show("four spellings", variants)
    show("after strip().lower()", normalised)
    show("all the same now?", normalised.count("latte") == len(normalised))

    print()
    show('"£3.80".replace("£", "")', "£3.80".replace("£", ""))
    show('"3,80".replace(",", ".")', "3,80".replace(",", "."))
    show('"1,250.00".replace(",", "")', "1,250.00".replace(",", ""))

    print()
    show('"latte,medium,3.80".split(",")', "latte,medium,3.80".split(","))
    show('"2026-09-14".split("-")', "2026-09-14".split("-"))
    show('"-".join(["2026", "09", "14"])', "-".join(["2026", "09", "14"]))
    show('", ".join(["latte", "tea"])', ", ".join(["latte", "tea"]))

    print()
    name = "flat white"
    show('name.startswith("flat")', name.startswith("flat"))
    show('name.endswith("tea")', name.endswith("tea"))
    show('"white" in name', "white" in name)
    show('"42".isdigit()', "42".isdigit())
    show('"4.2".isdigit()', "4.2".isdigit())


# ---------------------------------------------------------------------------
# 7. Why CSV gives you text, and what to do about it
# ---------------------------------------------------------------------------


def parse_price_cents(text):
    """
    Turn a messy price string into whole cents, or None if it isn't a price.

    Normalise first, handle "missing" explicitly, decide what a comma means,
    then convert. Every step is a decision about THIS data; when you meet a
    new file, expect to revisit them.
    """
    cleaned = text.strip().lower().replace("£", "")
    if cleaned in ("", "n/a", "free"):
        return None
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")     # "3,90" -> "3.90" (European decimal)
    else:
        cleaned = cleaned.replace(",", "")      # "1,250.00" -> "1250.00" (thousands)
    return round(float(cleaned) * 100)


def section_csv_text():
    heading("7. Why CSV gives you text, and what to do about it")

    raw_prices = ["3.80", " 4.20", "£1.15", "3,90", "free", "", "N/A", "1,250.00"]
    print("  One price column, three export systems, eight opinions:")
    for raw in raw_prices:
        print(f"  {raw!r:>12} -> {parse_price_cents(raw)!r}")

    print()
    print("  Why round() at the end, and not int()? Because floats leave crumbs:")
    show('float("1.15") * 100', float("1.15") * 100)
    show('int(float("1.15") * 100)   (wrong!)', int(float("1.15") * 100))
    show('round(float("1.15") * 100)', round(float("1.15") * 100))

    print()
    usable = 0
    total_cents = 0
    for raw in raw_prices:
        cents = parse_price_cents(raw)
        if cents is not None:
            usable += 1
            total_cents += cents
    print(f"  {usable} usable prices out of {len(raw_prices)}. "
          f"Their total: £{total_cents / 100:,.2f} (exact, because cents).")


def main():
    print("=" * 66)
    print("  Lesson 003: Numbers, strings and the things that bite")
    print("=" * 66)
    section_three_kinds()
    section_floats()
    section_money()
    section_rounding_and_formats()
    section_fstrings()
    section_string_methods()
    section_csv_text()
    print()
    print("That's the toolkit for messy values. Now do the exercises in exercises.md.")
    print()


if __name__ == "__main__":
    main()
