"""
Lesson 006 - Functions that don't lie

Print vs return, one job per function, names that tell the truth, return
values that don't surprise, default arguments (and the one trap in them),
not changing what you were given, and assert as a one-line honesty check.
Finishes by rebuilding the week-of-orders report out of small functions
that can each be trusted on their own.

Run it with:

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.
"""

from collections import defaultdict

# The same week of orders: (date, drink, size, price_in_cents, quantity).
ORDERS = [
    ("2026-09-07", "latte",      "medium", 380, 2),
    ("2026-09-07", "espresso",   "small",  220, 1),
    ("2026-09-07", "cappuccino", "large",  420, 1),
    ("2026-09-08", "latte",      "large",  430, 1),
    ("2026-09-08", "flat white", "medium", 390, 3),
    ("2026-09-08", "espresso",   "small",  220, 2),
    ("2026-09-09", "cappuccino", "medium", 370, 2),
    ("2026-09-09", "latte",      "medium", 380, 1),
    ("2026-09-09", "tea",        "medium", 250, 1),
    ("2026-09-10", "flat white", "medium", 390, 1),
    ("2026-09-10", "latte",      "small",  330, 2),
    ("2026-09-10", "espresso",   "small",  220, 1),
    ("2026-09-11", "cappuccino", "large",  420, 2),
    ("2026-09-11", "latte",      "medium", 380, 2),
    ("2026-09-11", "tea",        "large",  290, 1),
]


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<44} -> {value!r}")


def pounds(cents):
    """Format whole cents as pounds for printing. Maths in cents, display in pounds."""
    return f"£{cents / 100:.2f}"


# ---------------------------------------------------------------------------
# 1. Print is for people, return is for code
# ---------------------------------------------------------------------------


def describe_order_badly(drink, cents):
    """Prints a label. Returns nothing. The caller gets None."""
    print(f"    {drink}: {pounds(cents)}")


def describe_order(drink, cents):
    """Returns a label. The caller decides what to do with it."""
    return f"{drink}: {pounds(cents)}"


def section_print_vs_return():
    heading("1. Print is for people, return is for code")

    print("  A function that prints:")
    label = describe_order_badly("latte", 380)
    show("label = describe_order_badly(...)", label)
    print("  It showed you something, but it handed the code nothing.")

    print()
    print("  A function that returns:")
    label = describe_order("latte", 380)
    show("label = describe_order(...)", label)
    show("label.upper()", label.upper())
    show("len(label)", len(label))
    print("  Now the caller can print it, store it, compare it, or put it in a list.")

    print()
    print("  A def with no return statement returns None. Every time.")

    def add_up(a, b):
        total = a + b          # ...and then forgets to return it

    show("add_up(2, 3)", add_up(2, 3))


# ---------------------------------------------------------------------------
# 2. One job, named honestly
# ---------------------------------------------------------------------------


def line_total(order):
    """The value of one order line, in cents."""
    _, _, _, price, qty = order
    return price * qty


def revenue_by_drink(orders):
    """Total cents per drink, as a plain dict. Does not change orders."""
    totals = defaultdict(int)
    for order in orders:
        totals[order[1]] += line_total(order)
    return dict(totals)


def is_large(order):
    """True if the order is a large size."""
    return order[2] == "large"


def section_one_job():
    heading("2. One job, named honestly")

    show("line_total(ORDERS[0])", line_total(ORDERS[0]))
    show("revenue_by_drink(ORDERS[:3])", revenue_by_drink(ORDERS[:3]))
    show("is_large(ORDERS[2])", is_large(ORDERS[2]))
    show("is_large(ORDERS[0])", is_large(ORDERS[0]))

    print()
    print("  Read the names out loud. Each one is a promise:")
    print("    line_total(order)        -> a number")
    print("    revenue_by_drink(orders) -> a dict, drink -> cents")
    print("    is_large(order)          -> True or False")
    print("  If you need 'and' to describe a function, it's two functions.")


# ---------------------------------------------------------------------------
# 3. Return values that don't surprise
# ---------------------------------------------------------------------------


def parse_quantity(text):
    """Whole-number text -> int. Anything else -> None. Never a string, never an error."""
    cleaned = text.strip()
    if not cleaned.isdigit():
        return None
    return int(cleaned)


def parse_quantity_liar(text):
    """DON'T do this: sometimes an int, sometimes a string, sometimes nothing."""
    cleaned = text.strip()
    if cleaned == "":
        return            # None, by accident rather than by design
    if not cleaned.isdigit():
        return "bad quantity"
    return int(cleaned)


def cheapest_and_dearest(orders):
    """Return (cheapest_order, dearest_order) by unit price."""
    return min(orders, key=lambda o: o[3]), max(orders, key=lambda o: o[3])


def section_return_values():
    heading("3. Return values that don't surprise")

    inputs = [" 2 ", "x", "", "10"]
    print("  A liar: three different kinds of answer for one question.")
    for raw in inputs:
        print(f"    {raw!r:<6} -> {parse_quantity_liar(raw)!r}")
    print("  Try adding these up and the string blows up in your face two lines later.")

    print()
    print("  Honest: an int, or None meaning 'no quantity'. Two kinds, both documented.")
    for raw in inputs:
        print(f"    {raw!r:<6} -> {parse_quantity(raw)!r}")

    print()
    print("  Two answers? Return a tuple and unpack it (lesson 004):")
    lo, hi = cheapest_and_dearest(ORDERS)
    show("cheapest", lo)
    show("dearest", hi)

    print()
    print("  A guard clause returns early so the happy path isn't indented:")
    print("    if not cleaned.isdigit():")
    print("        return None")
    print("    return int(cleaned)")


# ---------------------------------------------------------------------------
# 4. Default arguments, and the one trap
# ---------------------------------------------------------------------------


def format_money(cents, symbol="£", decimals=2):
    """Cents -> a money string. Defaults suit the cafe; override for anything else."""
    return f"{symbol}{cents / 100:.{decimals}f}"


def top_n(totals, n=3):
    """The n biggest (key, value) pairs from a dict, biggest first."""
    return sorted(totals.items(), key=lambda item: item[1], reverse=True)[:n]


def add_cup_trap(drink, cups=[]):
    """DON'T do this. The [] is created once and shared by every call."""
    cups.append(drink)
    return cups


def add_cup(drink, cups=None):
    """A fresh list each call, unless the caller hands one in."""
    if cups is None:
        cups = []
    cups.append(drink)
    return cups


def section_defaults():
    heading("4. Default arguments, and the one trap")

    show("format_money(380)", format_money(380))
    show('format_money(380, symbol="$")', format_money(380, symbol="$"))
    show("format_money(380, decimals=0)", format_money(380, decimals=0))
    show('format_money(380, "€", 1)  (positional)', format_money(380, "€", 1))

    print()
    revenue = revenue_by_drink(ORDERS)
    show("top_n(revenue)", top_n(revenue))
    show("top_n(revenue, n=1)", top_n(revenue, n=1))
    print("  Keywords make a call read like a sentence: top_n(revenue, n=1).")

    print()
    print("  THE TRAP. A mutable default is built once, when the def runs:")
    morning = add_cup_trap("latte")
    afternoon = add_cup_trap("tea")
    show("morning   (expected ['latte'])", morning)
    show("afternoon (expected ['tea'])", afternoon)
    show("morning is afternoon", morning is afternoon)

    print()
    print("  The fix: default to None, make the list inside.")
    morning = add_cup("latte")
    afternoon = add_cup("tea")
    show("morning", morning)
    show("afternoon", afternoon)
    show("morning is afternoon", morning is afternoon)


# ---------------------------------------------------------------------------
# 5. Don't change what you were given
# ---------------------------------------------------------------------------


def top_three_sneaky(prices):
    """DON'T do this. It sorts the caller's list behind their back."""
    prices.sort(reverse=True)
    return prices[:3]


def top_three(prices):
    """The three highest prices, as a new list. prices is left alone."""
    return sorted(prices, reverse=True)[:3]


def section_dont_mutate():
    heading("5. Don't change what you were given")

    prices = [380, 220, 420, 430, 390]
    show("prices before", prices)
    result = top_three_sneaky(prices)
    show("top_three_sneaky(prices)", result)
    show("prices after  (rearranged!)", prices)

    print()
    prices = [380, 220, 420, 430, 390]
    result = top_three(prices)
    show("top_three(prices)", result)
    show("prices after  (untouched)", prices)

    print()
    print("  Python's own library follows the rule:")
    print("    list.sort()  changes the list, returns None    (a verb: 'sort it')")
    print("    sorted()     returns a new list, changes nothing (an adjective: 'sorted copy')")
    print("  Do one or the other. Never both. Let the name say which.")

    print()
    print("  Pure functions: same inputs, same output, nothing else touched.")
    show("line_total(ORDERS[4])", line_total(ORDERS[4]))
    show("line_total(ORDERS[4])  (again)", line_total(ORDERS[4]))
    print("  You can call a pure function a thousand times and nothing changes but your answer.")


# ---------------------------------------------------------------------------
# 6. Pass it in, hand it back
# ---------------------------------------------------------------------------

discount_rate = 0.10       # module-level, for the sake of the bad example


def discounted_sneaky(cents):
    """Reads a variable from outside. Works until someone renames or changes it."""
    return round(cents * (1 - discount_rate))


def discounted(cents, rate=0.10):
    """Everything it needs arrives through the door."""
    return round(cents * (1 - rate))


def section_scope():
    heading("6. Pass it in, hand it back")

    show("discounted_sneaky(380)", discounted_sneaky(380))
    show("discounted(380)", discounted(380))
    show("discounted(380, rate=0.25)", discounted(380, rate=0.25))
    print("  The second can be read, tested and reused without knowing anything")
    print("  about the rest of the file. The first can't.")

    print()
    print("  Names made inside a function stay inside it:")

    def make_label():
        inner = "only here"
        return inner

    show("make_label()", make_label())
    try:
        inner
    except NameError as err:
        print(f"    NameError: {err}")


# ---------------------------------------------------------------------------
# 7. assert: a one-line honesty check
# ---------------------------------------------------------------------------


def section_assert():
    heading("7. assert: a one-line honesty check")

    assert line_total(("d", "latte", "medium", 380, 2)) == 760
    assert parse_quantity(" 3 ") == 3
    assert parse_quantity("x") is None
    assert parse_quantity("") is None
    assert top_three([1, 5, 3, 4, 2]) == [5, 4, 3]
    assert format_money(1250) == "£12.50"
    assert revenue_by_drink(ORDERS[:2]) == {"latte": 760, "espresso": 220}
    print("  Seven asserts passed silently. Silence is the good outcome.")

    print()
    print("  When one fails, it fails loudly and points at the line:")
    try:
        assert format_money(380) == "£3.8", "format_money should always show 2 decimals"
    except AssertionError as err:
        print(f"    AssertionError: {err}")
    print("  A handful of these under a function is the cheapest insurance in programming.")


# ---------------------------------------------------------------------------
# 8. Putting it together: a report made of small, honest pieces
# ---------------------------------------------------------------------------


def format_row(drink: str, cents: int, star_from_cents: int) -> str:
    """One line of the report. Adds a star for big earners."""
    star = " *" if cents >= star_from_cents else ""
    return f"    {drink:<12} {pounds(cents):>7}{star}"


def report_lines(orders, top: int = 5, star_from_cents: int = 2000) -> list[str]:
    """Build the report as a list of strings. Nothing printed, nothing changed."""
    revenue = revenue_by_drink(orders)
    lines = [f"    {'drink':<12} {'revenue':>7}"]
    for drink, cents in top_n(revenue, n=top):
        lines.append(format_row(drink, cents, star_from_cents))
    lines.append(f"    {'total':<12} {pounds(sum(revenue.values())):>7}")
    lines.append(f"    * = {pounds(star_from_cents)} or more")
    return lines


def print_report(orders, top: int = 5, star_from_cents: int = 2000) -> None:
    """The one function that talks to a human. It only prints."""
    for line in report_lines(orders, top=top, star_from_cents=star_from_cents):
        print(line)


def section_report():
    heading("8. Putting it together: a report made of small, honest pieces")

    print("  print_report(ORDERS):")
    print_report(ORDERS)

    print()
    print("  print_report(ORDERS[:6], top=2, star_from_cents=1000):")
    print_report(ORDERS[:6], top=2, star_from_cents=1000)

    print()
    print("  Because report_lines RETURNS its lines, we can test it without reading a screen:")
    lines = report_lines(ORDERS, top=1)
    assert lines[1].strip().startswith("latte"), "latte should be the top earner"
    assert len(lines) == 4, "header + 1 row + total + legend"
    show("len(report_lines(ORDERS, top=1))", len(lines))
    show("ORDERS[0] still first?", ORDERS[0][1] == "latte")


def main():
    print("=" * 66)
    print("  Lesson 006: Functions that don't lie")
    print("=" * 66)
    section_print_vs_return()
    section_one_job()
    section_return_values()
    section_defaults()
    section_dont_mutate()
    section_scope()
    section_assert()
    section_report()
    print()
    print("Small, named, one job, honest about what comes back. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
