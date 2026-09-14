"""
Lesson 004 - Lists and tuples, properly

Slicing, copies vs aliases, the list methods you'll actually use, sorting
with key=, unpacking, readable comprehensions, and when a tuple is the
better fit. Finishes with a week of coffee orders held as a list of tuples.

Run it with:

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.
"""

# Friday's coffee_orders.csv, typed in as records. Each row is a tuple:
# (date, drink, size, price_in_cents, quantity). A list OF tuples is a table.
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
    print(f"  {label:<36} -> {value!r}")


def pounds(cents):
    """Format whole cents as pounds for printing. Maths in cents, display in pounds."""
    return f"£{cents / 100:.2f}"


# ---------------------------------------------------------------------------
# 1. Indexing and slicing
# ---------------------------------------------------------------------------


def section_slicing():
    heading("1. Indexing and slicing")

    prices = [380, 220, 420, 430, 390, 220, 370]
    show("prices", prices)
    show("prices[0]", prices[0])
    show("prices[-1]", prices[-1])
    show("prices[1:4]   (stop not included)", prices[1:4])
    show("prices[:3]    (first three)", prices[:3])
    show("prices[3:]    (the rest)", prices[3:])
    show("prices[-2:]   (last two)", prices[-2:])
    show("prices[::2]   (every second)", prices[::2])
    show("prices[::-1]  (reversed)", prices[::-1])
    show("prices[5:100] (slices never crash)", prices[5:100])

    print()
    print("  A slice is a NEW list. Change it, the original is untouched:")
    first_three = prices[:3]
    first_three[0] = 0
    show("first_three after change", first_three)
    show("prices", prices)


# ---------------------------------------------------------------------------
# 2. Two names, one list
# ---------------------------------------------------------------------------


def section_aliases_and_copies():
    heading("2. Two names, one list")

    a = [380, 220, 420]
    b = a                    # NOT a copy. Two names, one list.
    b.append(999)
    show("a after b.append(999)", a)
    show("a is b", a is b)

    print()
    a = [380, 220, 420]
    b = a.copy()             # or list(a), or a[:]
    b.append(999)
    show("a after b.copy() then append", a)
    show("b", b)
    show("a is b", a is b)


# ---------------------------------------------------------------------------
# 3. Changing a list
# ---------------------------------------------------------------------------


def section_changing_lists():
    heading("3. Changing a list")

    queue = ["latte", "tea"]
    show("start", queue)
    queue.append("espresso")
    show('append("espresso")', queue)
    queue.extend(["mocha", "tea"])
    show('extend(["mocha", "tea"])', queue)
    queue.insert(0, "flat white")
    show('insert(0, "flat white")', queue)
    served = queue.pop(0)
    show("pop(0) returned", served)
    last = queue.pop()
    show("pop() returned", last)
    queue.remove("tea")
    show('remove("tea")  (first one only)', queue)
    show('"mocha" in queue', "mocha" in queue)
    show('queue.index("mocha")', queue.index("mocha"))

    print()
    print("  The classic mistake: append where you meant extend.")
    oops = ["latte"]
    oops.append(["mocha", "tea"])
    show("after append(a list)", oops)
    print("  And remember these return None; don't write queue = queue.append(x).")
    show('["a"].append("b")', ["a"].append("b"))


# ---------------------------------------------------------------------------
# 4. Sorting
# ---------------------------------------------------------------------------


def section_sorting():
    heading("4. Sorting: sorted vs .sort, and key=")

    prices = [380, 220, 420]
    ranked = sorted(prices)
    show("sorted(prices)", ranked)
    show("prices (unchanged)", prices)
    result = prices.sort()
    show("prices.sort() returned", result)
    show("prices (changed in place)", prices)

    print()
    orders = [("latte", 380, 2), ("tea", 250, 1), ("cappuccino", 420, 1)]
    show("cheapest first", sorted(orders, key=lambda o: o[1]))
    show("biggest line total first",
         sorted(orders, key=lambda o: o[1] * o[2], reverse=True))
    show("alphabetical", sorted(orders, key=lambda o: o[0]))
    show("max by line total", max(orders, key=lambda o: o[1] * o[2]))

    print()
    print("  Sort by several things with a tuple key: size, then price descending.")
    stock = [("medium", 380), ("small", 220), ("medium", 370), ("small", 330)]
    show("key=lambda s: (s[0], -s[1])", sorted(stock, key=lambda s: (s[0], -s[1])))


# ---------------------------------------------------------------------------
# 5. Unpacking
# ---------------------------------------------------------------------------


def section_unpacking():
    heading("5. Unpacking: several names at once")

    order = ("latte", 380, 2)
    drink, price, qty = order
    show("drink, price, qty = order", (drink, price, qty))

    a, b = 1, 2
    a, b = b, a
    show("a, b = b, a", (a, b))

    first, *rest = [380, 220, 420]
    show("first, *rest = [380, 220, 420]", (first, rest))
    *most, last = [380, 220, 420]
    show("*most, last = [380, 220, 420]", (most, last))

    print()
    print("  enumerate gives you the position too:")
    for i, drink in enumerate(["latte", "tea", "mocha"], start=1):
        print(f"    {i}. {drink}")

    print("  zip walks two lists in step:")
    drinks = ["latte", "tea"]
    cups = [7, 12]
    for drink, n in zip(drinks, cups):
        print(f"    {drink}: {n} cups")


# ---------------------------------------------------------------------------
# 6. List comprehensions
# ---------------------------------------------------------------------------


def section_comprehensions():
    heading("6. List comprehensions, kept readable")

    prices = [380, 220, 420, 430, 390, 220, 370]

    big_loop = []
    for p in prices:
        if p > 400:
            big_loop.append(p)
    big_comp = [p for p in prices if p > 400]
    show("the loop version", big_loop)
    show("[p for p in prices if p > 400]", big_comp)

    print()
    orders = [("latte", 380, 2), ("tea", 250, 1), ("cappuccino", 420, 1)]
    show("[p / 100 for p in prices]", [p / 100 for p in prices])
    show("[o[0].title() for o in orders]", [o[0].title() for o in orders])
    show("[price * qty for _, price, qty in orders]",
         [price * qty for _, price, qty in orders])
    show("sum(price * qty for _, price, qty in orders)",
         sum(price * qty for _, price, qty in orders))
    print("  If you can't say a comprehension out loud in one breath, write the loop.")


# ---------------------------------------------------------------------------
# 7. When a tuple is the better fit
# ---------------------------------------------------------------------------


def cheapest_and_dearest(prices):
    """Return two values. This is quietly a tuple."""
    return min(prices), max(prices)


def section_tuples():
    heading("7. When a tuple is the better fit")

    order = ("2026-09-14", "latte", "medium", 380, 2)
    show("a record", order)
    show("order[1]", order[1])
    show("order[3:]", order[3:])
    print("  Try to change it and Python refuses:")
    try:
        order[3] = 400
    except TypeError as err:
        print(f"    TypeError: {err}")

    print()
    print("  Tuples can be dict keys. Group by two things at once:")
    by_day_and_drink = {}
    for date, drink, _, price, qty in ORDERS[:6]:
        key = (date, drink)
        by_day_and_drink[key] = by_day_and_drink.get(key, 0) + price * qty
    for key, cents in by_day_and_drink.items():
        print(f"    {key!s:<28} {pounds(cents):>7}")

    print()
    lo, hi = cheapest_and_dearest([380, 220, 420])
    show("lo, hi = cheapest_and_dearest(...)", (lo, hi))
    show("(380)   is just a number", (380))
    show("(380,)  is a one-item tuple", (380,))


# ---------------------------------------------------------------------------
# 8. Putting it together
# ---------------------------------------------------------------------------


def section_week_of_orders():
    heading("8. Putting it together: a week of orders")

    print(f"  {len(ORDERS)} orders. First row: {ORDERS[0]}")

    line_totals = [price * qty for _, _, _, price, qty in ORDERS]
    print(f"  Week revenue: {pounds(sum(line_totals))}   "
          f"(biggest single line: {pounds(max(line_totals))})")

    print()
    print("  Top three orders by line total:")
    top3 = sorted(ORDERS, key=lambda o: o[3] * o[4], reverse=True)[:3]
    for i, (date, drink, size, price, qty) in enumerate(top3, start=1):
        print(f"    {i}. {date} {drink:<11} {size:<7} {qty} x {pounds(price):>5} = {pounds(price * qty):>7}")

    print()
    seen = []
    for _, drink, *_ in ORDERS:
        if drink not in seen:
            seen.append(drink)
    show("distinct drinks, first-seen order", seen)
    show("same, alphabetical", sorted(seen))

    print()
    print("  Revenue by (date, drink), tuple keys, biggest first:")
    by_day_and_drink = {}
    for date, drink, _, price, qty in ORDERS:
        key = (date, drink)
        by_day_and_drink[key] = by_day_and_drink.get(key, 0) + price * qty
    ranked = sorted(by_day_and_drink.items(), key=lambda item: item[1], reverse=True)
    for (date, drink), cents in ranked[:5]:
        print(f"    {date}  {drink:<11} {pounds(cents):>7}")
    print(f"    ... and {len(ranked) - 5} more")


def main():
    print("=" * 66)
    print("  Lesson 004: Lists and tuples, properly")
    print("=" * 66)
    section_slicing()
    section_aliases_and_copies()
    section_changing_lists()
    section_sorting()
    section_unpacking()
    section_comprehensions()
    section_tuples()
    section_week_of_orders()
    print()
    print("Lists for many of a thing, tuples for one thing with parts. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
