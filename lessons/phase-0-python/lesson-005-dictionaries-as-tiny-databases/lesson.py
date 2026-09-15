"""
Lesson 005 - Dictionaries as tiny databases

Look things up by key, ask politely for keys that might not be there, count
things with collections.Counter, group things with defaultdict, and use a
small dict as a lookup table to join two sets of facts together. Finishes
with the week of coffee orders answered five ways.

Run it with:

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.
"""

from collections import Counter, defaultdict

# Yesterday's table again: (date, drink, size, price_in_cents, quantity).
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

# The menu board, medium size, in cents. A dict: drink -> price.
MENU = {
    "espresso": 220,
    "tea": 250,
    "cappuccino": 370,
    "latte": 380,
    "flat white": 390,
}

# A second tiny table: drink -> category. Joining ORDERS to this is section 7.
CATEGORY = {
    "espresso": "black coffee",
    "latte": "milky coffee",
    "flat white": "milky coffee",
    "cappuccino": "milky coffee",
    "tea": "tea",
}

# Spellings staff actually type -> the one spelling we want to keep.
ALIASES = {
    "flat-white": "flat white",
    "flatwhite": "flat white",
    "capp": "cappuccino",
    "cap": "cappuccino",
}


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<40} -> {value!r}")


def pounds(cents):
    """Format whole cents as pounds for printing. Maths in cents, display in pounds."""
    return f"£{cents / 100:.2f}"


# ---------------------------------------------------------------------------
# 1. A dict is a lookup table
# ---------------------------------------------------------------------------


def section_lookup_table():
    heading("1. A dict is a lookup table")

    show("MENU", MENU)
    show('MENU["latte"]', MENU["latte"])
    show("len(MENU)", len(MENU))
    show('"tea" in MENU', "tea" in MENU)
    show('"mocha" in MENU', "mocha" in MENU)

    print()
    print('  Ask for a key that isn\'t there with [] and Python refuses:')
    try:
        MENU["mocha"]
    except KeyError as err:
        print(f"    KeyError: {err}")

    print()
    print("  Keys must be unchangeable. Strings, numbers and tuples are fine; lists are not:")
    ok = {("latte", "medium"): 380}
    show("{('latte', 'medium'): 380}", ok)
    try:
        {["latte", "medium"]: 380}
    except TypeError as err:
        print(f"    TypeError: {err}")


# ---------------------------------------------------------------------------
# 2. .get: asking politely
# ---------------------------------------------------------------------------


def section_get():
    heading("2. .get: asking politely")

    show('MENU.get("latte")', MENU.get("latte"))
    show('MENU.get("mocha")', MENU.get("mocha"))
    show('MENU.get("mocha", 0)', MENU.get("mocha", 0))
    show('MENU.get("mocha", "not on the menu")', MENU.get("mocha", "not on the menu"))

    print()
    print("  The counting pattern you've used since Friday, spelled out once:")
    cups = {}
    for _, drink, _, _, qty in ORDERS[:6]:
        cups[drink] = cups.get(drink, 0) + qty
    show("cups after six orders", cups)

    print()
    print("  Use [] when a missing key is a BUG you want to hear about.")
    print("  Use .get when a missing key is NORMAL and you know what to do instead.")


# ---------------------------------------------------------------------------
# 3. Changing a dict, and looping over one
# ---------------------------------------------------------------------------


def section_changing_and_looping():
    heading("3. Changing a dict, and looping over one")

    menu = dict(MENU)                 # a copy, so the original stays clean
    menu["mocha"] = 400
    show('after menu["mocha"] = 400', menu)
    menu["tea"] = 260
    show('after menu["tea"] = 260  (overwrites)', menu)
    removed = menu.pop("espresso")
    show('menu.pop("espresso") returned', removed)
    show('menu.pop("hot choc", None)', menu.pop("hot choc", None))
    menu.update({"hot chocolate": 350, "tea": 270})
    show("after .update({...})", menu)
    del menu["mocha"]
    show('after del menu["mocha"]', menu)

    print()
    print("  Three ways to loop. .items() is the one you want almost every time:")
    show("list(MENU.keys())", list(MENU.keys()))
    show("list(MENU.values())", list(MENU.values()))
    for drink, cents in MENU.items():
        print(f"    {drink:<12} {pounds(cents):>6}")

    print()
    print("  Dicts remember insertion order. Sorting one means sorting its items:")
    by_price_desc = sorted(MENU.items(), key=lambda item: item[1], reverse=True)
    show("dearest first", by_price_desc)
    show("keys only, alphabetical", sorted(MENU))

    print()
    print("  A dict comprehension builds a dict the way a list comprehension builds a list:")
    in_pounds = {drink: cents / 100 for drink, cents in MENU.items()}
    show("{drink: cents / 100 for ...}", in_pounds)
    under_3 = {d: c for d, c in MENU.items() if c < 300}
    show("{... if c < 300}", under_3)
    by_price = {cents: drink for drink, cents in MENU.items()}
    show("flipped: {cents: drink ...}", by_price)


# ---------------------------------------------------------------------------
# 4. Counting: Counter
# ---------------------------------------------------------------------------


def section_counter():
    heading("4. Counting: Counter")

    drinks = [drink for _, drink, *_ in ORDERS]
    show("drinks (one per order)", drinks)

    by_hand = {}
    for drink in drinks:
        by_hand[drink] = by_hand.get(drink, 0) + 1
    show("counted by hand", by_hand)

    orders_per_drink = Counter(drinks)
    show("Counter(drinks)", orders_per_drink)
    show(".most_common(2)", orders_per_drink.most_common(2))
    show('["latte"]', orders_per_drink["latte"])
    show('["mocha"]  (missing -> 0, not an error)', orders_per_drink["mocha"])
    show(".total()", orders_per_drink.total())

    print()
    print("  Counting rows is rarely what you want. Count CUPS by adding quantities:")
    cups = Counter()
    for _, drink, _, _, qty in ORDERS:
        cups[drink] += qty
    show("cups per drink", cups)
    show("cups.most_common(1)", cups.most_common(1))

    print()
    print("  Counter arithmetic. Last week's cups vs this week's:")
    last_week = Counter({"latte": 6, "espresso": 5, "cappuccino": 4, "flat white": 5, "tea": 2})
    show("cups - last_week  (only what grew)", cups - last_week)
    show("last_week - cups  (only what shrank)", last_week - cups)
    show("cups + last_week", cups + last_week)


# ---------------------------------------------------------------------------
# 5. Grouping: defaultdict
# ---------------------------------------------------------------------------


def section_defaultdict():
    heading("5. Grouping: defaultdict")

    print("  Grouping by hand: check, create an empty list, then append.")
    by_date_hand = {}
    for order in ORDERS:
        date = order[0]
        if date not in by_date_hand:
            by_date_hand[date] = []
        by_date_hand[date].append(order[1])
    show("by_date_hand['2026-09-08']", by_date_hand["2026-09-08"])

    print()
    print("  defaultdict(list) does the 'if not there, make an empty list' for you:")
    by_date = defaultdict(list)
    for date, drink, *_ in ORDERS:
        by_date[date].append(drink)
    for date, drinks in by_date.items():
        print(f"    {date}  {len(drinks)} orders  {drinks}")

    print()
    print("  defaultdict(int) is 'if not there, start at 0'. Revenue per drink:")
    revenue = defaultdict(int)
    for _, drink, _, price, qty in ORDERS:
        revenue[drink] += price * qty
    for drink, cents in sorted(revenue.items(), key=lambda item: item[1], reverse=True):
        print(f"    {drink:<12} {pounds(cents):>7}")

    print()
    print("  The trap: READING a missing key creates it.")
    show("len(revenue) before", len(revenue))
    show('revenue["mocha"]', revenue["mocha"])
    show("len(revenue) after  (oops)", len(revenue))
    show('"mocha" in revenue', "mocha" in revenue)
    print("  When you've finished building, turn it into a plain dict:")
    plain = dict(revenue)
    show("type(dict(revenue)).__name__", type(plain).__name__)


# ---------------------------------------------------------------------------
# 6. Nested dicts vs tuple keys
# ---------------------------------------------------------------------------


def section_nested():
    heading("6. Nested dicts vs tuple keys")

    print("  Yesterday's shape: one flat dict, (date, drink) tuple keys.")
    flat = defaultdict(int)
    for date, drink, _, price, qty in ORDERS:
        flat[(date, drink)] += price * qty
    show("flat[('2026-09-08', 'latte')]", flat[("2026-09-08", "latte")])
    show("len(flat)", len(flat))

    print()
    print("  The nested shape: a dict of dicts, date -> drink -> cents.")
    nested = defaultdict(dict)
    for date, drink, _, price, qty in ORDERS:
        nested[date][drink] = nested[date].get(drink, 0) + price * qty
    show("nested['2026-09-08']", nested["2026-09-08"])
    show("nested['2026-09-08']['latte']", nested["2026-09-08"]["latte"])
    show("sum(nested['2026-09-08'].values())", sum(nested["2026-09-08"].values()))

    print()
    print("  Same facts, two shapes. Flat is easier to build and sort;")
    print("  nested is easier when you keep asking 'everything about one date'.")
    print("  Pick the shape that matches the question you'll ask most.")


# ---------------------------------------------------------------------------
# 7. Lookups as joins
# ---------------------------------------------------------------------------


def section_joins():
    heading("7. Lookups as joins")

    print("  ORDERS knows drinks. CATEGORY knows what kind of drink each one is.")
    print("  Joining them is one .get per row:")
    cups_by_category = Counter()
    revenue_by_category = defaultdict(int)
    for _, drink, _, price, qty in ORDERS:
        category = CATEGORY.get(drink, "unknown")
        cups_by_category[category] += qty
        revenue_by_category[category] += price * qty
    for category, cups in cups_by_category.most_common():
        print(f"    {category:<14} {cups:>3} cups  {pounds(revenue_by_category[category]):>7}")

    print()
    print("  The same trick cleans data. Look the messy name up; fall back to itself:")
    typed = ["latte", "flat-white", "CAPP ", "tea", "Flatwhite", "mocha"]
    for raw in typed:
        key = raw.strip().lower()
        clean = ALIASES.get(key, key)
        print(f"    {raw!r:<14} -> {clean!r}")
    print("  .get(key, key) means 'translate it if you know it, otherwise leave it alone'.")


# ---------------------------------------------------------------------------
# 8. Putting it together
# ---------------------------------------------------------------------------


def section_week_of_orders():
    heading("8. Putting it together: five questions, five dicts")

    cups = Counter()
    revenue = defaultdict(int)
    by_date = defaultdict(list)
    sizes_per_drink = defaultdict(Counter)
    for date, drink, size, price, qty in ORDERS:
        cups[drink] += qty
        revenue[drink] += price * qty
        by_date[date].append((drink, price * qty))
        sizes_per_drink[drink][size] += qty

    print("  Cups per drink, most popular first:")
    for drink, n in cups.most_common():
        print(f"    {drink:<12} {n:>3}")

    print()
    print("  Revenue per drink, biggest first:")
    for drink, cents in sorted(revenue.items(), key=lambda item: item[1], reverse=True):
        print(f"    {drink:<12} {pounds(cents):>7}")

    print()
    print("  Revenue per day, with the day's best seller:")
    for date in sorted(by_date):
        day_total = sum(cents for _, cents in by_date[date])
        best_drink, best_cents = max(by_date[date], key=lambda item: item[1])
        print(f"    {date}  {pounds(day_total):>7}   best line: {best_drink} {pounds(best_cents)}")

    print()
    print("  Which sizes does each drink sell in? (a dict of Counters)")
    for drink in sorted(sizes_per_drink):
        print(f"    {drink:<12} {dict(sizes_per_drink[drink])}")

    print()
    cups_per_day = Counter()
    for date, _, _, _, qty in ORDERS:
        cups_per_day[date] += qty
    busiest_date, busiest_n = cups_per_day.most_common(1)[0]
    print(f"  Cups per day: {dict(cups_per_day)}")
    print(f"  Busiest day: {busiest_date} ({busiest_n} cups)")
    print(f"  Week total: {pounds(sum(revenue.values()))} across {cups.total()} cups")


def main():
    print("=" * 66)
    print("  Lesson 005: Dictionaries as tiny databases")
    print("=" * 66)
    section_lookup_table()
    section_get()
    section_changing_and_looping()
    section_counter()
    section_defaultdict()
    section_nested()
    section_joins()
    section_week_of_orders()
    print()
    print("Look up, count, group, join. That's most of what a database does. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
