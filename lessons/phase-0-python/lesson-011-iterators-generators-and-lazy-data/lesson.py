"""
Lesson 011 - Iterators, generators and lazy data

Find out what a for loop really does, why a file or a DictReader can only be
looped over once, write generator functions with `yield`, compare generator
expressions with list comprehensions, build a pipeline of small lazy steps,
meet the handful of itertools you'll actually use, and process a till export
far bigger than we need to hold in memory, one row at a time.

Run it with:

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script WRITES one file, output/till_year.csv (about 8 MB), next to itself,
the first time it runs. Delete it whenever you like; it will be made again.
"""

import csv
import itertools
import random
import sys
import tracemalloc
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders.csv"                  # the clean week, 15 rows
OUTPUT_DIR = HERE / "output"
BIG_CSV = OUTPUT_DIR / "till_year.csv"            # a made-up year of orders, 250,000 rows
BIG_ROWS = 250_000

MENU = {"espresso": 220, "tea": 250, "cappuccino": 370, "latte": 380, "flat white": 390}
SIZE_EXTRA = {"small": -50, "medium": 0, "large": 50}


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def pounds(cents):
    return f"£{cents / 100:,.2f}"


# ---------------------------------------------------------------------------
# 1. What a for loop really does
# ---------------------------------------------------------------------------


def section_for_loop():
    heading("1. What a for loop really does")

    drinks = ["latte", "tea", "espresso"]
    it = iter(drinks)                      # ask the list for an iterator
    show("it = iter(drinks); type(it).__name__", type(it).__name__)
    show("next(it)", next(it))
    show("next(it)", next(it))
    show("next(it)", next(it))
    try:
        next(it)
    except StopIteration:
        print("  next(it)                                       -> StopIteration (nothing left)")

    print()
    print("  `for drink in drinks:` is exactly that: iter() once, next() until StopIteration.")
    print("  Anything you can call iter() on is an ITERABLE: lists, strings, dicts, files...")
    show("list(iter('tea'))", list(iter("tea")))
    show("list(iter({'latte': 380, 'tea': 250}))", list(iter({"latte": 380, "tea": 250})))


# ---------------------------------------------------------------------------
# 2. Iterators get used up
# ---------------------------------------------------------------------------


def section_used_up():
    heading("2. Iterators get used up")

    with open(ORDERS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_pass = sum(1 for _ in reader)
        second_pass = sum(1 for _ in reader)
    show("rows on the first loop over the DictReader", first_pass)
    show("rows on the second loop over the same one", second_pass)
    print("  A DictReader (like a file) is an ITERATOR: it hands out each row once, then it's")
    print("  empty. No error, just nothing. A list, by contrast, can be looped over forever.")

    print()
    it = iter([1, 2, 3])
    show("it = iter([1, 2, 3]); 2 in it", 2 in it)
    show("list(it)  (what's left after `in` looked)", list(it))
    print("  Even `in` uses it up as it searches. If you need the data twice, make a list,")
    print("  or open the file again.")


# ---------------------------------------------------------------------------
# 3. Generator functions: yield
# ---------------------------------------------------------------------------


def countdown(n):
    """A generator: each `yield` hands out one value, then pauses right here."""
    print(f"    (countdown starting from {n})")
    while n > 0:
        yield n
        n -= 1
    print("    (countdown finished)")


def read_orders(path):
    """Yield one clean order dict at a time. Never holds more than one row."""
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            yield {
                "date": raw["date"],
                "drink": raw["drink"],
                "size": raw["size"],
                "price_cents": round(float(raw["price"]) * 100),
                "quantity": int(raw["quantity"]),
            }


def section_generators():
    heading("3. Generator functions: yield")

    gen = countdown(3)
    show("gen = countdown(3); type(gen).__name__", type(gen).__name__)
    print("  Nothing printed yet! Calling a generator function runs NONE of its body.")
    print("  Now next(gen), three times, then a fourth:")
    show("next(gen)", next(gen))
    show("next(gen)", next(gen))
    show("next(gen)", next(gen))
    show("next(gen, 'done')  (a default instead of StopIteration)", next(gen, "done"))

    print()
    orders = read_orders(ORDERS_CSV)
    show("orders = read_orders(ORDERS_CSV)", type(orders).__name__)
    show("next(orders)", next(orders))
    total = sum(row["price_cents"] * row["quantity"] for row in orders)
    show("the other 14 rows, summed as they stream past", pounds(total))
    print("  One row in memory at a time. The file is closed when the generator finishes.")


# ---------------------------------------------------------------------------
# 4. Generator expressions vs list comprehensions
# ---------------------------------------------------------------------------


def section_expressions():
    heading("4. Generator expressions vs list comprehensions")

    as_list = [n * n for n in range(1_000_000)]
    as_gen = (n * n for n in range(1_000_000))
    show("sys.getsizeof([n * n for n in range(1e6)])", f"{sys.getsizeof(as_list):,} bytes")
    show("sys.getsizeof((n * n for n in range(1e6)))", f"{sys.getsizeof(as_gen):,} bytes")
    show("sum(as_list) == sum(as_gen)", sum(as_list) == sum(as_gen))
    print("  Square brackets build every value now. Round brackets promise to make them")
    print("  one at a time, on request. Same answer; wildly different memory.")

    print()
    show("sum(q for q in [2, 1, 1])  (no extra brackets)", sum(q for q in [2, 1, 1]))
    show("any(d == 'mocha' for d in MENU)", any(d == "mocha" for d in MENU))
    show("max(len(d) for d in MENU)", max(len(d) for d in MENU))
    show("range(10**12)  (a trillion, instantly)", range(10**12))
    print("  sum, any, all, max, min all take generators. range, zip, enumerate and")
    print("  dict.items() are lazy too, which is why they're cheap.")


# ---------------------------------------------------------------------------
# 5. Pipelines of small lazy steps
# ---------------------------------------------------------------------------


def only_drink(rows, drink):
    """Pass through only the rows for one drink."""
    for row in rows:
        if row["drink"] == drink:
            yield row


def with_total(rows):
    """Add a total_cents field to each row, without changing the original dict."""
    for row in rows:
        yield {**row, "total_cents": row["price_cents"] * row["quantity"]}


def section_pipeline():
    heading("5. Pipelines of small lazy steps")

    rows = read_orders(ORDERS_CSV)
    lattes = only_drink(rows, "latte")
    priced = with_total(lattes)
    print("  rows -> only_drink(latte) -> with_total. Nothing has been read yet.")
    for row in priced:
        print(f"    {row['date']}  {row['quantity']} x {row['size']:<6} {pounds(row['total_cents']):>7}")
    print("  Each row travels the WHOLE pipeline before the next one is read. Small steps,")
    print("  each easy to test on its own, and memory stays flat however big the file is.")


# ---------------------------------------------------------------------------
# 6. The itertools you'll actually use
# ---------------------------------------------------------------------------


def section_itertools():
    heading("6. The itertools you'll actually use")

    show("list(islice(read_orders(...), 2))  (peek)",
         [row["drink"] for row in itertools.islice(read_orders(ORDERS_CSV), 2)])

    week_one = ["latte", "tea"]
    week_two = ["espresso"]
    show("list(chain(week_one, week_two))", list(itertools.chain(week_one, week_two)))

    cups = [row["quantity"] for row in read_orders(ORDERS_CSV)]
    show("list(accumulate(cups))  (running total)", list(itertools.accumulate(cups)))

    print()
    print("  groupby: consecutive rows with the same key. The data MUST be sorted by it.")
    for day, rows in itertools.groupby(read_orders(ORDERS_CSV), key=lambda row: row["date"]):
        cents = sum(row["price_cents"] * row["quantity"] for row in rows)
        print(f"    {day}  {pounds(cents):>7}")
    unsorted = ["tea", "latte", "tea"]
    show("[k for k, _ in groupby(['tea','latte','tea'])]", [k for k, _ in itertools.groupby(unsorted)])
    print("  'tea' twice: groupby only merges NEIGHBOURS. Sort first, or use a dict (lesson 005).")

    print()
    show("list(zip(count(1), ['a', 'b', 'c']))", list(zip(itertools.count(1), ["a", "b", "c"])))
    print("  count() goes on forever. That's fine, because zip stops at the shortest input.")


# ---------------------------------------------------------------------------
# 7. Putting it together: a year of orders, one row at a time
# ---------------------------------------------------------------------------


def fake_orders(n, seed=11):
    """Yield n made-up orders. A generator can write a file bigger than we'd want in memory."""
    rng = random.Random(seed)
    drinks = list(MENU)
    day = date(2025, 9, 1)
    for i in range(n):
        if i and i % 700 == 0:
            day += timedelta(days=1)
        drink = rng.choice(drinks)
        size = rng.choice(list(SIZE_EXTRA))
        price = MENU[drink] + SIZE_EXTRA[size]
        yield {"date": day.isoformat(), "drink": drink, "size": size,
               "price": f"{price / 100:.2f}", "quantity": rng.choice([1, 1, 1, 2, 2, 3])}


def ensure_big_file():
    if BIG_CSV.exists():
        return False
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BIG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "drink", "size", "price", "quantity"])
        writer.writeheader()
        writer.writerows(fake_orders(BIG_ROWS))      # writerows happily takes a generator
    return True


def revenue_streaming(path):
    totals = {}
    for row in read_orders(path):
        totals[row["drink"]] = totals.get(row["drink"], 0) + row["price_cents"] * row["quantity"]
    return totals


def revenue_all_in_memory(path):
    rows = list(read_orders(path))                    # every row, at once
    totals = {}
    for row in rows:
        totals[row["drink"]] = totals.get(row["drink"], 0) + row["price_cents"] * row["quantity"]
    return totals


def peak_memory(func, *args):
    """Run func and return (result, peak memory in MB) as measured by tracemalloc."""
    tracemalloc.start()
    result = func(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1_000_000


def section_together():
    heading("7. Putting it together: a year of orders, one row at a time")

    made = ensure_big_file()
    size_mb = BIG_CSV.stat().st_size / 1_000_000
    print(f"  {BIG_CSV.relative_to(HERE)}: {size_mb:.1f} MB, {BIG_ROWS:,} rows"
          f"{' (just written)' if made else ' (already there)'}")

    streamed, streamed_mb = peak_memory(revenue_streaming, BIG_CSV)
    listed, listed_mb = peak_memory(revenue_all_in_memory, BIG_CSV)
    for drink, cents in sorted(streamed.items(), key=lambda item: item[1], reverse=True):
        print(f"    {drink:<12} {pounds(cents):>12}")
    print(f"    {'total':<12} {pounds(sum(streamed.values())):>12}")

    print()
    print(f"  Peak memory, streaming with a generator:   {streamed_mb:6.2f} MB")
    print(f"  Peak memory, list(...) of every row first: {listed_mb:6.2f} MB")
    assert streamed == listed, "both ways must give the same answer"
    assert streamed_mb * 10 < listed_mb, "streaming should use a small fraction of the memory"
    print("  Same totals. The streaming version's memory doesn't grow with the file;")
    print("  make it ten times bigger and it would still need well under 1 MB.")


def main():
    print("=" * 66)
    print("  Lesson 011: Iterators, generators and lazy data")
    print("=" * 66)
    section_for_loop()
    section_used_up()
    section_generators()
    section_expressions()
    section_pipeline()
    section_itertools()
    section_together()
    print()
    print("Ask for one value at a time, and memory stops being your problem. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
