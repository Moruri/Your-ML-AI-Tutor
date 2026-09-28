"""
Lesson 024 - Correlation, causation and the traps in between

Compute Pearson correlation, read a scatter of two columns, and practise the
habits that stop "these move together" turning into "this caused that". Uses
the September shop orders for real numbers, then a few tiny constructed
examples for the classic traps (confounders, aggregation, and a nonsense
pair that still correlates).

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders_september.csv"

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)
RNG = np.random.default_rng(24)   # reproducible toy examples (lesson 016)


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        print(f"  {label}:")
        for line in value.to_string().splitlines():
            print(f"      {line}")
    else:
        print(f"  {label:<46} -> {value!r}")


def load_orders(path):
    orders = pd.read_csv(path, parse_dates=["timestamp"])
    orders["revenue"] = orders["price"] * orders["quantity"]
    orders["size_n"] = orders["size"].map({"small": 1, "medium": 2, "large": 3})
    orders["hour"] = orders["timestamp"].dt.hour
    return orders


# ---------------------------------------------------------------------------
# 1. What correlation measures
# ---------------------------------------------------------------------------

def section_what(orders):
    heading("1. What correlation measures")
    r = orders["quantity"].corr(orders["revenue"])
    show("quantity.corr(revenue)  (Pearson r)", round(float(r), 3))
    show("same with np.corrcoef",
         round(float(np.corrcoef(orders["quantity"], orders["revenue"])[0, 1]), 3))
    print("  r lives between -1 and +1. Close to +1: when one is high, the other tends")
    print("  to be high. Close to -1: one high, the other low. Near 0: no straight-line")
    print("  link. Cups and revenue are strongly linked (r ≈ 0.86) — more cups, bigger")
    print("  bill. That is association, not yet a cause.")
    return r


# ---------------------------------------------------------------------------
# 2. Reading a few real pairs from the shop
# ---------------------------------------------------------------------------

def section_shop_pairs(orders):
    heading("2. Reading a few real pairs from the shop")
    pairs = {
        "quantity vs revenue": orders["quantity"].corr(orders["revenue"]),
        "size_n vs price": orders["size_n"].corr(orders["price"]),
        "price vs revenue": orders["price"].corr(orders["revenue"]),
        "hour vs revenue": orders["hour"].corr(orders["revenue"]),
        "quantity vs price": orders["quantity"].corr(orders["price"]),
    }
    table = pd.Series({k: round(float(v), 3) for k, v in pairs.items()})
    show("Pearson r for a few pairs", table)
    print("  Strong: cups↔revenue, size↔price (the menu really does charge more for")
    print("  large). Mild: unit price↔revenue. Near zero: hour↔revenue *per order*,")
    print("  and cups↔unit price. 'Near zero' means no straight line — the morning can")
    print("  still be the busy time once you *aggregate* (section 5).")

    by_drink = (
        orders.groupby("drink")
        .apply(lambda g: g["quantity"].corr(g["revenue"]), include_groups=False)
        .round(3)
        .sort_values(ascending=False)
    )
    show("quantity vs revenue inside each drink", by_drink)
    print("  Inside every drink the cups↔revenue link is even tighter. Espresso is")
    print("  exactly 1.0 here: one size, so revenue is just price × cups.")


# ---------------------------------------------------------------------------
# 3. Correlation is not causation: a sunny-day confounder
# ---------------------------------------------------------------------------

def section_confounder():
    heading("3. Correlation is not causation: a sunny-day confounder")
    n = 60
    sun = RNG.uniform(0, 10, size=n)                         # hours of sun
    iced = (3 + 1.2 * sun + RNG.normal(0, 1.5, size=n)).clip(0)
    tips = (40 + 4.0 * sun + RNG.normal(0, 5, size=n)).clip(0)
    # a useless 'cause': number of pigeons outside
    pigeons = RNG.integers(5, 40, size=n)

    demo = pd.DataFrame({"sun": sun, "iced_drinks": iced, "tips": tips, "pigeons": pigeons})
    show("corr iced_drinks vs tips", round(float(demo["iced_drinks"].corr(demo["tips"])), 3))
    show("corr sun vs iced_drinks", round(float(demo["sun"].corr(demo["iced_drinks"])), 3))
    show("corr sun vs tips", round(float(demo["sun"].corr(demo["tips"])), 3))
    show("corr pigeons vs tips", round(float(demo["pigeons"].corr(demo["tips"])), 3))
    print("  Iced drinks and tips move together. Did iced drinks *cause* bigger tips?")
    print("  No: sunshine drove both (a confounder). Pigeons are a nonsense partner —")
    print("  their r is near zero, as it should be. High r is a clue to investigate,")
    print("  not a verdict.")


# ---------------------------------------------------------------------------
# 4. Three more traps: range, nonsense time-twins, and reversing the arrow
# ---------------------------------------------------------------------------

def section_traps():
    heading("4. Three more traps: range, nonsense time-twins, and reversing the arrow")
    # (a) restricted range: only look at large drinks
    size = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=float)
    price = 2.0 + 0.9 * size + RNG.normal(0, 0.05, size=len(size))
    show("r size vs price (all sizes)", round(float(np.corrcoef(size, price)[0, 1]), 3))
    mask = size == 3
    # All large → size has no variance, so Pearson r is undefined (not zero).
    show("r size vs price (large only)", "undefined (no variance in size)")
    print("  Shrink the range and the correlation can vanish. Espresso in the shop has")
    print("  one size, so size↔price is undefined there — not 'no relationship'.")

    # (b) two things that both rise over time
    t = np.arange(30)
    sales = 200 + 3 * t + RNG.normal(0, 5, size=30)
    followers = 1000 + 40 * t + RNG.normal(0, 20, size=30)
    show("r daily sales vs Instagram followers",
         round(float(np.corrcoef(sales, followers)[0, 1]), 3))
    print("  Both drift up over the month, so r looks huge. Time is the hidden partner.")
    print("  Correlate *changes* (today minus yesterday), or detrend, before you brag.")

    # (c) reverse arrow
    print("  Reverse-arrow thought: 'busy hours cause more staff' sounds as plausible")
    print("  as 'more staff cause busier hours' if all you have is a correlation. The")
    print("  number cannot pick the direction. You need a story, an experiment, or")
    print("  time order that only runs one way.")


# ---------------------------------------------------------------------------
# 5. Aggregation changes the answer
# ---------------------------------------------------------------------------

def section_aggregation(orders):
    heading("5. Aggregation changes the answer")
    r_order = orders["hour"].corr(orders["revenue"])
    by_hour = orders.groupby("hour").agg(orders=("order_id", "count"),
                                         revenue=("revenue", "sum"),
                                         avg_ticket=("revenue", "mean"))
    # morning-ness as a number: treat hour as x, total revenue as y
    r_hour_totals = by_hour.reset_index()["hour"].corr(by_hour.reset_index()["revenue"])
    show("r hour vs revenue  (per order)", round(float(r_order), 3))
    show("r hour vs total revenue  (per hour-of-day)", round(float(r_hour_totals), 3))
    show("orders and revenue by hour", by_hour)
    print("  Per order, hour barely matters (a 7am latte costs what an 3pm latte costs).")
    print("  Per hour-of-day, morning hours take far more money because *more orders*")
    print("  land then. Same columns, different grain, different story. Always say")
    print("  what one row represents before you quote r.")
    return by_hour


# ---------------------------------------------------------------------------
# 6. Putting it together: what correlation is for
# ---------------------------------------------------------------------------

def section_together(orders, r_cups, by_hour):
    heading("6. Putting it together: what correlation is for")
    strong = (
        orders[["quantity", "revenue", "price", "size_n"]]
        .corr()
        .round(3)
    )
    show("correlation matrix (shop numeric columns)", strong)
    show("hour with most revenue", int(by_hour["revenue"].idxmax()))
    show("hour with least revenue", int(by_hour["revenue"].idxmin()))

    assert abs(r_cups - 0.861) < 0.01
    assert orders["size_n"].corr(orders["price"]) > 0.6
    assert abs(orders["hour"].corr(orders["revenue"])) < 0.1
    assert by_hour["revenue"].idxmax() == 8
    print("  Use correlation to find pairs worth a closer look, to sanity-check a")
    print("  merge, or to spot leakage before modelling. Do not use it as a verdict")
    print("  about causes. For that you need design, not just r.")


def main():
    print("=" * 66)
    print("  Lesson 024: Correlation, causation and the traps in between")
    print("=" * 66)
    orders = load_orders(ORDERS_CSV)
    r_cups = section_what(orders)
    section_shop_pairs(orders)
    section_confounder()
    section_traps()
    by_hour = section_aggregation(orders)
    section_together(orders, r_cups, by_hour)
    print()
    print("r measures straight-line association. Causes need a story the data alone")
    print("cannot finish. Next up: probability intuition. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
