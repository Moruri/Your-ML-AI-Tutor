"""
Lesson 023 - Distributions and summary statistics

Look at a pile of numbers and say something honest about it. Mean and median
tell different stories when the data is skewed; standard deviation and the
IQR measure spread differently; percentiles and a quick look at the shape
stop you trusting the wrong number. Uses the September shop orders, then
zooms out to daily revenue.

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
    """The clean month, with a revenue column and a few helpful bits."""
    orders = pd.read_csv(path, parse_dates=["timestamp"])
    orders["revenue"] = orders["price"] * orders["quantity"]
    return orders


# ---------------------------------------------------------------------------
# 1. A distribution is a shape, not a single number
# ---------------------------------------------------------------------------

def section_shape(orders):
    heading("1. A distribution is a shape, not a single number")
    rev = orders["revenue"]
    show("orders, count", len(rev))
    show("revenue: min, max", (float(rev.min()), float(rev.max())))
    counts, edges = np.histogram(rev, bins=[0, 3, 5, 8, 12, 20])
    bands = pd.Series(counts, index=[f"£{edges[i]:g}–{edges[i+1]:g}" for i in range(len(counts))])
    show("rough bins of order revenue", bands)
    print("  Most orders sit in the cheap-to-middling band. A few big ones stretch the")
    print("  right side. That stretched shape is a right-skewed distribution, and it")
    print("  decides which summary you should trust.")


# ---------------------------------------------------------------------------
# 2. Mean, median, and when they disagree
# ---------------------------------------------------------------------------

def section_centre(orders):
    heading("2. Mean, median, and when they disagree")
    rev = orders["revenue"]
    show("mean   (the balance point)", round(float(rev.mean()), 3))
    show("median (the middle order)", round(float(rev.median()), 3))
    show("mode of quantity  (most common)", int(orders["quantity"].mode().iloc[0]))
    print("  Mean £5.46, median £4.20. The mean is pulled up by the large orders.")
    print("  For a typical customer's bill, trust the median. For total money into the")
    print("  till (mean × count), the mean is the right tool.")

    daily = orders.set_index("timestamp")["revenue"].resample("D").sum()
    trading = daily[daily > 0]
    show("daily revenue mean / median (trading days)",
         (round(float(trading.mean()), 2), round(float(trading.median()), 2)))
    print("  Daily totals are nearly symmetric, so mean and median almost agree. The")
    print("  gap between them is a quick skew detector: big gap → skewed; small → not.")
    return rev, trading


# ---------------------------------------------------------------------------
# 3. Spread: standard deviation, IQR and the five-number summary
# ---------------------------------------------------------------------------

def section_spread(rev, trading):
    heading("3. Spread: standard deviation, IQR and the five-number summary")
    show("revenue std (sample)", round(float(rev.std()), 3))
    q1, q3 = rev.quantile(0.25), rev.quantile(0.75)
    show("Q1, Q3", (round(float(q1), 2), round(float(q3), 2)))
    show("IQR  (Q3 - Q1)", round(float(q3 - q1), 2))
    five = rev.quantile([0, 0.25, 0.5, 0.75, 1.0]).round(2)
    five.index = ["min", "Q1", "median", "Q3", "max"]
    show("five-number summary of order revenue", five)
    show("pandas describe() for order revenue", rev.describe().round(3))
    print("  std asks 'typical distance from the mean' and gets dragged by outliers,")
    print("  same as the mean. IQR is the width of the middle half: stubborn against")
    print("  extremes. Pair mean with std; pair median with IQR.")
    show("daily revenue: std and IQR",
         (round(float(trading.std()), 2),
          round(float(trading.quantile(0.75) - trading.quantile(0.25)), 2)))


# ---------------------------------------------------------------------------
# 4. Percentiles, and what 'typical' really means
# ---------------------------------------------------------------------------

def section_percentiles(rev):
    heading("4. Percentiles, and what 'typical' really means")
    pct = rev.quantile([0.10, 0.50, 0.90, 0.95, 0.99]).round(2)
    pct.index = ["p10", "p50 (median)", "p90", "p95", "p99"]
    show("order revenue percentiles", pct)
    show("share of orders under the mean (£5.46)",
         round(float((rev < rev.mean()).mean()), 3))
    print("  Half of orders are at or below £4.20 (p50). Nine in ten are at or below")
    print("  £9.97 (p90). More than half sit under the mean — another sign of a long")
    print("  right tail. Percentiles answer 'how unusual is this?' better than mean±std")
    print("  when the shape isn't a neat bell.")


# ---------------------------------------------------------------------------
# 5. Skew, and a tiny experiment with outliers
# ---------------------------------------------------------------------------

def section_skew(rev, trading):
    heading("5. Skew, and a tiny experiment with outliers")
    show("skew of order revenue", round(float(rev.skew()), 3))
    show("skew of daily trading revenue", round(float(trading.skew()), 3))
    print("  Positive skew = long right tail (mean > median). Near zero = roughly")
    print("  symmetric. Negative would mean a long left tail.")

    spiked = pd.concat([rev, pd.Series([500.0])], ignore_index=True)
    show("one £500 catering order added: new mean", round(float(spiked.mean()), 3))
    show("... new median", round(float(spiked.median()), 3))
    show("... new std", round(float(spiked.std()), 3))
    show("... new IQR", round(float(spiked.quantile(0.75) - spiked.quantile(0.25)), 2))
    print("  One wild value moved the mean and std a lot, and barely touched the")
    print("  median and IQR. That's why 'robust' summaries exist.")


# ---------------------------------------------------------------------------
# 6. Putting it together: what to tell the owner
# ---------------------------------------------------------------------------

def section_together(orders, rev, trading):
    heading("6. Putting it together: what to tell the owner")
    by_drink = orders.groupby("drink")["revenue"].agg(
        n="count", mean="mean", median="median", iqr=lambda s: s.quantile(0.75) - s.quantile(0.25)
    ).round(2).sort_values("median", ascending=False)
    show("revenue per order, by drink", by_drink)
    show("busiest day's revenue", round(float(trading.max()), 2))
    show("quietest day's revenue", round(float(trading.min()), 2))
    show("median trading day", round(float(trading.median()), 2))

    assert len(orders) == 1214
    assert abs(rev.mean() - 5.459) < 0.01
    assert abs(rev.median() - 4.2) < 0.01
    assert rev.mean() > rev.median()
    assert rev.skew() > 1.0
    assert abs(trading.mean() - trading.median()) < 15
    assert by_drink.loc["espresso", "median"] < by_drink.loc["latte", "median"]
    print("  Honest summary: a typical order is about £4.20; the average is higher")
    print("  (£5.46) because of multi-cup tickets. A typical trading day takes about")
    print(f"  £{trading.median():.0f}, swinging roughly £{trading.std():.0f} either side.")


def main():
    print("=" * 66)
    print("  Lesson 023: Distributions and summary statistics")
    print("=" * 66)
    orders = load_orders(ORDERS_CSV)
    section_shape(orders)
    rev, trading = section_centre(orders)
    section_spread(rev, trading)
    section_percentiles(rev)
    section_skew(rev, trading)
    section_together(orders, rev, trading)
    print()
    print("Mean for totals, median for 'typical', IQR when outliers lurk, and always")
    print("glance at the shape before you trust any single number. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
