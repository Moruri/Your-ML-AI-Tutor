"""
Lesson 022 - Plotting that tells the truth

Make the four charts you'll use most (line, bar, scatter, histogram) with
matplotlib, each labelled so it stands on its own: a title that says what it
shows, axis labels with units, bars that start at zero. Shows how the same
numbers can mislead (a truncated axis, a bad bin count) and finishes with a
one-page September summary for the owner.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes PNG files to output/ (next to this file). Open them to see
the charts. The folder is ignored by git.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                  # draw to files, no window needed (works everywhere)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders_september.csv"      # the clean September from lessons 017-021
OUT = HERE / "output"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SOURCE = "Source: till export, September 2026"

pd.set_option("display.width", 110)


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
    """The clean month, timestamps parsed, with a revenue column."""
    orders = pd.read_csv(path, parse_dates=["timestamp"])
    orders["revenue"] = orders["price"] * orders["quantity"]
    return orders


def save(fig, name):
    """Save a figure to output/, close it, and report the file."""
    path = OUT / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)                     # free the memory; matters when you make many
    show(f"saved output/{name}", f"{path.stat().st_size:,} bytes")
    return path


# ---------------------------------------------------------------------------
# 1. A figure, an axes, and the labels that make it stand alone
# ---------------------------------------------------------------------------

def section_anatomy():
    heading("1. A figure, an axes, and the labels that make it stand alone")
    x = np.linspace(0, 4, 5)           # 5 cups: 0, 1, 2, 3, 4
    y = 3.80 * x                        # the bill for that many lattes
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o")
    ax.set_title("A latte order costs £3.80 per cup")
    ax.set_xlabel("Cups in the order")
    ax.set_ylabel("Bill (£)")
    ax.grid(alpha=0.3)
    show("fig, ax = plt.subplots()  ->  ax is a", type(ax).__name__)
    print("  fig = the whole image. ax = one chart inside it. You draw on ax, save fig.")
    print("  Title says what it shows; axis labels say what AND in what units.")
    return save(fig, "01_anatomy.png")


# ---------------------------------------------------------------------------
# 2. Line charts: change over time
# ---------------------------------------------------------------------------

def daily_revenue(orders):
    daily = orders.set_index("timestamp")["revenue"].resample("D").sum()
    return daily[daily > 0]            # trading days only (lesson 021)


def section_line(orders):
    heading("2. Line charts: change over time")
    trading = daily_revenue(orders)
    smooth = trading.rolling(5).mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(trading.index, trading, marker="o", linewidth=1, alpha=0.6, label="Daily revenue")
    ax.plot(smooth.index, smooth, linewidth=2.5, label="5-trading-day average")
    ax.annotate("Two rainy days", xy=(pd.Timestamp("2026-09-16 12:00"), 236),
                xytext=(pd.Timestamp("2026-09-10"), 215),
                arrowprops={"arrowstyle": "->"})
    ax.set_title("Daily revenue, September 2026 (trading days only)")
    ax.set_ylabel("Revenue (£)")
    ax.set_ylim(0, trading.max() * 1.15)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()                # tilt the dates so they don't overlap
    fig.text(0.01, -0.02, SOURCE, fontsize=8, color="grey")
    show("points plotted", len(trading))
    print("  One line per series, a legend to tell them apart, and an annotation for the")
    print("  one thing you want the reader to notice. The line skips weekends because")
    print("  there are no points there; it doesn't mean Saturday sold £300.")
    return save(fig, "02_line_daily_revenue.png")


# ---------------------------------------------------------------------------
# 3. Bar charts: comparing categories, and the axis that lies
# ---------------------------------------------------------------------------

def section_bar(orders):
    heading("3. Bar charts: comparing categories, and the axis that lies")
    by_drink = orders.groupby("drink")["revenue"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(by_drink.index, by_drink.values)
    for y, value in enumerate(by_drink.values):
        ax.text(value + 20, y, f"£{value:,.0f}", va="center")
    ax.set_title("Revenue by drink, September 2026")
    ax.set_xlabel("Revenue (£)")
    ax.set_xlim(0, by_drink.max() * 1.18)
    show("revenue by drink (sorted)", by_drink)
    print("  Sorted bars make the ranking obvious; horizontal bars give long names room.")
    save(fig, "03_bar_by_drink.png")

    trading = daily_revenue(orders)
    per_weekday = trading.groupby(trading.index.day_name()).mean().reindex(WEEKDAYS)
    fig, (bad, good) = plt.subplots(1, 2, figsize=(10, 4))
    for ax in (bad, good):
        ax.bar([d[:3] for d in WEEKDAYS], per_weekday.values)
        ax.set_ylabel("Average revenue per day (£)")
    bad.set_ylim(270, 370)
    bad.set_title("Misleading: axis starts at £270")
    good.set_ylim(0, 400)
    good.set_title("Honest: axis starts at £0")
    fig.suptitle("Average takings by weekday, September 2026")
    real = per_weekday["Friday"] / per_weekday["Wednesday"]
    looks = (per_weekday["Friday"] - 270) / (per_weekday["Wednesday"] - 270)
    show("Friday / Wednesday, real ratio", round(float(real), 2))
    show("Friday / Wednesday, bar heights on the £270 axis", round(float(looks), 1))
    print("  A bar's LENGTH is the message. Cut the axis and a 34% difference looks 20 times bigger.")
    print("  Bars start at zero. (Lines may zoom in; bars may not.)")
    save(fig, "04_bar_truncated_vs_honest.png")
    return real, looks


# ---------------------------------------------------------------------------
# 4. Scatter plots: two numbers per thing
# ---------------------------------------------------------------------------

def section_scatter(orders):
    heading("4. Scatter plots: two numbers per thing")
    per_day = (orders.set_index("timestamp")
               .resample("D").agg({"order_id": "count", "revenue": "sum"}))
    per_day = per_day[per_day["order_id"] > 0]
    x, y = per_day["order_id"].to_numpy(), per_day["revenue"].to_numpy()
    slope, intercept = np.polyfit(x, y, deg=1)     # the best straight line (more in Phase 2)
    line_x = np.linspace(x.min(), x.max(), 50)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, alpha=0.7)
    ax.plot(line_x, slope * line_x + intercept, color="grey", linestyle="--",
            label=f"trend: about £{slope:.2f} per extra order")
    ax.set_title("Busier days take more money (one dot per trading day)")
    ax.set_xlabel("Orders that day")
    ax.set_ylabel("Revenue that day (£)")
    ax.legend()
    ax.grid(alpha=0.3)
    show("trading days (dots)", len(per_day))
    show("slope of the trend line (£ per order)", round(float(slope), 2))
    show("correlation (lesson 024 explains it)", round(float(np.corrcoef(x, y)[0, 1]), 3))
    print("  np.linspace gives 50 evenly spaced x values to draw the trend line through.")
    save(fig, "05_scatter_orders_vs_revenue.png")
    return slope


# ---------------------------------------------------------------------------
# 5. Histograms: the shape of one number
# ---------------------------------------------------------------------------

def section_hist(orders):
    heading("5. Histograms: the shape of one number")
    values = orders["revenue"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
    for ax, bins, title in [(axes[0], 4, "4 bins: too few"),
                            (axes[1], np.arange(0, 21, 1), "£1 bins: about right"),
                            (axes[2], 200, "200 bins: too many")]:
        ax.hist(values, bins=bins, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("Order value (£)")
    axes[0].set_ylabel("Number of orders")
    fig.suptitle("How much is one order worth? The same 1,214 orders, three ways")
    counts, edges = np.histogram(values, bins=np.arange(0, 21, 1))
    top = counts.argmax()
    show("most common £1 band", f"£{edges[top]:.0f}-£{edges[top + 1]:.0f}: {counts[top]} orders")
    show("median order value", round(float(values.median()), 2))
    show("orders over £10", int((values > 10).sum()))
    print("  A histogram counts how many values land in each band (bin). The bin width")
    print("  changes the picture: pick one that means something (here, £1) and say so.")
    save(fig, "06_hist_order_values.png")
    return counts


# ---------------------------------------------------------------------------
# 6. Putting it together: one page for the owner
# ---------------------------------------------------------------------------

def section_together(orders):
    heading("6. Putting it together: one page for the owner")
    trading = daily_revenue(orders)
    by_drink = orders.groupby("drink")["revenue"].sum().sort_values()
    by_hour = orders.groupby(orders["timestamp"].dt.hour)["revenue"].sum()
    milky = orders[orders["milk"] != "none"]
    oat = (milky.set_index("timestamp")["milk"] == "oat").resample("W-SUN").mean()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(trading.index, trading, marker="o", alpha=0.6)
    ax.plot(trading.index, trading.rolling(5).mean(), linewidth=2.5)
    ax.set(title="Daily revenue (line: 5-day average)", ylabel="Revenue (£)",
           ylim=(0, trading.max() * 1.15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))   # "07 Sep", not the full date
    ax = axes[0, 1]
    ax.barh(by_drink.index, by_drink.values)
    ax.set(title="Revenue by drink", xlabel="Revenue (£)")
    ax = axes[1, 0]
    ax.bar(by_hour.index, by_hour.values)
    ax.set(title="Revenue by hour of day (whole month)", xlabel="Hour the order was placed",
           ylabel="Revenue (£)", xticks=by_hour.index)
    ax = axes[1, 1]
    labels = oat.index.strftime("w/e %d %b")
    ax.bar(labels, oat.values * 100)
    ax.set(title="Oat milk share of milky orders, by week", ylabel="Share (%)", ylim=(0, 35))
    ax.tick_params(axis="x", labelrotation=30)
    for axis in axes.flat:
        axis.grid(alpha=0.3)
    fig.suptitle(f"The coffee shop, September 2026: £{orders['revenue'].sum():,.2f} "
                 f"from {len(orders):,} orders", fontsize=14)
    fig.tight_layout()
    fig.text(0.01, -0.01, SOURCE + ". Weeks end Sunday; the last week is Mon-Wed only.",
             fontsize=8, color="grey")
    print("  Four charts, each with its own title and units, one headline on top, and a")
    print("  footnote for the caveat (the short last week) instead of hiding it.")
    return save(fig, "07_september_summary.png")


def main():
    print("=" * 66)
    print("  Lesson 022: Plotting that tells the truth")
    print("=" * 66)
    OUT.mkdir(exist_ok=True)
    orders = load_orders(ORDERS_CSV)
    section_anatomy()
    section_line(orders)
    real, looks = section_bar(orders)
    slope = section_scatter(orders)
    counts = section_hist(orders)
    section_together(orders)

    pngs = sorted(OUT.glob("*.png"))
    assert len(pngs) >= 7 and all(p.stat().st_size > 5000 for p in pngs)
    assert 1.3 < real < 1.4 and looks > 10
    assert slope > 0
    assert counts.sum() == len(orders)
    print()
    print(f"Seven charts in {OUT.name}/. Open them. A chart should make one point, honestly,")
    print("and make sense to someone who never saw your code. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
