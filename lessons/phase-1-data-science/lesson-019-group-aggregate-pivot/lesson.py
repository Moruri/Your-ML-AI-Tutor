"""
Lesson 019 - Group, aggregate, pivot

Answer "average X per Y" questions. Split a table into groups, apply a
summary to each, and combine the answers (groupby), several summaries at once
(agg), groups of groups (several keys, unstack), the spreadsheet-style
summary table (pivot_table) and its counting cousin (crosstab), and
transform, which puts each group's answer back on every row. Finishes with
the owner's weekday-and-hour questions about September.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders_september.csv"      # the clean September from lessons 017 and 018

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SIZES = ["small", "medium", "large"]

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
    """The clean month, with a revenue column and a couple of handy date parts."""
    orders = pd.read_csv(path, parse_dates=["timestamp"])
    orders["revenue"] = orders["price"] * orders["quantity"]
    orders["weekday"] = orders["timestamp"].dt.day_name()      # 'Monday', ... (lesson 021 has more)
    orders["hour"] = orders["timestamp"].dt.hour
    return orders


# ---------------------------------------------------------------------------
# 1. Split, apply, combine
# ---------------------------------------------------------------------------


def section_groupby(orders):
    heading("1. Split, apply, combine")

    by_drink = orders.groupby("drink")["revenue"].sum()
    show("orders.groupby('drink')['revenue'].sum()", by_drink)
    show("orders.groupby('drink')['quantity'].mean().round(2)", orders.groupby("drink")["quantity"].mean().round(2))
    show("orders.groupby('drink').size()  (rows per group)", orders.groupby("drink").size())
    show("...sort_values(ascending=False)", by_drink.sort_values(ascending=False))
    print("  Split the rows by drink, apply sum to each group's revenue, combine the")
    print("  answers into one Series indexed by drink. Lesson 005's dict loop, in a line.")


# ---------------------------------------------------------------------------
# 2. Several summaries at once: agg
# ---------------------------------------------------------------------------


def section_agg(orders):
    heading("2. Several summaries at once: agg")

    show("groupby('drink')['revenue'].agg(['count', 'sum', 'mean', 'median'])",
         orders.groupby("drink")["revenue"].agg(["count", "sum", "mean", "median"]).round(2))

    summary = orders.groupby("drink").agg(
        orders=("order_id", "count"),
        cups=("quantity", "sum"),
        revenue=("revenue", "sum"),
        avg_order=("revenue", "mean"),
        biggest=("revenue", "max"),
    ).round(2).sort_values("revenue", ascending=False)
    show("named aggregation: new_name=(column, function)", summary)
    print("  Named aggregation gives each result column a name you chose, from whichever")
    print("  column and function you like. It's the form to use in real code.")
    return summary


# ---------------------------------------------------------------------------
# 3. Groups of groups
# ---------------------------------------------------------------------------


def section_multi(orders):
    heading("3. Groups of groups")

    two_keys = orders.groupby(["drink", "size"])["quantity"].sum()
    show("groupby(['drink', 'size'])['quantity'].sum().head(6)", two_keys.head(6))
    print("  Two keys give a two-level index (a MultiIndex): drink, then size within it.")
    show("...unstack()  (move the inner level into columns)", two_keys.unstack()[SIZES])
    show("...reset_index().head(4)  (back to plain columns)", two_keys.reset_index().head(4))
    print("  unstack turns a long list of (drink, size) totals into a drink-by-size table.")
    print("  reset_index turns the index levels back into ordinary columns.")


# ---------------------------------------------------------------------------
# 4. pivot_table: the summary table
# ---------------------------------------------------------------------------


def section_pivot(orders):
    heading("4. pivot_table: the summary table")

    table = orders.pivot_table(index="drink", columns="size", values="revenue",
                               aggfunc="sum", margins=True, margins_name="total")
    show("pivot_table(index='drink', columns='size', values='revenue', aggfunc='sum', margins=True)",
         table[SIZES + ["total"]].round(2))
    avg = orders.pivot_table(index="weekday", columns="milk", values="price", aggfunc="mean")
    show("average price, weekday x milk", avg.reindex(WEEKDAYS).round(2))
    print("  index = rows, columns = columns, values = what to summarise, aggfunc = how.")
    print("  margins=True adds totals. reindex puts weekdays in calendar order, not A-Z.")


# ---------------------------------------------------------------------------
# 5. Counting combinations: crosstab
# ---------------------------------------------------------------------------


def section_crosstab(orders):
    heading("5. Counting combinations: crosstab")

    show("pd.crosstab(orders['drink'], orders['milk'])", pd.crosstab(orders["drink"], orders["milk"]))
    milky = orders[orders["milk"] != "none"]
    shares = pd.crosstab(milky["drink"], milky["milk"], normalize="index")
    show("same, milky drinks only, normalize='index'  (row shares)", shares.round(3))
    print("  crosstab counts how often each pair turns up. normalize='index' makes each")
    print("  row add up to 1, so you compare shares, not raw counts.")


# ---------------------------------------------------------------------------
# 6. transform: a group's answer on every row
# ---------------------------------------------------------------------------


def section_transform(orders):
    heading("6. transform: a group's answer on every row")

    orders["drink_avg"] = orders.groupby("drink")["revenue"].transform("mean")
    orders["vs_drink_avg"] = orders["revenue"] - orders["drink_avg"]
    show("orders[['drink', 'revenue', 'drink_avg', 'vs_drink_avg']].head()",
         orders[["drink", "revenue", "drink_avg", "vs_drink_avg"]].head().round(2))
    print("  agg gives ONE row per group. transform gives the group's answer back on EVERY")
    print("  row, lined up with the original, so you can compare each row with its group.")

    orders["day"] = orders["timestamp"].dt.date
    orders["share_of_day"] = orders["revenue"] / orders.groupby("day")["revenue"].transform("sum")
    show("share_of_day for each day sums to", orders.groupby("day")["share_of_day"].sum().round(6).unique())
    biggest = orders.loc[orders["share_of_day"].idxmax(), ["order_id", "day", "drink", "quantity", "revenue", "share_of_day"]]
    show("the single order that was the biggest share of its day", biggest)


# ---------------------------------------------------------------------------
# 7. Putting it together: weekdays and hours
# ---------------------------------------------------------------------------


def section_together(orders):
    heading("7. Putting it together: weekdays and hours")

    orders["day"] = orders["timestamp"].dt.date
    daily = orders.groupby(["day", "weekday"], as_index=False)["revenue"].sum()
    per_weekday = daily.groupby("weekday")["revenue"].agg(days="count", avg_day="mean").reindex(WEEKDAYS)
    show("average DAY's revenue, per weekday", per_weekday.round(2))
    print("  Two steps: total each day first, THEN average the days. Averaging the orders")
    print("  instead would answer a different question (the average order, per weekday).")

    cups = orders.pivot_table(index="hour", columns="weekday", values="quantity", aggfunc="sum")
    cups = cups[WEEKDAYS] / per_weekday["days"]                 # broadcast: divide each column by its day count
    show("average cups per hour, per weekday", cups.round(1))
    peak = cups.idxmax()
    print(f"  Peak hour every weekday: {sorted(set(peak))[0]}:00" if peak.nunique() == 1
          else f"  Peak hours: {dict(peak)}")

    best = per_weekday["avg_day"].idxmax()
    assert round(daily["revenue"].sum(), 2) == 6627.50
    assert best == "Friday", "Fridays are the busiest days"
    print(f"  Best weekday: {best}, at £{per_weekday.loc[best, 'avg_day']:.2f} on an average day.")
    print("  Every number above is a groupby or a pivot_table. 'Average X per Y' is one line,")
    print("  once you've decided exactly what X and Y are.")


def main():
    print("=" * 66)
    print("  Lesson 019: Group, aggregate, pivot")
    print("=" * 66)
    orders = load_orders(ORDERS_CSV)
    section_groupby(orders)
    section_agg(orders)
    section_multi(orders)
    section_pivot(orders)
    section_crosstab(orders)
    section_transform(orders)
    section_together(orders)
    print()
    print("Split, apply, combine: most summary questions are this one idea. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
