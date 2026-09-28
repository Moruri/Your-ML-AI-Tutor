"""
Lesson 020 - Joining tables and reshaping

Bring information from two tables together on a shared key (merge), learn
the four kinds of join and how to see which rows didn't match, protect
yourself from the join that quietly multiplies rows, stack tables on top of
each other (concat), and move between wide tables (nice to read) and long
tables (nice to compute with) with melt and pivot. Finishes with profit per
drink and cups-versus-target per week.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders_september.csv"     # the clean September
MENU_CSV = HERE / "menu.csv"                   # one row per drink: category, price, cost per cup
TARGETS_CSV = HERE / "targets_wide.csv"        # the owner's weekly cup targets, one column per week
TOTAL_PROFIT = 4936.05                         # September's revenue minus cost of the cups, checked at the end

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
    orders = pd.read_csv(path, parse_dates=["timestamp"])
    orders["revenue"] = orders["price"] * orders["quantity"]
    orders["week"] = orders["timestamp"].dt.isocalendar().week.astype(int)
    return orders


# ---------------------------------------------------------------------------
# 1. Two tables, one question
# ---------------------------------------------------------------------------


def section_why(orders, menu):
    heading("1. Two tables, one question")

    show("menu", menu)
    show("orders[['order_id', 'drink', 'quantity']].head(3)", orders[["order_id", "drink", "quantity"]].head(3))
    print("  'What did each order cost us to make?' needs the quantity from orders AND the")
    print("  cost per cup from the menu. The column they share, drink, is the KEY.")
    cost_lookup = menu.set_index("drink")["cost_per_cup"]
    show("orders['drink'].map(cost_lookup).head(3)  (a lookup)", orders["drink"].map(cost_lookup).head(3))
    print("  map() works for one column (lesson 005's dict lookup). For whole rows, merge.")


# ---------------------------------------------------------------------------
# 2. merge
# ---------------------------------------------------------------------------


def section_merge(orders, menu):
    heading("2. merge")

    joined = orders.merge(menu, on="drink", how="left", validate="many_to_one")
    show("orders.shape, menu.shape, joined.shape", (orders.shape, menu.shape, joined.shape))
    show("joined[['order_id', 'drink', 'quantity', 'category', 'cost_per_cup']].head()",
         joined[["order_id", "drink", "quantity", "category", "cost_per_cup"]].head())
    print("  Every order found its drink in the menu and picked up that row's columns.")
    print("  Same number of rows as orders: a left join keeps every row on the left.")
    print("  validate='many_to_one' checks that each drink appears ONCE in the menu.")
    return joined


# ---------------------------------------------------------------------------
# 3. The four joins, and seeing what didn't match
# ---------------------------------------------------------------------------


def section_join_types(orders, menu):
    heading("3. The four joins, and seeing what didn't match")

    sold = orders.groupby("drink", as_index=False)["quantity"].sum().rename(columns={"quantity": "cups"})
    sold = sold[sold["drink"] != "tea"]                         # pretend tea's sales went missing
    sold = pd.concat([sold, pd.DataFrame({"drink": ["chai"], "cups": [12]})], ignore_index=True)
    small_menu = menu[["drink", "cost_per_cup"]]
    show("sold  (no tea; and chai, which the menu file doesn't know yet)", sold)
    show("small_menu  (has mocha, which sold nothing)", small_menu)
    for how in ["inner", "left", "right", "outer"]:
        result = sold.merge(small_menu, on="drink", how=how)
        print(f"    how={how!r:<8} -> {len(result)} rows: {', '.join(result['drink'])}")
    print("  inner: only keys in BOTH. left: every row of the left table. right: every row")
    print("  of the right. outer: everything from either side. Missing partners get NaN.")

    checked = sold.merge(small_menu, on="drink", how="outer", indicator=True)
    show("outer join with indicator=True", checked)
    show("checked['_merge'].value_counts()", checked["_merge"].value_counts())
    print("  _merge says where each row came from. After any important join, look at it.")


# ---------------------------------------------------------------------------
# 4. Different key names, and the join that multiplies rows
# ---------------------------------------------------------------------------


def section_keys(orders, menu):
    heading("4. Different key names, and the join that multiplies rows")

    prices = menu.rename(columns={"drink": "item"})[["item", "medium_price"]]
    both = orders.merge(prices, left_on="drink", right_on="item", how="left")
    show("merge(left_on='drink', right_on='item').shape", both.shape)
    print("  When the key has different names in each table, name both.")

    print()
    oops_menu = pd.concat([menu, menu[menu["drink"] == "latte"]])   # latte listed twice by mistake
    exploded = orders.merge(oops_menu, on="drink", how="left")
    show("rows before, rows after joining a menu with latte twice", (len(orders), len(exploded)))
    show("revenue before -> after (!)", (round(float(orders["revenue"].sum()), 2), round(float(exploded["revenue"].sum()), 2)))
    try:
        orders.merge(oops_menu, on="drink", how="left", validate="many_to_one")
    except pd.errors.MergeError as err:
        print(f"    with validate='many_to_one': MergeError: {err}")
    print("  A duplicated key on the 'one' side copies every matching row. No error, just")
    print("  more rows and bigger totals. validate= turns that silent bug into a loud one.")


# ---------------------------------------------------------------------------
# 5. Stacking tables: concat
# ---------------------------------------------------------------------------


def section_concat(orders):
    heading("5. Stacking tables: concat")

    first_half = orders[orders["timestamp"] < "2026-09-16"]
    second_half = orders[orders["timestamp"] >= "2026-09-16"]
    show("len(first_half), len(second_half)", (len(first_half), len(second_half)))
    whole = pd.concat([first_half, second_half], ignore_index=True)
    show("len(pd.concat([first_half, second_half]))", len(whole))
    show("whole.equals(orders)", whole.equals(orders))
    print("  concat stacks tables with the same columns, one under another: two exports,")
    print("  two shops, two months. ignore_index=True renumbers the rows 0, 1, 2...")
    labelled = pd.concat([first_half, second_half], keys=["early", "late"], names=["half", None])
    show("with keys=: revenue per half", labelled.groupby(level="half")["revenue"].sum().round(2))


# ---------------------------------------------------------------------------
# 6. Wide and long: melt and pivot
# ---------------------------------------------------------------------------


def section_reshape(targets_wide):
    heading("6. Wide and long: melt and pivot")

    show("targets_wide  (a column per week: easy to read)", targets_wide)
    targets_long = targets_wide.melt(id_vars="drink", var_name="week", value_name="target")
    targets_long["week"] = targets_long["week"].str.removeprefix("week_").astype(int)
    show("melt(id_vars='drink', var_name='week', value_name='target').head(7)", targets_long.head(7))
    show("targets_long.shape", targets_long.shape)
    print("  Wide: one row per drink, one column per week. Long: one row per (drink, week).")
    print("  Same 25 numbers. Long is what groupby, merge and plotting libraries want.")

    back = targets_long.pivot(index="drink", columns="week", values="target")
    show("targets_long.pivot(index='drink', columns='week', values='target')", back)
    print("  pivot is melt's inverse. (pivot_table is pivot plus an aggfunc, for when a")
    print("  combination appears more than once and needs summarising.)")
    return targets_long


# ---------------------------------------------------------------------------
# 7. Putting it together: profit, and cups against target
# ---------------------------------------------------------------------------


def section_together(orders, menu, targets_long):
    heading("7. Putting it together: profit, and cups against target")

    joined = orders.merge(menu[["drink", "category", "cost_per_cup"]], on="drink",
                          how="left", validate="many_to_one")
    assert len(joined) == len(orders) and joined["cost_per_cup"].notna().all()
    joined["cost"] = joined["cost_per_cup"] * joined["quantity"]
    joined["profit"] = joined["revenue"] - joined["cost"]
    profit = joined.groupby("drink").agg(revenue=("revenue", "sum"), cost=("cost", "sum"),
                                         profit=("profit", "sum"))
    profit["margin"] = profit["profit"] / profit["revenue"]
    show("profit per drink", profit.sort_values("profit", ascending=False).round(2))

    actual = orders.groupby(["drink", "week"], as_index=False)["quantity"].sum()
    vs = actual.merge(targets_long, on=["drink", "week"], how="outer", validate="one_to_one", indicator=True)
    assert (vs["_merge"] == "both").all(), "every drink-week has both an actual and a target"
    vs["pct_of_target"] = vs["quantity"] / vs["target"]
    report = vs.pivot(index="drink", columns="week", values="pct_of_target")
    show("cups as a share of target, drink x week", (report * 100).round(0).astype(int).astype(str) + "%")

    missed = vs[vs["pct_of_target"] < 0.9]
    print(f"  Drink-weeks more than 10% under target: {len(missed)} of {len(vs)}")
    for row in missed.itertuples():
        print(f"    week {row.week}  {row.drink:<11} {row.quantity:>4} of {row.target} cups")

    total_profit = profit["profit"].sum()
    assert round(total_profit, 2) == TOTAL_PROFIT
    print(f"  September profit on drinks (before rent, wages...): £{total_profit:,.2f}")
    print("  Two joins, one melt, one pivot. Neither question could be answered from one table.")


def main():
    print("=" * 66)
    print("  Lesson 020: Joining tables and reshaping")
    print("=" * 66)
    orders = load_orders(ORDERS_CSV)
    menu = pd.read_csv(MENU_CSV)
    targets_wide = pd.read_csv(TARGETS_CSV)
    section_why(orders, menu)
    section_merge(orders, menu)
    section_join_types(orders, menu)
    section_keys(orders, menu)
    section_concat(orders)
    targets_long = section_reshape(targets_wide)
    section_together(orders, menu, targets_long)
    print()
    print("Join on a key, check what matched, and keep data long until someone needs to read it. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
