"""
Lesson 017 - pandas: Series and DataFrames

Meet pandas' two building blocks: the Series (a NumPy array with labels) and
the DataFrame (a table of Series that share one set of row labels). Load a
month of till data with read_csv, get to know it with head, info and
describe, pick out columns and rows (by label, by position, and with the
boolean masks from lesson 015), add a column, and answer the owner's first
questions about September.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders_september.csv"      # every order in September 2026, weekdays only

pd.set_option("display.width", 100)             # let tables use a wider terminal
pd.set_option("display.max_columns", 12)


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    """Print a label, then the value. Tables and Series get their own indented block."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        print(f"  {label}:")
        for line in value.to_string().splitlines():
            print(f"      {line}")
    else:
        print(f"  {label:<46} -> {value!r}")


# ---------------------------------------------------------------------------
# 1. A Series is an array with labels
# ---------------------------------------------------------------------------


def section_series():
    heading("1. A Series is an array with labels")

    prices = pd.Series([3.80, 2.20, 3.70, 3.90, 2.50],
                       index=["latte", "espresso", "cappuccino", "flat white", "tea"],
                       name="medium_price")
    show("prices", prices)
    show("prices['latte']", float(prices["latte"]))
    show("prices.index", list(prices.index))
    show("prices.values  (the NumPy array underneath)", prices.values)
    show("prices * 1.05  (vectorised, labels kept)", (prices * 1.05).round(2))
    show("prices[prices > 3]  (a mask, as in lesson 015)", prices[prices > 3])

    print()
    cups = pd.Series({"tea": 2, "latte": 8, "espresso": 4})      # different order, fewer drinks
    show("cups  (made from a dict)", cups)
    show("prices * cups  (lined up by LABEL)", prices * cups)
    print("  pandas matched tea with tea and latte with latte, whatever the order. Labels")
    print("  with no partner (cappuccino, flat white) give NaN: 'not a number', i.e. missing.")


# ---------------------------------------------------------------------------
# 2. A DataFrame is a table of Series
# ---------------------------------------------------------------------------


def section_dataframe():
    heading("2. A DataFrame is a table of Series")

    week = pd.DataFrame({
        "drink": ["latte", "espresso", "cappuccino", "latte", "flat white"],
        "size": ["medium", "small", "large", "large", "medium"],
        "price": [3.80, 2.20, 4.20, 4.30, 3.90],
        "quantity": [2, 1, 1, 1, 3],
    })
    show("week", week)
    show("week.shape  (rows, columns)", week.shape)
    show("list(week.columns)", list(week.columns))
    show("week.index", week.index)
    show("type(week['price']).__name__", type(week["price"]).__name__)
    show("week.dtypes", week.dtypes)
    print("  Each column is a Series with its own dtype. All share one index: the row labels")
    print("  down the left (0, 1, 2... unless you choose otherwise).")


# ---------------------------------------------------------------------------
# 3. Loading a CSV and getting to know it
# ---------------------------------------------------------------------------


def section_read_csv():
    heading("3. Loading a CSV and getting to know it")

    orders = pd.read_csv(ORDERS_CSV)
    show("orders = pd.read_csv(ORDERS_CSV); orders.shape", orders.shape)
    show("orders.head()", orders.head())
    show("orders.tail(3)", orders.tail(3))

    print()
    print("  orders.info():")
    info_lines = []

    class Collect:                                   # info() prints; catch its lines to indent them
        def write(self, text):
            info_lines.append(text)

    orders.info(buf=Collect())
    for line in "".join(info_lines).splitlines():
        print(f"      {line}")

    print()
    show("orders.describe()  (numeric columns)", orders.describe().round(2))
    show("orders.describe(include='object')  (text columns)", orders.describe(include="object"))
    print("  shape, head, info, describe: four calls, and you know the size, the columns,")
    print("  the types, the gaps, and the rough range of every number. Do this every time.")
    return orders


# ---------------------------------------------------------------------------
# 4. Choosing columns
# ---------------------------------------------------------------------------


def section_columns(orders):
    heading("4. Choosing columns")

    show("orders['drink'].head(3)  (one column: a Series)", orders["drink"].head(3))
    show("orders[['drink', 'price']].head(3)  (a list: a DataFrame)", orders[["drink", "price"]].head(3))
    show("orders.price.head(3)  (attribute style also works)", orders.price.head(3))
    print("  Single brackets and a name: a Series. Double brackets (a list of names): a table.")
    print("  orders.price is handy for typing, but breaks for names with spaces, and never")
    print("  use it to CREATE a column. orders['name'] always works.")


# ---------------------------------------------------------------------------
# 5. Choosing rows: masks, .loc and .iloc
# ---------------------------------------------------------------------------


def section_rows(orders):
    heading("5. Choosing rows: masks, .loc and .iloc")

    lattes = orders[orders["drink"] == "latte"]
    show("orders[orders['drink'] == 'latte'].shape", lattes.shape)
    big_oat = orders[(orders["milk"] == "oat") & (orders["quantity"] >= 3)]
    show("oat orders of 3+ cups  (& and brackets, as in 015)", big_oat.head(4))
    show("orders[orders['drink'].isin(['tea', 'espresso'])].shape",
         orders[orders["drink"].isin(["tea", "espresso"])].shape)

    print()
    show("orders.loc[0]  (the row LABELLED 0)", orders.loc[0])
    show("orders.loc[orders['quantity'] == 4, ['order_id', 'drink', 'quantity']].head(3)",
         orders.loc[orders["quantity"] == 4, ["order_id", "drink", "quantity"]].head(3))
    show("orders.iloc[-1]  (the LAST row, by position)", orders.iloc[-1])
    show("orders.iloc[:3, :4]  (first 3 rows, first 4 columns)", orders.iloc[:3, :4])
    print("  .loc[rows, columns] works with labels and masks. .iloc[rows, columns] works with")
    print("  positions, like a NumPy array. Right now labels and positions happen to agree;")
    print("  after filtering or sorting they won't, which is why the two exist.")

    print()
    indexed = orders.set_index("order_id")
    show("orders.set_index('order_id').loc['S0100', ['drink', 'price']]",
         indexed.loc["S0100", ["drink", "price"]])


# ---------------------------------------------------------------------------
# 6. New columns and quick summaries
# ---------------------------------------------------------------------------


def section_new_columns(orders):
    heading("6. New columns and quick summaries")

    orders["revenue"] = orders["price"] * orders["quantity"]
    show("orders[['drink', 'price', 'quantity', 'revenue']].head(3)",
         orders[["drink", "price", "quantity", "revenue"]].head(3))
    show("round(orders['revenue'].sum(), 2)", round(float(orders["revenue"].sum()), 2))
    show("orders['drink'].value_counts()", orders["drink"].value_counts())
    show("orders['milk'].value_counts(normalize=True).round(3)",
         orders["milk"].value_counts(normalize=True).round(3))
    show("orders.sort_values('revenue', ascending=False).head(3)",
         orders.sort_values("revenue", ascending=False).head(3)[["order_id", "drink", "size", "quantity", "revenue"]])
    print("  A new column is one line of vectorised maths. value_counts is lesson 005's")
    print("  Counter, sorted, as a Series. sort_values sorts the whole table by one column.")


# ---------------------------------------------------------------------------
# 7. Putting it together: first questions about September
# ---------------------------------------------------------------------------


def section_together(orders):
    heading("7. Putting it together: first questions about September")

    revenue = orders["revenue"].sum()
    cups = orders["quantity"].sum()
    oat_share = (orders.loc[orders["milk"] != "none", "milk"] == "oat").mean()
    card_share = orders.loc[orders["payment"] == "card", "revenue"].sum() / revenue
    top_drink = orders.groupby("drink")["revenue"].sum().idxmax()   # a sneak peek at lesson 019
    print(f"  Orders:                 {len(orders):,}")
    print(f"  Cups:                   {cups:,}")
    print(f"  Revenue:                £{revenue:,.2f}")
    print(f"  Average order:          £{orders['revenue'].mean():.2f}  (median £{orders['revenue'].median():.2f})")
    print(f"  Oat share of milky:     {oat_share:.0%}")
    print(f"  Paid by card:           {card_share:.0%} of revenue")
    print(f"  Top drink by revenue:   {top_drink}")

    assert len(orders) == 1214, "September has 1,214 orders"
    assert round(revenue, 2) == 6627.50, "and took £6,627.50"
    print("  Seven answers, each one line. Lesson 012 needed a page of code for less.")


def main():
    print("=" * 66)
    print("  Lesson 017: pandas: Series and DataFrames")
    print("=" * 66)
    show("pd.__version__", pd.__version__)
    section_series()
    section_dataframe()
    orders = section_read_csv()
    section_columns(orders)
    section_rows(orders)
    section_new_columns(orders)
    section_together(orders)
    print()
    print("Labels on every row and column, and a whole table in one call. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
