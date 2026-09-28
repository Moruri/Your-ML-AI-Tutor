"""
Lesson 018 - Cleaning data: missing values, types and duplicates

The same September as lesson 017, but as it really arrived from the till:
blanks, stray spaces, three spellings of 'latte', prices with £ signs,
quantities written as words, and rows exported twice. Find each problem with
pandas before fixing it, fix what can be fixed honestly, refuse to invent
what can't, and check at the end that the clean table is exactly right.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes one file, output/orders_september_clean.csv, next to itself.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_CSV = HERE / "orders_september_raw.csv"
OUTPUT_DIR = HERE / "output"
CLEAN_CSV = OUTPUT_DIR / "orders_september_clean.csv"

MEDIUM_PRICE = {"latte": 3.80, "cappuccino": 3.70, "flat white": 3.90, "espresso": 2.20, "tea": 2.50}
SIZE_EXTRA = {"small": -0.50, "medium": 0.00, "large": 0.50}
OAT_EXTRA = 0.40
DRINK_ALIASES = {"capp": "cappuccino", "flat-white": "flat white"}
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4"}

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 12)


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


# ---------------------------------------------------------------------------
# 1. First look: shape, dtypes, and a surprise
# ---------------------------------------------------------------------------


def section_first_look():
    heading("1. First look: shape, dtypes, and a surprise")

    raw = pd.read_csv(RAW_CSV)
    show("raw.shape", raw.shape)
    show("raw.dtypes", raw.dtypes)
    print("  price and quantity are 'object', not numbers: something in each column")
    print("  isn't a number. And 1,229 rows, when lesson 017's month had 1,214.")
    show("raw['drink'].unique()", pd.Series(sorted(raw["drink"].unique())))
    print("  Five drinks on the menu, sixteen spellings in the file.")
    return raw


# ---------------------------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------------------------


def section_missing(raw):
    heading("2. Missing values")

    show("raw.isna().sum()  (missing per column)", raw.isna().sum())
    show("raw.isna().any(axis=1).sum()  (rows with any gap)", int(raw.isna().any(axis=1).sum()))
    show("raw[raw['price'].isna()].head(3)", raw[raw["price"].isna()].head(3))
    print("  read_csv turned every empty cell into NaN. isna() gives a True/False table;")
    print("  .sum() counts the Trues per column (lesson 015's reductions, with labels).")

    print()
    milk_gaps = raw.loc[raw["milk"].isna(), "drink"].str.strip().str.lower().value_counts()
    show("drinks where milk is missing", milk_gaps)
    print("  Every missing milk is an espresso or a tea: drinks that never have milk. The")
    print("  gap means 'none', written as nothing. Knowing WHY a value is missing tells")
    print("  you what to do about it. (Section 6 does it.)")

    print()
    s = pd.Series([2.0, np.nan, 4.0])
    show("pd.Series([2, NaN, 4]).mean()  (NaN skipped)", float(s.mean()))
    show("np.array([2, nan, 4]).mean()   (NaN spreads)", float(np.array([2, np.nan, 4]).mean()))
    show("np.nan == np.nan", bool(np.nan == np.nan))
    print("  pandas skips NaN in sums and means; NumPy doesn't. And NaN never equals")
    print("  anything, itself included, so always test with .isna(), never == np.nan.")


# ---------------------------------------------------------------------------
# 3. Duplicates
# ---------------------------------------------------------------------------


def section_duplicates(raw):
    heading("3. Duplicates")

    show("raw.duplicated().sum()  (rows identical to an earlier row)", int(raw.duplicated().sum()))
    show("raw['order_id'].duplicated().sum()", int(raw["order_id"].duplicated().sum()))
    example = raw.loc[raw["order_id"].duplicated(keep=False)].head(4)
    show("raw.loc[raw['order_id'].duplicated(keep=False)].head(4)", example)
    print("  Each order ID should appear once. Fifteen appear twice, and the two copies are")
    print("  identical: the till exported them twice. Keeping both counts the money twice.")
    deduped = raw.drop_duplicates()
    show("raw.drop_duplicates().shape", deduped.shape)
    return deduped


# ---------------------------------------------------------------------------
# 4. Tidying text
# ---------------------------------------------------------------------------


def section_text(df):
    heading("4. Tidying text")

    before = df["drink"].nunique()
    df["drink"] = df["drink"].str.strip().str.lower().replace(DRINK_ALIASES)
    show("drink spellings before -> after", (before, df["drink"].nunique()))
    show("df['drink'].value_counts()", df["drink"].value_counts())
    unknown = set(df["drink"]) - set(MEDIUM_PRICE)
    show("drinks not on the menu", unknown)
    print("  .str gives every string method from lesson 003, applied to a whole column.")
    print("  .replace(dict) swaps known aliases. Then CHECK nothing unknown is left.")
    return df


# ---------------------------------------------------------------------------
# 5. Fixing types
# ---------------------------------------------------------------------------


def section_types(df):
    heading("5. Fixing types")

    as_number = pd.to_numeric(df["price"], errors="coerce")
    problems = df.loc[as_number.isna() & df["price"].notna(), "price"]
    show("prices that aren't numbers (and aren't blank)", problems.value_counts().head(3))
    df["price"] = pd.to_numeric(df["price"].str.replace("£", "", regex=False))
    show("price dtype after removing £", str(df["price"].dtype))

    print()
    quantity_text = df["quantity"].str.strip()
    bad = quantity_text[~quantity_text.str.isdigit()]
    show("quantities that aren't digits", bad.value_counts())
    df["quantity"] = quantity_text.replace(NUMBER_WORDS).astype(int)
    show("quantity dtype, min, max", (str(df["quantity"].dtype), int(df["quantity"].min()), int(df["quantity"].max())))

    print()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M")
    show("timestamp dtype", str(df["timestamp"].dtype))
    print("  to_numeric(errors='coerce') is the detective: whatever can't be read becomes")
    print("  NaN, so you can SEE the problem values before deciding what to do. Then fix")
    print("  them on purpose, and convert for real, letting any surprise raise an error.")
    return df


# ---------------------------------------------------------------------------
# 6. Filling gaps honestly
# ---------------------------------------------------------------------------


def expected_price(df):
    """What each row's price should be, from the menu: medium price + size extra + oat."""
    return (df["drink"].map(MEDIUM_PRICE)
            + df["size"].map(SIZE_EXTRA)
            + np.where(df["milk"] == "oat", OAT_EXTRA, 0.0)).round(2)


def section_fill(df):
    heading("6. Filling gaps honestly")

    df["milk"] = df["milk"].fillna("none")
    show("milk: filled 'none' (we know why it was blank)", df["milk"].value_counts())

    expected = expected_price(df)
    known = df["price"].notna()
    mismatches = int((~np.isclose(df.loc[known, "price"], expected[known])).sum())
    show("known prices that disagree with the menu", mismatches)
    missing = int(df["price"].isna().sum())
    df["price"] = df["price"].fillna(expected)
    show(f"price: {missing} blanks filled from the menu", int(df["price"].isna().sum()))
    print("  We only fill price from the menu because every KNOWN price matches the menu")
    print("  exactly. That check is what makes the fill defensible, not a guess.")

    print()
    mean_fill = pd.Series([3.80, np.nan, 1.70]).fillna(pd.Series([3.80, 1.70]).mean())
    show("filling a latte's missing price with the mean", mean_fill.round(2).tolist())
    print("  The lazy alternative: fill with the column mean. It 'works', and prices a")
    print("  latte at £2.75, a price nothing on the menu has. Never fill without a reason.")

    print()
    df["payment"] = df["payment"].fillna("unknown")
    show("payment: left as 'unknown' (no way to know)", df["payment"].value_counts())
    print("  We can't know how someone paid, so we don't pretend. 'unknown' keeps the row")
    print("  (its money is real) and keeps the honesty (it's not counted as card or cash).")
    return df


# ---------------------------------------------------------------------------
# 7. Putting it together: a cleaning function you can trust
# ---------------------------------------------------------------------------


def clean(raw):
    """Raw till export -> clean DataFrame. Each step is one line; the checks are at the end."""
    df = raw.drop_duplicates().copy()
    df["drink"] = df["drink"].str.strip().str.lower().replace(DRINK_ALIASES)
    df["price"] = pd.to_numeric(df["price"].str.replace("£", "", regex=False))
    df["quantity"] = df["quantity"].str.strip().replace(NUMBER_WORDS).astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M")
    df["milk"] = df["milk"].fillna("none")
    df["price"] = df["price"].fillna(expected_price(df))
    df["payment"] = df["payment"].fillna("unknown")
    return df.reset_index(drop=True)


def check(df):
    """Raise AssertionError if the clean table breaks any rule we rely on."""
    assert df["order_id"].is_unique, "one row per order"
    assert not df[["timestamp", "drink", "size", "milk", "price", "quantity"]].isna().any().any()
    assert set(df["drink"]) <= set(MEDIUM_PRICE), "only drinks on the menu"
    assert set(df["size"]) <= set(SIZE_EXTRA)
    assert (df["quantity"] >= 1).all()
    assert np.isclose(df["price"], expected_price(df)).all(), "every price matches the menu"


def section_together():
    heading("7. Putting it together: a cleaning function you can trust")

    raw = pd.read_csv(RAW_CSV)
    orders = clean(raw)
    check(orders)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    orders.to_csv(CLEAN_CSV, index=False)

    revenue = (orders["price"] * orders["quantity"]).sum()
    print(f"  raw rows {len(raw):,} -> clean rows {len(orders):,}  (15 duplicates removed)")
    print(f"  revenue £{revenue:,.2f} from {orders['quantity'].sum():,} cups")
    show("orders.dtypes", orders.dtypes)
    print(f"  Written to {CLEAN_CSV.relative_to(HERE)}.")

    assert len(orders) == 1214 and round(revenue, 2) == 6627.50
    print("  1,214 orders and £6,627.50: exactly lesson 017's month. Every problem fixed,")
    print("  nothing invented, and the checks in check() will shout if next month's file")
    print("  brings a problem we haven't seen yet.")


def main():
    print("=" * 66)
    print("  Lesson 018: Cleaning data: missing values, types and duplicates")
    print("=" * 66)
    raw = section_first_look()
    section_missing(raw)
    df = section_duplicates(raw).copy()
    df = section_text(df)
    df = section_types(df)
    section_fill(df)
    section_together()
    print()
    print("Find it, understand it, fix it on purpose, then check. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
