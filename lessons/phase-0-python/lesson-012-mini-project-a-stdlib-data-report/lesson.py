"""
Lesson 012 - Mini-project: a stdlib data report

Everything from Phase 0 in one small, real job. Take two weeks of messy till
exports, look at the raw file before trusting it, normalise the mess that can
be fixed, reject (and record) what can't, turn good rows into Order objects,
compute summary statistics with the standard library's `statistics` module,
and write a tidy report a shop owner could read over breakfast.

Run it with:

    python lesson.py
    python lesson.py --csv some_other_export.csv

Read it alongside README.md in this folder. Each numbered step here matches a
numbered step there.

This script WRITES two files next to itself: output/report.txt and
output/rejected_rows.csv.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_CSV = HERE / "orders_fortnight.csv"         # two weeks of till exports, 7-18 September
OUTPUT_DIR = HERE / "output"
REPORT_TXT = OUTPUT_DIR / "report.txt"
REJECTS_CSV = OUTPUT_DIR / "rejected_rows.csv"
TOTAL_CENTS = 68150                             # the fortnight's revenue, £681.50, checked at the end

MENU = {"espresso": 220, "tea": 250, "cappuccino": 370, "latte": 380, "flat white": 390}
SIZE_EXTRA = {"small": -50, "medium": 0, "large": 50}

# Mess we know how to fix: spellings the till (or the staff) use for the same drink.
DRINK_ALIASES = {"capp": "cappuccino", "flat-white": "flat white", "flatwhite": "flat white"}

# The columns we need, and the header names the till actually uses for them.
COLUMNS = {"order id": "order_id", "date": "date", "time": "time", "drink": "drink",
           "size": "size", "price": "price", "qty": "quantity", "payment": "payment"}


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def pounds(cents):
    return f"£{cents / 100:,.2f}"


# ---------------------------------------------------------------------------
# Step 1. Look before you trust
# ---------------------------------------------------------------------------


def peek(path, lines=4):
    """Print the first few raw lines exactly as they are in the file."""
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            print(f"    {number:>2} | {line.rstrip(chr(10))}")
            if number == lines:
                break


def step_look(path):
    heading("Step 1. Look before you trust")
    print(f"  {path.name}, first four lines, exactly as written:")
    peek(path)
    with open(path, encoding="utf-8") as f:
        all_lines = f.read().splitlines()
    blanks = sum(1 for line in all_lines if not line.strip())
    print(f"  {len(all_lines)} lines in total, {blanks} of them blank.")
    print("  Already visible: spaces in the header, 'Qty' not 'quantity', capital letters,")
    print("  a £ sign here and there. None of that is a reason to reject a row.")


# ---------------------------------------------------------------------------
# Step 2. Normalise what can be fixed
# ---------------------------------------------------------------------------


def normalise(raw):
    """
    One raw DictReader row -> a dict with our column names and tidy text.
    Fixes spacing, case, £ signs and known aliases. Converts nothing; rejects nothing.
    """
    row = {}
    for header, value in raw.items():
        if header is None:                         # extra fields on a long row: ignore
            continue
        name = COLUMNS.get(header.strip().lower())
        if name:
            row[name] = (value or "").strip()
    row["drink"] = row.get("drink", "").lower()
    row["drink"] = DRINK_ALIASES.get(row["drink"], row["drink"])
    row["size"] = row.get("size", "").lower()
    row["payment"] = row.get("payment", "").lower()
    row["price"] = row.get("price", "").replace("£", "")
    return row


def step_normalise(path):
    heading("Step 2. Normalise what can be fixed")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        print(f"  raw header:        {reader.fieldnames}")
        first = next(reader)
    print(f"  first raw row:     {first}")
    print(f"  after normalise(): {normalise(first)}")
    print("  Our names, no stray spaces, lower case, no £. Still all strings: converting")
    print("  and judging come next, in one place.")


# ---------------------------------------------------------------------------
# Step 3. Validate into Orders, and record the rejects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Order:
    order_id: str
    day: date
    at: time
    drink: str
    size: str
    price_cents: int
    quantity: int
    payment: str

    def total_cents(self):
        return self.price_cents * self.quantity


def to_order(row):
    """A normalised row -> an Order. Raises ValueError naming the column and the value."""
    missing = [name for name in COLUMNS.values() if not row.get(name)]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    try:
        day = date.fromisoformat(row["date"])
    except ValueError as err:
        raise ValueError(f"date {row['date']!r} is not YYYY-MM-DD") from err
    try:
        at = time.fromisoformat(row["time"])
    except ValueError as err:
        raise ValueError(f"time {row['time']!r} is not HH:MM") from err
    if row["drink"] not in MENU:
        raise ValueError(f"drink {row['drink']!r} is not on the menu")
    if row["size"] not in SIZE_EXTRA:
        raise ValueError(f"size {row['size']!r} is not one of {', '.join(SIZE_EXTRA)}")
    try:
        price_cents = round(float(row["price"]) * 100)
    except ValueError as err:
        raise ValueError(f"price {row['price']!r} is not a number") from err
    expected = MENU[row["drink"]] + SIZE_EXTRA[row["size"]]
    if price_cents != expected:
        raise ValueError(f"price {row['price']} doesn't match the menu ({pounds(expected)})")
    try:
        quantity = int(row["quantity"])
    except ValueError as err:
        raise ValueError(f"quantity {row['quantity']!r} is not a whole number") from err
    if quantity < 1:
        raise ValueError(f"quantity {quantity} must be at least 1")
    if row["payment"] not in ("card", "cash"):
        raise ValueError(f"payment {row['payment']!r} is not card or cash")
    return Order(row["order_id"], day, at, row["drink"], row["size"], price_cents, quantity, row["payment"])


def load(path):
    """Returns (orders, rejects). Skips blank lines, rejects duplicates by order id."""
    orders, rejects, seen = [], [], {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            line = reader.line_num                  # the real line in the file, blanks included
            row = normalise(raw)
            try:
                order = to_order(row)
                if order.order_id in seen:
                    raise ValueError(f"duplicate of {order.order_id} on line {seen[order.order_id]}")
            except ValueError as err:
                rejects.append({"line": line, "reason": str(err), **row})
                continue
            seen[order.order_id] = line
            orders.append(order)
    return orders, rejects


def write_rejects(rejects, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["line", "reason", *COLUMNS.values()])
        writer.writeheader()
        writer.writerows(rejects)


def step_validate(path):
    heading("Step 3. Validate into Orders, and record the rejects")
    orders, rejects = load(path)
    print(f"  {len(orders)} good orders, {len(rejects)} rejected:")
    for reject in rejects:
        print(f"    line {reject['line']:>3}  {reject['reason']}")
    write_rejects(rejects, REJECTS_CSV)
    print(f"  Written to {REJECTS_CSV.relative_to(HERE)}, with the raw values, for a human to fix.")
    return orders, rejects


# ---------------------------------------------------------------------------
# Step 4. Summary statistics
# ---------------------------------------------------------------------------


def summarise(values):
    """Count, total, mean, median, standard deviation, min and max of some numbers."""
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def step_statistics(orders):
    heading("Step 4. Summary statistics")
    spend = [order.total_cents() for order in orders]
    stats = summarise(spend)
    print("  Spend per order, in pounds:")
    for name in ["count", "mean", "median", "stdev", "min", "max"]:
        value = stats[name]
        shown = value if name == "count" else pounds(value)
        print(f"    {name:<7} {shown:>8}")
    print("  The mean is above the median: a few big orders (four lattes at once, say)")
    print("  pull the average up. Half of all orders were at or below the median.")
    quartiles = statistics.quantiles(spend, n=4)
    print(f"  Quartiles (25%, 50%, 75%): {', '.join(pounds(q) for q in quartiles)}")
    return stats


# ---------------------------------------------------------------------------
# Step 5. Break it down
# ---------------------------------------------------------------------------


def breakdowns(orders):
    by_drink, by_day, by_payment = defaultdict(int), defaultdict(int), defaultdict(int)
    cups_by_hour = Counter()
    for order in orders:
        by_drink[order.drink] += order.total_cents()
        by_day[order.day] += order.total_cents()
        by_payment[order.payment] += order.total_cents()
        cups_by_hour[order.at.hour] += order.quantity
    return by_drink, by_day, by_payment, cups_by_hour


def step_breakdowns(orders):
    heading("Step 5. Break it down")
    by_drink, by_day, by_payment, cups_by_hour = breakdowns(orders)
    best_day = max(by_day, key=by_day.get)
    busiest_hour, cups = cups_by_hour.most_common(1)[0]
    print(f"  best day:      {best_day:%A %d %B} ({pounds(by_day[best_day])})")
    print(f"  busiest hour:  {busiest_hour:02d}:00-{busiest_hour + 1:02d}:00 ({cups} cups)")
    card_share = by_payment["card"] / sum(by_payment.values())
    print(f"  paid by card:  {card_share:.0%} of revenue")
    return by_drink, by_day, by_payment, cups_by_hour


# ---------------------------------------------------------------------------
# Step 6. The report
# ---------------------------------------------------------------------------


def bar(value, biggest, width=24):
    """A text bar chart segment: # characters in proportion to value."""
    return "#" * round(width * value / biggest) if biggest else ""


def build_report(orders, rejects, stats, parts):
    by_drink, by_day, by_payment, cups_by_hour = parts
    first, last = min(o.day for o in orders), max(o.day for o in orders)
    cups = sum(o.quantity for o in orders)
    lines = [
        "COFFEE SHOP REPORT",
        f"{first:%a %d %b} to {last:%a %d %b %Y}",
        "=" * 44,
        f"Revenue          {pounds(stats['total']):>12}",
        f"Orders           {stats['count']:>12}",
        f"Cups             {cups:>12}",
        f"Average order    {pounds(stats['mean']):>12}   (median {pounds(stats['median'])})",
        f"Rows rejected    {len(rejects):>12}   (see rejected_rows.csv)",
        "",
        "Revenue by drink",
    ]
    top = max(by_drink.values())
    for drink, cents in sorted(by_drink.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"  {drink:<11} {pounds(cents):>9}  {bar(cents, top)}")
    lines += ["", "Revenue by day"]
    top = max(by_day.values())
    for day, cents in sorted(by_day.items()):
        lines.append(f"  {day:%a %d %b}  {pounds(cents):>9}  {bar(cents, top)}")
    lines += ["", "Cups by hour"]
    top = max(cups_by_hour.values())
    for hour in range(min(cups_by_hour), max(cups_by_hour) + 1):
        lines.append(f"  {hour:02d}:00  {cups_by_hour[hour]:>4}  {bar(cups_by_hour[hour], top)}")
    lines += ["", "Payment"]
    total = sum(by_payment.values())
    for method, cents in sorted(by_payment.items()):
        lines.append(f"  {method:<5} {pounds(cents):>9}  {cents / total:>4.0%}")
    return "\n".join(lines) + "\n"


def step_report(orders, rejects, stats, parts):
    heading("Step 6. The report")
    report = build_report(orders, rejects, stats, parts)
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text(report, encoding="utf-8")
    for line in report.splitlines():
        print(f"    {line}")
    print(f"  Saved to {REPORT_TXT.relative_to(HERE)}.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Two-week report from a till export.")
    parser.add_argument("--csv", type=Path, default=RAW_CSV, help="raw till export to read")
    args = parser.parse_args(argv)

    print("=" * 66)
    print("  Lesson 012: Mini-project: a stdlib data report")
    print("=" * 66)
    step_look(args.csv)
    step_normalise(args.csv)
    orders, rejects = step_validate(args.csv)
    stats = step_statistics(orders)
    parts = step_breakdowns(orders)
    step_report(orders, rejects, stats, parts)

    if args.csv == RAW_CSV:
        assert len(orders) + len(rejects) == 130, "every non-blank row is either kept or rejected"
        assert len(rejects) == 7, "six unfixable rows and one duplicate"
        assert stats["total"] == TOTAL_CENTS, "the fortnight's revenue"
        print()
        print(f"  Checked: 130 rows in, {len(orders)} kept, 7 rejected, {pounds(TOTAL_CENTS)} total.")
    print()
    print("That's Phase 0: plain Python, turning a messy file into something worth reading.")
    print()


if __name__ == "__main__":
    main()
