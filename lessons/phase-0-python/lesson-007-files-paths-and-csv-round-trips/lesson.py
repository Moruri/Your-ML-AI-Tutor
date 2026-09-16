"""
Lesson 007 - Files, paths and CSV round-trips

Find files with pathlib no matter where you run from, read and write text
without encoding surprises, read a messy till export with csv.DictReader,
clean it with the habits from lessons 003-006, write it back out with
csv.DictWriter, and prove the round trip by reading your own file back.

Run it with:

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script WRITES files. Everything it writes goes into an `output/` folder
next to this script, so nothing you were given gets touched.
"""

import csv
from pathlib import Path

# The folder this script lives in, as an absolute path. Everything else hangs
# off it, so the script works from the repo root, from this folder, or from
# anywhere else.
HERE = Path(__file__).resolve().parent

RAW_CSV = HERE / "orders_raw.csv"            # given to us; we only read it
OUTPUT_DIR = HERE / "output"                 # ours; we create it and write into it
CLEAN_CSV = OUTPUT_DIR / "orders_clean.csv"
REPORT_TXT = OUTPUT_DIR / "report.txt"

# Spellings staff actually type -> the one spelling we keep (lesson 005).
ALIASES = {
    "flat-white": "flat white",
    "flatwhite": "flat white",
    "capp": "cappuccino",
    "cap": "cappuccino",
}

# The columns of the CLEAN file, in the order we want them written.
CLEAN_FIELDS = ["date", "drink", "size", "price_cents", "quantity", "note"]


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def pounds(cents):
    """Format whole cents as pounds for printing. Maths in cents, display in pounds."""
    return f"£{cents / 100:.2f}"


# ---------------------------------------------------------------------------
# 1. Where am I? Paths with pathlib
# ---------------------------------------------------------------------------


def section_paths():
    heading("1. Where am I? Paths with pathlib")

    show("Path.cwd()  (where you RAN python from)", Path.cwd())
    show("Path('orders_raw.csv').exists()  (from cwd)", Path("orders_raw.csv").exists())
    show("Path(__file__)  (this script)", Path(__file__))
    show("HERE = Path(__file__).resolve().parent", HERE)
    print("  cwd changes with every terminal, so the bare 'orders_raw.csv' is only found")
    print("  if you happen to be standing in this folder. HERE doesn't change. Build off HERE.")

    print()
    show("RAW_CSV = HERE / 'orders_raw.csv'", RAW_CSV)
    show("RAW_CSV.name", RAW_CSV.name)
    show("RAW_CSV.stem", RAW_CSV.stem)
    show("RAW_CSV.suffix", RAW_CSV.suffix)
    show("RAW_CSV.parent.name", RAW_CSV.parent.name)
    show("RAW_CSV.with_suffix('.txt')", RAW_CSV.with_suffix(".txt").name)
    show("RAW_CSV.with_name('orders_clean.csv')", RAW_CSV.with_name("orders_clean.csv").name)

    print()
    show("RAW_CSV.exists()", RAW_CSV.exists())
    show("(HERE / 'nope.csv').exists()", (HERE / "nope.csv").exists())
    show("RAW_CSV.is_file()", RAW_CSV.is_file())
    show("HERE.is_dir()", HERE.is_dir())

    print()
    print("  What's in this folder? iterdir() lists everything, glob() filters:")
    for path in sorted(HERE.iterdir()):
        kind = "folder" if path.is_dir() else f"{path.stat().st_size:>5} bytes"
        print(f"    {path.name:<20} {kind}")
    show("[p.name for p in HERE.glob('*.csv')]", [p.name for p in HERE.glob("*.csv")])


# ---------------------------------------------------------------------------
# 2. Reading a text file
# ---------------------------------------------------------------------------


def section_read_text():
    heading("2. Reading a text file")

    text = RAW_CSV.read_text(encoding="utf-8")
    show("type(text).__name__", type(text).__name__)
    show("len(text)  (characters)", len(text))
    show("text[:40]", text[:40])

    lines = text.splitlines()
    show("len(text.splitlines())", len(lines))
    show("lines[0]  (the header)", lines[0])
    show("lines[1]", lines[1])
    show("lines[7]  (a blank line in the export)", lines[7])

    print()
    print("  The same file with open() and a with-block, one line at a time:")
    with open(RAW_CSV, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if number > 3:
                break
            show(f"line {number}, raw", line)
    print("  Each line arrives with its newline still attached. .rstrip('\\n') takes it off.")
    print("  The with-block closes the file for you when the block ends, even on an error.")

    print()
    print("  Encoding. There's a £ in this file. Look at the bytes on disk:")
    raw_bytes = RAW_CSV.read_bytes()
    show("RAW_CSV.read_bytes()[36:66]  (line 2)", raw_bytes[36:66])
    show("'£'.encode('utf-8')", "£".encode("utf-8"))
    show("b'\\xc2\\xa3'.decode('utf-8')", b"\xc2\xa3".decode("utf-8"))
    show("b'\\xc2\\xa3'.decode('cp1252')  (a Windows default)", b"\xc2\xa3".decode("cp1252"))
    print("  Same bytes, two readings. Always say encoding='utf-8' and this never happens to you.")


# ---------------------------------------------------------------------------
# 3. Writing a text file (and a folder to put it in)
# ---------------------------------------------------------------------------


def section_write_text():
    heading("3. Writing a text file (and a folder to put it in)")

    OUTPUT_DIR.mkdir(exist_ok=True)
    show("OUTPUT_DIR.mkdir(exist_ok=True); exists?", OUTPUT_DIR.exists())

    note = OUTPUT_DIR / "hello.txt"
    text = "Week of 2026-09-07\nTotal: £79.70\n"
    note.write_text(text, encoding="utf-8")
    show("note.write_text(text); note.exists()", note.exists())
    show("note.read_text()", note.read_text(encoding="utf-8"))
    show("len(text)  (characters)", len(text))
    show("note.stat().st_size  (bytes on disk)", note.stat().st_size)
    print("  33 characters, 34 bytes: £ is two bytes in UTF-8. Characters and bytes differ.")

    print()
    print("  write_text REPLACES the whole file. There is no undo:")
    note.write_text("Overwritten.\n", encoding="utf-8")
    show("after a second write_text", note.read_text(encoding="utf-8"))

    print()
    print("  For many lines, open in 'w' mode and write as you go:")
    cups = [("latte", 8), ("cappuccino", 5), ("espresso", 4)]
    with open(note, "w", encoding="utf-8") as f:
        f.write("drink,cups\n")
        for drink, n in cups:
            f.write(f"{drink},{n}\n")
    show("read back, split into lines", note.read_text(encoding="utf-8").splitlines())

    print()
    print("  'a' APPENDS to the end instead of replacing:")
    with open(note, "a", encoding="utf-8") as f:
        f.write("tea,2\n")
    show("read back again", note.read_text(encoding="utf-8").splitlines())
    print("  'r' read (the default), 'w' write-and-replace, 'a' append. Pick on purpose.")


# ---------------------------------------------------------------------------
# 4. Reading a CSV with DictReader
# ---------------------------------------------------------------------------


def read_raw_rows(path):
    """Every row of a CSV as a dict, exactly as the file has it. Strings throughout."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def section_dictreader():
    heading("4. Reading a CSV with DictReader")

    with open(RAW_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        show("reader.fieldnames", reader.fieldnames)
        rows = list(reader)

    show("len(rows)  (16 lines, 1 blank, no header)", len(rows))
    show("rows[0]", rows[0])
    show("rows[2][' Note']  (a quoted comma, kept)", rows[2][" Note"])
    show("rows[10][' Qty']  (a stray space, kept)", rows[10][" Qty"])
    show("rows[6][' Drink']  (shouting, kept)", rows[6][" Drink"])

    print()
    print("  Three things to notice: every value is a str; the header's spaces are")
    print("  now IN the keys; and the blank line was skipped for you.")
    try:
        rows[0]["drink"]
    except KeyError as err:
        print(f"  rows[0]['drink'] -> KeyError: {err}   (the key is ' Drink', space and capital)")

    print()
    print("  Why not just split on commas yourself?")
    line = 'cappuccino,large,4.20 ,1,"oat milk, extra hot"'
    show("line.split(',')  (6 pieces: WRONG)", line.split(","))
    show("next(csv.reader([line]))  (5 pieces)", next(csv.reader([line])))
    print("  The csv module knows about quotes. Your .split doesn't.")


# ---------------------------------------------------------------------------
# 5. Cleaning the rows
# ---------------------------------------------------------------------------


def tidy(row):
    """Lower-case and strip every column name and value. A missing value becomes ''."""
    return {key.strip().lower(): (value or "").strip() for key, value in row.items()}


def parse_price_cents(text):
    """Messy price text -> whole cents, or None if it isn't a price. From lesson 003."""
    cleaned = text.strip().lower().replace("£", "")
    if cleaned in ("", "n/a", "free"):
        return None
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    return round(float(cleaned) * 100)


def clean_drink(text):
    """One spelling per drink: normalise, then translate known variants. From lesson 005."""
    key = text.strip().lower()
    return ALIASES.get(key, key)


def clean_row(raw):
    """One raw till row -> one clean row with the CLEAN_FIELDS columns and real types."""
    row = tidy(raw)
    return {
        "date": row["date"],
        "drink": clean_drink(row["drink"]),
        "size": row["size"].lower(),
        "price_cents": parse_price_cents(row["price"]),
        "quantity": int(row["qty"]),
        "note": row["note"],
    }


def section_cleaning():
    heading("5. Cleaning the rows")

    rows = read_raw_rows(RAW_CSV)
    print("  tidy() fixes the keys and the stray spaces in one go:")
    show("rows[6]  (raw)", rows[6])
    show("tidy(rows[6])", tidy(rows[6]))

    print()
    print("  clean_row() puts lessons 003, 005 and 006 in one honest function:")
    for i in (0, 2, 6, 10):
        print(f"    {dict(rows[i])}")
        print(f"      -> {clean_row(rows[i])}")

    clean = [clean_row(r) for r in rows]
    show("len(clean)", len(clean))
    show("sorted({r['drink'] for r in clean})", sorted({r["drink"] for r in clean}))
    show("type(clean[0]['price_cents']).__name__", type(clean[0]["price_cents"]).__name__)
    print("  Five spellings became five drinks. Prices are ints. Quantities are ints.")


# ---------------------------------------------------------------------------
# 6. Writing a CSV with DictWriter
# ---------------------------------------------------------------------------


def write_rows(rows, path, fieldnames):
    """Write a list of dicts as a CSV with the given columns. Creates the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def section_dictwriter():
    heading("6. Writing a CSV with DictWriter")

    clean = [clean_row(r) for r in read_raw_rows(RAW_CSV)]
    written = write_rows(clean, CLEAN_CSV, CLEAN_FIELDS)
    show("write_rows(clean, CLEAN_CSV, CLEAN_FIELDS)", written)
    show("CLEAN_CSV", CLEAN_CSV)

    print()
    print("  The first four lines of what we wrote:")
    for line in CLEAN_CSV.read_text(encoding="utf-8").splitlines()[:4]:
        print(f"    {line}")
    print("  The header came from fieldnames. The note with a comma got quotes. Not by us.")

    print()
    print("  DictWriter is strict about columns, and that's a feature:")
    try:
        with open(OUTPUT_DIR / "scratch.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "drink"])
            writer.writeheader()
            writer.writerow({"date": "2026-09-07", "drink": "latte", "size": "medium"})
    except ValueError as err:
        print(f"    ValueError: {err}")
    print("  A key that isn't in fieldnames is an error, so a typo can't silently vanish.")
    (OUTPUT_DIR / "scratch.csv").unlink()

    print()
    print("  The plain (non-dict) pair, for when you have lists instead of dicts:")
    with open(OUTPUT_DIR / "cups.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["drink", "cups"])
        writer.writerows([["latte", 8], ["flat white", 4]])
    with open(OUTPUT_DIR / "cups.csv", newline="", encoding="utf-8") as f:
        show("list(csv.reader(f))", list(csv.reader(f)))


# ---------------------------------------------------------------------------
# 7. The round trip, and what CSV forgets
# ---------------------------------------------------------------------------


def load_orders(path):
    """
    Read a CLEAN orders CSV into a list of dicts with real types.

    The conversion happens here, once, so nothing downstream has to remember
    that CSV hands you strings.
    """
    rows = read_raw_rows(path)
    for row in rows:
        row["price_cents"] = int(row["price_cents"])
        row["quantity"] = int(row["quantity"])
    return rows


def section_round_trip():
    heading("7. The round trip, and what CSV forgets")

    clean = [clean_row(r) for r in read_raw_rows(RAW_CSV)]
    write_rows(clean, CLEAN_CSV, CLEAN_FIELDS)

    back = read_raw_rows(CLEAN_CSV)
    show("back[0]  (straight from DictReader)", back[0])
    show("type(back[0]['price_cents']).__name__", type(back[0]["price_cents"]).__name__)
    show("back[0] == clean[0]", back[0] == clean[0])
    print("  CSV has no types. 380 went out as the text '380' and came back as the text '380'.")

    print()
    orders = load_orders(CLEAN_CSV)
    show("load_orders(CLEAN_CSV)[0]", orders[0])
    show("orders == clean", orders == clean)
    assert orders == clean, "the round trip should give back exactly what we wrote"
    print("  With one conversion step, the round trip is exact. The assert proves it.")

    print()
    print("  Now the newline thing. Look at the raw bytes of the file we wrote:")
    show("CLEAN_CSV.read_bytes()[:60]", CLEAN_CSV.read_bytes()[:60])
    print("  The csv module ends lines with \\r\\n (the CSV standard). Opening with")
    print("  newline='' tells Python 'don't translate line endings, csv will handle them'.")
    print("  Forget it on Windows and every row gets a blank row after it.")


# ---------------------------------------------------------------------------
# 8. Putting it together: raw in, clean out, report
# ---------------------------------------------------------------------------


def clean_csv(src, dst):
    """Read the raw till export at src, clean every row, write it to dst. Returns the clean rows."""
    clean = [clean_row(r) for r in read_raw_rows(src)]
    write_rows(clean, dst, CLEAN_FIELDS)
    return clean


def revenue_by_drink(orders):
    """Total cents per drink, as a plain dict. Does not change orders."""
    totals = {}
    for row in orders:
        totals[row["drink"]] = totals.get(row["drink"], 0) + row["price_cents"] * row["quantity"]
    return totals


def report_lines(orders):
    """The week's report as a list of strings. Nothing printed, nothing changed."""
    revenue = revenue_by_drink(orders)
    cups = sum(row["quantity"] for row in orders)
    lines = [f"{'drink':<12} {'revenue':>8}"]
    for drink, cents in sorted(revenue.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"{drink:<12} {pounds(cents):>8}")
    lines.append(f"{'total':<12} {pounds(sum(revenue.values())):>8}   ({cups} cups, {len(orders)} orders)")
    return lines


def section_together():
    heading("8. Putting it together: raw in, clean out, report")

    clean = clean_csv(RAW_CSV, CLEAN_CSV)
    print(f"  {RAW_CSV.name} -> {len(clean)} clean rows -> {CLEAN_CSV.relative_to(HERE)}")

    orders = load_orders(CLEAN_CSV)
    lines = report_lines(orders)
    print()
    for line in lines:
        print(f"    {line}")

    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print(f"  The same report, saved to {REPORT_TXT.relative_to(HERE)}.")

    assert sum(revenue_by_drink(orders).values()) == 7970, "the week should total £79.70"
    assert sum(r["quantity"] for r in orders) == 23, "the week should be 23 cups"
    print("  £79.70 and 23 cups: the same numbers as lessons 005 and 006, from a file this time.")

    print()
    print("  What's in output/ now:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {path.name:<20} {path.stat().st_size:>5} bytes")


def main():
    print("=" * 66)
    print("  Lesson 007: Files, paths and CSV round-trips")
    print("=" * 66)
    section_paths()
    section_read_text()
    section_write_text()
    section_dictreader()
    section_cleaning()
    section_dictwriter()
    section_round_trip()
    section_together()
    print()
    print("Find it, read it, clean it, write it, read it back. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
