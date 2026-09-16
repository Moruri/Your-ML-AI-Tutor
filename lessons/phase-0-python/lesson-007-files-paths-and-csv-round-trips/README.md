# Lesson 007 - Files, paths and CSV round-trips

**Phase 0 - Python foundations for data work** | Week 2, Day 3 | Wednesday 2026-09-16

> **Goal:** read and write text and CSV files safely with `pathlib` and
> `csv.DictReader`/`DictWriter`, so that "load this, clean it, save it, and
> prove the saved copy is right" is something you can do without a search
> engine open.

Time: about 50 minutes. No installs. Python 3.10+.

---

Morning. For four lessons the coffee orders have lived inside `lesson.py` as
a list called `ORDERS`, typed in by hand. Real data doesn't arrive like that.
It arrives as a file: a till export, a spreadsheet someone saved as CSV, a
download from a website. It's in a folder you didn't choose, it has a header
row somebody else named, and it's full of the small human mess you've spent
the week learning to clean.

Today we go and get it. You met `csv.DictReader` and `Path(__file__).parent`
back in lesson 002 without much explanation. Now you get the explanation,
plus the other half: writing files. Reading a CSV and never writing one is
like learning to listen but not to talk. By the end you'll read a messy
export, clean it with the functions you already know how to write, save a
tidy copy, and then read your own copy back to prove nothing got lost. That
read-write-read loop is the **round trip** in the title, and it is the
single most useful habit in data work: never trust a file you wrote until
you've read it.

Next to this README is `orders_raw.csv`: the same week of orders, as the
till actually exported them. Have a look at it in a text editor before you
run anything. Notice the spaces in the header, the `£` signs, the blank
line, and the note with a comma in it. Every one of those is a lesson.

## How to follow along

Terminal in the repo root, `python`, type along at `>>>`. Full script:

```bash
python lessons/phase-0-python/lesson-007-files-paths-and-csv-round-trips/lesson.py
```

Or from inside the lesson folder, plain `python lesson.py`. It works from
both, and section 1 is about why.

**This script writes files.** Everything it writes goes into an `output/`
folder next to the script, which it creates. Nothing you were given is
touched. Delete `output/` any time; running the script again rebuilds it.

## 1. Where am I? Paths with pathlib

Every file has an address. A **path** is that address, and `pathlib.Path`
is how Python represents one:

```python
from pathlib import Path

Path.cwd()                  # the folder your terminal is standing in
Path(__file__)              # the script that's running
Path(__file__).resolve().parent   # the folder that script lives in
```

`cwd` is "current working directory": wherever you were when you typed
`python`. The trap is that it changes with every terminal. Open a file as
`open("orders_raw.csv")` and Python looks in `cwd`, which is fine if you
happen to be standing in the lesson folder and a `FileNotFoundError` if
you're at the repo root. Run `lesson.py` from both places and watch the
`Path('orders_raw.csv').exists()` line flip between `True` and `False`.

`__file__` is the fix. It's the path of the script itself, and it doesn't
care where you launched from. So every script that needs files does this at
the top:

```python
HERE = Path(__file__).resolve().parent
RAW_CSV = HERE / "orders_raw.csv"
```

That `/` is not division. On a `Path` it means "and then inside that", and
it uses the right slash for whatever operating system you're on. You never
type `"folder" + "/" + "file"` again. `.resolve()` turns the path into a full
absolute one, which is worth doing once so that later error messages tell
you exactly where Python looked.

Paths know things about themselves:

```python
RAW_CSV.name        # 'orders_raw.csv'
RAW_CSV.stem        # 'orders_raw'
RAW_CSV.suffix      # '.csv'
RAW_CSV.parent      # the folder
RAW_CSV.exists()    # True (or False, no error, for a path that isn't there)
RAW_CSV.is_file()   # True
RAW_CSV.with_suffix(".txt")            # same place, different extension
RAW_CSV.with_name("orders_clean.csv")  # same folder, different file
```

And folders know what's in them: `HERE.iterdir()` gives you everything,
`HERE.glob("*.csv")` gives you only the CSVs. (`*` means "anything here";
`rglob` goes into subfolders too.) You'll use `glob` every time someone
hands you "a folder of exports".

## 2. Reading a text file

The two-line way, for a file that fits in memory (which, for now, is all of
them):

```python
text = RAW_CSV.read_text(encoding="utf-8")
lines = text.splitlines()
```

`text` is one long string with `\n` between the lines. `.splitlines()` cuts
it into a list, one string per line, newlines removed. That's often all you
need.

The longer way, which you'll see in every codebase, and which handles files
too big to hold at once:

```python
with open(RAW_CSV, encoding="utf-8") as f:
    for line in f:
        print(line.rstrip("\n"))
```

`with open(...) as f:` opens the file and promises to close it when the
block ends, even if something inside goes wrong. Always use `with`. A file
left open is a small leak on your machine and, on Windows, a file nobody
else can delete. Looping over `f` gives you one line at a time, each with
its newline still attached, hence the `.rstrip("\n")`.

**The encoding gotcha.** Files are bytes on disk. Turning bytes into text
needs a rule, an **encoding**, and if the reader guesses a different rule
from the writer, you get garbage. Look at the `£` in `orders_raw.csv`:

```python
"£".encode("utf-8")               # b'\xc2\xa3'    two bytes
b"\xc2\xa3".decode("utf-8")       # '£'
b"\xc2\xa3".decode("cp1252")      # 'Â£'           what Windows might show you
```

Same two bytes. Read with the right rule, a pound sign. Read with the wrong
one, `Â£`. If you've ever opened a file and seen `Ã©` where an `é` should
be, this is why. UTF-8 is the rule almost everything uses now, and Python
uses it by default on Mac and Linux, but not on every Windows setup. So:
**always write `encoding="utf-8"`**, reading and writing. It's eleven extra
characters and it removes a whole category of "works on my machine".

## 3. Writing a text file (and a folder to put it in)

```python
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

note = OUTPUT_DIR / "hello.txt"
note.write_text("Week of 2026-09-07\nTotal: £79.70\n", encoding="utf-8")
```

`mkdir(exist_ok=True)` makes the folder, and doesn't complain if it's
already there. (Add `parents=True` if you need several levels made at
once.) Keeping what you *write* in a separate folder from what you were
*given* is a habit worth starting now: it means you can delete `output/`
and be back where you started, and you can never overwrite your input by
accident.

`write_text` writes the whole string and replaces whatever was in the file.
**There is no undo.** Write to `orders_raw.csv` by mistake and the till
export is gone. Which is the other reason for the `output/` folder.

For many lines, the `with open` form again, this time in write mode:

```python
with open(note, "w", encoding="utf-8") as f:
    f.write("drink,cups\n")
    for drink, n in cups:
        f.write(f"{drink},{n}\n")
```

`f.write` does not add a newline for you. You add the `\n`. The second
argument to `open` is the **mode**:

| Mode | Means | If the file exists | If it doesn't |
|------|-------|--------------------|---------------|
| `"r"` | read (the default) | reads it | `FileNotFoundError` |
| `"w"` | write | **erases it**, then writes | creates it |
| `"a"` | append | adds to the end | creates it |
| `"x"` | create | `FileExistsError` | creates it |

`"a"` is for logs and anything you add to over time. `"x"` is for when
overwriting would be a disaster and you'd rather crash. Pick on purpose.

## 4. Reading a CSV with DictReader

You could `.split(",")` each line yourself. Don't. Here's line 4 of the raw
file:

```
2026-09-07,cappuccino,large,4.20 ,1,"oat milk, extra hot"
```

Six fields if you split on commas. Five if you read it as CSV, because the
quotes around the note mean "this comma is part of the value". The `csv`
module knows that rule and a dozen others (quotes inside quotes, different
delimiters, newlines inside a field). You don't want to.

```python
import csv

with open(RAW_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    reader.fieldnames     # ['Date', ' Drink', ' Size', ' Price', ' Qty', ' Note']
    rows = list(reader)

len(rows)                 # 15
rows[0]                   # {'Date': '2026-09-07', ' Drink': 'latte', ' Size': 'medium', ' Price': '£3.80', ' Qty': '2', ' Note': ''}
```

`DictReader` uses the first line as column names and hands you one dict per
row, keyed by those names. It skipped the blank line for you. And it gives
you **exactly what the file says**, which means three things you need to
see coming:

- **Every value is a string.** `'2'`, not `2`. `'£3.80'`, not `3.8`. Lesson
  003 was about what to do with that.
- **The header's spaces are in the keys.** The file says `Date, Drink, ...`
  with a space after each comma, so the keys are `'Date'`, `' Drink'`,
  `' Size'`. `rows[0]["drink"]` is a `KeyError`. This bites everyone once.
  (`csv.DictReader(f, skipinitialspace=True)` strips those spaces, but
  you'll still want to lower-case the keys, so we'll tidy them ourselves in
  section 5.)
- **Stray spaces inside values are kept.** `' 2'` and `'CAPP '` arrive
  exactly like that.

**`newline=""`**, every time you open a file for the `csv` module, reading
or writing. Python normally translates line endings as it reads; the `csv`
module wants to see them raw so it can handle quoted fields that span
lines. Leave it out and on Windows you'll get a blank row after every real
one when writing, and quiet corruption in the rare file that has newlines
inside quotes. Nothing visibly breaks on Mac or Linux, which is exactly why
it's a habit and not a thing you'll notice on your own.

## 5. Cleaning the rows

This is where the week pays off. One function to tidy the keys and values,
then the helpers you already know:

```python
def tidy(row):
    """Lower-case and strip every column name and value. A missing value becomes ''."""
    return {key.strip().lower(): (value or "").strip() for key, value in row.items()}
```

A dict comprehension (lesson 005) over `.items()`. The `(value or "")` is
there because a row with too few fields gives `None` for the missing ones,
and `None.strip()` is an error; tomorrow's lesson is about exactly that kind
of thing. After `tidy`, the keys are `date`, `drink`, `size`, `price`,
`qty`, `note`, and nothing has spaces on its ends.

Then the row itself:

```python
def clean_row(raw):
    row = tidy(raw)
    return {
        "date": row["date"],
        "drink": clean_drink(row["drink"]),        # ALIASES.get(key, key), lesson 005
        "size": row["size"].lower(),
        "price_cents": parse_price_cents(row["price"]),   # lesson 003
        "quantity": int(row["qty"]),
        "note": row["note"],
    }
```

`parse_price_cents` is lesson 003's function, copied in whole. `clean_drink`
is lesson 005's alias trick. `clean_row` is a lesson 006 function: one job,
honest name, returns the same shape every time, changes nothing it was
given. Run it over the fifteen rows and `'flat-white'`, `'Flatwhite'`,
`'CAPP '` and `'Espresso'` collapse into five drinks with integer prices and
integer quantities.

Notice we renamed columns on the way through: `Price` became `price_cents`
and `Qty` became `quantity`. The clean file is *ours*, so it gets names that
say what they mean. Nobody reading `price_cents` will divide by 100 twice.

## 6. Writing a CSV with DictWriter

The mirror image of `DictReader`:

```python
CLEAN_FIELDS = ["date", "drink", "size", "price_cents", "quantity", "note"]

with open(CLEAN_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=CLEAN_FIELDS)
    writer.writeheader()
    writer.writerows(clean)
```

You tell it the columns and their order; `writeheader()` writes that as the
first line; `writerows` takes your list of dicts. It puts quotes around any
value that needs them (the note with the comma) and nowhere else. You
never think about quoting again.

Two things worth knowing:

- **It's strict about columns.** A dict with a key that isn't in
  `fieldnames` raises `ValueError: dict contains fields not in fieldnames`.
  That's good. A typo in a key can't silently drop a column. (If you
  genuinely want extra keys ignored, `extrasaction="ignore"` does that; a
  key that's *missing* from a row is written as empty.)
- **Numbers become text.** `380` is written as `380`. There's no such thing
  as an integer in a CSV file; there's text that looks like one. Section 7
  is about the consequence.

For lists instead of dicts, `csv.writer` and `csv.reader` are the plainer
pair: `writer.writerow(["drink", "cups"])`. Same rules, no header handling.
Use the Dict versions when the file has a header, which is nearly always.

## 7. The round trip, and what CSV forgets

Now read back what you just wrote:

```python
with open(CLEAN_CSV, newline="", encoding="utf-8") as f:
    back = list(csv.DictReader(f))

back[0]
# {'date': '2026-09-07', 'drink': 'latte', 'size': 'medium', 'price_cents': '380', 'quantity': '2', 'note': ''}
back[0] == clean[0]     # False
```

`False`. The `380` you wrote as an int came back as the string `'380'`.
**CSV has no types.** It stores text, full stop. Which means every time you
load a CSV, even one you wrote yourself five seconds ago, you convert:

```python
def load_orders(path):
    rows = read_raw_rows(path)
    for row in rows:
        row["price_cents"] = int(row["price_cents"])
        row["quantity"] = int(row["quantity"])
    return rows

load_orders(CLEAN_CSV) == clean     # True
```

That's the round trip: write, read, convert, compare. Put an `assert` on
the end of it and you've proved your file is right rather than believing
it is. When you save anything you'll load again (cleaned data, results, a
report), do the round trip once before you rely on the file.

A last look at the bytes, because it explains the `newline=""` rule:

```python
CLEAN_CSV.read_bytes()[:60]
# b'date,drink,size,price_cents,quantity,note\r\n2026-09-07,latte,'
```

`\r\n`. The `csv` module ends lines the way the CSV standard says to, with
two characters. On Windows, Python's normal text mode *also* turns `\n`
into `\r\n`, so without `newline=""` you'd get `\r\r\n`: a blank line after
every row. With `newline=""` Python steps back and lets `csv` do it. The
files you wrote today open cleanly in Excel, Numbers and pandas alike.

## 8. Putting it together: raw in, clean out, report

```python
def clean_csv(src, dst):
    """Read the raw till export at src, clean every row, write it to dst. Returns the clean rows."""
    clean = [clean_row(r) for r in read_raw_rows(src)]
    write_rows(clean, dst, CLEAN_FIELDS)
    return clean

clean = clean_csv(RAW_CSV, CLEAN_CSV)
orders = load_orders(CLEAN_CSV)          # the round trip, not the in-memory list
for line in report_lines(orders):
    print(line)
REPORT_TXT.write_text("\n".join(report_lines(orders)) + "\n", encoding="utf-8")
```

```
drink         revenue
latte          £29.90
cappuccino     £20.00
flat white     £15.60
espresso        £8.80
tea             £5.40
total          £79.70   (23 cups, 15 orders)
```

£79.70 and 23 cups. Those are the exact numbers from lessons 005 and 006,
where the orders were typed in by hand. Same facts, but this time they came
from a messy file, through a cleaner you wrote, out to a tidy file, and
back in again. The two `assert`s at the end of `lesson.py` check it.

Look in `output/` when it's done: `orders_clean.csv` (the tidy data),
`report.txt` (something a human could read), and two small scratch files
from sections 3 and 6. Open `orders_clean.csv` in a spreadsheet if you have
one. It'll look right, because it is.

One design note. `read_raw_rows`, `clean_row`, `write_rows`, `load_orders`
and `report_lines` are each one job. The only functions that touch the disk
are the ones with "read", "write" or "load" in their names. The one that
prints is `section_together`. That's lesson 006's advice applied to files:
keep the disk at the edges, keep the thinking pure in the middle, and every
piece can be checked on its own.

## What you can do now

- Build a path off `Path(__file__).resolve().parent` so a script finds its
  files no matter where it's run from; join with `/`; ask a path its
  `.name`, `.stem`, `.suffix`, `.parent`, and whether it `.exists()`.
- List a folder with `iterdir()` and `glob()`.
- Read a whole text file with `read_text` or line by line with `with open`,
  and say why `with` matters.
- Write a text file with `write_text` or `open(..., "w")`, append with
  `"a"`, and make a folder first with `mkdir(exist_ok=True)`.
- Pass `encoding="utf-8"` and `newline=""` without having to remember why,
  and explain both when asked.
- Read a CSV with `DictReader`, tidy its keys and values, and convert its
  strings to real types in one place.
- Write a CSV with `DictWriter`, in the column order you choose.
- Do the round trip: write, read back, convert, `assert` equal.

## What to do now

1. Run `lesson.py` from the repo root, then `cd` into the lesson folder and
   run it again. Read section 1's output both times.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   takes the loyalty card scans from lesson 005 through a full round trip
   of their own.
3. Then move on to
   [Lesson 008 - Errors, and what to do about them](../lesson-008-errors-and-what-to-do-about-them/README.md),
   where the till export gets rows that *can't* be cleaned, and you learn
   what to do instead of crashing.
