# Lesson 007 - Exercises

Three quick checks, one hands-on round trip with data you've met before, and
an optional wander through the repo with `glob`. Predict first, then run.

For the hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-0-python/lesson-007-files-paths-and-csv-round-trips/lesson.py my_lesson_007.py
python my_lesson_007.py
```

Because the copy lives in a different folder, `HERE` now points at the repo
root, so change the `HERE = ...` line near the top to point at the lesson
folder directly:

```python
HERE = Path("lessons/phase-0-python/lesson-007-files-paths-and-csv-round-trips").resolve()
```

Everything else (`RAW_CSV`, `OUTPUT_DIR`, the helpers) hangs off `HERE`, so
that one change is enough. Your files will land in the lesson's `output/`
folder, same as the script's.

---

## 1. Read the path

```python
from pathlib import Path
p = Path("lessons/phase-0-python") / "lesson-007-files-paths-and-csv-round-trips" / "orders_raw.csv"
```

Without running it, what are these?

```python
p.name
p.suffix
p.stem
p.parent.name
p.with_suffix(".txt").name
p.exists()
```

(For the last one: the answer depends on something. What?)

<details>
<summary>Check yourself</summary>

```
'orders_raw.csv'
'.csv'
'orders_raw'
'lesson-007-files-paths-and-csv-round-trips'
'orders_raw.txt'
True or False, depending on your cwd
```

`p` is a *relative* path, so `.exists()` asks "is this here, starting from
the folder my terminal is standing in?" From the repo root, `True`. From
anywhere else, `False`, with no error to tell you why. That's the whole
reason `lesson.py` builds its paths off `Path(__file__).resolve().parent`
instead.

</details>

## 2. Which mode, and what else goes in the brackets?

For each job, which mode goes in `open(path, ___)`, and which other keyword
arguments should be there? One of them is a trap.

- (a) Read a CSV with `DictReader`.
- (b) Add one line to the end of a log file, every time the script runs.
- (c) Write a fresh report, replacing last week's.
- (d) Save results to a file that must never be overwritten by accident.
- (e) `open("orders_raw.csv", "w")` to "open the export so I can read it".

<details>
<summary>Check yourself</summary>

- (a) `"r"` (or leave it out), plus `newline=""` **and** `encoding="utf-8"`.
- (b) `"a"`, plus `encoding="utf-8"`.
- (c) `"w"`, plus `encoding="utf-8"`. (`newline=""` too if it's a CSV.)
- (d) `"x"`, plus `encoding="utf-8"`. `"x"` raises `FileExistsError` if the
  file is already there, which is exactly what you want.
- (e) **The trap.** `"w"` erases the file the instant you open it, before
  you've read a byte. The export is gone. Reading is `"r"`, and it's the
  default, so just `open("orders_raw.csv", encoding="utf-8")`. This is also
  why the lesson keeps everything it writes in `output/`: a mistake there
  costs nothing.

</details>

## 3. Split or csv?

```python
line = 'flat white,medium,"3,90",1,"milk, no sugar"'
```

How many pieces does `line.split(",")` give? How many does
`next(csv.reader([line]))` give, and what's the third one?

<details>
<summary>Check yourself</summary>

`split(",")` gives **seven** pieces: it cuts at every comma, including the
two inside quotes, and leaves the quote characters stuck to the fragments.

`csv.reader` gives **five**: `['flat white', 'medium', '3,90', '1', 'milk,
no sugar']`. The third is `'3,90'`, quotes removed, comma intact. That's a
European-style price which `parse_price_cents` from lesson 003 already knows
how to read; the `csv` module got it to you in one piece so that it could.

Splitting on commas by hand works right up until the first quoted field,
and then it silently gives you the wrong number of columns. Use `csv`.

</details>

## 4. Hands-on: the loyalty card round trip

Lesson 005's loyalty scans, as a Python list. Paste it into your copy:

```python
scans = [
    ("2026-09-07", "Amira", "latte"),
    ("2026-09-07", "ben", "espresso"),
    ("2026-09-07", "Chloe ", "flat-white"),
    ("2026-09-08", "amira", "latte"),
    ("2026-09-08", "Ben", "espresso"),
    ("2026-09-08", "chloe", "flat white"),
    ("2026-09-08", "Dev", "capp"),
    ("2026-09-09", "AMIRA", "cappuccino"),
    ("2026-09-09", "ben", "espresso"),
    ("2026-09-09", "chloe", "latte"),
    ("2026-09-10", "amira", "latte"),
    ("2026-09-10", "Ben ", "latte"),
    ("2026-09-10", "dev", "cappuccino"),
    ("2026-09-11", "amira", "latte"),
    ("2026-09-11", "ben", "espresso"),
    ("2026-09-11", "chloe", "flat-white"),
    ("2026-09-11", "dev", "capp"),
    ("2026-09-11", "Amira", "tea"),
]
```

Pretend this is what the loyalty card machine exports, mess and all. Take
it through the whole loop, one step at a time, and check each step before
the next:

**a) Write the raw file.** Save `scans` to `OUTPUT_DIR / "scans_raw.csv"`
with `DictWriter`, columns `date`, `customer`, `drink`, exactly as they are
(don't clean anything yet). Check: the file has 19 lines.

**b) Read it back and clean it.** Read `scans_raw.csv` with `DictReader`.
Write a `clean_scan(row)` function that returns a new dict with the customer
`strip().lower()`ed and the drink through `ALIASES.get(key, key)`. Write the
cleaned rows to `OUTPUT_DIR / "scans_clean.csv"`. Check: the distinct
customers are `['amira', 'ben', 'chloe', 'dev']`.

**c) Prove the round trip.** Read `scans_clean.csv` back with `DictReader`
and `assert` it equals your cleaned list. (Why does this one pass without a
conversion step, when section 7 of the lesson needed one?)

**d) A summary file.** Count cups per customer with a `Counter`, and write
`OUTPUT_DIR / "cups_per_customer.csv"` with columns `customer`, `cups`,
`free_coffees` (cups `// 5`), most cups first.

**e) Round-trip the summary.** Read `cups_per_customer.csv` back and
`assert` the cups add up to 18. Which column needs `int()` before you can
add it?

Hints, if you want them:

- For (a), `DictWriter` wants dicts. `{"date": d, "customer": c, "drink": k}`
  inside a `for d, c, k in scans:` loop, or a list comprehension of them.
- For (b), you can reuse `read_raw_rows` from the lesson to read, and
  `write_rows` to write. That's what they're for.
- For (d), `cups.most_common()` is already in the right order.
  `[{"customer": name, "cups": n, "free_coffees": n // 5} for name, n in ...]`.

<details>
<summary>Expected results</summary>

```
a) 19 lines            <- header + 18 scans
b) ['amira', 'ben', 'chloe', 'dev']
c) assert passes; 18 rows; drinks are ['cappuccino', 'espresso', 'flat white', 'latte', 'tea']
d) customer,cups,free_coffees
   amira,6,1
   ben,5,1
   chloe,4,0
   dev,3,0
e) 18 cups
```

(c) passes without conversion because every value in a scan is text
already: dates, names and drinks are strings going out and strings coming
back. The lesson's round trip needed `int()` because `price_cents` and
`quantity` were numbers going out and text coming back. (e) is the same
story: `cups` is a number you wrote, so it's text you read.

</details>

<details>
<summary>One way to write it</summary>

```python
from collections import Counter

SCAN_FIELDS = ["date", "customer", "drink"]

# a)
raw_scans = [{"date": d, "customer": c, "drink": k} for d, c, k in scans]
raw_path = OUTPUT_DIR / "scans_raw.csv"
write_rows(raw_scans, raw_path, SCAN_FIELDS)
print(f"a) {len(raw_path.read_text(encoding='utf-8').splitlines())} lines")

# b)
def clean_scan(row):
    """One raw scan -> one clean scan. Name normalised, drink through the aliases."""
    key = row["drink"].strip().lower()
    return {
        "date": row["date"],
        "customer": row["customer"].strip().lower(),
        "drink": ALIASES.get(key, key),
    }

clean = [clean_scan(row) for row in read_raw_rows(raw_path)]
clean_path = OUTPUT_DIR / "scans_clean.csv"
write_rows(clean, clean_path, SCAN_FIELDS)
print(f"b) {sorted({row['customer'] for row in clean})}")

# c)
back = read_raw_rows(clean_path)
assert back == clean, "the clean file should read back exactly as written"
print(f"c) {len(back)} rows, drinks: {sorted({row['drink'] for row in back})}")

# d)
cups = Counter(row["customer"] for row in back)
summary = [{"customer": name, "cups": n, "free_coffees": n // 5}
           for name, n in cups.most_common()]
summary_path = OUTPUT_DIR / "cups_per_customer.csv"
write_rows(summary, summary_path, ["customer", "cups", "free_coffees"])
print("d)")
print(summary_path.read_text(encoding="utf-8"))

# e)
total = sum(int(row["cups"]) for row in read_raw_rows(summary_path))
assert total == 18
print(f"e) {total} cups")
```

Three files, three round trips, two `assert`s, and not one `open()` written
by hand, because `read_raw_rows` and `write_rows` already say
`newline=""` and `encoding="utf-8"` so you don't have to remember. Wrapping
the fiddly bits in a function once is how you stop getting them wrong.

</details>

## 5. (Optional) A walk through the repo

`HERE.parent` is the `phase-0-python` folder. Using `rglob("lesson.py")`,
print the folder name and the line count of every `lesson.py` in Phase 0,
in order. Then: which is the longest so far?

<details>
<summary>Check yourself</summary>

```python
for path in sorted(HERE.parent.rglob("lesson.py")):
    lines = len(path.read_text(encoding="utf-8").splitlines())
    print(f"{path.parent.name:<52} {lines:>4} lines")
```

You should see one line per lesson published so far, 001 upward, and the
counts should creep up through the week: each lesson leans on the last, so
each script has a bit more to say. `sorted()` on paths sorts by the text of
the path, which is why the lesson numbers come out in order; that's a good
reason to zero-pad numbers in folder names.

`rglob` is `glob` that also searches subfolders. Between the two of them,
"do this to every CSV in that folder" is a two-line loop, and that's most of
what people mean when they say a script "processes a batch of files".

</details>

---

Done? Go to
[Lesson 008 - Errors, and what to do about them](../lesson-008-errors-and-what-to-do-about-them/README.md).
