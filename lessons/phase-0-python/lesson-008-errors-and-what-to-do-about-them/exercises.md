# Lesson 008 - Exercises

Three quick checks, one hands-on task that makes the validator stricter and
the file nastier, and an optional design question. Predict first, then run.

For the hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-0-python/lesson-008-errors-and-what-to-do-about-them/lesson.py my_lesson_008.py
python my_lesson_008.py
```

As in lesson 007, the copy lives in a different folder, so change the
`HERE = ...` line near the top to point at the lesson folder:

```python
HERE = Path("lessons/phase-0-python/lesson-008-errors-and-what-to-do-about-them").resolve()
```

Everything else (`MESSY_CSV`, `OUTPUT_DIR`, `validate_row`,
`load_orders_carefully`, `write_rejects`) hangs off `HERE` and is there for
you to reuse.

---

## 1. Read this traceback

Someone ran their report and got this. Don't look at any code; you don't
have it. Answer from the traceback alone.

```
Traceback (most recent call last):
  File "my_report.py", line 31, in <module>
    print_report(load_orders(CSV_PATH))
  File "my_report.py", line 18, in load_orders
    row["price_cents"] = int(row["price_cents"])
KeyError: 'price_cents'
```

- (a) What kind of error, and what is Python saying in plain English?
- (b) Which line should they open first?
- (c) It's a `KeyError`, not a `ValueError`, so `int()` never ran. Where
  did the problem *actually* start, and what's your best guess at the fix?

<details>
<summary>Check yourself</summary>

**(a)** `KeyError: 'price_cents'`: "the dict `row` has no key called
`price_cents`". Nothing about numbers; `int()` was never reached.

**(b)** Line 18, in `load_orders`. That's the bottom frame and it's in their
own file.

**(c)** `row` came from a `DictReader`, so its keys are whatever the CSV's
header row says. The header doesn't say `price_cents`. Most likely they
pointed `CSV_PATH` at the *raw* export (header `price`) instead of the
*clean* file from lesson 007 (header `price_cents`), or the header has a
stray space or capital in it. The fix is at line 31 (or wherever `CSV_PATH`
is set), not line 18. The traceback shows where the error *surfaced*; the
mistake is upstream. A `print(reader.fieldnames)` would settle it in one run.

</details>

## 2. Which error?

For each line, name the error it raises. One of them raises nothing. Which,
and what does it give you instead?

```python
int("3.80")
{"price": "3.80"}["Price"]
open("orders.csv")              # run from the repo root
None.strip()
[][0]
"2" + 2
{"price": "3.80"}.get("Price")
```

<details>
<summary>Check yourself</summary>

```
ValueError          <- right type (str), wrong value: int() won't take a decimal point
KeyError            <- 'Price' with a capital isn't 'price'
FileNotFoundError   <- no orders.csv in the repo root; cwd strikes again (lesson 007)
AttributeError      <- None has no .strip; ask where the None came from
IndexError          <- an empty list has no position 0
TypeError           <- str + int; Python won't guess whether you meant "22" or 4
(nothing)           <- .get returns None for a missing key, quietly
```

The last one is lesson 005's rule again: `[]` shouts when a key is
missing, `.get` doesn't. Both are right in the right place. The trouble
starts when you use `.get` to make a `KeyError` go away without asking why
the key was missing.

</details>

## 3. Spot the swallow

Five `except` blocks. For each: which of the three honest things is it
doing (recover / record / add context and re-raise), or is it swallowing?

```python
# (a)
try:
    qty = int(text)
except ValueError:
    qty = None

# (b)
try:
    rows = load_orders(path)
except:
    rows = []

# (c)
try:
    average = revenue / cups
except ZeroDivisionError:
    average = 0.0

# (d)
try:
    text = path.read_text(encoding="utf-8")
except FileNotFoundError as err:
    raise FileNotFoundError(f"Expected this week's export at {path}") from err

# (e)
try:
    print_report(rows)
except Exception as err:
    print(err)
```

<details>
<summary>Check yourself</summary>

**(a) Recover.** Honest. `None` means "not a quantity", it's documented by
the function's shape, and it can't be mistaken for a real count.

**(b) Swallow.** Bare `except`, and an empty list that looks exactly like a
real week with no sales. A missing file, a typo in `load_orders`, a
permissions problem: all become "zero rows" with no message. This is the
`load_export_quietly` from section 6 wearing a different hat.

**(c) Half a swallow.** The `except` is narrow, which is good, but `0.0` is
a *real* average and this isn't one. "No cups" has no average. `None` is
honest; `0.0` will get plotted on a chart next to real numbers and nobody
will know. Same error, better fallback: `average = None`.

**(d) Add context and re-raise.** Honest. Still stops, still shows the
original error via `from err`, but the first line the reader sees now says
what was expected.

**(e) Swallow, dressed up.** It prints the message, which feels responsible,
but then the program *carries on* as if the report was fine. `except
Exception` also catches typos, `KeyError`s, everything. If `print_report`
can fail for a reason you expect, catch that one. If it can't, don't wrap
it at all and let the traceback do its job.

</details>

## 4. Hands-on: a stricter door, a nastier file

The shop is closed at weekends, and nobody has ever ordered two hundred of
anything. `validate_row` doesn't know either of those things yet.

**a) Build the nastier file.** Read `orders_messy.csv` as text, add these
four lines to the end, and write the result to
`OUTPUT_DIR / "orders_nastier.csv"`. (Never edit the original; that's what
`output/` is for.) Check: the new file has 24 lines.

```
2026-09-12,latte,medium,3.80,1
2026-09-11,latte,medium,3.80,200
2026-09-11,Latte,MEDIUM,3.80,1
2026-09-11,mocha,medium,4.00,1
```

**b) Baseline.** Run `load_orders_carefully` on the nastier file *before*
changing any rules. How many good, how many rejected, and which of the four
new rows got through that shouldn't have? Predict before you run.

**c) Two new rules.** Write `validate_row_strict(raw)` that calls
`validate_row` first (so all the existing checks still happen) and then
rejects, with a message in the same style:

- any date that falls on a Saturday or Sunday (`date.fromisoformat(...)`
  gives you a `date` object; `.weekday()` is `5` or `6` at the weekend, and
  `.strftime("%A")` gives you the day's name for the message);
- any quantity over 20, as a probable typo.

Then a `load_orders_strictly(path)` that uses it. (Copying
`load_orders_carefully` and changing one name is fine. Lesson 009 shows a
tidier way.)

**d) Write the rejects** to `OUTPUT_DIR / "rejected_nastier.csv"` with
`write_rejects`. Open the file. Every reason should tell a human what to fix
without them reading your code.

**e) Prove the numbers.** Total revenue and cups from the good rows. Then
two `assert`s: the total, and the number of rejects.

Hints, if you want them:

- For (a), `MESSY_CSV.read_text(encoding="utf-8") + extra` where `extra` is
  a triple-quoted string of the four lines, then `.write_text(...)`. The
  original ends with a newline, so the join is clean.
- For (c), `validate_row` returns a dict with `"date"` as a string and
  `"quantity"` as an int, so `date.fromisoformat(row["date"])` and
  `row["quantity"] > 20` are all you need.
- The third new row (`Latte,MEDIUM`) is *supposed* to pass. `validate_row`
  lower-cases both. If it's in your rejects, something's wrong.

<details>
<summary>Expected results</summary>

```
a) 24 lines
b) 18 good, 5 rejected
   line 24  drink 'mocha' is not on the menu        <- the only new row caught
   (the Saturday order and the 200 lattes both got through)
c) 16 good, 7 rejected
   line  8  missing price
   line 12  date '09/09/2026' is not YYYY-MM-DD
   line 16  missing price, quantity
   line 20  quantity -1 must be at least 1
   line 21  date 2026-09-12 is a Saturday; the shop is closed
   line 22  quantity 200 looks like a typo (more than 20)
   line 24  drink 'mocha' is not on the menu
d) 7 rows in rejected_nastier.csv
e) £83.50, 24 cups        <- the week's £79.70 plus one honest Thursday latte
```

If (b) surprised you: that's the point. A validator only catches what
someone thought to tell it. `-1` cups was rejected because lesson 008's
author thought of it; `200` wasn't because they didn't. Every rule in
`validate_row` is a decision about *this* data, and the list grows every
time the data teaches you something new.

</details>

<details>
<summary>One way to write it</summary>

```python
# a)
extra = """2026-09-12,latte,medium,3.80,1
2026-09-11,latte,medium,3.80,200
2026-09-11,Latte,MEDIUM,3.80,1
2026-09-11,mocha,medium,4.00,1
"""
NASTIER_CSV = OUTPUT_DIR / "orders_nastier.csv"
OUTPUT_DIR.mkdir(exist_ok=True)
NASTIER_CSV.write_text(MESSY_CSV.read_text(encoding="utf-8") + extra, encoding="utf-8")
print(f"a) {len(NASTIER_CSV.read_text(encoding='utf-8').splitlines())} lines")

# b)
orders, rejects = load_orders_carefully(NASTIER_CSV)
print(f"b) {len(orders)} good, {len(rejects)} rejected")

# c)
def validate_row_strict(raw):
    """validate_row, plus the shop's own rules: no weekends, no absurd quantities."""
    row = validate_row(raw)
    when = date.fromisoformat(row["date"])
    if when.weekday() >= 5:
        raise ValueError(f"date {row['date']} is a {when.strftime('%A')}; the shop is closed")
    if row["quantity"] > 20:
        raise ValueError(f"quantity {row['quantity']} looks like a typo (more than 20)")
    return row


def load_orders_strictly(path):
    """Like load_orders_carefully, with the stricter validator."""
    orders, rejects = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for line_number, raw in enumerate(csv.DictReader(f), start=2):
            try:
                orders.append(validate_row_strict(raw))
            except ValueError as err:
                rejects.append({"line": line_number, "reason": str(err),
                                **{field: raw.get(field) or "" for field in RAW_FIELDS}})
    return orders, rejects


orders, rejects = load_orders_strictly(NASTIER_CSV)
print(f"c) {len(orders)} good, {len(rejects)} rejected")
for reject in rejects:
    print(f"   line {reject['line']:>2}  {reject['reason']}")

# d)
print(f"d) {write_rejects(rejects, OUTPUT_DIR / 'rejected_nastier.csv')} rows written")

# e)
revenue = revenue_by_drink(orders)
cups = sum(row["quantity"] for row in orders)
print(f"e) {pounds(sum(revenue.values()))}, {cups} cups")
assert sum(revenue.values()) == 8350
assert len(rejects) == 7
```

`validate_row_strict` calling `validate_row` first is worth a second look.
The new function doesn't repeat any of the old checks; it *adds* to them,
and if `validate_row` raises, the `raise` passes straight through
`validate_row_strict` to the `except` in the loader. Errors travel up
through every function that doesn't catch them until one does. That's what
the frames in a traceback are: the list of functions the error passed
through on its way to you.

</details>

## 5. (Optional) Look first, or leap?

Two ways to open a file that might not be there:

```python
# Look before you leap
if path.exists():
    text = path.read_text(encoding="utf-8")
else:
    text = None

# Easier to ask forgiveness than permission
try:
    text = path.read_text(encoding="utf-8")
except FileNotFoundError:
    text = None
```

And two ways to turn text into a number:

```python
if text.strip().isdigit():
    qty = int(text)

try:
    qty = int(text)
except ValueError:
    qty = None
```

Which would you pick in each pair, and why? Is there a case the "look
first" version gets wrong?

<details>
<summary>Check yourself</summary>

**Files: lean towards `try`.** `exists()` answers a different question from
"can I read it": the file might exist and be unreadable, or be a folder,
or (rarely, but really) vanish between the check and the read. The `try`
version asks the only question that matters, "did the read work?", and
handles the answer. `exists()` is still fine for a friendly early message
("no export found, nothing to do") when you'd rather not open the file at
all.

**Numbers: `try`, clearly.** `.isdigit()` says no to `"-1"`, `" 3"`
(without the strip), `"+2"` and `"1_000"`, all of which `int()` accepts, so
you'd be re-implementing `int`'s rules and getting them slightly wrong.
Let `int()` decide and deal with the refusals. Then, separately, apply
*your* rules to the number you got (`if qty < 1`), which is what
`validate_row` does.

The general shape: use `try` for "will this conversion or operation work?",
because the operation knows its own rules better than you do. Use `if` for
"is this value acceptable to *me*?", because those are your rules and
nothing else knows them.

</details>

---

That's Week 2, Day 3 done. Lessons 009 and 010 arrive tomorrow (Thursday).
If one thing sticks from today, let it be: *read the last line first, catch
only what you expect, and never make red text go away without knowing why
it was there*.
