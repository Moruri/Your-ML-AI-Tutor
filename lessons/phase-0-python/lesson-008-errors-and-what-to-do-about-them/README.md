# Lesson 008 - Errors, and what to do about them

**Phase 0 - Python foundations for data work** | Week 2, Day 3 | Wednesday 2026-09-16

> **Goal:** read a traceback calmly, use `try`/`except` on purpose, and
> validate input before it hurts, so that bad data gets caught at the door
> with a message worth reading, and real bugs stay loud enough to find.

Time: about 50 minutes. No installs. Python 3.10+.

---

You've seen a lot of red text this week. Some of it on purpose (`MENU["mocha"]`
in lesson 005, the `KeyError` in lesson 007's DictReader section), most of it
not. Every one of those was Python trying to tell you something, and if the
first thing you feel when you see a traceback is a small jolt of dread,
you're normal, and this lesson is for you.

Here's the reframe. An error is not Python being difficult. It's Python
refusing to guess. `int("two")` could return `0`, or `2`, or `None`, and each
guess would be wrong for someone. So it stops and tells you exactly where
and exactly why. That is a *good* thing, and the biggest mistake beginners
make is to make the red text go away without understanding it, which leaves
the problem in place and removes the only warning you had.

This morning's file had mess that could be cleaned. This afternoon's file,
`orders_messy.csv`, has rows that can't: a price that isn't there, a date
in the wrong format, a quantity of `two`, a line with fields missing, a
quantity of `-1`. Run lesson 007's cleaner on it and you'd get a traceback
on row seven and nothing else. By the end of today you'll get fifteen good
rows, a file listing the four bad ones with a reason each, a report with
the right totals, and no crash. And you'll know why the *missing-file* case
should still crash.

## How to follow along

Terminal in the repo root, `python`, type along at `>>>`. Full script:

```bash
python lessons/phase-0-python/lesson-008-errors-and-what-to-do-about-them/lesson.py
```

The script writes one file, `output/rejected_rows.csv`, next to itself. Same
`output/` habit as this morning.

## 1. Reading a traceback, bottom up

Two functions, one of which trusts its input a bit too much:

```python
def quantity_of(row):
    return int(row["quantity"])

def cups_in(rows):
    total = 0
    for row in rows:
        total += quantity_of(row)
    return total

cups_in([{"drink": "latte", "quantity": "2"}, {"drink": "tea", "quantity": "two"}])
```

```
Traceback (most recent call last):
  File ".../lesson.py", line 88, in section_traceback
    cups_in(rows)
  File ".../lesson.py", line 73, in cups_in
    total += quantity_of(row)
             ^^^^^^^^^^^^^^^^
  File ".../lesson.py", line 66, in quantity_of
    return int(row["quantity"])
           ^^^^^^^^^^^^^^^^^^^^
ValueError: invalid literal for int() with base 10: 'two'
```

The title says "most recent call last", which is a polite way of saying
**read it from the bottom**:

1. **The last line is *what* went wrong.** `ValueError`, and the message:
   `int()` was handed `'two'` and couldn't make a number of it. Read the
   whole message. It nearly always names the value that caused it.
2. **The line above is *where*.** `line 66, in quantity_of`, and the exact
   code: `int(row["quantity"])`. Python 3.11+ even puts `^^^^` under the
   part that failed.
3. **Every line above that is *who called whom*.** `cups_in` called
   `quantity_of`. The section called `cups_in`. Walk up until you reach code
   you wrote, and that's where to start looking.

Two habits. First, when a traceback has twenty frames and most are inside
some library, skip them: find the last frame that's in *your* file. Second,
the line Python points at is where the error *surfaced*, not always where
the mistake *is*. `int()` did nothing wrong here; the `'two'` was already in
the data. Section 5 is about catching it before it gets that far.

## 2. A field guide to the errors you'll meet

Nine errors cover almost everything you'll see this month. Each one is a
specific sentence, and once you know the sentence the fix is usually
obvious:

| Error | Python is saying | Usually means |
|-------|------------------|---------------|
| `ValueError` | "Right type, wrong value." `int("3.5")`, `int("two")` | Text from a file that isn't the number you assumed. |
| `TypeError` | "Wrong type entirely." `3 + " lattes"`, `int(None)` | You mixed text and numbers, or a `None` got in. |
| `KeyError` | "That key isn't in this dict." `MENU["mocha"]` | A typo in a column name, or a value you assumed was there. |
| `IndexError` | "That position isn't in this list." `orders[5]` | An empty list, or an off-by-one. |
| `AttributeError` | "That thing has no such method." `None.strip()`, `"latte".push()` | A `None` where you expected a string (very common with short CSV rows), or a method from another language. |
| `FileNotFoundError` | "No file at that path." | Wrong folder, wrong name, or `cwd` (lesson 007, section 1). |
| `ZeroDivisionError` | "You divided by zero." | An empty group: `total / len(rows)` with no rows. |
| `NameError` | "I've never heard of that name." `pirce` | A typo, or a variable used before it's assigned. |
| `SyntaxError` | "I can't read this line at all." `if cups = 3:` | A typo in the *code*, found before anything runs. |

`SyntaxError` is the odd one out: it happens when Python *reads* your file,
not when it runs it, so there's nothing to catch and no traceback to walk.
Read the message (3.10+ messages are genuinely helpful: `Maybe you meant
'==' or ':=' instead of '='?`), fix the line, run again.

`AttributeError: 'NoneType' object has no attribute ...` deserves a special
mention because you *will* meet it with CSVs. A row with too few fields
gives `None` for the missing ones, and `None.strip()` is exactly that
error. When you see `'NoneType'` in a message, the question is never "why
doesn't None have strip?" but "where did the None come from?"

`lesson.py` runs all nine so you can see the real messages. The demo uses
`except Exception` to catch anything, which is fine for a demo and a bad
habit in real code, for reasons the next section is about.

## 3. `try`/`except` on purpose

```python
def parse_quantity(text):
    """Whole-number text -> int. Anything else -> None."""
    try:
        return int(text)
    except ValueError:
        return None
```

"Try this. If it raises a `ValueError`, do that instead." The `try` block is
the one line that can fail. The `except` names the one error you expect and
know how to handle. This is lesson 006's `parse_quantity` with `try` in
place of `.isdigit()`, and it's better: `int()` knows more about numbers
than `.isdigit()` does (`" 3 "`, `"-1"`, `"+2"`), so you let it decide and
only deal with the refusals.

Now the version you'll see in other people's code and must not copy:

```python
def cups_from_bad(text):
    try:
        return int(txet)        # typo: txet
    except:                     # bare except: catches EVERYTHING
        return 0
```

There's a typo in the `try`. `txet` doesn't exist, so Python raises a
`NameError`. The bare `except:` catches it, along with every other error
in the world, and returns `0`. Every call returns `0`. There's no red text.
The function is completely broken and it will tell you nothing until
someone notices the weekly report says zero cups.

**Catch the error you expect, by name.** `except ValueError:` would have
let the `NameError` through, you'd have seen the traceback on the first
run, and you'd have fixed the typo in ten seconds. Broad excepts (`except:`
and `except Exception:`) don't make code safer. They make it quieter, which
is the opposite.

Two shapes you'll need soon:

```python
except (TypeError, ValueError) as err:     # several kinds at once
    ...
except KeyError as err:                    # the error object, to reuse its message
    print(f"not on the menu: {err}")
```

`as err` gives you the error as a value: `str(err)` is the message,
`type(err).__name__` is the kind. You'll use it to build better messages in
section 5.

## 4. `else`, `finally`, and why `with` exists

The full shape, which you'll see occasionally and write rarely:

```python
try:
    f = open(path, encoding="utf-8")
except FileNotFoundError:
    return None                   # the try failed
else:
    header = f.readline()         # the try succeeded; runs only then
finally:
    print("runs no matter what")  # success, failure, or return: always
```

`else` is for "the risky line worked, now do the thing that depended on
it", keeping the `try` block down to the single line that can actually
fail. That matters: a `try` wrapped around ten lines catches errors from
all ten, and you no longer know which one you were expecting.

`finally` is for cleanup: closing files, tidying up. It runs even if the
`try` returned early or raised something you didn't catch. "Close the file
whatever happens" is so common that Python gave it its own keyword, and
that keyword is `with`:

```python
with open(path, encoding="utf-8") as f:
    header = f.readline()
# f is closed here, even if readline raised
```

You've been writing `with open(...)` since lesson 002. Now you know what
it's short for. Anything that says "do this at the end, without fail" and
isn't a file probably wants `finally`; anything that is a file wants `with`.

## 5. Validate before it hurts

The traceback in section 1 surfaced in `int()`, three functions deep. The
`'two'` had been sitting in the row since it was read. The further a bad
value travels before anyone checks it, the harder it is to work out where
it came from, so **check at the door**: the moment data comes in from
outside (a file, a user, a website), decide whether it's acceptable, and
refuse it with a clear message if it isn't.

```python
def validate_row(raw):
    """One raw CSV row -> one clean dict. Raises ValueError naming the column and the bad value."""
    row = {key: (value or "").strip() for key, value in raw.items() if key is not None}

    missing = [field for field in RAW_FIELDS if not row.get(field)]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")

    try:
        when = date.fromisoformat(row["date"])
    except ValueError as err:
        raise ValueError(f"date {row['date']!r} is not YYYY-MM-DD") from err

    drink = row["drink"].lower()
    if drink not in MENU:
        raise ValueError(f"drink {row['drink']!r} is not on the menu")

    # ... size, price, quantity the same way ...

    quantity = int(row["quantity"])          # inside its own try, message names the column
    if quantity < 1:
        raise ValueError(f"quantity {quantity} must be at least 1")

    return {"date": when.isoformat(), "drink": drink, "size": size,
            "price_cents": price_cents, "quantity": quantity}
```

Things to notice:

- **`raise` is how you throw an error yourself.** `raise ValueError("...")`
  stops the function exactly like `int("two")` would, but with *your*
  message. Use `ValueError` for "the data is wrong", the same word Python
  uses for it.
- **The message names the column and quotes the value.** `date '09/09/2026'
  is not YYYY-MM-DD` tells you which row to open and what to change,
  without reading a line of code. `"bad row"` tells you nothing. The `!r`
  puts quotes round the value so that `' 2'` and `''` are visible.
- **Cheapest checks first, first failure wins.** Missing fields are checked
  before anything is parsed, so a short row gets `missing price, quantity`
  rather than a confusing `None` error. A row with two problems reports the
  first; the person fixing it will find the second on the next run.
- **`raise ... from err`.** When you catch one error and raise a clearer
  one, `from err` keeps the original attached. A traceback then shows both,
  joined by *"The above exception was the direct cause of the following
  exception"*. You get your friendly message *and* Python's precise one.
- **A wrong type is not the same as a wrong value.** `int("-1")` works.
  `-1` cups is still wrong. Type conversion and business rules are two
  separate checks, and validation is where both live.
- **It returns the clean row.** Validation and conversion happen in one
  place, once. Everything after `validate_row` can trust its input
  completely, which is what lets `quantity_of`-style functions stay short.

The classic name for checking first is **look before you leap**; for trying
and catching it's **easier to ask forgiveness than permission**. Python
leans towards the second for conversions (`try: int(x)` beats trying to
predict what `int` will accept) and the first for rules (`if quantity < 1`).
Use whichever reads more clearly. Just do one of them at the door.

## 6. When not to swallow an error

Every `except` block does one of exactly three honest things:

1. **Recover, with a fallback you could defend out loud.** `parse_quantity`
   returns `None` for "not a quantity". You can explain that to a colleague
   in one sentence, and `None` can't be mistaken for a real count.
2. **Record it and move on.** Write the bad row and the reason somewhere
   a human will see, then continue with the next row. Section 7.
3. **Add context and re-raise.** You can't fix it, but you can make the
   message better:

```python
def load_export(path):
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"No till export at {path}. Copy this week's export into "
            f"{path.parent.name}/ and name it {path.name}."
        ) from err
```

Still an error. Still stops the program. But now it says what was expected
and what to do, which is what the next person (usually you) needs at 11pm.

And the one dishonest thing: `except: pass`, or its cousins `return 0`,
`return ""`, `return None` with no thought behind them. The error still
happened. You've just removed the evidence:

```python
def load_export_quietly(path):
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

text = load_export_quietly(missing_file)    # ''
len(text.splitlines())                      # 0 rows this week
```

Zero rows, zero revenue, no error. The weekly report goes out saying the
shop sold nothing, and nobody finds out until the owner phones. Swallowing
an error doesn't remove the bang. It moves it somewhere else, later, where
nobody's looking.

**Rule of thumb: catch an error where you can *do* something about it.** If
you can't, don't catch it. A loud crash with a good traceback is a gift to
whoever debugs it next.

## 7. Putting it together: good rows, rejected rows, nothing hidden

```python
def load_orders_carefully(path):
    """Returns (orders, rejects). A missing file is NOT caught: the caller should hear about it."""
    orders, rejects = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for line_number, raw in enumerate(csv.DictReader(f), start=2):
            try:
                orders.append(validate_row(raw))
            except ValueError as err:
                rejects.append({"line": line_number, "reason": str(err),
                                **{field: raw.get(field) or "" for field in RAW_FIELDS}})
    return orders, rejects

orders, rejects = load_orders_carefully(MESSY_CSV)
write_rejects(rejects, REJECTS_CSV)
```

```
orders_messy.csv: 19 rows read, 15 good, 4 rejected.

Rejected, with the line number in the file and the reason:
  line  8  missing price
  line 12  date '09/09/2026' is not YYYY-MM-DD
  line 16  missing price, quantity
  line 20  quantity -1 must be at least 1
4 rows written to output/rejected_rows.csv for a human to fix.

latte          £29.90
cappuccino     £20.00
flat white     £15.60
espresso        £8.80
tea             £5.40
total          £79.70   (23 cups, 15 orders)
```

£79.70 and 23 cups: the same week as lessons 005, 006 and 007. The four bad
rows didn't crash the report *and* didn't quietly change the numbers.
Both halves of that matter.

Look at the decisions, because they're the whole lesson:

- **The `try` wraps one line**, `validate_row(raw)`, and catches one kind,
  `ValueError`, the kind `validate_row` promises to raise for bad data. A
  `KeyError` from a typo in `RAW_FIELDS` would come straight through, loud,
  as it should.
- **Rejects are recorded, not dropped.** Line number (`enumerate(...,
  start=2)`, because line 1 is the header), the reason, and the raw values,
  written to a CSV with this morning's `DictWriter`. The owner can open it
  in a spreadsheet and fix four rows in two minutes.
- **The missing-file case is not caught.** Bad *rows* are normal; we
  planned for them. A missing *file* means something is wrong with the
  setup, and the right response is to stop and say so. The last thing
  `lesson.py` does is prove that a wrong path still raises
  `FileNotFoundError`.
- **The asserts at the end** check the totals *and* the reject count. A
  validator that rejected nothing would be as wrong as one that rejected
  everything.

This is the shape of every data loader you'll write from here on: validate
at the door, keep the good, record the bad, crash on the unexpected. Lesson
012's mini-project is this with more columns.

## What you can do now

- Read a traceback from the bottom: what, where, who called it; and find
  the last frame that's in your own code.
- Recognise `ValueError`, `TypeError`, `KeyError`, `IndexError`,
  `AttributeError`, `FileNotFoundError`, `ZeroDivisionError`, `NameError`
  and `SyntaxError`, and say what each one usually means.
- Write `try`/`except` around one line, naming the one error you expect,
  and explain why a bare `except:` is worse than no `except` at all.
- Use `else`, `finally`, and know that `with` is `finally` for files.
- Validate input at the boundary with a function that returns clean data
  or raises `ValueError` with a message naming the column and the value.
- Use `raise ... from err` to add context without losing the original.
- Name the three honest things an `except` block can do, and choose one
  on purpose.

## What to do now

1. Run `lesson.py` and read the traceback in section 1 slowly, bottom to
   top, with the code open next to it.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   gives `validate_row` two new rules and a nastier file.
3. That's Week 2, Day 3. Lessons 009 (modules, scripts and `__main__`) and
   010 (a little bit of classes) arrive tomorrow. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Today was the day the coffee shop's data started living in files, and the
day you stopped being afraid of red text. The first makes everything from
here on realistic. The second makes everything from here on possible.
