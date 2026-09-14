# Lesson 003 - Exercises

Three quick checks, then one proper hands-on task with a genuinely messy
input. Do the checks in your head first, then in `python` to confirm.

For the hands-on exercise, work in a copy so the original stays clean. From
the repo root:

```bash
cp lessons/phase-0-python/lesson-003-numbers-strings-and-the-things-that-bite/lesson.py my_lesson_003.py
python my_lesson_003.py
```

(`my_lesson_003.py` is yours to mess up. This lesson has no data file, so the
copy works from anywhere.)

---

## 1. Predict the conversion

Without running them, what does each line give you? One of them is an error.
Which, and why?

```python
int("7")
int(7.9)
float("7")
int("7.0")
str(7) + "0"
```

<details>
<summary>Check yourself</summary>

```
7
7          <- int() truncates, it does not round
7.0
ValueError <- int() from a string only accepts whole-number text
"70"       <- string + string is glue, not addition
```

The last one is the Friday bug again: forget to convert and Python happily
concatenates. If you wanted 7, you needed `int("7.0")`... which fails, so
you needed `int(float("7.0"))`.

</details>

## 2. Close enough

```python
0.1 * 3 == 0.3
```

1. True or False? Type it to check.
2. Write two different lines that give `True` for "are these the same
   amount?".

<details>
<summary>Check yourself</summary>

1. `False`. `0.1 * 3` is `0.30000000000000004`.
2. Either of:

```python
import math
math.isclose(0.1 * 3, 0.3)          # True
round(0.1 * 3, 2) == round(0.3, 2)  # True
```

Or, if it's money, don't be in floats in the first place: `10 * 3 == 30` in
cents is exactly `True` with nothing to think about.

</details>

## 3. Read the format spec

With `s = "tea"` and `n = 3.5`, what exactly does this print? Count the
spaces.

```python
print(f"{s:>6}|{n:.2f}|{n:05.1f}")
```

And what do these two give?

```python
f"{0.5:.0%}"
f"{1234567:,}"
```

<details>
<summary>Check yourself</summary>

```
   tea|3.50|003.5
```

`>6` right-aligns `tea` in six characters (three spaces, then `tea`). `.2f`
is two decimals. `05.1f` is one decimal, padded with zeros to five characters
wide (the dot counts as one).

`f"{0.5:.0%}"` is `'50%'` and `f"{1234567:,}"` is `'1,234,567'`. You don't
need to memorise these; you need to know they exist so you look them up
instead of writing loops to pad strings by hand.

</details>

## 4. Hands-on: the till export from hell

The cafe's till exports orders as lines of text separated by `|` (a "pipe").
Except the staff type the drink names by hand, the price column has been
through two software upgrades, and someone put a letter in the quantity
field. Here's a sample:

```python
raw_lines = [
    "  LATTE | Medium | £3.80 | 2 ",
    "espresso|small|2.20|1",
    "Flat White | medium | 3,90 | 3",
    "TEA|large|free|1",
    "cappuccino | Large | £4.20 | 2",
    "latte|Small| 3.30 |x",
]
```

Paste that into your copy of the script, then, step by step:

**a) Split and strip.** Turn each line into a list of four clean fields.
Print each list with `!r` so you can *see* there's no whitespace left.

**b) Normalise the text.** Lowercase the drink and size so `"LATTE"`,
`"Latte"` and `"latte"` will group together later.

**c) Convert the numbers.** Use the lesson's `parse_price_cents` for the
price. Write a matching `parse_quantity(text)` that returns an `int`, or
`None` if the text isn't a whole number. (`str.isdigit()` is your friend.)

**d) Report.** Skip any line where the price or quantity came back as
`None`, and for the rest print an aligned table: drink, size, unit price in
pounds, quantity, line total. Finish with the number of lines you skipped
and the grand total in pounds. Do the maths in cents, format at the end.

Hints, if you want them:

- For (a), `line.split("|")` gets you most of the way. You still need to
  `strip()` each piece. A small loop that appends `piece.strip()` to a new
  list is fine.
- For (c), `parse_quantity` is three lines: strip, check `isdigit()`, return
  `int(...)` or `None`.
- For (d), format specs like `{drink:<12}` and `{pounds:>7.2f}` line the
  columns up. Remember `cents / 100` to get pounds.

<details>
<summary>Expected results</summary>

```
latte        medium    3.80  x 2  =    7.60
espresso     small     2.20  x 1  =    2.20
flat white   medium    3.90  x 3  =   11.70
cappuccino   large     4.20  x 2  =    8.40
skipped 2 lines (tea: no price; latte: bad quantity)
total: £29.90
```

</details>

<details>
<summary>One way to write it</summary>

```python
def parse_quantity(text):
    cleaned = text.strip()
    if cleaned.isdigit():
        return int(cleaned)
    return None


def split_line(line):
    fields = []
    for piece in line.split("|"):
        fields.append(piece.strip())
    return fields


skipped = 0
total_cents = 0
for line in raw_lines:
    drink, size, price_text, qty_text = split_line(line)     # (a)
    drink, size = drink.lower(), size.lower()                # (b)
    cents = parse_price_cents(price_text)                    # (c)
    qty = parse_quantity(qty_text)
    if cents is None or qty is None:                         # (d)
        skipped += 1
        continue
    line_cents = cents * qty
    total_cents += line_cents
    print(f"{drink:<12} {size:<8} {cents / 100:>5.2f}  x {qty}  = {line_cents / 100:>7.2f}")

print(f"skipped {skipped} lines")
print(f"total: £{total_cents / 100:.2f}")
```

Two things worth noticing. `drink, size, price_text, qty_text = split_line(line)`
pulls four things out of a four-item list in one go; that's called
*unpacking* and lesson 004 makes a lot of use of it. And `continue` means
"stop working on this item and go round the loop again", which is the
tidiest way to skip bad rows.

If your version prints the same numbers with different spacing, it's fine.

</details>

## 5. (Optional) Rounding is weirder than it looks

Try these and explain the results using what you know from section 2:

```python
round(2.675, 2)
round(0.125, 2)
round(0.375, 2)
```

<details>
<summary>Check yourself</summary>

`2.67`, `0.12`, `0.38`.

You might expect `2.68` from the first one. But `2.675` can't be stored
exactly as a float; the nearest value is `2.67499999999999982...`, which is
below the halfway point, so it rounds down. It's not banker's rounding here,
it's the float being slightly less than it looks.

`0.125` *can* be stored exactly (it's 1/8), so that one really is a tie, and
banker's rounding sends it to the even neighbour: `0.12`. `0.375` (3/8) is
also an exact tie, and its even neighbour is `0.38`.

If any of this matters for your numbers (it does for invoices, it doesn't
for temperatures), use `Decimal` and its `quantize` method, which lets you
choose the rounding rule explicitly. You'll meet it when we get to reports.

</details>

---

Done? Go to
[Lesson 004 - Lists and tuples, properly](../lesson-004-lists-and-tuples-properly/README.md).
