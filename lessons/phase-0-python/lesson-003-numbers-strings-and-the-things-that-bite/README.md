# Lesson 003 - Numbers, strings and the things that bite

**Phase 0 - Python foundations for data work** | Week 2, Day 1 | Monday 2026-09-14

> **Goal:** get comfortable with ints vs floats, rounding, f-strings, and the
> string methods you'll use daily, so you can turn messy text like
> `" £3.80 "` into a number you can trust.

Time: about 45 minutes. No installs. Python 3.10+.

---

Welcome back. Hope the weekend was restful.

On Friday you read a CSV and every value came back as text. You converted
`"3.80"` with `float()` and moved on. That works right up until the file has
`"£3.80"`, or `" 3.80 "`, or `"3.80 "` with a sneaky space, or `"free"`. Real
data files are full of this. Not because people are careless, but because
data gets typed by humans, exported by five different programs, and edited in
spreadsheets along the way.

So today is about the three kinds of value you already met (`int`, `float`,
`str`), the surprising ways they behave, and the small toolkit of string
methods that turns messy text into clean numbers. It's not glamorous. It is,
genuinely, half of all data work.

## How to follow along

Same as last time: open a terminal in the repo root, start `python`, and type
the examples at the `>>>` prompt. The full script is `lesson.py` in this
folder:

```bash
python lessons/phase-0-python/lesson-003-numbers-strings-and-the-things-that-bite/lesson.py
```

## 1. Three kinds of value, and moving between them

You know these already:

```python
quantity = 2          # int   - whole numbers
price = 3.80          # float - numbers with a decimal point
drink = "latte"       # str   - text, always in quotes
```

If you're ever unsure what you've got, ask:

```python
type(price)           # <class 'float'>
type("3.80")          # <class 'str'>
```

Converting between them is done with functions named after the type:

```python
int("2")              # 2
float("3.80")         # 3.8
str(3.8)              # '3.8'
int(3.99)             # 3      <- chops off the decimals. Doesn't round!
float(2)              # 2.0
```

Two of these bite people regularly:

- `int(3.99)` is `3`, not `4`. `int()` *truncates*. If you want rounding, use
  `round()`, which we'll get to.
- `int("3.80")` is an **error**, even though `float("3.80")` is fine. `int()`
  from a string only accepts things that already look like whole numbers. If
  you have `"3.80"` and want an int, go via float: `int(float("3.80"))`.

Notice also that `3.8` and `3.80` are the same number. Python just prints the
shortest version. The trailing zero was formatting, not information. If you
want it back, that's what f-strings are for (section 5).

## 2. Floats: the surprise, and why

Type this and look carefully at the result:

```python
0.1 + 0.2             # 0.30000000000000004
```

That is not a bug in Python. Every language on every computer does this. Here
is the honest explanation, once, so it never bothers you again.

Computers store numbers in binary. In decimal, some fractions can't be written
exactly: 1/3 is 0.3333... forever. In binary, *0.1* is one of those forever
fractions. So the computer stores the closest value it can, which is a hair
off. Add two hairs together and you can see the gap.

For most data work this is harmless. A sensor reading of 21.4 degrees isn't
accurate to 17 decimal places anyway. But there are two places it will hurt
you if you don't know:

**Comparing floats with `==`.**

```python
0.1 + 0.2 == 0.3      # False. Ouch.
```

Never ask whether two floats are *exactly* equal. Ask whether they're *close
enough*:

```python
import math
math.isclose(0.1 + 0.2, 0.3)      # True
```

Or round both sides first: `round(0.1 + 0.2, 2) == round(0.3, 2)`.

**Money.** Add `3.90` to itself ten times in a loop and you get
`38.99999999999999`, not `39.0`. Add up a thousand prices and the crumbs
accumulate, and a till that's out by a penny is a till that's wrong. Section
3 is about that.

## 3. Money: use cents, or use `Decimal`

Two good options. Pick one and be consistent.

**Option A: store money as whole cents (or pence) in an `int`.** Integers are
exact. Always. `380 + 220` is `600`, full stop. You convert to pounds only
when printing.

```python
price_cents = 380
quantity = 2
total_cents = price_cents * quantity           # 760, exactly
print(f"£{total_cents / 100:.2f}")             # £7.60
```

This is what a lot of real payment systems do. It's boring and it works.

**Option B: the `decimal` module.** It stores numbers in base 10, the way you
think about them, so `0.1 + 0.2` really is `0.3`.

```python
from decimal import Decimal
Decimal("0.1") + Decimal("0.2")     # Decimal('0.3')
Decimal("3.80") * 2                 # Decimal('7.60')
```

One rule: **build a `Decimal` from a string, not from a float.**
`Decimal("0.1")` is exactly a tenth. `Decimal(0.1)` faithfully copies the
float's tiny error and you're back where you started. Since CSV values arrive
as strings anyway, this is easy to get right.

For this course: use plain floats for measurements and statistics, and cents
or `Decimal` when a number is money that has to add up. When we get to pandas
in Phase 1 you'll mostly use floats and round at the end, and that's fine for
analysis. Just know the difference exists.

## 4. Rounding, and telling Python how to print numbers

`round()` does what you'd hope, mostly:

```python
round(3.14159, 2)     # 3.14
round(2.5)            # 2    <- wait, what?
round(3.5)            # 4
```

`round()` on a `.5` goes to the nearest *even* number. It's called banker's
rounding, and it exists so that rounding a long column of numbers doesn't
drift upward. It surprises everyone exactly once. Now it's surprised you.

Far more often, you don't want to *change* a number, you want to *display* it
tidily. That's a **format spec**: the bit after the colon inside an f-string's
curly braces. You met `:.2f` on Friday. Here's the family:

```python
n = 1234.5678
f"{n:.2f}"        # '1234.57'      2 decimal places
f"{n:,.2f}"       # '1,234.57'     thousands separators too
f"{n:.0f}"        # '1235'         no decimals (this one does round)
f"{0.256:.1%}"    # '25.6%'        as a percentage
f"{n:>12.2f}"     # '     1234.57' right-aligned in 12 characters
f"{'latte':<10}|" # 'latte     |'  left-aligned in 10 characters
f"{7:03d}"        # '007'          zero-padded to 3 digits
```

Read `:>12.2f` as: `>` align right, `12` characters wide, `.2f` two decimal
places. The alignment ones are how you make columns line up when printing a
table, which you did on Friday without quite knowing why it worked.

## 5. f-strings, properly

You've been using f-strings. Three things you might not know yet.

**You can put any expression in the braces**, not just a variable name:

```python
price, qty = 3.80, 3
print(f"{qty} cups = £{price * qty:.2f}")     # 3 cups = £11.40
```

**`!r` shows the value the way Python sees it.** Invaluable for spotting
whitespace you can't otherwise see:

```python
raw = " latte "
print(f"got {raw}")       # got  latte       <- looks fine, isn't
print(f"got {raw!r}")     # got ' latte '    <- ah. Spaces.
```

**`=` prints the expression and its value.** Pure convenience when
debugging:

```python
total = 7.6
print(f"{total=}")        # total=7.6
```

You'll use `!r` a lot today. When a string "looks right" but a conversion
fails, `!r` is how you find the invisible character causing it.

## 6. String methods for messy data

Strings come with methods: functions you call *on* the string with a dot.
Here are the ones you'll reach for almost every day. Notice that none of them
change the original string. They all hand you a new one, so you keep the
result.

```python
raw = "  Flat White  "

raw.strip()                 # 'Flat White'      remove spaces at both ends
raw.lower()                 # '  flat white  '  all lowercase
raw.upper()                 # '  FLAT WHITE  '
raw.strip().lower()         # 'flat white'      chain them, left to right
```

`strip().lower()` is so common you'll type it in your sleep. It's the standard
way to make `"Latte"`, `"latte "` and `" LATTE"` all become the same thing so
they group together properly.

```python
"£3.80".replace("£", "")    # '3.80'            swap one piece for another
"3,80".replace(",", ".")    # '3.80'            handy for European decimals
"1,250.00".replace(",", "") # '1250.00'
```

```python
"latte,medium,3.80".split(",")     # ['latte', 'medium', '3.80']
"2026-09-14".split("-")            # ['2026', '09', '14']
"-".join(["2026", "09", "14"])     # '2026-09-14'   <- the opposite of split
", ".join(["latte", "tea"])        # 'latte, tea'
```

`split` gives you a list; `join` is called on the *separator* and takes a
list. That backwards-looking `", ".join(...)` trips people up, but it means
join works on any list of strings without the list having to know about it.

```python
name = "flat white"
name.startswith("flat")            # True
name.endswith("tea")               # False
"white" in name                    # True     <- substring check
```

`in` is the one you'll use most for filtering: "does this description mention
'refund'?", "does the filename contain '2026'?".

And two small ones for checking what you've got before converting it:

```python
"42".isdigit()        # True
"4.2".isdigit()       # False   (the dot isn't a digit)
"".strip() == ""      # True    (a blank cell, once you strip it)
```

## 7. Why CSV gives you text, and what to do about it

Now the payoff. Why does the `csv` module hand you strings for everything?
Because a CSV file has no idea what its columns mean. `3.80` in a file is
just the characters `3`, `.`, `8`, `0`. So is `2026-09-14`. So is `latte`.
The *file* can't tell you which of those is a number. Only you can. Which
means cleaning and converting is your job, every time, and the tools above
are how you do it.

Here's a realistic mess. Prices exported from three different systems into
one column:

```python
raw_prices = ["3.80", " 4.20", "£1.15", "3,90", "free", "", "N/A", "1,250.00"]
```

Write down what you *want* out of each one before coding: `380`, `420`, `115`,
`390`, `125000`, and something that means "no price" for `"free"`, `""` and
`"N/A"` (`"free"` is arguably `0`, but we'll treat it as missing and let a
human decide). We'll work in cents, so each answer is an `int` or `None`.
`None` is Python's word for "nothing here", and it's the right thing to
return when there's no sensible number.

```python
def parse_price_cents(text):
    """Turn a messy price string into whole cents, or None if it isn't a price."""
    cleaned = text.strip().lower().replace("£", "")
    if cleaned in ("", "n/a", "free"):
        return None
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")     # "3,90" -> "3.90"
    else:
        cleaned = cleaned.replace(",", "")      # "1,250.00" -> "1250.00"
    return round(float(cleaned) * 100)
```

Read it top to bottom:

1. **Normalise first.** Strip, lowercase, remove the currency symbol. Now
   every input is in the same shape and the rest of the function has fewer
   cases to think about.
2. **Handle "no value" explicitly.** A little tuple of the things that mean
   "missing". You'll add to this list as you meet new files.
3. **Decide what a comma means.** If there's a comma and no dot, it's a
   European decimal. Otherwise it's a thousands separator. That's a judgement
   call about *this* data, and you should write it down, which is what the
   comment is for.
4. **Convert, then go to cents.** Careful here. `float("1.15") * 100` is
   `114.99999999999999` (section 2 again), and `int()` of that is `114`.
   Wrong by a penny, silently. `round()` gives the `115` you meant. This is
   the single most common money bug, and now you've seen it.

Run it over the list:

```python
for raw in raw_prices:
    print(f"{raw!r:>12} -> {parse_price_cents(raw)}")
```

The `!r:>12` combination (repr, then right-align) is exactly the kind of
format spec you'd use to eyeball messy input. Look at the output and check
each line against what you wrote down. That's the whole job: decide what
clean looks like, write a small function, check it against real examples,
fix, repeat.

## What you can do now

- Convert between `int`, `float` and `str`, and predict when it'll fail.
- Explain `0.1 + 0.2` calmly, and compare floats with `math.isclose`.
- Keep money exact with cents or `Decimal`.
- Round on purpose, and format numbers into tidy columns with format specs.
- Use `strip`, `lower`, `replace`, `split`, `join`, `startswith`, `in` to
  clean text.
- Write a small "parse this messy field" function and check it against real
  examples.

If someone hands you a CSV with a horrible price column tomorrow, you can
deal with it. That was the goal.

## What to do now

1. Run `lesson.py` and read the output next to this README.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one gives
   you a messier list than the one above.
3. Then move on to
   [Lesson 004 - Lists and tuples, properly](../lesson-004-lists-and-tuples-properly/README.md).
