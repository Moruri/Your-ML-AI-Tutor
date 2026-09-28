# Lesson 014 - NumPy arrays: why not just lists?

**Phase 1 - Data science basics** | Week 3, Day 1 | Monday 2026-09-21

> **Goal:** create arrays, understand shape and dtype, and see why
> vectorised code is faster and clearer, so that "do this to every
> number" stops needing a loop.

Time: about 45 minutes. Needs the venv from lesson 013 (`numpy`).

---

Here's a line of Python that surprises everyone once:

```python
prices = [3.80, 2.20, 4.20]
prices * 2          # [3.8, 2.2, 4.2, 3.8, 2.2, 4.2]
```

You wanted every price doubled. Python gave you the list twice. That's
not a bug: a list is a general container that can hold anything (numbers,
strings, other lists, a mixture), so `*` can't sensibly mean "multiply
every element". It means "repeat". To double the prices, you write a loop
or a comprehension, and you've been doing that for two weeks.

Data science is mostly "do this to every number". Double these prices.
Add these two columns. Which of these are over 20? Loops work, but they
are slow for big data, and they bury the idea ("revenue is price times
quantity") under the machinery (`for p, q in zip(...)`).

NumPy's array fixes both. It's a container built for one job, holding lots
of numbers of the same type, and on an array, `*` means multiply. Nearly
every data and ML library in Python is built on top of it: pandas stores
its columns as NumPy arrays, scikit-learn takes them as input, PyTorch
tensors are their close cousins. Learn arrays now and a lot of later
code will look familiar.

## How to follow along

Activate your venv (lesson 013), start `python` in the repo root, and type
along. Everyone writes the import the same way:

```python
import numpy as np
```

Full script:

```bash
python lessons/phase-1-data-science/lesson-014-numpy-arrays-why-not-just-lists/lesson.py
```

It writes no files. If you get `ModuleNotFoundError: No module named
'numpy'`, your venv isn't active: lesson 013, section 5.

## 1. Lists and arrays look alike, and behave very differently

```python
arr = np.array([3.80, 2.20, 4.20])
arr * 2         # array([7.6, 4.4, 8.4])
arr + 0.10      # array([3.9, 2.3, 4.3])
```

`np.array(some_list)` makes an array. Printed, it looks almost like a
list. But now `* 2` doubles every element, and `+ 0.10` adds ten pence to
every price. No loop, no comprehension. This style, where one operation
applies to a whole array, is called **vectorised** code, and it's the
main thing to take from today.

## 2. Making arrays

You'll mostly get arrays from data (a file, pandas, a function that
returns one), but these come up constantly:

```python
np.array([2, 1, 1, 1])      # from a list
np.arange(7, 17)            # 7, 8, ..., 16: like range(), as an array
np.arange(0, 1, 0.25)       # 0, 0.25, 0.5, 0.75: steps can be floats
np.linspace(0, 1, 5)        # 5 evenly spaced points from 0 to 1, BOTH ends included
np.zeros(3)                 # [0., 0., 0.]
np.ones(3, dtype=int)       # [1, 1, 1]
np.full(5, 380)             # [380, 380, 380, 380, 380]
```

`arange` follows `range`'s rule: the stop value is excluded. `linspace`
includes both ends and asks for *how many* points, not the step. Use
`linspace` when you want "N points between A and B", which, when you start
plotting in lesson 022, is often.

## 3. Shape, size and dimensions

Every array has a **shape**: a tuple saying how long it is in each
direction.

```python
cups = np.array([4, 2, 3, 1, 2])
cups.shape      # (5,)       one dimension, five long
cups.ndim       # 1
cups.size       # 5          total number of elements
```

`(5,)` with the trailing comma is how Python writes a one-element tuple.
Read it as "five, and that's the only dimension".

Arrays can have more dimensions. A table of cups sold, with a row per day
and a column per drink, is two-dimensional:

```python
week = np.array([
    [2, 1, 1, 0, 0],      # Monday:    latte, espresso, cappuccino, flat white, tea
    [1, 2, 0, 3, 0],      # Tuesday
    [1, 0, 2, 0, 1],
    [2, 1, 0, 1, 0],
    [2, 0, 2, 0, 1],      # Friday
])
week.shape      # (5, 5)      5 rows, 5 columns
week.size       # 25
week.sum()      # 23          every cell added up: our familiar 23 cups
```

Shape is always **(rows, columns)**, in that order. `reshape` rearranges
the same elements into a different shape: `np.arange(12).reshape(3, 4)`
is 0 to 11 laid out in three rows of four.

Unlike a list of lists, an array must be rectangular. Every row the same
length. Ask for `np.array([[1, 2], [3]], dtype=int)` and NumPy refuses.
That strictness is what makes the rest possible.

When a shape isn't what you expected, it's almost always the key to the
bug. Printing `.shape` is the NumPy equivalent of lesson 008's "read the
last line first".

## 4. dtype: one type for every element

A list can hold `[1, "two", 3.0]`. An array holds **one type**, recorded
in its **dtype**:

```python
np.array([1, 2, 3]).dtype        # int64       whole numbers, 64 bits each
np.array([1.5, 2, 3]).dtype      # float64     one float makes them all floats
np.array([1, "two", 3])          # array(['1', 'two', '3'], dtype='<U21')   strings!
```

If you mix types, NumPy picks one type that can hold everything. Ints
and floats become all floats, which is harmless. A string in the mix
turns *everything* into strings, and now `arr * 2` fails. When maths on an
array fails unexpectedly, check `.dtype`: a stray bit of text in a column
is the usual culprit.

To convert, use `.astype`:

```python
np.array(["3.80", "2.20", "4.20"]).astype(float)    # [3.8, 2.2, 4.2]
np.array([3.7, 2.2]).astype(int)                    # [3, 2]    truncated, NOT rounded
```

That second line is lesson 003's `int(3.7)` again: converting to int chops
off the decimals. Use `np.round(...)` first if you mean "nearest".

Two more things fixed types bring:

- **Overflow.** Each dtype has a fixed size. `int8` holds -128 to 127, so
  `np.array([120], dtype=np.int8) + 10` wraps round to `-126`, silently.
  Plain Python ints grow as big as you need, so this is new. The default
  `int64` goes up to about nine quintillion, so you'll rarely hit it, but
  if a count ever turns negative for no reason, this is why.
- **Float precision.** `np.array([0.1, 0.2]).sum() == 0.3` is `False`, just
  as in lesson 003. NumPy's answer is `np.isclose(a, b)`, "equal to within
  a tiny tolerance". Use it whenever you compare floats.

## 5. Vectorised maths and comparisons

Two arrays of the same shape combine element by element:

```python
price_cents = np.array([380, 220, 420, 430, 390])
quantity    = np.array([2, 1, 1, 1, 3])

price_cents * quantity            # [760, 220, 420, 430, 1170]
(price_cents * quantity).sum()    # 3000
```

The first price times the first quantity, the second times the second,
and so on. That line reads exactly like the sentence "revenue is price
times quantity". That's the clarity half of the argument for NumPy, and
it matters as much as the speed.

Comparisons work the same way, and give arrays of `True`/`False`:

```python
quantity > 1              # [ True, False, False, False,  True]
(quantity > 1).sum()      # 2      True counts as 1, False as 0
```

"How many orders were for more than one cup?" is one short line. Lesson
015 uses these True/False arrays to *select* elements, which is where
they get really useful.

Arrays also have methods for the usual summaries: `.sum()`, `.mean()`,
`.min()`, `.max()`, `.std()`. They return NumPy number types (`np.int64`,
`np.float64`), which behave like ordinary numbers; `int(...)` or
`float(...)` turns them into plain Python ones if you need to.

And if shapes don't match, NumPy refuses:

```python
np.array([1, 2, 3]) + np.array([1, 2])
# ValueError: operands could not be broadcast together with shapes (3,) (2,)
```

Good. A silent guess would be worse. (Some mismatched shapes *are*
allowed, through a rule called broadcasting. `arr + 0.10` was already
one. Lesson 015 explains it.)

## 6. Why it's faster

`lesson.py` works out revenue for a million orders both ways:

```
  Revenue from 1,000,000 orders, plain Python:    42.0 ms
  Revenue from 1,000,000 orders, NumPy:            1.1 ms
  About 38x faster, same answer.
```

(Your numbers will differ. The ratio is usually somewhere between 20 and
100.) Why?

A Python list is a row of *pointers*, each leading to a separate Python
object somewhere in memory, and each object carries its type and other
bookkeeping. Adding two of them means Python looks up both types, checks
what `*` means for them, makes a brand-new object for the answer, and
does that a million times.

A NumPy array is one solid block of raw numbers, all the same type,
side by side: `8,000,000 bytes` for a million `int64`s, exactly eight
each. Since NumPy knows every element is an `int64`, it hands the whole
block to a tight loop written in C that just multiplies numbers. No
lookups, no new objects. The loop still happens; it just happens in the
fastest possible place.

That's the trade: an array gives up the flexibility of a list (any type,
any length per row) in exchange for speed and compactness. For columns
of numbers, which is what data is, that's an excellent trade.

## 7. Putting it together

`lesson.py` loads `orders.csv` (the clean week from Phase 0) with the
`csv` module you know, turns each column into an array, and does the
maths vectorised:

```python
cols = load_columns(ORDERS_CSV)
revenue = cols["price_cents"] * cols["quantity"]
revenue.sum()                 # 7970
cols["quantity"].sum()        # 23

for drink in np.unique(cols["drink"]):
    is_drink = cols["drink"] == drink
    revenue[is_drink].sum()
```

```
  total £79.70 from 23 cups; average order £5.31

    cappuccino    £20.00   (3 orders)
    espresso       £8.80   (3 orders)
    flat white    £15.60   (2 orders)
    latte         £29.90   (5 orders)
    tea            £5.40   (2 orders)
```

Compare that with lesson 005's `revenue_by_drink`: the per-order
arithmetic is gone, `np.unique` gives the distinct drinks, and
`revenue[is_drink]` sneaks in a preview of tomorrow's lesson (pick
out the elements where the mask is True). Same £79.70 as always.

## What you can do now

- Explain why `list * 2` repeats and `array * 2` multiplies.
- Create arrays with `np.array`, `arange`, `linspace`, `zeros`, `ones` and
  `full`.
- Read `.shape`, `.ndim`, `.size` and `.dtype`, and use shape to think about
  rows and columns.
- Say what happens when types are mixed, convert with `.astype`, and watch
  out for truncation, overflow and float comparisons.
- Write vectorised arithmetic and comparisons, and summarise with `.sum()`,
  `.mean()`, `.min()`, `.max()`.
- Explain, roughly, why NumPy is so much faster than a Python loop.

## What to do now

1. Run `lesson.py`. Then, in section 6, change `n` to `10_000_000` and see
   whether the speed-up grows or shrinks.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   rewrites a Phase 0 calculation in NumPy.
3. That's Week 3, Day 1. Lessons 015 (indexing, broadcasting and
   reductions) and 016 (randomness you can reproduce) arrive tomorrow. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

From today, when you catch yourself writing `for x in numbers:` to do
arithmetic, pause and ask: could this be one line on an array? Usually,
it can.
