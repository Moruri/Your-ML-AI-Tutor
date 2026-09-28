# Lesson 014 - Exercises

Three quick checks, a hands-on task that works out a price rise in NumPy,
and an optional question about when a list is still the right tool.
Predict first, then run.

Activate your venv first. For the hands-on task, work in a copy. From the
repo root:

```bash
cp lessons/phase-1-data-science/lesson-014-numpy-arrays-why-not-just-lists/lesson.py my_lesson_014.py
python my_lesson_014.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-014-numpy-arrays-why-not-just-lists").resolve()
```

`load_columns` and `ORDERS_CSV` are there to reuse.

---

## 1. What do you get?

Predict the result (value *and* dtype where it matters) of each:

```python
import numpy as np

a = np.array([1, 2, 3])
a * a
a / 2
a // 2
a + [10, 20, 30]
np.array([1, 2, 3]) == np.array([1, 5, 3])
np.array(["1", "2"]) * 2
```

<details>
<summary>Check yourself</summary>

```
array([1, 4, 9])                  int64: element by element
array([0.5, 1. , 1.5])            float64: / always gives floats, as in plain Python
array([0, 1, 1])                  int64: // is floor division
array([11, 22, 33])               the list is turned into an array first
array([ True, False,  True])      == compares element by element
UFuncTypeError (or TypeError)     you can't multiply strings by 2 in NumPy
```

The last one is the "stray text in a column" problem from section 4.
`np.array(["1", "2"]).astype(int) * 2` gives `array([2, 4])`.

</details>

## 2. Shapes

What's the `.shape` of each?

```python
np.array([5, 6, 7, 8])
np.array([[5, 6, 7, 8]])
np.array([[5], [6], [7], [8]])
np.zeros((2, 3))
np.arange(24).reshape(4, 6)
np.arange(24).reshape(2, 3, 4)
```

<details>
<summary>Check yourself</summary>

```
(4,)          one dimension, four long
(1, 4)        one row, four columns: note the double brackets
(4, 1)        four rows, one column
(2, 3)        np.zeros takes the shape as a tuple
(4, 6)
(2, 3, 4)     three dimensions: two blocks of 3 rows by 4 columns
```

`(4,)`, `(1, 4)` and `(4, 1)` all hold the same four numbers, and NumPy
treats them differently. That difference is behind a lot of confusing
errors, and it's exactly what lesson 015's broadcasting rules are about.

</details>

## 3. Spot the bug

```python
import numpy as np

prices = np.array([3.80, 2.20, 4.20])
price_cents = prices.astype(int) * 100
print(price_cents)
```

The author expected `[380 220 420]`. What did they get, and what's the
fix?

<details>
<summary>Check yourself</summary>

`[300 200 400]`. `.astype(int)` truncated 3.80 to 3 *before* multiplying.
Swap the order and round: `np.round(prices * 100).astype(int)` gives
`[380 220 420]`. The `np.round` matters too: `2.20 * 100` is
`220.00000000000003` in floating point, and something like
`0.29 * 100` is `28.999999999999996`, which would truncate to 28. Always
round before converting floats to ints. (It's what `load_columns` in
`lesson.py` does.)

</details>

## 4. Hands-on: the price rise

The owner wants to put every price up by 5%, rounded to the nearest 10p,
and asks what last week would have taken at the new prices.

**a) Load.** Use `load_columns(ORDERS_CSV)` to get the three arrays.

**b) New prices.** Make `new_cents`: each price times 1.05, rounded to
the nearest 10p, as an `int` array. (Rounding to the nearest 10 is: divide
by 10, round, multiply by 10.)

**c) The difference.** Revenue at new prices, and how much more that is
than £79.70. Which single order gains the most?

**d) Big orders.** How many orders were over £5 at the *old* prices, and
what fraction of all orders is that? (A comparison, then `.sum()` and
`.mean()`.)

**e) Mean vs median.** The mean order at old prices, and the median
(`np.median(...)`). Which is bigger, and does that match what lesson 012
found?

**f) Check.** `assert` the new total and the count from (d).

Hints, if you want them:

- For (b): `np.round(price_cents * 1.05 / 10) * 10`, then `.astype(int)`.
- For (c), `np.argmax(gain)` gives the *position* of the biggest value.
  Use it to look up the drink: `cols["drink"][np.argmax(gain)]`.
- For (d), `(revenue > 500).mean()` is the fraction directly: the mean of
  True/False values is the share that are True.

<details>
<summary>Expected results</summary>

```
b) new_cents: [400 230 440 450 410 230 390 400 260 410 350 230 440 400 300]
c) new total £83.70, which is £4.00 more
   biggest gain: flat white (3 x 20p = 60p)
d) 6 orders over £5, 40% of orders
e) mean £5.31, median £4.30: mean is bigger
```

Overall, the rise came out almost exactly right: £4.00 on £79.70 is
5.0%. But not order by order. Rounding to 10p pushed some prices up by more
than 5% (a £3.30 small latte becomes £3.50, a 6% rise) and others by less
(£4.30 becomes £4.50, only 4.7%). When someone says "5%", it's worth
checking what rounding does to it.

(e) matches lesson 012: a few multi-drink orders pull the mean above
the median.

</details>

<details>
<summary>One way to write it</summary>

```python
cols = load_columns(ORDERS_CSV)
price_cents, quantity, drink = cols["price_cents"], cols["quantity"], cols["drink"]

# b)
new_cents = (np.round(price_cents * 1.05 / 10) * 10).astype(int)
print(new_cents)

# c)
old_rev = price_cents * quantity
new_rev = new_cents * quantity
gain = new_rev - old_rev
print(f"new total £{new_rev.sum() / 100:.2f}, which is £{gain.sum() / 100:.2f} more")
i = np.argmax(gain)
print(f"biggest gain: {drink[i]} ({quantity[i]} x {new_cents[i] - price_cents[i]}p = {gain[i]}p)")

# d)
big = old_rev > 500
print(f"{big.sum()} orders over £5, {big.mean():.0%} of orders")

# e)
print(f"mean £{old_rev.mean() / 100:.2f}, median £{np.median(old_rev) / 100:.2f}")

# f)
assert new_rev.sum() == 8370
assert big.sum() == 6
```

No loops anywhere. Every line is "do this to every order" written once.

</details>

## 5. (Optional) When is a list still better?

Name two situations where you'd keep a plain list rather than convert to a
NumPy array.

<details>
<summary>Check yourself</summary>

Good answers include:

- **Mixed types**, like a row with a date, a drink name and a price. An
  array would turn it all into strings. (Tables of mixed columns are what
  pandas is for, from lesson 017.)
- **Growing one item at a time.** `list.append` is cheap. Arrays have a
  fixed size, so `np.append` makes a whole new copy every time, which is
  slow in a loop. Collect into a list, then convert once at the end.
- **Small, non-numeric collections**: a list of file paths, drink names to
  loop over, lines of a report. NumPy's speed only matters for lots of
  numbers.
- **Ragged data**, where each item has a different length (each
  customer's list of orders). Arrays must be rectangular.

A rule of thumb: *lists for collecting and for things; arrays for doing
maths on many numbers at once.*

</details>

---

That's Week 3, Day 1 done. Lessons 015 and 016 arrive tomorrow
(Tuesday). If one thing sticks from today, let it be: *an array is many
numbers of one type, and maths on it happens to every element at once*.
