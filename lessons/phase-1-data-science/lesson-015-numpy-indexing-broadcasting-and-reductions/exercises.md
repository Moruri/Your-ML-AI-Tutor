# Lesson 015 - Exercises

Three prediction questions (the real test for today), a hands-on task on
the week's revenue table, and an optional puzzle about views. Predict
first, *then* run. For broadcasting, write the shapes down.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-015-numpy-indexing-broadcasting-and-reductions/lesson.py my_lesson_015.py
python my_lesson_015.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-015-numpy-indexing-broadcasting-and-reductions").resolve()
```

`WEEK`, `DAYS`, `DRINKS`, `MEDIUM_CENTS` and `load_week_revenue` are there
to reuse.

---

## 1. Will it broadcast?

For each pair, say whether `a + b` works, and if so, the shape of the
result.

```
(a)  a: (5, 5)     b: (5,)
(b)  a: (5, 5)     b: (1, 5)
(c)  a: (4, 3)     b: (4,)
(d)  a: (4, 3)     b: (4, 1)
(e)  a: (3,)       b: (4, 1)
(f)  a: (2, 3, 4)  b: (3, 1)
```

<details>
<summary>Check yourself</summary>

Right-align, then compare pairs from the right:

```
(a) (5, 5) + (5,)      -> (5, 5)   5=5; missing stretches
(b) (5, 5) + (1, 5)    -> (5, 5)   5=5; 1 stretches
(c) (4, 3) + (4,)      -> ERROR    3 vs 4: not equal, neither is 1
(d) (4, 3) + (4, 1)    -> (4, 3)   3 vs 1: stretches; 4=4
(e) (3,)   + (4, 1)    -> (4, 3)   3 vs 1: stretches; missing vs 4: stretches
(f) (2, 3, 4) + (3, 1) -> (2, 3, 4)   4 vs 1; 3=3; missing vs 2
```

(c) is the classic: four *row* values against a table with four rows
doesn't work, because a `(4,)` lines up with the *columns*. (d) is the
fix. (e) surprises people: two small arrays make a bigger one, an
"outer" sum where every value of `a` meets every value of `b`. It's
genuinely useful whenever you need every combination of two lists.

</details>

## 2. Which axis?

`sales` has shape `(52, 7)`: a row per week of the year, a column per day
of the week. What's the shape of each result, and what does it mean in
words?

```python
sales.sum(axis=0)
sales.sum(axis=1)
sales.mean(axis=0).argmax()
sales.sum(axis=1, keepdims=True)
sales / sales.sum(axis=1, keepdims=True)
```

<details>
<summary>Check yourself</summary>

```
(7,)       total per day of the week, over the whole year
(52,)      total per week
a number   which day of the week is busiest on average (0 = the first column)
(52, 1)    weekly totals, kept as a column
(52, 7)    each day as a share of its own week; every row sums to 1
```

The axis you name disappears: `axis=0` removes the 52 weeks, `axis=1`
removes the 7 days.

</details>

## 3. Masks

```python
q = np.array([2, 1, 3, 1, 2])
p = np.array([380, 220, 420, 250, 390])
```

Predict each (or say which one raises an error, and why):

```python
p[q == 1]
p[(q > 1) & (p < 400)]
p[q > 1 and p < 400]
q[p > 1000]
(p > 300).mean()
```

<details>
<summary>Check yourself</summary>

```
[220, 250]
[380, 390]
ValueError: The truth value of an array with more than one element is ambiguous...
[]                  an empty array: nothing matched, which is not an error
0.6                 three of five prices are over 300
```

The third needs `&` and brackets. Whenever you see "truth value ...
ambiguous", look for `and`, `or`, `not`, or an `if` that's been handed a
whole array.

</details>

## 4. Hands-on: questions for the revenue table

Start from `table = load_week_revenue(ORDERS_CSV)` (5 days by 5 drinks,
in pence). Answer each with array operations only; no `for` loops over
the numbers.

**a) Share of the week.** Each day's revenue as a share of the week's
total. Print as percentages.

**b) Cappuccino beat latte.** On which days did cappuccino take more than
latte? (Compare two columns, then use the mask on `np.array(DAYS)`.)

**c) Above average.** Each drink's average daily revenue is
`table.mean(axis=0)`. Make `diff = table - table.mean(axis=0)`: each
cell's difference from its drink's average. Before running, write down
the two shapes and say which way the broadcast goes. Then find the single
cell that's furthest *above* its drink's average: which day, which drink,
by how much?

**d) The Friday what-if.** If Friday had been 50% busier (every drink),
what would the week have taken? Use a `(5, 1)` factor array.

**e) Check.** `assert` your answers to (b) and (d).

Hints, if you want them:

- For (c), `diff.argmax()` gives a position in the *flattened* array (0
  to 24). `np.unravel_index(diff.argmax(), diff.shape)` turns it back
  into `(row, column)`.
- For (d), `np.array([1, 1, 1, 1, 1.5]).reshape(5, 1)`. Using the `(5,)`
  version would run without an error on this square table, and scale
  *tea* instead. Try both and compare.

<details>
<summary>Expected results</summary>

```
a) Mon 18%, Tue 26%, Wed 17%, Thu 16%, Fri 24%
b) ['Wed' 'Fri']
c) table (5, 5) - mean (5,) -> per column: each drink minus its own average
   furthest above average: Tue, flat white, +858p (£8.58)
d) £89.15
```

(c) makes sense when you look at the data: Tuesday's three flat whites
(£11.70) are by far the week's biggest flat white sale (Thursday sold one,
and no other day any), so Tuesday stands far above that drink's daily
average of £3.12. (d) with the wrong-shaped factor gives £82.40 instead, because it scales the tea column: a quiet,
plausible-looking, wrong answer.

</details>

<details>
<summary>One way to write it</summary>

```python
table = load_week_revenue(ORDERS_CSV)
days, drinks = np.array(DAYS), np.array(DRINKS)

# a)
share = table.sum(axis=1) / table.sum()
print(", ".join(f"{d} {s:.0%}" for d, s in zip(DAYS, share)))

# b)
capp_won = table[:, 2] > table[:, 0]
print(days[capp_won])

# c)  (5, 5) - (5,): right-aligned, 5 = 5, the missing dimension stretches down the rows,
#     so every ROW has the column averages subtracted. Per drink. That's what we want.
diff = table - table.mean(axis=0)
row, col = np.unravel_index(diff.argmax(), diff.shape)
print(f"{days[row]}, {drinks[col]}, +{diff[row, col]:.0f}p")

# d)
friday = np.array([1, 1, 1, 1, 1.5]).reshape(5, 1)
what_if = (table * friday).sum()
print(f"£{what_if / 100:.2f}")

# e)
assert list(days[capp_won]) == ["Wed", "Fri"]
assert what_if == 8915
```

The `zip` in (a) is only for printing: the maths is one line. Loops for
*display* are fine. It's loops for *arithmetic* that NumPy replaces.

</details>

## 5. (Optional) View or copy?

For each, does changing `b` afterwards change `a`?

```python
a = np.arange(10)
b = a[2:6]           # (i)
b = a[[2, 3, 4, 5]]  # (ii)
b = a[a > 5]         # (iii)
b = a                # (iv)
b = a.copy()         # (v)
```

<details>
<summary>Check yourself</summary>

- **(i) Yes.** A slice is a view.
- **(ii) No.** Indexing with a list of positions makes a copy.
- **(iii) No.** Masks make copies too.
- **(iv) Yes.** That's not even a view; it's the same array with two
  names, like lesson 004's list aliasing.
- **(v) No.** That's what `.copy()` is for.

The honest summary: *slices share, masks and position lists copy*. And
the reassuring bit: `a[a > 5] = 0` (assigning *through* a mask) still
changes `a`, because that's a single operation on `a`, not a change to a
separate copy. When in doubt, `np.shares_memory(a, b)` tells you.

</details>

---

That's the morning of Week 3, Day 2 done. This afternoon, lesson 016
uses arrays to simulate chance. If one thing sticks, let it be: *mask with
`&` and brackets, right-align shapes before you broadcast, and the axis
you name is the one that disappears*.
