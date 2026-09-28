# Lesson 011 - Exercises

Three quick checks, a hands-on task that streams several files as one, and
an optional puzzle about when laziness surprises you. Predict first, then
run.

For the hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-0-python/lesson-011-iterators-generators-and-lazy-data/lesson.py my_lesson_011.py
python my_lesson_011.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-0-python/lesson-011-iterators-generators-and-lazy-data").resolve()
```

`read_orders`, `fake_orders`, `only_drink`, `with_total` and `pounds` are
all there to reuse. (Running the copy also rebuilds the big file if it's
missing; comment out `section_together()` in `main()` if you'd rather skip
it while you experiment.)

---

## 1. What prints?

```python
def squares(n):
    print("start")
    for i in range(n):
        yield i * i
    print("end")

gen = squares(3)
print("made it")
print(list(gen))
print(list(gen))
```

<details>
<summary>Check yourself</summary>

```
made it
start
end
[0, 1, 4]
[]
```

`squares(3)` runs none of the body, so `made it` comes first. `list(gen)`
pulls every value, printing `start` before the first and `end` after the
last, and *then* the list is printed. The second `list(gen)` gets nothing:
the generator is used up. No error, just empty.

</details>

## 2. List or generator?

For each, would you write `[...]` or `(...)`, and why?

- (a) `total = sum(row["quantity"] ___ for row in read_orders(path) ___)`
- (b) You want the drinks from a file so you can print how many there are
  and then sort them.
- (c) You want to know whether *any* order in a million-row file has a
  quantity over 20.

<details>
<summary>Check yourself</summary>

**(a) Generator** (and, as the only argument, no extra brackets at all:
`sum(row["quantity"] for row in read_orders(path))`). One pass, keep
nothing.

**(b) List.** You need `len()` and `sorted()`, and a generator has no
length and could only be sorted once anyway. `sorted()` would build a list
behind the scenes regardless.

**(c) Generator**, inside `any(...)`. `any` stops at the first `True`, so
if row 12 has a quantity of 200 it reads twelve rows, not a million. A list
comprehension would read the entire file before `any` even started.

</details>

## 3. The groupby surprise

```python
from itertools import groupby

drinks = ["latte", "latte", "tea", "latte", "tea", "tea"]
print([(d, len(list(g))) for d, g in groupby(drinks)])
```

What prints? How would you get `{'latte': 3, 'tea': 3}` instead?

<details>
<summary>Check yourself</summary>

```
[('latte', 2), ('tea', 1), ('latte', 1), ('tea', 2)]
```

Four groups, because `groupby` only joins neighbours. For counts across
the whole list, either sort first (`groupby(sorted(drinks))`) or, simpler,
`Counter(drinks)` from lesson 005. `groupby` is the right tool when the data
is *already* sorted and too big to sort in memory; otherwise a dict or
`Counter` is clearer.

</details>

## 4. Hands-on: four weeks, one stream

The owner has four weekly exports and wants a month's report without
gluing the files together by hand.

**a) Make the files.** Use `fake_orders` to write four small files to
`OUTPUT_DIR`: `week_1.csv` to `week_4.csv`, 500 rows each, with seeds
`1`, `2`, `3` and `4`. (Use `csv.DictWriter` and `writerows`, as
`ensure_big_file` does.)

**b) One stream.** Write a generator function `read_many(paths)` that
yields every order from every file in turn. Use `read_orders` for each
file and `itertools.chain`, or a plain loop with `yield from` (see the
hint).

**c) The report.** In a single pass over `read_many(...)`, work out total
revenue and total cups. Then, separately, use `only_drink` and
`with_total` on a fresh stream to find the single most expensive latte
order (the biggest `total_cents`), with `max(..., key=...)`.

**d) Peek.** Print the first three orders of the month with `islice`,
without reading the rest.

**e) Prove it.** `assert` that the month's cups equal the sum of the four
weeks' cups worked out file by file.

Hints, if you want them:

- `yield from some_iterable` means "yield everything it produces, one at a
  time". So `for path in paths: yield from read_orders(path)` is the whole
  body of `read_many`. `yield from chain.from_iterable(read_orders(p) for p
  in paths)` is the itertools version.
- A generator can only be used once, so (c) needs a fresh `read_many(...)`
  for each question. That's cheap: it's just reading files again.
- `fake_orders` starts every file on 2025-09-01, so all four "weeks" share
  dates. That's fine for this exercise; we only care about totals.

<details>
<summary>Expected results</summary>

```
a) 4 files, 500 rows each
c) month: £10,678.40 from 3,305 cups
   priciest latte order: 3 x large at £4.30 = £12.90
d) first three: tea, cappuccino, latte
e) per-file cups: [832, 837, 801, 835] -> 3,305  (matches)
```

The exact numbers depend on the seeds; if yours differ, check you used
`seed=1` to `seed=4` and 500 rows. The priciest latte is always a large
latte times three: the most any single order can cost with our menu, and
with 2,000 orders at least one turns up.

</details>

<details>
<summary>One way to write it</summary>

```python
# a)
week_paths = []
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for week in range(1, 5):
    path = OUTPUT_DIR / f"week_{week}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "drink", "size", "price", "quantity"])
        writer.writeheader()
        writer.writerows(fake_orders(500, seed=week))
    week_paths.append(path)

# b)
def read_many(paths):
    for path in paths:
        yield from read_orders(path)

# c)
revenue = cups = 0
for row in read_many(week_paths):
    revenue += row["price_cents"] * row["quantity"]
    cups += row["quantity"]
print(f"month: {pounds(revenue)} from {cups:,} cups")

lattes = with_total(only_drink(read_many(week_paths), "latte"))
priciest = max(lattes, key=lambda row: row["total_cents"])
print(f"priciest latte order: {priciest['quantity']} x {priciest['size']} "
      f"at {pounds(priciest['price_cents'])} = {pounds(priciest['total_cents'])}")

# d)
print([row["drink"] for row in itertools.islice(read_many(week_paths), 3)])

# e)
per_file = [sum(row["quantity"] for row in read_orders(p)) for p in week_paths]
print(per_file, sum(per_file))
assert cups == sum(per_file)
```

Look at (c) again: `max` over a generator pipeline. Four files' worth of
rows flow through `read_many`, `only_drink` and `with_total`, and `max`
keeps only the best row seen so far. At no point is there a list of
lattes. When the files become a year of files, the code doesn't change.

</details>

## 5. (Optional) Laziness that bites

```python
from pathlib import Path

def read_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")

lines = read_lines(Path("does_not_exist.csv"))
print("got the lines")
for line in lines:
    print(line)
```

Where does the `FileNotFoundError` appear, and why might that confuse
someone reading the traceback?

<details>
<summary>Check yourself</summary>

`got the lines` prints first, happily. The error appears at the `for`
loop, because `read_lines(...)` didn't run any of its body, including the
`open`. The generator only starts when something asks it for a value.

So the traceback points at the loop, not the line that named the bad
path, which can send you looking in the wrong place. Two habits help:
remember that errors inside generators surface *where the values are
used*, and, when it matters, check inputs *before* making the generator,
in an ordinary function (for example, `if not path.exists(): raise
FileNotFoundError(...)`) that then returns the generator.

</details>

---

That's the morning of Week 2, Day 5 done. This afternoon's lesson 012
puts all of Phase 0 together in one small project. If one thing sticks,
let it be: *`yield` hands out one value and waits; a generator gets used
up; lists are for keeping, generators are for passing through*.
