# Lesson 005 - Exercises

Three quick checks, one hands-on task with a new (and slightly messy) log,
and an optional stretch. Predict first, then run.

For the hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-0-python/lesson-005-dictionaries-as-tiny-databases/lesson.py my_lesson_005.py
python my_lesson_005.py
```

(No data file today, so the copy runs from anywhere. The `ALIASES` dict in
the script is useful for the hands-on task.)

---

## 1. Loud or quiet?

With `prices = {"latte": 380, "tea": 250}`, what does each line give? One of
them is an error. Which?

```python
prices.get("latte")
prices.get("mocha")
prices.get("mocha", 0)
prices["mocha"]
"mocha" in prices
len(prices)
```

<details>
<summary>Check yourself</summary>

```
380
None          <- .get with no default is None, quietly
0
KeyError      <- [] on a missing key shouts
False         <- in checks keys
2             <- none of the above added anything to the dict
```

The point of the last line: `.get` on a plain dict does *not* create the key.
Compare with exercise 3.

</details>

## 2. Counter, in your head

```python
from collections import Counter
c = Counter("mississippi")
```

What are these? (A string is a sequence of characters, so `Counter` counts
letters.)

```python
c.most_common(2)
c["z"]
len(c)
c.total()
sorted(c)
```

<details>
<summary>Check yourself</summary>

```
[('i', 4), ('s', 4)]      <- ties keep first-seen order; i was seen before s
0                         <- missing key in a Counter is 0, not an error
4                         <- four DISTINCT letters: m, i, s, p
11                        <- eleven letters in total
['i', 'm', 'p', 's']      <- sorting a dict gives you its keys
```

`len` versus `.total()` is the one people mix up: how many *kinds* of thing
versus how many things.

</details>

## 3. The quiet key

Read, don't run yet:

```python
from collections import defaultdict
groups = defaultdict(list)
groups["a"].append(1)
x = groups["b"]
```

What is `len(groups)` now? What does `dict(groups)` look like? Then, if you
add `groups.get("c")`, does `len(groups)` change?

<details>
<summary>Check yourself</summary>

`len(groups)` is `2`. Reading `groups["b"]` created an empty list under
`"b"`, so `dict(groups)` is `{'a': [1], 'b': []}`.

`groups.get("c")` returns `None` and does **not** add `"c"`: `.get` never
triggers the default factory. `len(groups)` is still `2`.

So: `[]` on a `defaultdict` creates; `.get` and `in` don't. If you only
meant to *look*, use `.get` or `in`, or convert to a plain `dict` once
you've finished building.

</details>

## 4. Hands-on: the loyalty card log

Every time a regular scans their loyalty card, the till logs
`(date, customer, drink)`. Staff type the names, so the names are a mess,
and the drink field has the usual creative spellings:

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

Paste it into your copy and answer, one at a time:

**a) Clean it first.** Build a new list `clean` of `(date, name, drink)`
tuples where the name is `strip().lower()`ed and the drink has been through
`ALIASES.get(key, key)` (also stripped and lowercased). Prove it worked by
printing the distinct names: there should be four, not nine.

**b) Cups per customer,** most first. (`Counter` over the names, then
`.most_common()`.)

**c) Each customer's favourite drink.** Group the drinks by customer with a
`defaultdict(list)`, then for each customer make a `Counter` of their drinks
and take `.most_common(1)`. Print the name, the favourite, and "n of total".

**d) Free coffees.** Every fifth cup is free. Using the `Counter` from (b)
and a dict comprehension, build `{name: free_coffees}` where free coffees is
cups `// 5`. Then list who has earned at least one.

**e) Busiest day.** Scans per date, and the single busiest date. This one
is two lines with `Counter`.

Hints, if you want them:

- For (a), a `for date, customer, drink in scans:` loop that appends a new
  tuple to `clean` is fine. Distinct names: `sorted(Counter(...))` or
  yesterday's "seen" list.
- For (c), `Counter(drinks).most_common(1)[0]` gives you a `(drink, n)`
  tuple, which you can unpack straight away.
- For (d), the shape is `{name: n // 5 for name, n in cups.items()}`.
- For (e), `Counter(date for date, *_ in clean)`.

<details>
<summary>Expected results</summary>

```
a) 4 customers: ['amira', 'ben', 'chloe', 'dev']
b) [('amira', 6), ('ben', 5), ('chloe', 4), ('dev', 3)]
c) amira  favourite: latte (4 of 6)
   ben    favourite: espresso (4 of 5)
   chloe  favourite: flat white (3 of 4)
   dev    favourite: cappuccino (3 of 3)
d) {'amira': 1, 'ben': 1, 'chloe': 0, 'dev': 0}
   earned a free coffee: ['amira', 'ben']
e) {'2026-09-07': 3, '2026-09-08': 4, '2026-09-09': 3, '2026-09-10': 3, '2026-09-11': 5}
   busiest: [('2026-09-11', 5)]
```

If you skipped (a) and counted the raw names, you'd have nine "customers"
and nobody would have earned a free coffee. Cleaning first isn't a nicety;
it changes the answer.

</details>

<details>
<summary>One way to write it</summary>

```python
# a)
clean = []
for date, customer, drink in scans:
    name = customer.strip().lower()
    key = drink.strip().lower()
    clean.append((date, name, ALIASES.get(key, key)))
names = sorted(Counter(name for _, name, _ in clean))
print(f"a) {len(names)} customers: {names}")

# b)
cups = Counter(name for _, name, _ in clean)
print(f"b) {cups.most_common()}")

# c)
drinks_by_customer = defaultdict(list)
for _, name, drink in clean:
    drinks_by_customer[name].append(drink)
for name in sorted(drinks_by_customer):
    favourite, n = Counter(drinks_by_customer[name]).most_common(1)[0]
    print(f"c) {name:<6} favourite: {favourite} ({n} of {len(drinks_by_customer[name])})")

# d)
free = {name: n // 5 for name, n in cups.items()}
earned = [name for name, n in free.items() if n > 0]
print(f"d) {free}")
print(f"   earned a free coffee: {earned}")

# e)
per_day = Counter(date for date, *_ in clean)
print(f"e) {dict(sorted(per_day.items()))}")
print(f"   busiest: {per_day.most_common(1)}")
```

Notice that (c) is a *dict of lists* turned into a *Counter per key*: group
first, then count inside each group. That two-step shape ("group by X, then
summarise each group") is what pandas calls `groupby`, and you've just done
it by hand. Notice also that `Counter(name for ...)` takes a generator
directly, no list in between, same as `sum(...)` did yesterday.

</details>

## 5. (Optional) Week on week

Using the `cups` Counter from `lesson.py` section 4 (this week's cups per
drink) and

```python
last_week = Counter({"latte": 6, "espresso": 5, "cappuccino": 4, "flat white": 5, "tea": 2})
```

1. Which drinks sold *more* this week than last, and by how much?
2. Which sold *fewer*?
3. Write one line that gives the change for *every* drink, including
   negatives and zeros. (Hint: a dict comprehension over the union of keys,
   using the fact that a missing key in a Counter is `0`.)

<details>
<summary>Check yourself</summary>

1. `cups - last_week` is `Counter({'latte': 2, 'cappuccino': 1})`.
2. `last_week - cups` is `Counter({'espresso': 1, 'flat white': 1})`.
3. Counter subtraction drops zeros and negatives, so for the full picture
   you build it yourself:

```python
change = {d: cups[d] - last_week[d] for d in sorted(cups.keys() | last_week.keys())}
# {'cappuccino': 1, 'espresso': -1, 'flat white': -1, 'latte': 2, 'tea': 0}
```

`cups.keys() | last_week.keys()` is "every key from either" (the `|` is a
set union; we'll meet sets properly later, but this one use is worth
knowing now). Because a `Counter` gives `0` for a missing key, a drink that
appears in only one week still gets a sensible number.

</details>

---

Done? Go to
[Lesson 006 - Functions that don't lie](../lesson-006-functions-that-dont-lie/README.md).
