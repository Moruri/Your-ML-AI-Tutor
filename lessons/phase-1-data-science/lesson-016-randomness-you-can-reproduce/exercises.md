# Lesson 016 - Exercises

Three quick checks, a hands-on simulation that answers a real question for
the owner, and an optional puzzle you can solve by simulation. Predict
first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-016-randomness-you-can-reproduce/lesson.py my_lesson_016.py
python my_lesson_016.py
```

This lesson has no data files, so no `HERE` change is needed.
`simulate_month`, `DRINKS`, `DRINK_SHARE` and `PRICE_CENTS` are there to
reuse.

---

## 1. Same or different?

For each pair, will `x` and `y` be identical?

```python
# (a)
x = np.random.default_rng(1).integers(0, 100, size=3)
y = np.random.default_rng(1).integers(0, 100, size=3)

# (b)
rng = np.random.default_rng(1)
x = rng.integers(0, 100, size=3)
y = rng.integers(0, 100, size=3)

# (c)
x = np.random.default_rng().integers(0, 100, size=3)
y = np.random.default_rng().integers(0, 100, size=3)

# (d)
x = np.random.default_rng(1).integers(0, 100, size=3)
y = np.random.default_rng(1).integers(0, 100, size=6)[:3]
```

<details>
<summary>Check yourself</summary>

- **(a) Identical.** Two generators, same seed, same call.
- **(b) Different** (almost certainly). One generator, called twice: the
  second call continues the sequence.
- **(c) Different.** No seed, so each generator starts somewhere
  unpredictable.
- **(d) Identical.** Asking for six numbers produces the same first three
  as asking for three. The sequence doesn't depend on how many you take.
  (This holds for `integers` like here; don't rely on it for every
  method, but it's a nice property when it's there.)

</details>

## 2. Which draw?

Which `rng` method would you use to simulate each?

- (a) The number of customers who arrive between 8 and 9am.
- (b) Whether each of 40 customers leaves a tip, if 15% do.
- (c) Picking 5 different customers from 200 for a survey.
- (d) Next month's average daily temperature, if it's usually 14°C give or
  take 3.
- (e) The order in which 12 cakes are displayed.

<details>
<summary>Check yourself</summary>

- **(a) `poisson`**: a count of independent arrivals in a fixed time.
- **(b) `binomial(40, 0.15)`** for how many tip, or
  `rng.random(40) < 0.15` for *which* ones (a True/False per customer,
  lesson 015's masks).
- **(c) `choice(200, size=5, replace=False)`**. Without `replace=False`
  you might survey someone twice.
- **(d) `normal(14, 3)`**: a value near a typical level, symmetric spread.
- **(e) `permutation(12)`** (or `permutation` of the list of cakes).

</details>

## 3. Spot the bug

```python
def simulate_day(customers=50):
    rng = np.random.default_rng(42)
    return rng.choice(DRINKS, size=customers, p=DRINK_SHARE)

week = [simulate_day() for _ in range(5)]
```

The author wanted five different, reproducible days. What did they get,
and how would you fix it?

<details>
<summary>Check yourself</summary>

Five *identical* days. The generator is created, with the same seed,
inside the function, so every call starts the sequence from the beginning.

Fix: make the generator once, outside, and pass it in (section 5's habit):

```python
def simulate_day(rng, customers=50):
    return rng.choice(DRINKS, size=customers, p=DRINK_SHARE)

rng = np.random.default_rng(42)
week = [simulate_day(rng) for _ in range(5)]
```

Five different days, and the same five every time you run it.

</details>

## 4. Hands-on: is a £4,500 month bad luck, or bad news?

Last month the shop took £4,480. The owner is worried. You know the shop
averages about 60 customers a day. Is £4,480 unusual for a normal month,
or well within the range that chance alone produces?

**a) Simulate.** With `rng = np.random.default_rng(2026)` (made once),
call `simulate_month(rng)` 2,000 times and collect each month's total
revenue into an array.

**b) Typical month.** The mean of your 2,000 totals, and the range that
the middle 90% falls in: `np.percentile(totals, [5, 95])`.

**c) How unusual?** What share of simulated months took *less than*
£4,500?

**d) What if?** A new sign outside is expected to bring the average up to
65 customers. Re-run (a) to (c) with `mean_customers=65`, using a fresh
`default_rng(2026)`. What share of months would now fall under £4,500?

**e) Sanity check.** Work out the *expected* monthly revenue without
simulating: 22 days x 60 customers x the average price of a drink, where
the average price is `(DRINK_SHARE * PRICE_CENTS).sum()`. Is it close to
your mean from (b)?

Hints, if you want them:

- `np.array([simulate_month(rng)[1].sum() for _ in range(2000)])` does (a)
  in one line. `simulate_month` returns two arrays; `[1]` is revenue.
- Work in cents throughout and divide by 100 only when printing.

<details>
<summary>Expected results</summary>

```
b) mean £4,541.53; middle 90% from £4,329.13 to £4,757.91
c) 36.5% of months took less than £4,500
d) with 65 customers: mean £4,922.88; 0.05% of months under £4,500
e) 22 x 60 x £3.442 = £4,543.44   (the simulation's £4,541.53 is very close)
```

So £4,480 is *not* unusual. More than a third of perfectly ordinary
simulated months come in under £4,500. The honest answer to the owner is
"that's within normal wobble; I wouldn't read anything into one month".

(d) is the flip side: if the sign really works, a month under £4,500
becomes almost impossible (one in two thousand). So if the sign goes up
and the next month is £4,480 again, *that* would be real evidence the
sign isn't doing much. Simulation tells you not only what's likely, but
what would count as a surprise. Lesson 027 turns this idea into
hypothesis testing.

</details>

<details>
<summary>One way to write it</summary>

```python
def simulate_totals(mean_customers, months=2000, seed=2026):
    rng = np.random.default_rng(seed)
    return np.array([simulate_month(rng, mean_customers=mean_customers)[1].sum()
                     for _ in range(months)])

for customers in (60, 65):
    totals = simulate_totals(customers)
    low, high = np.percentile(totals, [5, 95])
    print(f"{customers} customers/day: mean £{totals.mean() / 100:,.2f}; "
          f"middle 90% £{low / 100:,.2f} to £{high / 100:,.2f}; "
          f"under £4,500 in {(totals < 450_000).mean():.2%} of months")

average_price = (DRINK_SHARE * PRICE_CENTS).sum()
print(f"expected: 22 x 60 x £{average_price / 100:.3f} = £{22 * 60 * average_price / 100:,.2f}")
```

Wrapping the simulation in a function with the seed as an argument means
both runs in (d) start from the same place, so the only difference between
them is the number of customers. That's how you make a fair comparison
with random numbers.

</details>

## 5. (Optional) The shared-birthday puzzle, coffee edition

The shop has 30 regulars. Each has a favourite day of the year to treat
themselves (any of 365, equally likely). What's the chance that at least
two regulars share the same day?

Guess first. Then simulate 100,000 groups of 30.

<details>
<summary>Check yourself</summary>

About **71%**. Most people guess something like 10%.

```python
rng = np.random.default_rng(16)
days = rng.integers(0, 365, size=(100_000, 30))     # a row per simulated group
sorted_days = np.sort(days, axis=1)
has_match = (np.diff(sorted_days, axis=1) == 0).any(axis=1)
print(has_match.mean())                              # about 0.705
```

Sorting each row puts equal days next to each other; `np.diff` is zero
exactly where two neighbours match; `.any(axis=1)` asks "any match in this
row?". Lessons 014 to 016 in five lines.

Why is it so high? Because it's not about *your* day matching someone's;
it's about *any* pair matching, and 30 people make 435 pairs. Our intuition
for chance is often badly wrong, which is exactly why simulating is such a
good habit.

</details>

---

That's Week 3, Day 2 done. Tomorrow, lessons 017 and 018 introduce
pandas. If one thing sticks from today, let it be: *seed it, pass the
rng in, and when you don't know the formula, simulate and count*.
