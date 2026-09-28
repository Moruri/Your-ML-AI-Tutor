# Lesson 016 - Randomness you can reproduce

**Phase 1 - Data science basics** | Week 3, Day 2 | Tuesday 2026-09-22

> **Goal:** use `numpy.random.default_rng`, seeds, and simple simulations
> to build intuition for chance, so that "how likely is that?" becomes
> something you can answer by trying it a hundred thousand times.

Time: about 45 minutes. Needs the venv from lesson 013 (`numpy`).

---

Randomness turns up all over data science and machine learning, often
where you wouldn't expect it:

- **Sampling**: picking 1,000 rows from a million to look at.
- **Splitting**: dividing data into a training part and a testing part
  (lesson 031), which has to be done randomly to be fair.
- **Training**: most models start from random numbers and some shuffle
  the data as they learn.
- **Simulation**: building a small fake world with known rules, to see
  what *could* happen.

That raises a puzzle. If an analysis uses random numbers, it gives a
slightly different answer every time you run it. So how does anyone check
your work? How do *you* check your own work, next week, when you've
forgotten the details?

The answer is a **seed**: a number that fixes the sequence of "random"
numbers, so the same code gives the same results every time, on any
machine. Today you'll learn to make randomness that's realistic *and*
repeatable, and then use it for something genuinely powerful: answering
probability questions without needing the formula.

## How to follow along

Venv active, `python` in the repo root, `import numpy as np`, and type
along. Full script:

```bash
python lessons/phase-1-data-science/lesson-016-randomness-you-can-reproduce/lesson.py
```

It writes no files. Every number it prints comes from seed `16`, so your
output should match the numbers below exactly, except for the one line that
deliberately uses no seed. (If you're on a very different NumPy version, a
few numbers might differ. The ideas won't.)

## 1. Random, but the same random

Computers can't really be random. They run instructions. What they can do
is follow a recipe that produces numbers that *look* random: no pattern
you could spot, each value equally likely. The recipe needs a starting
point, and that's the seed.

```python
rng = np.random.default_rng(16)
rng.integers(1, 7, size=8)        # [4, 4, 5, 3, 5, 1, 4, 3]   eight dice rolls

rng2 = np.random.default_rng(16)
rng2.integers(1, 7, size=8)       # [4, 4, 5, 3, 5, 1, 4, 3]   the SAME eight rolls
```

`default_rng(seed)` makes a **random number generator**, conventionally
called `rng`. Two generators with the same seed produce the same sequence.
Each call moves along the sequence, so calling the *same* generator again
gives the *next* numbers (`[1, 4, 3, 1, ...]`), but running the whole
script again replays everything from the start.

Without a seed, `default_rng()` starts from a fresh, unpredictable point
each run. That's right for a game and wrong for analysis. **If an
analysis uses randomness, it gets a seed, and the seed goes in the code**
where anyone can see it. Which number you pick doesn't matter at all (42
is a popular joke; the year is common). What matters is writing it down.

One honest warning: you may see older code using `np.random.seed(42)` and
`np.random.rand()`. That's the previous style, with one hidden generator
shared by your whole program. It still works, but `default_rng` is the
recommended way now, and section 5 explains why.

## 2. The draws you'll use

A generator has a method for each kind of randomness. These cover nearly
everything in this course:

```python
rng.random(4)                           # floats between 0 and 1 (1 excluded)
rng.integers(0, 10, size=6)             # whole numbers, 0 to 9 (10 excluded)
rng.choice(DRINKS, size=5)              # pick from a list, all equally likely
rng.choice(DRINKS, size=5, p=DRINK_SHARE)   # pick with weights
rng.choice(10, size=4, replace=False)   # four DIFFERENT numbers from 0-9
rng.permutation(DRINKS)                 # a shuffled copy
rng.normal(60, 8, size=5)               # bell curve: mean 60, spread 8
rng.poisson(60, size=5)                 # counts around 60, like customers per day
rng.binomial(50, 0.33, size=5)          # how many of 50 orders are lattes, at 33% each
```

The first six are "pick things". The last three are **distributions**,
recipes for particular *shapes* of randomness:

- **`normal`**: the bell curve. Most values near the mean, fewer further
  out, symmetrical. Heights, measurement errors, lots of natural things.
- **`poisson`**: counts of independent events in a fixed time. Customers
  through the door in a day, emails per hour. Always whole numbers, never
  negative.
- **`binomial`**: "out of N tries, each with chance p, how many
  succeeded?" Lattes out of 50 orders, heads out of 10 coin flips.

You don't need the maths behind them yet (lesson 023 comes back to
distributions). You just need to recognise which one fits the story you're
simulating.

`p=` for `choice` must be a list of probabilities that add up to 1, one per
option. `DRINK_SHARE = [0.33, 0.25, 0.18, 0.13, 0.11]` says a third of
orders are lattes, a quarter cappuccinos, and so on.

## 3. Big samples settle down

If a third of orders are lattes, what share of lattes will you see in a
sample? `lesson.py` simulates samples of growing size:

```
    n =        10   latte share  30.0%   (off by 3.0%)
    n =       100   latte share  31.0%   (off by 2.0%)
    n =     1,000   latte share  32.9%   (off by 0.1%)
    n =    10,000   latte share  32.4%   (off by 0.6%)
    n =   100,000   latte share  32.9%   (off by 0.1%)
    n = 1,000,000   latte share  32.9%   (off by 0.1%)
```

Small samples wobble; big ones settle close to the true 33%. This is the
**law of large numbers**, and it's the single most useful idea about
chance you'll ever learn. It's also a warning. With ten orders, 30% is
perfectly normal, and so would be 10% or 60%. The coffee shop's two weeks
of data from lesson 012 are a small sample. Any pattern in them might be
real, or might be wobble.

Notice that it doesn't settle *smoothly*: 1,000 happened to land closer
than 10,000. The wobble shrinks as samples grow, but any single sample can
be lucky or unlucky. (Lesson 026 is about putting a number on "how much
wobble should I expect?")

## 4. Simulation answers "how likely?"

Here's a real question for the shop. They make oat-milk lattes, and prep
enough oat milk each morning for 20. On a day with 50 orders, where each
order has a 33% chance of being a latte, how often will they run out?

There's a formula for this (the binomial distribution has one). But you
don't need it. You can *simulate* a hundred thousand days and count:

```python
lattes = rng.binomial(n=50, p=0.33, size=100_000)    # lattes on each simulated day
(lattes > 20).mean()                                 # 0.1164
```

About 11.6% of days, or roughly one day in nine. And once you have the
simulated days, "what if" questions are free:

```
    stock for 20 lattes -> run out on 11.64% of days
    stock for 22 lattes -> run out on  3.72% of days
    stock for 24 lattes -> run out on  0.94% of days
    stock for 26 lattes -> run out on  0.18% of days
```

Now the owner can make a decision: prep for 24, and they'll run out on
roughly two or three days a *year* (0.94% of about 250 trading days).
That's the kind of answer people can act on.

This technique is called **Monte Carlo simulation** (after the casino),
and its power is that it works for problems far too messy for formulas.
Add queues, staff breaks, rainy days and a weekly delivery, and the formula
becomes impossible. The simulation is just a few more lines. The recipe is
always the same: **write down the rules, simulate many times, count how
often the thing happens.**

With 100,000 trials, the answer is good to about a tenth of a percentage
point. More trials, better answer, and NumPy makes 100,000 trials take a
blink.

## 5. Habits that keep randomness reproducible

Seeding is easy to get *mostly* right and still end up with results you
can't reproduce. Four habits prevent that:

1. **Make one generator, with a seed, near the top of your script**, and
   use it everywhere.
2. **Pass it into functions that need randomness**:

   ```python
   def roll_day(rng, customers=50):
       return rng.choice(DRINKS, size=customers, p=DRINK_SHARE)
   ```

   It's lesson 006's "functions that don't lie" again: the function's
   randomness comes in through the front door, visible in its signature,
   rather than from somewhere hidden. You can test it by passing a
   seeded generator and checking the answer.
3. **Avoid the old global style** (`np.random.seed`, `np.random.rand`).
   It's one generator shared by everything in your program, including any
   library you import, so adding an unrelated line of code can silently
   change every random number after it.
4. **Need several independent streams?** `rng.spawn(3)` gives three child
   generators, independent of each other and all reproducible from the one
   seed. Useful for simulating several shops, or running experiments in
   parallel.

## 6. Putting it together: a simulated month

`simulate_month` uses a Poisson draw for customers each day (around 60)
and a weighted `choice` for what each customer orders, then looks up
prices with lesson 015's fancy indexing:

```python
customers = rng.poisson(mean_customers, size=days)
for day, n in enumerate(customers):
    drinks = rng.choice(len(DRINKS), size=n, p=DRINK_SHARE)
    revenue[day] = PRICE_CENTS[drinks].sum()
```

```
  22 weekdays, 1,355 customers, £4,631.60 revenue
  quietest day £154.70, busiest £279.30, mean £210.53
```

Two things to notice. First, a loop over *days* is fine: there are only
22. The work inside (hundreds of drinks) is vectorised. NumPy doesn't ban
loops; it removes the ones over many numbers.

Second, the busiest day took nearly twice the quietest, and all the rules
were identical every day. That spread is *pure chance*. When the real shop
has a slow Tuesday, some of that is just this. Keep it in mind before
explaining every dip.

The script re-runs the month with the same seed and asserts it gets the
same numbers, to the penny. Change `SEED` at the top and you'll get a
different, equally plausible month.

## What you can do now

- Create a generator with `np.random.default_rng(seed)` and explain why
  analysis code should always be seeded.
- Draw integers, floats, weighted choices, samples without replacement
  and shuffles; and use `normal`, `poisson` and `binomial` for the
  situations they fit.
- Explain the law of large numbers, and why small samples can mislead.
- Answer a "how likely is it?" question by simulation: rules, many
  trials, count.
- Keep randomness reproducible: one seeded `rng`, passed into functions;
  `spawn` for independent streams; no global `np.random.seed`.

## What to do now

1. Run `lesson.py`. Then change `SEED` to your birthday and run it again.
   Which numbers changed, and which conclusions didn't?
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   simulates two thousand months to answer an owner's question.
3. That's Week 3, Day 2. Lessons 017 (pandas: Series and DataFrames) and
   018 (cleaning data) arrive tomorrow. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

A seeded simulation is one of the most honest tools in data science: it
shows how much of what you see could be luck, and anyone can re-run it
and check.
