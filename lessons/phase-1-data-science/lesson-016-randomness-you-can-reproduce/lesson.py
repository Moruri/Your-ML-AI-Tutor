"""
Lesson 016 - Randomness you can reproduce

Make random numbers with numpy.random.default_rng, and make them the SAME
random numbers every time with a seed. Draw the kinds of randomness data work
needs (integers, choices with weights, shuffles, bell curves, counts), watch
averages settle down as samples grow, answer a "how likely is it?" question by
simulating it thousands of times, and finish with a whole simulated month at
the coffee shop that anyone can re-run and get exactly the same numbers.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

import numpy as np

SEED = 16                       # any whole number works; what matters is writing it down
EXPECTED_CUSTOMERS = 1355       # what seed 16 gives for the simulated month in section 6

DRINKS = np.array(["latte", "cappuccino", "flat white", "espresso", "tea"])
DRINK_SHARE = np.array([0.33, 0.25, 0.18, 0.13, 0.11])     # roughly what the shop sells
PRICE_CENTS = np.array([380, 370, 390, 220, 250])


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    if isinstance(value, np.ndarray):
        text = np.array2string(value, separator=", ", precision=3).replace("\n", "\n" + " " * 52)
    else:
        text = repr(value)
    print(f"  {label:<46} -> {text}")


# ---------------------------------------------------------------------------
# 1. Random, but the same random
# ---------------------------------------------------------------------------


def section_seeds():
    heading("1. Random, but the same random")

    rng = np.random.default_rng(SEED)
    first = rng.integers(1, 7, size=8)
    show("default_rng(16).integers(1, 7, size=8)", first)
    rng_again = np.random.default_rng(SEED)
    again = rng_again.integers(1, 7, size=8)
    show("a NEW default_rng(16), same call", again)
    show("identical?", bool((first == again).all()))

    print()
    show("the first rng, asked again", rng.integers(1, 7, size=8))
    print("  Same seed -> same sequence. Each call moves along the sequence, so asking the")
    print("  same rng twice gives different numbers, but re-running the script gives the")
    print("  same numbers in the same order. That's what 'reproducible' means.")

    print()
    unseeded = np.random.default_rng()
    show("default_rng() with NO seed", unseeded.integers(1, 7, size=8))
    print("  No seed: fresh numbers every run. Fine for games. Not for analysis you want")
    print("  someone else (or you, next month) to be able to check.")


# ---------------------------------------------------------------------------
# 2. The draws you'll use
# ---------------------------------------------------------------------------


def section_draws():
    heading("2. The draws you'll use")

    rng = np.random.default_rng(SEED)
    show("rng.random(4)  (floats in [0, 1))", rng.random(4))
    show("rng.integers(0, 10, size=6)  (10 excluded)", rng.integers(0, 10, size=6))
    show("rng.choice(DRINKS, size=5)", rng.choice(DRINKS, size=5))
    show("rng.choice(DRINKS, size=5, p=DRINK_SHARE)", rng.choice(DRINKS, size=5, p=DRINK_SHARE))
    show("rng.choice(10, size=4, replace=False)", rng.choice(10, size=4, replace=False))
    show("rng.permutation(DRINKS)  (a shuffled copy)", rng.permutation(DRINKS))
    show("rng.normal(60, 8, size=5).round(1)", rng.normal(60, 8, size=5).round(1))
    show("rng.poisson(60, size=5)", rng.poisson(60, size=5))
    show("rng.binomial(50, 0.33, size=5)", rng.binomial(50, 0.33, size=5))
    print("  choice picks from a list (p= for weights; replace=False for no repeats).")
    print("  normal: a bell curve around a mean. poisson: counts of things that happen")
    print("  independently, like customers through the door. binomial: 'how many of N")
    print("  said yes', like lattes out of 50 orders.")


# ---------------------------------------------------------------------------
# 3. Big samples settle down
# ---------------------------------------------------------------------------


def section_large_numbers():
    heading("3. Big samples settle down")

    rng = np.random.default_rng(SEED)
    print("  Share of orders that are lattes, in simulated samples of growing size")
    print(f"  (the true share is {DRINK_SHARE[0]:.0%}):")
    for n in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
        orders = rng.choice(len(DRINKS), size=n, p=DRINK_SHARE)
        share = (orders == 0).mean()
        print(f"    n = {n:>9,}   latte share {share:6.1%}   (off by {abs(share - DRINK_SHARE[0]):.1%})")
    print("  Small samples wobble a lot. Big ones settle close to the truth. That's the")
    print("  law of large numbers, and it's why ten days of data can mislead you.")


# ---------------------------------------------------------------------------
# 4. Simulation answers "how likely?"
# ---------------------------------------------------------------------------


def section_simulation():
    heading("4. Simulation answers 'how likely?'")

    rng = np.random.default_rng(SEED)
    print("  The shop keeps oat milk for 20 lattes a day. On a day with 50 orders, where")
    print("  each order is a latte with chance 33%, how often do they run out?")
    trials = 100_000
    lattes = rng.binomial(n=50, p=0.33, size=trials)          # one number per simulated day
    run_out = (lattes > 20).mean()
    show("lattes[:10]  (ten simulated days)", lattes[:10])
    show(f"share of {trials:,} days with more than 20", round(float(run_out), 4))
    print(f"  About {run_out:.1%} of days, or roughly one day in {1 / run_out:.0f}.")

    print()
    for stock in [20, 22, 24, 26]:
        print(f"    stock for {stock} lattes -> run out on {(lattes > stock).mean():6.2%} of days")
    print("  No formulas: simulate the situation many times, count how often it happens.")
    print("  With 100,000 trials the answer is good to about a tenth of a percent.")


# ---------------------------------------------------------------------------
# 5. Habits that keep randomness reproducible
# ---------------------------------------------------------------------------


def roll_day(rng, customers=50):
    """Simulate one day's drinks. Takes the rng as an argument: no hidden global state."""
    return rng.choice(DRINKS, size=customers, p=DRINK_SHARE)


def section_habits():
    heading("5. Habits that keep randomness reproducible")

    a = roll_day(np.random.default_rng(SEED))[:6]
    b = roll_day(np.random.default_rng(SEED))[:6]
    show("roll_day(default_rng(16))[:6]", a)
    show("same again", b)
    print("  1. Make ONE rng with a seed at the top of your script, and pass it in.")
    print("  2. Functions that need randomness take `rng` as an argument.")
    print("  3. Don't use the old np.random.seed() / np.random.rand() style: it's one hidden")
    print("     global generator that any library can quietly move along.")

    print()
    parent = np.random.default_rng(SEED)
    children = parent.spawn(3)                                 # independent streams
    show("rng.spawn(3): first draw from each child", np.array([c.integers(0, 100) for c in children]))
    print("  4. Need several independent streams (one per shop, one per experiment)?")
    print("     spawn() gives you as many as you like, all reproducible from one seed.")


# ---------------------------------------------------------------------------
# 6. Putting it together: a simulated month
# ---------------------------------------------------------------------------


def simulate_month(rng, days=22, mean_customers=60):
    """Return (customers_per_day, revenue_cents_per_day) for a month of weekdays."""
    customers = rng.poisson(mean_customers, size=days)
    revenue = np.zeros(days, dtype=int)
    for day, n in enumerate(customers):                       # a loop over DAYS is fine; drinks are vectorised
        drinks = rng.choice(len(DRINKS), size=n, p=DRINK_SHARE)
        revenue[day] = PRICE_CENTS[drinks].sum()
    return customers, revenue


def section_together():
    heading("6. Putting it together: a simulated month")

    rng = np.random.default_rng(SEED)
    customers, revenue = simulate_month(rng)
    show("customers per day", customers)
    print(f"  22 weekdays, {customers.sum():,} customers, £{revenue.sum() / 100:,.2f} revenue")
    print(f"  quietest day £{revenue.min() / 100:.2f}, busiest £{revenue.max() / 100:.2f},"
          f" mean £{revenue.mean() / 100:.2f}")

    rerun_customers, rerun_revenue = simulate_month(np.random.default_rng(SEED))
    assert (customers == rerun_customers).all() and (revenue == rerun_revenue).all()
    assert customers.sum() == EXPECTED_CUSTOMERS, "same seed, same month (with the same NumPy version)"
    print("  Run it again with seed 16: the very same month, to the penny. Change SEED at")
    print("  the top and you get a different, equally plausible month.")


def main():
    print("=" * 66)
    print("  Lesson 016: Randomness you can reproduce")
    print("=" * 66)
    section_seeds()
    section_draws()
    section_large_numbers()
    section_simulation()
    section_habits()
    section_together()
    print()
    print("Random enough to be realistic, seeded so anyone can check your work. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
