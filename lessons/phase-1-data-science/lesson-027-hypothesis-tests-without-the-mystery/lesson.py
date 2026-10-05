"""
Lesson 027 - Hypothesis tests without the mystery

Two numbers differ. Is the gap real, or is it the kind of wobble you saw in
lesson 026? This script answers that twice: once with a permutation test you
build yourself (shuffle the labels, see what chance alone does), and once with
Welch's t-test from scipy. Then it shows what a p-value does and doesn't tell
you, using the same September shop orders as lessons 017-026.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders_september.csv"

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)
RNG = np.random.default_rng(27)   # reproducible shuffles (lesson 016)


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        print(f"  {label}:")
        for line in value.to_string().splitlines():
            print(f"      {line}")
    else:
        print(f"  {label:<46} -> {value!r}")


def load_orders(path):
    orders = pd.read_csv(path, parse_dates=["timestamp"])
    orders["revenue"] = orders["price"] * orders["quantity"]
    return orders


def permutation_test(a, b, n_perm=5000, rng=RNG):
    """Two-sided permutation test for a difference in means.

    Returns (observed difference, null differences, p-value).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(pooled)
        null[i] = shuffled[:n_a].mean() - shuffled[n_a:].mean()
    # how often does chance alone give a gap at least this big (either way)?
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, null, p


# ---------------------------------------------------------------------------
# 1. Two numbers differ. So what?
# ---------------------------------------------------------------------------

def section_question(orders):
    heading("1. Two numbers differ. So what?")
    by_payment = orders.groupby("payment")["revenue"].agg(["count", "mean", "std"]).round(3)
    show("order revenue by payment method", by_payment)
    gap = by_payment.loc["cash", "mean"] - by_payment.loc["card", "mean"]
    show("cash minus card, mean revenue (£)", round(float(gap), 3))
    print("  Cash orders look a little bigger. Before anyone redesigns the")
    print("  till, ask the boring question: could a gap this size turn up")
    print("  just from which orders happened to land in which group?")


# ---------------------------------------------------------------------------
# 2. A permutation test: let chance have a go
# ---------------------------------------------------------------------------

def section_permutation(orders):
    heading("2. A permutation test: let chance have a go")
    cash = orders.loc[orders["payment"] == "cash", "revenue"]
    card = orders.loc[orders["payment"] == "card", "revenue"]
    observed, null, p = permutation_test(cash, card)
    show("observed gap, cash - card (£)", round(float(observed), 3))
    show("shuffles of the payment labels", len(null))
    show("middle 95% of shuffled gaps (£)",
         tuple(round(float(x), 3) for x in np.percentile(null, [2.5, 97.5])))
    show("share of shuffles with a gap this big", round(float(p), 3))
    print("  The null hypothesis says payment method makes no difference.")
    print("  If that were true, the labels are arbitrary, so shuffling them")
    print("  shows what gaps chance alone produces. Our real gap sits")
    print("  comfortably inside that cloud. That share is the p-value.")
    return cash, card, p


# ---------------------------------------------------------------------------
# 3. The same question with a t-test
# ---------------------------------------------------------------------------

def section_ttest(cash, card, p_perm):
    heading("3. The same question with a t-test")
    result = stats.ttest_ind(cash, card, equal_var=False)   # Welch's t-test
    show("Welch t statistic", round(float(result.statistic), 3))
    show("Welch p-value", round(float(result.pvalue), 3))
    show("permutation p-value (section 2)", round(float(p_perm), 3))
    print("  The t-test reaches the same verdict with a formula instead of")
    print("  shuffles. 'Welch' means it doesn't assume both groups have the")
    print("  same spread, which is the safer default. With hundreds of")
    print("  orders per group the two methods agree closely.")
    print("  Verdict: no good evidence that cash orders are bigger.")
    assert abs(result.pvalue - p_perm) < 0.1


# ---------------------------------------------------------------------------
# 4. A difference that survives the shuffle
# ---------------------------------------------------------------------------

def section_real_difference(orders):
    heading("4. A difference that survives the shuffle")
    large = orders.loc[orders["size"] == "large", "revenue"]
    small = orders.loc[orders["size"] == "small", "revenue"]
    observed, null, p = permutation_test(large, small)
    result = stats.ttest_ind(large, small, equal_var=False)
    show("large-cup orders / small-cup orders", (len(large), len(small)))
    show("observed gap, large - small (£)", round(float(observed), 3))
    show("largest shuffled gap seen (£)", round(float(np.abs(null).max()), 3))
    show("permutation p-value", round(float(p), 4))
    show("Welch p-value", float(f"{result.pvalue:.2g}"))

    # bootstrap a CI for the gap, the lesson 026 way
    boot = np.array([
        RNG.choice(large.to_numpy(), len(large)).mean()
        - RNG.choice(small.to_numpy(), len(small)).mean()
        for _ in range(2000)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    show("95% bootstrap CI for the gap (£)", (round(float(lo), 3), round(float(hi), 3)))
    print("  Not one shuffle got close, so the p-value is as small as 5,000")
    print("  shuffles can measure (1 in 5,001). The interval says how big the")
    print("  gap probably is. Report both: the test says 'probably real', the")
    print("  interval says 'and roughly this large'.")
    assert p < 0.01 and lo > 0


# ---------------------------------------------------------------------------
# 5. What a p-value is, and what it is not
# ---------------------------------------------------------------------------

def section_p_values(orders):
    heading("5. What a p-value is, and what it is not")
    rev = orders["revenue"].to_numpy()
    n_tests = 200
    p_values = np.empty(n_tests)
    for i in range(n_tests):
        # split orders into two groups at random: the null is TRUE by design
        mask = RNG.random(len(rev)) < 0.5
        p_values[i] = stats.ttest_ind(rev[mask], rev[~mask], equal_var=False).pvalue
    false_alarms = int(np.sum(p_values < 0.05))
    show("random splits where nothing is going on", n_tests)
    show("splits with p < 0.05 anyway", false_alarms)
    show("share of false alarms", round(false_alarms / n_tests, 3))

    any_hit = np.mean([
        np.any(RNG.random(20) < 0.05) for _ in range(5000)
    ])
    show("chance of >=1 'hit' when you run 20 null tests", round(float(any_hit), 3))
    print("  A p-value is: 'if there were no real difference, how often would")
    print("  chance give a gap at least this big?' It is NOT the chance the")
    print("  null is true, and NOT the chance your result is a fluke.")
    print("  At the 0.05 line, about 1 in 20 pure-noise tests 'passes'. Run")
    print("  20 tests and a false alarm is more likely than not.")
    assert 0.0 < false_alarms / n_tests < 0.12


# ---------------------------------------------------------------------------
# 6. Significant is not the same as important
# ---------------------------------------------------------------------------

def section_significance_vs_size(orders):
    heading("6. Significant is not the same as important")
    orders = orders.assign(morning=orders["timestamp"].dt.hour < 11)
    am = orders.loc[orders["morning"], "revenue"]
    pm = orders.loc[~orders["morning"], "revenue"]
    observed, _, p = permutation_test(am, pm)
    show("morning orders / later orders", (len(am), len(pm)))
    show("gap, morning - later (£ per order)", round(float(observed), 3))
    show("permutation p-value", round(float(p), 3))

    # pretend we had ten times the data, with the same gap and spread
    am10 = np.tile(am.to_numpy(), 10)
    pm10 = np.tile(pm.to_numpy(), 10)
    p10 = stats.ttest_ind(am10, pm10, equal_var=False).pvalue
    show("same gap with 10x the orders, p-value", float(f"{p10:.2g}"))
    show("gap x later orders (£ per month)",
         round(float(observed * len(pm)), 2))
    print("  The gap is the same in both rows; only the amount of data")
    print("  changed. With enough data almost any gap becomes 'significant'.")
    print("  So the useful question is never just 'is p small?' but 'is the")
    print("  effect big enough to matter, and how sure are we of its size?'")


def main():
    print("=" * 66)
    print("  Lesson 027: Hypothesis tests without the mystery")
    print("=" * 66)
    orders = load_orders(ORDERS_CSV)
    section_question(orders)
    cash, card, p_perm = section_permutation(orders)
    section_ttest(cash, card, p_perm)
    section_real_difference(orders)
    section_p_values(orders)
    section_significance_vs_size(orders)
    print()
    print("Shuffle the labels to see what chance does, then ask how big the")
    print("effect is. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
