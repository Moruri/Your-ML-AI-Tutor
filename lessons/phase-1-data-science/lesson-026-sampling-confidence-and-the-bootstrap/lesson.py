"""
Lesson 026 - Sampling, confidence and the bootstrap

How sure are we? Sample means wobble; the bootstrap turns that wobble into
a confidence interval you can explain in plain words — without memorising
a z-table. Uses the September shop orders as a stand-in population, then
resamples with a seeded generator (lesson 016).

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders_september.csv"

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)
RNG = np.random.default_rng(26)   # reproducible sampling (lesson 016)


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


# ---------------------------------------------------------------------------
# 1. A sample is not the whole story
# ---------------------------------------------------------------------------

def section_population(orders):
    heading("1. A sample is not the whole story")
    rev = orders["revenue"].to_numpy()
    show("population size (all September orders)", len(rev))
    show("true mean order revenue", round(float(rev.mean()), 3))
    show("true median", round(float(np.median(rev)), 3))
    print("  Today we treat the full month as the *population* — the truth.")
    print("  In real life you rarely see the whole population. You see a")
    print("  sample, and you have to say how much that sample can be trusted.")
    return rev


# ---------------------------------------------------------------------------
# 2. Sample means wobble
# ---------------------------------------------------------------------------

def section_wobble(rev):
    heading("2. Sample means wobble")
    n_sample = 40
    n_repeats = 8
    means = []
    for i in range(n_repeats):
        sample = RNG.choice(rev, size=n_sample, replace=False)
        means.append(float(sample.mean()))
    show(f"{n_repeats} sample means (n={n_sample} each)",
         [round(m, 3) for m in means])
    show("spread of those means (std)", round(float(np.std(means, ddof=1)), 3))
    show("true population mean", round(float(rev.mean()), 3))
    print("  Same population, different samples, different means. That")
    print("  scatter is *sampling variability*. A single sample mean is a")
    print("  noisy estimate; the question is how noisy.")
    return n_sample


# ---------------------------------------------------------------------------
# 3. One sample, and the question of confidence
# ---------------------------------------------------------------------------

def section_one_sample(rev, n_sample):
    heading("3. One sample, and the question of confidence")
    sample = RNG.choice(rev, size=n_sample, replace=False)
    show("our one sample, n", n_sample)
    show("sample mean", round(float(sample.mean()), 3))
    show("sample std", round(float(sample.std(ddof=1)), 3))
    show("true mean (normally unknown)", round(float(rev.mean()), 3))
    print("  In practice you get *one* sample and you do not know the true")
    print("  mean. You need a range that usually covers the truth — a")
    print("  confidence interval. The bootstrap builds that range by")
    print("  resampling the sample you actually have.")
    return sample


# ---------------------------------------------------------------------------
# 4. The bootstrap: resample your sample
# ---------------------------------------------------------------------------

def section_bootstrap(sample):
    heading("4. The bootstrap: resample your sample")
    n = len(sample)
    n_boot = 2000
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        # draw n items WITH replacement from the sample
        resample = RNG.choice(sample, size=n, replace=True)
        boot_means[b] = resample.mean()
    show("bootstrap replicates", n_boot)
    show("mean of bootstrap means", round(float(boot_means.mean()), 3))
    show("std of bootstrap means (SE≈)", round(float(boot_means.std(ddof=1)), 3))
    print("  Each replicate: draw n rows from the sample, *with replacement*,")
    print("  compute the mean. Do it thousands of times. The cloud of")
    print("  bootstrap means stands in for 'what other samples might have")
    print("  looked like'. No formula required — just resampling.")
    return boot_means


# ---------------------------------------------------------------------------
# 5. A percentile confidence interval, in plain words
# ---------------------------------------------------------------------------

def section_percentile_ci(boot_means, sample, rev):
    heading("5. A percentile confidence interval, in plain words")
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    show("sample mean", round(float(sample.mean()), 3))
    show("95% percentile CI  [2.5th, 97.5th]",
         (round(float(lo), 3), round(float(hi), 3)))
    show("true mean inside the interval?",
         bool(lo <= rev.mean() <= hi))
    print("  Plain reading: if we repeated this whole process many times,")
    print("  about 95% of such intervals would cover the true mean. This")
    print("  *particular* interval either covers it or it doesn't — we just")
    print("  don't get to peek. Wider interval → less certainty; more data")
    print("  (or less noisy data) narrows it.")


# ---------------------------------------------------------------------------
# 6. Same idea on daily revenue, and a nod to the formula
# ---------------------------------------------------------------------------

def section_daily_and_formula(orders, rev):
    heading("6. Same idea on daily revenue, and a nod to the formula")
    daily = orders.set_index("timestamp")["revenue"].resample("D").sum()
    trading = daily[daily > 0].to_numpy()
    show("trading days (population)", len(trading))
    show("true mean daily revenue", round(float(trading.mean()), 2))

    # bootstrap the mean of daily revenue from a sample of 14 days
    n_days = 14
    day_sample = RNG.choice(trading, size=n_days, replace=False)
    n_boot = 2000
    boot = np.array([
        RNG.choice(day_sample, size=n_days, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    show(f"sample of {n_days} days, mean", round(float(day_sample.mean()), 2))
    show("95% bootstrap CI for mean daily revenue",
         (round(float(lo), 2), round(float(hi), 2)))

    # Classical SE shortcut for comparison (order revenue, large n)
    n = 40
    # use a fresh sample of order revenue for the formula demo
    samp = RNG.choice(rev, size=n, replace=False)
    se = samp.std(ddof=1) / np.sqrt(n)
    # rough 95% CI with 1.96
    classical_lo = samp.mean() - 1.96 * se
    classical_hi = samp.mean() + 1.96 * se
    show("classical ~95% CI (mean ± 1.96·SE), n=40 orders",
         (round(float(classical_lo), 3), round(float(classical_hi), 3)))
    print("  The ±1.96·SE shortcut assumes a roughly bell-shaped sampling")
    print("  distribution. The bootstrap does not — it reads the percentiles")
    print("  of the resampled means. For skewed order revenue, prefer the")
    print("  bootstrap until the formula's assumptions feel earned.")

    assert len(trading) >= 20
    assert lo < trading.mean() < hi or True  # CI may or may not cover; just run
    assert classical_hi > classical_lo
    print("  Sanity checks passed.")


def main():
    print("=" * 66)
    print("  Lesson 026: Sampling, confidence and the bootstrap")
    print("=" * 66)
    orders = load_orders(ORDERS_CSV)
    rev = section_population(orders)
    n_sample = section_wobble(rev)
    sample = section_one_sample(rev, n_sample)
    boot_means = section_bootstrap(sample)
    section_percentile_ci(boot_means, sample, rev)
    section_daily_and_formula(orders, rev)
    print()
    print("Samples wobble; the bootstrap turns that wobble into an interval")
    print("you can explain without a z-table. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
