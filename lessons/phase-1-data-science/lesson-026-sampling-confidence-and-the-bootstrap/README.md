# Lesson 026 - Sampling, confidence and the bootstrap

**Phase 1 - Data science basics** | Week 4, Day 2 | Tuesday 2026-09-29

> **Goal:** Quantify "how sure are we?" with resampling instead of
> formulas you don't trust yet — so a confidence interval is something
> you can build and explain, not a black-box output from a textbook.

Time: about 55 minutes. Needs the venv from lesson 013 (`pandas`, `numpy`).

---

Yesterday's lesson taught you how to bet. Today asks the follow-up every
honest analyst gets: *how sure are we?*

A sample mean is a noisy estimate of a truth you usually cannot see. The
**bootstrap** turns that noise into a confidence interval by resampling
the data you actually have — no z-table, no "assume normality" leap. You
will still meet the classical formulas later; this lesson makes sure you
understand what they are trying to say before you memorise them.

## How to follow along

Venv active, `python` in the repo root. Same `orders_september.csv` as
lessons 017–025. Full script:

```bash
python lessons/phase-1-data-science/lesson-026-sampling-confidence-and-the-bootstrap/lesson.py
```

It writes no files. Sampling uses a seeded generator (lesson 016), so
your numbers will match.

## 1. A sample is not the whole story

```
population size (all September orders)         -> 1214
true mean order revenue                        -> 5.459
true median                                    -> 4.2
```

Today we treat the full month as the **population** — the truth. In real
life you rarely see the whole population. You see a sample, and you have
to say how much that sample can be trusted. (In production ML the
"population" is often *future* users you have not met yet. Same problem.)

## 2. Sample means wobble

```
8 sample means (n=40 each)  -> [5.678, 4.675, 5.112, 5.707, 5.232, 6.065, 5.723, 4.600]
spread of those means (std)                    -> 0.531
true population mean                           -> 5.459
```

Same population, different samples, different means. That scatter is
**sampling variability**. A single sample mean is a noisy estimate; the
question is how noisy. Larger `n` tightens the scatter; noisier data
widens it.

## 3. One sample, and the question of confidence

```
our one sample, n                              -> 40
sample mean                                    -> 5.152
sample std                                     -> 2.934
true mean (normally unknown)                   -> 5.459
```

In practice you get **one** sample and you do not know the true mean.
You need a range that usually covers the truth — a **confidence
interval**. The bootstrap builds that range by resampling the sample you
actually have.

## 4. The bootstrap: resample your sample

```
bootstrap replicates                           -> 2000
mean of bootstrap means                        -> 5.152
std of bootstrap means (SE≈)                   -> 0.452
```

Each replicate: draw `n` rows from the sample, **with replacement**,
compute the mean. Do it thousands of times. The cloud of bootstrap means
stands in for "what other samples might have looked like". No formula
required — just resampling. The spread of that cloud is a data-driven
estimate of the standard error.

## 5. A percentile confidence interval, in plain words

```
sample mean                                    -> 5.152
95% percentile CI  [2.5th, 97.5th]             -> (4.317, 6.065)
true mean inside the interval?                 -> True
```

Take the 2.5th and 97.5th percentiles of the bootstrap means. That band
is a **95% percentile confidence interval**.

Plain reading: if we repeated this whole process many times, about 95%
of such intervals would cover the true mean. This *particular* interval
either covers it or it doesn't — we just don't get to peek. Wider
interval → less certainty; more data (or less noisy data) narrows it.

(Here we *can* peek, because we pretended the month was the population.
The true mean £5.46 sits inside. Lucky this time — and that is fine.)

## 6. Same idea on daily revenue, and a nod to the formula

```
trading days (population)                      -> 22
true mean daily revenue                        -> 301.25
sample of 14 days, mean                        -> 292.34
95% bootstrap CI for mean daily revenue        -> (268.88, 315.93)
classical ~95% CI (mean ± 1.96·SE), n=40 orders -> (5.207, 7.233)
```

Same recipe on daily totals: sample a handful of days, bootstrap the
mean, read the percentiles. The classical shortcut `mean ± 1.96·SE`
assumes a roughly bell-shaped sampling distribution. The bootstrap does
not — it reads the percentiles of the resampled means. For skewed order
revenue, prefer the bootstrap until the formula's assumptions feel
earned. Later lessons will show when the shortcut is good enough.

## What you can do now

- Treat a full dataset as a stand-in population and draw samples from it.
- See sampling variability as the reason one mean is not enough.
- Build a bootstrap distribution by resampling with replacement.
- Form a percentile confidence interval from bootstrap percentiles.
- Explain in plain words what "95% confident" does and does not mean.
- Contrast the bootstrap with the classical `mean ± 1.96·SE` shortcut.

## What to do now

1. Run `lesson.py`. Change `n_sample` from 40 to 10 and to 200; watch the
   CI widen and shrink.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   bootstraps a median, not a mean.
3. Next, lesson 027: hypothesis tests without the mystery. Intervals say
   where a number probably sits; tests ask whether a difference is
   believable. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

Samples wobble. Resample the wobble, and you get an interval you can
explain without a z-table.
