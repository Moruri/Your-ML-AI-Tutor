# Lesson 023 - Distributions and summary statistics

**Phase 1 - Data science basics** | Week 4, Day 1 | Monday 2026-09-28

> **Goal:** read means, medians, spreads and skew, and know which summary
> to trust for a given dataset — so "the average order is £X" is a claim
> you can defend, not a number you copied from `.describe()`.

Time: about 50 minutes. Needs the venv from lesson 013 (`pandas`, `numpy`).

---

You've cleaned, grouped, dated and plotted the September orders. Now the
owner asks the question that looks easiest and isn't: "What's a typical
order? What's a typical day?"

"Typical" is not one number. A pile of values has a **shape** — where the
mass sits, how wide it is, whether a long tail pulls one side out. The
mean, the median, the standard deviation and the IQR each summarise a
different part of that shape. Use the wrong one and you tell a tidy lie.

This lesson is about reading the shape first, then picking the summary
that matches the question.

## How to follow along

Venv active, `python` in the repo root. The data is `orders_september.csv`,
the same clean month as lessons 017–022. Full script:

```bash
python lessons/phase-1-data-science/lesson-023-distributions-and-summary-statistics/lesson.py
```

It writes no files.

## 1. A distribution is a shape, not a single number

```
orders, count                                  -> 1214
revenue: min, max                              -> (1.7, 19.2)
rough bins of order revenue:
£0–3      147
£3–5      619
£5–8      219
£8–12     162
£12–20     67
```

Most orders sit in the cheap-to-middling band. A few big tickets stretch
the right side. That stretched shape is a **right-skewed** distribution,
and it decides which summary you should trust. Before you quote any
average, glance at a histogram (lesson 022) or a quick set of bins like
these. Shape first; numbers second.

## 2. Mean, median, and when they disagree

```
mean   (the balance point)                     -> 5.459
median (the middle order)                      -> 4.2
```

The **mean** is the balance point: add everything up, divide by the count.
The **median** is the middle value once the list is sorted. Here they
disagree by more than a pound, because the large multi-cup orders pull
the mean up.

| Question | Prefer |
|----------|--------|
| What does a *typical* customer pay? | Median |
| How much money came in? (mean × count) | Mean |
| What's the most common cup count? | Mode |

Daily revenue is a useful contrast:

```
daily revenue mean / median (trading days)     -> (301.25, 293.1)
```

Nearly the same. Daily totals are roughly symmetric, so either summary
is fine. **The gap between mean and median is a quick skew detector**:
big gap → skewed; small → not.

## 3. Spread: standard deviation, IQR and the five-number summary

Centre without spread is half a story. "About £4" means something different
if every order is £3.90–£4.10 than if they run from £1.70 to £19.

```
revenue std (sample)                           -> 3.189
Q1, Q3                                         -> (3.4, 7.4)
IQR  (Q3 - Q1)                                 -> 4.0
five-number summary of order revenue:
min 1.7 | Q1 3.4 | median 4.2 | Q3 7.4 | max 19.2
```

- **Standard deviation** asks "typical distance from the mean". It gets
  dragged by outliers, same as the mean.
- **IQR** (interquartile range) is the width of the middle half of the
  data: stubborn against extremes.
- The **five-number summary** (min, Q1, median, Q3, max) is the skeleton
  of a box plot. `Series.describe()` prints most of it in one call.

Rule of thumb: pair **mean with std**; pair **median with IQR**.

## 4. Percentiles, and what "typical" really means

```
p10  2.50 | p50  4.20 | p90  9.97 | p95 12.60 | p99 15.60
share of orders under the mean (£5.46)         -> 0.664
```

Half of orders are at or below £4.20. Nine in ten are at or below £9.97.
About two-thirds sit *under* the mean — another sign of a long right
tail. Percentiles answer "how unusual is this?" better than "mean ± std"
when the shape isn't a neat bell. In ML you'll meet them again as
quantile losses, calibration curves and robust scalers.

## 5. Skew, and a tiny experiment with outliers

```
skew of order revenue                          -> 1.424
skew of daily trading revenue                  -> 0.331
```

**Skew** (pandas: `.skew()`) summarises the asymmetry. Positive means a
long right tail (mean > median). Near zero means roughly symmetric.

Add one absurd £500 catering order and watch what moves:

```
new mean  5.866   (was 5.459)
new median 4.2    (unchanged)
new std   14.541  (was 3.189)
new IQR    4.0    (unchanged)
```

One wild value moved the mean and std a lot, and barely touched the
median and IQR. That's why people call the second pair **robust**.

## 6. Putting it together: what to tell the owner

```
revenue per order, by drink (median, sorted):
flat white  £4.40 | latte £4.30 | cappuccino £4.20 | tea £3.00 | espresso £1.70
```

An honest summary for the owner:

- A **typical order** is about **£4.20** (median). The average is higher
  (£5.46) because of multi-cup tickets — useful for totals, misleading
  as "what people usually pay".
- A **typical trading day** takes about **£293**, swinging roughly **£48**
  either side (std). Busiest day £407, quietest £227.
- Espresso tickets are small and tight; flat whites and lattes spread
  more because of size and cup count.

Same dataset, different summaries, different sentences. The skill is
matching the sentence to the shape.

## What you can do now

- Glance at a distribution (bins or a histogram) before quoting an average.
- Choose mean vs median from the question and the skew.
- Report spread with std (alongside mean) or IQR (alongside median).
- Read percentiles and a five-number summary.
- Use the mean–median gap and `.skew()` as quick shape checks.
- Predict what a single outlier will do to each summary.

## What to do now

1. Run `lesson.py`. Try `.describe()` on `quantity` and on daily revenue;
   notice which one looks skewed.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   compares drink sizes.
3. Next, lesson 024: correlation, causation and the traps in between.
   Summaries describe one column; correlation asks how two move together.
   See [PROGRESS.md](../../../curriculum/PROGRESS.md).

A number without a shape is a rumour. Look first, then summarise.
