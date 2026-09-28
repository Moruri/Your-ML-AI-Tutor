# Lesson 024 - Correlation, causation and the traps in between

**Phase 1 - Data science basics** | Week 4, Day 1 | Monday 2026-09-28

> **Goal:** compute and interpret correlation, and learn the classic ways
> it misleads — so "these two move together" stays a careful claim, not a
> story about causes you haven't earned.

Time: about 55 minutes. Needs the venv from lesson 013 (`pandas`, `numpy`).

---

Yesterday's lesson summarised **one** column. Today asks a sharper
question: when *two* columns move, do they move together?

"People who buy more cups spend more" sounds obvious. "Iced-drink sales
track tips" sounds like a discovery until you notice the sun. Correlation
is one of the most useful numbers in data work, and one of the most
abused. This lesson computes it properly, reads it on the shop data, then
walks through the traps that turn a tidy `r = 0.8` into a bad decision.

## How to follow along

Venv active, `python` in the repo root. Same `orders_september.csv` as
lessons 017–023. Full script:

```bash
python lessons/phase-1-data-science/lesson-024-correlation-causation-and-the-traps-in-between/lesson.py
```

It writes no files. The sunny-day example uses a seeded generator (lesson
016), so your numbers will match.

## 1. What correlation measures

```python
orders["quantity"].corr(orders["revenue"])   # Pearson r ≈ 0.861
```

**Pearson r** lives between −1 and +1:

| r | Meaning |
|---|---------|
| near +1 | when one is high, the other tends to be high (straight line up) |
| near −1 | when one is high, the other tends to be low |
| near 0 | no *straight-line* link |

It measures **linear** association only. A perfect U-shape can have r ≈ 0.
It is also symmetric: `a.corr(b)` equals `b.corr(a)`. And it says nothing
about slopes in pounds — only about how tightly points hug a straight
line after each variable is scaled.

Cups and revenue are strongly linked. That is **association**, not yet a
cause. The bill is *defined* as price × cups, so of course they move
together. Useful check; not a discovery about nature.

## 2. Reading a few real pairs from the shop

```
quantity vs revenue    0.861
size_n vs price        0.680
price vs revenue       0.420
hour vs revenue       -0.028
quantity vs price     -0.033
```

(`size_n` is small=1, medium=2, large=3 — a numeric stand-in so
correlation can run.)

Strong where the menu says there should be a link (cups↔revenue,
size↔price). Near zero for hour↔revenue *per order*: a 7am latte costs
what a 3pm latte costs. That near-zero is honest at the order grain; it
does **not** mean mornings aren't busy (section 5).

Inside each drink, cups↔revenue is even tighter (espresso hits 1.0: one
size, so revenue is just price × cups).

## 3. Correlation is not causation: a sunny-day confounder

A tiny simulated month:

```
corr iced_drinks vs tips     ≈ 0.81
corr sun vs iced_drinks      ≈ 0.91
corr sun vs tips             ≈ 0.92
corr pigeons vs tips         ≈ 0.03
```

Iced drinks and tips move together. Did iced drinks *cause* bigger tips?
No. **Sunshine** drove both — a **confounder**. Pigeons are a nonsense
partner; their r sits near zero, as it should.

The habit: when two things correlate, ask "what else could be driving
both?" before you write a cause. High r is a clue to investigate, not a
verdict.

## 4. Three more traps

**Restricted range.** Look only at large drinks and size↔price becomes
undefined (no variance in size), not "no relationship". Espresso in the
real data has one size for the same reason. Correlation needs spread.

**Nonsense time-twins.** Daily sales and Instagram followers can show
r ≈ 0.98 simply because both drift up over the month. Time is the hidden
partner. Correlate day-to-day *changes*, or detrend, before you brag.

**Reverse arrow.** "Busy hours cause more staff" and "more staff cause
busier hours" are both compatible with the same r. The number cannot pick
the direction. You need a story, an experiment, or time order that only
runs one way.

## 5. Aggregation changes the answer

```
r hour vs revenue  (per order)              ≈ -0.028
r hour vs total revenue  (per hour-of-day)  ≈ -0.77
```

```
hour   orders   revenue
7        159     872.5
8        263    1506.4
...
16        45     244.5
```

Per order, hour barely matters. Per hour-of-day, later hours take less
money because *fewer orders* land then — so r with the hour number turns
clearly negative. Same column names, different **grain**, different
story. Always say what one row represents before you quote r.

(This is cousin to Simpson's paradox: the pattern at one aggregation
level can weaken, vanish or flip at another.)

## 6. Putting it together: what correlation is for

```
correlation matrix:
            quantity  revenue  price  size_n
quantity       1.000    0.861 -0.033  -0.019
revenue        0.861    1.000  0.420   0.286
price         -0.033    0.420  1.000   0.680
size_n        -0.019    0.286  0.680   1.000
```

Good uses of correlation in this course:

- find pairs worth a scatter plot or a closer look
- sanity-check a merge ("do these IDs actually line up?")
- spot leakage before modelling (a feature that correlates with the
  target because it *is* the target in disguise)

Bad use: a verdict about causes. Causes need design — experiments,
natural experiments, or domain knowledge the spreadsheet does not contain.

## What you can do now

- Compute Pearson r with `Series.corr` or `np.corrcoef`.
- Read r as linear association, not slope and not destiny.
- Name confounders, restricted range, time-twins and reverse arrows.
- Check whether aggregating changes the correlation.
- Build a small correlation matrix with `DataFrame.corr()`.

## What to do now

1. Run `lesson.py`. Change the seed in `RNG = ...` and watch the sunny-day
   r wobble; the shop r values won't.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   hunts a real confounder-shaped pattern in the orders.
3. Next, lesson 025: probability intuition for ML. Correlation told you
   two things move together; probability asks how to bet. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

r measures straight-line association. Causes need a story the data alone
cannot finish.
