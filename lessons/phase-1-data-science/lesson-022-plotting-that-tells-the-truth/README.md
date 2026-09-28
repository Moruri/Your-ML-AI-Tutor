# Lesson 022 - Plotting that tells the truth

**Phase 1 - Data science basics** | Week 3, Day 5 | Friday 2026-09-25

> **Goal:** make line, bar, scatter and histogram charts with matplotlib,
> each labelled well enough to stand alone, and learn the handful of habits
> that stop a chart from misleading anyone (including you).

Time: about 55 minutes. Needs the venv from lesson 013 (`matplotlib`,
`pandas`, `numpy`).

---

For a week you've been printing tables. Tables are precise, but people
don't read them; they *glance* at them. Show the owner 22 numbers of daily
revenue and they'll squint. Show them a line, and in two seconds they'll
say "what happened mid-month?".

That speed is a chart's power, and its danger. A reader believes a chart
before they check it. Cut off the bottom of a bar chart and a small
difference looks huge. Pick the wrong bins and a histogram hides its shape.
Nobody has to be lying on purpose: the defaults and a rush are enough.

So this lesson covers two things at once: the four charts you'll make most
often, and the habits that keep them honest. Lesson 012 promised "real
charts" in Phase 1. Here they are.

## How to follow along

Venv active, `python` in the repo root. The data is `orders_september.csv`,
the same clean month as lessons 017-021. Full script:

```bash
python lessons/phase-1-data-science/lesson-022-plotting-that-tells-the-truth/lesson.py
```

It saves seven PNG images in `output/` next to the script. **Open them**:
this lesson is about the pictures, and the printed output only describes
them. (`output/` is ignored by git.)

## 1. A figure, an axes, and the labels that make it stand alone

```python
import matplotlib
matplotlib.use("Agg")              # draw to files; no window needed
import matplotlib.pyplot as plt

x = np.linspace(0, 4, 5)           # 0, 1, 2, 3, 4 cups
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, 3.80 * x, marker="o")
ax.set_title("A latte order costs £3.80 per cup")
ax.set_xlabel("Cups in the order")
ax.set_ylabel("Bill (£)")
fig.savefig("output/01_anatomy.png", dpi=100, bbox_inches="tight")
plt.close(fig)
```

Two objects to know:

- **`fig`** (the Figure) is the whole image, the thing you save.
- **`ax`** (the Axes) is one chart inside it, the thing you draw on. A
  figure can hold several (section 6).

`plt.subplots()` gives you both. Almost everything else is a method on
`ax`: `plot`, `bar`, `scatter`, `hist`, and `set_title`, `set_xlabel`, and
so on. `matplotlib.use("Agg")` makes matplotlib draw straight to files,
which works on every machine, including ones without a screen. (In a
notebook or with a window, you'd use `plt.show()` instead of saving.)

`np.linspace` from lesson 014 earns its keep here: "N evenly spaced points
from A to B" is exactly what you need to draw a smooth line.

**The stand-alone test.** Imagine the image forwarded to someone with no
context. Can they tell what it shows, what's on each axis, in what units,
and where the data came from? That means:

- a **title** that says what it shows (better still, what it *means*),
- **axis labels with units**: "Revenue (£)", not "rev",
- a **legend** when there's more than one series,
- a **source line** for anything that leaves your laptop.

## 2. Line charts: change over time

A line says "these points are connected, in order". Use it for things that
change over **time**.

```python
ax.plot(trading.index, trading, marker="o", alpha=0.6, label="Daily revenue")
ax.plot(smooth.index, smooth, linewidth=2.5, label="5-trading-day average")
ax.annotate("Two rainy days", xy=(pd.Timestamp("2026-09-16 12:00"), 236),
            xytext=(pd.Timestamp("2026-09-10"), 215), arrowprops={"arrowstyle": "->"})
ax.legend()
fig.autofmt_xdate()
```

This is lesson 021's daily revenue with its 5-day rolling mean. Markers on
the raw line show where the real data points are; the thick line carries
the trend. One annotation points at the one thing you want noticed. The
line jumps straight from Friday to Monday; there's no point for the
weekend, so it doesn't claim anything about it. `fig.autofmt_xdate()` tilts
the date labels so they don't collide.

Lines are allowed to "zoom in" on the y-axis: the reader judges a line by
its *slope*, not its height. This chart starts at zero anyway, because
there's room and it keeps the mid-month dip in proportion.

## 3. Bar charts: comparing categories, and the axis that lies

Bars compare **categories**: drinks, weekdays, shops.

```python
by_drink = orders.groupby("drink")["revenue"].sum().sort_values()
ax.barh(by_drink.index, by_drink.values)
```

Two cheap wins: **sort** the bars so the ranking reads at a glance, and use
**horizontal** bars (`barh`) when the labels are words. Writing the value
on each bar with `ax.text` saves the reader from squinting at the axis.

Now the most common chart lie. Here are average takings per weekday, drawn
twice from the *same numbers*:

```
Friday / Wednesday, real ratio                 -> 1.34
Friday / Wednesday, bar heights on the £270 axis -> 21.7
```

On the left, the axis starts at £270 (`ax.set_ylim(270, 370)`); on the
right, at £0. Friday takes 34% more than Wednesday. On the truncated chart,
Friday's bar is about *20 times* taller. Open
`04_bar_truncated_vs_honest.png` and look at how different they feel.

A bar shows a quantity by its **length**, so the length has to start at
zero. **Bars start at zero, always.** If the differences are too small to
see from zero, that *is* the finding (they're small), or you should be
plotting the differences themselves.

## 4. Scatter plots: two numbers per thing

A scatter plot puts one dot per *thing* (here, per trading day), with one
number across and another up. It answers "when this is high, is that high
too?"

```python
ax.scatter(x, y, alpha=0.7)                         # orders vs revenue per day
slope, intercept = np.polyfit(x, y, deg=1)          # the best straight line
line_x = np.linspace(x.min(), x.max(), 50)
ax.plot(line_x, slope * line_x + intercept, linestyle="--")
```

```
slope of the trend line (£ per order)          -> 5.35
correlation (lesson 024 explains it)           -> 0.875
```

Busier days take more money, about £5.35 per extra order, close to the
average order value of £5.46. Not a surprise, but the chart shows it
in a way a table can't, *and* it shows how much the days scatter around the
line. `np.polyfit(..., deg=1)` fits a straight line; Phase 2 explains how. The
title says what one dot is, which is the first thing a reader of a scatter
plot needs.

Don't join scatter points with lines: the order of the rows means nothing
here. `alpha` (transparency) lets overlapping dots show up darker.

## 5. Histograms: the shape of one number

A histogram shows how **one** number is spread out: it cuts the range into
**bins** and draws how many values land in each.

```python
ax.hist(orders["revenue"], bins=np.arange(0, 21, 1), edgecolor="white")
```

```
most common £1 band                            -> '£3-£4: 382 orders'
median order value                             -> 4.2
orders over £10                                -> 117
```

Most orders are a single drink (£3-£4), with a long tail of group orders
out to £19.20. The figure draws the same 1,214 values three ways: 4 bins
(all the shape is lost), £1 bins (clear), and 200 bins (spiky noise). The
number of bins is a choice that changes the story, so choose one that
*means* something, like £1 bands, and say what it is in the axis label.
Lesson 023 builds on exactly this picture.

Histogram or bar chart? A **bar chart** has categories on the x-axis
(drinks). A **histogram** has a number, cut into ranges (order value).

## 6. Putting it together: one page for the owner

`07_september_summary.png` puts four charts on one figure with
`plt.subplots(2, 2)`: daily revenue with its trend, revenue by drink,
revenue by hour of day, and oat-milk share by week. Each panel has its own
title and units, one headline sits on top (`fig.suptitle`), and the
caveat, that the last week is only Monday to Wednesday, goes in a
footnote rather than being left for the owner to trip over.
`fig.tight_layout()` stops the panels overlapping.

That's the whole checklist in one picture:

| Habit | Why |
|-------|-----|
| Pick the chart for the question: line for time, bar for categories, scatter for two numbers, histogram for one number's spread. | The wrong chart implies things that aren't true (a line through categories implies an order). |
| A title that says what it shows, and axis labels with units. | It has to survive being forwarded without you. |
| Bars start at zero. | Length is the message. |
| Choose bins, windows and ranges on purpose, and say what they are. | Defaults are choices too; just not yours. |
| Show the caveat (partial weeks, missing days, small samples). | A footnote is honest; a surprise later isn't. |
| One point per chart; annotate it. | The reader should see what you saw in two seconds. |

## What you can do now

- Make a figure with `plt.subplots`, draw on its axes, and save it with
  `savefig` using the `Agg` backend.
- Choose between line, bar, scatter and histogram for a question.
- Label a chart so it stands alone: title, axis labels with units, legend,
  source.
- Spot and avoid the truncated bar axis, and choose histogram bins on
  purpose.
- Add a trend line with `np.polyfit` and `np.linspace`, and an annotation
  with `ax.annotate`.
- Build a multi-panel summary with `plt.subplots(rows, cols)`.

## What to do now

1. Run `lesson.py` and open all seven images in `output/`. Change one
   thing (a title, the bins, the colours) and run it again.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one is
   a chart for the owner about card payments.
3. That's Week 3 done: all of Phase 1's core tools, NumPy, pandas, dates
   and plotting. Next week starts with lesson 023, distributions and
   summary statistics. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

A good chart makes one point, honestly, to someone who never saw your
code. Everything in this lesson is in service of that.
