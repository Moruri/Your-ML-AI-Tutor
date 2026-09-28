# Lesson 022 - Exercises

Three quick checks, a hands-on chart for the owner, and an optional
critique of a chart that misleads. Open every image you make: the
picture is the answer.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-022-plotting-that-tells-the-truth/lesson.py my_lesson_022.py
python my_lesson_022.py
```

Change the `HERE = ...` line near the top so the copy finds the data (the
charts will land in that lesson folder's `output/`):

```python
HERE = Path("lessons/phase-1-data-science/lesson-022-plotting-that-tells-the-truth").resolve()
```

`load_orders`, `daily_revenue`, `save` and `ORDERS_CSV` are there to reuse.

---

## 1. Which chart?

Pick line, bar, scatter or histogram for each:

- (a) How many cups were sold each hour of one Friday, in order through the
  day.
- (b) Total September revenue for each of three branches.
- (c) For each order, the number of cups vs the time taken to make it.
- (d) How long customers wait, across 500 orders.

<details>
<summary>Check yourself</summary>

- **(a) Line** (or bars in time order): it's change over time.
- **(b) Bar**: three categories, compared by size, starting at zero.
- **(c) Scatter**: two numbers per order.
- **(d) Histogram**: the spread of one number.

</details>

## 2. What's wrong?

A chart has the title "Sales", a y-axis labelled "value" running from 950
to 1,000, and three bars for three shops. List everything you'd fix.

<details>
<summary>Check yourself</summary>

- The **title** doesn't say what, where or when: "Revenue by shop, September
  2026" at least.
- **"value"** has no meaning or units: "Revenue (£)".
- The **axis starts at 950**, so bars of £960 and £990 look wildly
  different when they're 3% apart. Bars start at zero.
- No **source**, if it's going anywhere.

If the shops really are within 3% of each other, the honest chart shows
three nearly equal bars. That's the finding.

</details>

## 3. What does the `alpha` do?

In section 4, `ax.scatter(x, y, alpha=0.7)`. What would change with
`alpha=0.1` if there were 10,000 dots instead of 22, and why might you
want that?

<details>
<summary>Check yourself</summary>

`alpha` is opacity: 1 is solid, 0 is invisible. With 10,000 solid dots,
the dense areas become one solid blob and you can't see where most points
are. With `alpha=0.1`, each dot is faint and overlaps add up, so the
crowded regions show darker. It turns overplotting into information.

</details>

## 4. Hands-on: the card question

Lesson 021 found card payments jumped in the second half of September. The
owner wants one chart they can show their accountant.

**a) The numbers.** For each trading day, the share of orders paid by card.
Then a 5-trading-day rolling mean of that share.

**b) The chart.** A line chart of the daily share (thin, with markers) and
its rolling mean (thick), with:

- a title that says what it shows,
- a y-axis in percent, from 0 to 100,
- a legend,
- a source line,

saved as `output/card_share.png`.

**c) A histogram.** How many cups are in one order? Draw a histogram of
`quantity` whose bars sit *centred* on 1, 2, 3 and 4, with labelled axes.
Save it as `output/cups_per_order.png`.

Hints, if you want them:

- For (a), `(by_time["payment"] == "card").resample("D").mean().dropna()`:
  the mean of True/False is the share, and `dropna` removes the weekends
  (the mean of nothing is `NaN`, lesson 021's exercises).
- Multiply by 100 for percent, or use
  `ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))`.
- For (c), bins at the half-way points, `[0.5, 1.5, 2.5, 3.5, 4.5]`, put
  each whole number in the middle of its bar. Then `ax.set_xticks([1, 2, 3, 4])`.

<details>
<summary>Expected results</summary>

```
a) 22 trading days
   lowest daily share:  62.7% on Mon 14 Sep
   highest daily share: 91.2% on Thu 24 Sep
   rolling mean: 72.8% at its first value (Mon 07), 86.1% on Wed 30
c) cups per order: 1 -> 722, 2 -> 320, 3 -> 128, 4 -> 44
```

On the chart, the daily line wobbles a lot (a few cash orders move a day's
share several points), while the rolling mean shows a clear step up in the
second half. Starting the y-axis at 0 makes the change look modest. It
*is* a line, so zooming to, say, 50-100% would be allowed. If you zoom in,
make sure the axis labels show it.

</details>

<details>
<summary>One way to write it</summary>

```python
orders = load_orders(ORDERS_CSV)
OUT.mkdir(exist_ok=True)
by_time = orders.set_index("timestamp")

# a)
card = (by_time["payment"] == "card").resample("D").mean().dropna() * 100
smooth = card.rolling(5).mean()
print(len(card), card.idxmin(), round(card.min(), 1), card.idxmax(), round(card.max(), 1))

# b)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(card.index, card, marker="o", linewidth=1, alpha=0.6, label="Daily share")
ax.plot(smooth.index, smooth, linewidth=2.5, label="5-trading-day average")
ax.set_title("Share of orders paid by card, September 2026")
ax.set_ylabel("Orders paid by card (%)")
ax.set_ylim(0, 100)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.text(0.01, -0.02, SOURCE, fontsize=8, color="grey")
save(fig, "card_share.png")

# c)
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(orders["quantity"], bins=[0.5, 1.5, 2.5, 3.5, 4.5], edgecolor="white")
ax.set_xticks([1, 2, 3, 4])
ax.set_title("Most orders are a single cup")
ax.set_xlabel("Cups in the order")
ax.set_ylabel("Number of orders")
save(fig, "cups_per_order.png")
```

(c) is really a bar chart of counts in disguise:
`orders["quantity"].value_counts().sort_index()` with `ax.bar` gives the
same picture. With only four possible values, either is fine.

</details>

## 5. (Optional) The chart in the newsletter

A café chain's newsletter shows a line of weekly revenue for one
branch, y-axis from £1,400 to £1,600, with the last point far below the
rest and the headline "Sales slump!". Using lesson 021, give two reasons
that chart might be misleading, and say what you'd ask before believing it.

<details>
<summary>Check yourself</summary>

- **A partial last week.** If the last bin covers three trading days,
  it's low because it's short (lesson 021, exercise 5).
- **A zoomed axis.** For a line that's allowed, but a drop from £1,560 to
  £1,420 is 9%, and on a £1,400-£1,600 axis it fills most of the chart.
  The headline leans on how it looks, not what it is.

Questions to ask: how many trading days are in each week? What does
revenue *per trading day* look like? What happened in the same weeks last
year? One chart rarely proves a slump; it tells you where to look.

</details>

---

That's Week 3 done: arrays, DataFrames, cleaning, grouping, joining, time,
and now pictures. If one thing sticks, let it be: *a chart should make one
point, honestly, to someone who never saw your code*.
