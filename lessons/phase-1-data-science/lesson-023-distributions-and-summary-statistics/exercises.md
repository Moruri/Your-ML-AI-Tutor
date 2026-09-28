# Lesson 023 - Exercises

Three quick checks, a hands-on look at drink sizes, and an optional
question about a summary that misleads. Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-023-distributions-and-summary-statistics/lesson.py my_lesson_023.py
python my_lesson_023.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-023-distributions-and-summary-statistics").resolve()
```

`load_orders` and `ORDERS_CSV` are there to reuse.

---

## 1. Which number?

For each question, would you quote the mean, the median, or either?

- (a) "What does a typical customer pay for one order?"
- (b) "If we multiply by 1,214 orders, roughly how much did we take?"
- (c) "What's a typical trading day's revenue?" (daily totals are nearly
  symmetric)

<details>
<summary>Check yourself</summary>

- **(a) Median** — "typical" and the order-revenue distribution is skewed.
- **(b) Mean** — totals are mean × count. (Or just sum; same idea.)
- **(c) Either** — mean and median almost agree when the shape is symmetric.

</details>

## 2. What moves?

You append one order of £200 to the September data. Which of these change
a lot, and which barely move: mean, median, std, IQR?

<details>
<summary>Check yourself</summary>

**Mean** and **std** jump. **Median** and **IQR** barely move (with 1,214
rows, one value can't shift the middle or the quartiles much). That's the
robust / non-robust split from section 5.

</details>

## 3. Read the five-number summary

```
min 1.7 | Q1 3.4 | median 4.2 | Q3 7.4 | max 19.2
```

Is this left-skewed, right-skewed, or roughly symmetric? How can you tell
without computing `.skew()`?

<details>
<summary>Check yourself</summary>

**Right-skewed.** The upper whisker (median→max, and even median→Q3) is
longer than the lower one (min→median). Also max is much further from the
median than min is. When in doubt: if mean > median, suspect a right tail.

</details>

## 4. Hands-on: sizes and a fair comparison

Using `orders_september.csv`:

**a)** For `price`, report mean, median, std and IQR.

**b)** Split by `size` (`small` / `medium` / `large`). For each size, report
the median price and the IQR of price.

**c)** Which size has the widest price spread (IQR), and why might that be
(look at which drinks appear in each size)?

**d)** Daily *cups* (`quantity` summed per trading day): are mean and median
close? What's the skew?

Hints:

- IQR: `s.quantile(0.75) - s.quantile(0.25)`.
- For (d), reuse the resample pattern from lesson 021:
  `orders.set_index("timestamp")["quantity"].resample("D").sum()`, then
  keep days `> 0`.

<details>
<summary>Expected results</summary>

```
a) mean ≈ 3.46, median 3.70, std ≈ 0.89, IQR 1.00
b) small:   median 3.20, IQR 1.60
   medium:  median 3.80, IQR 0.20
   large:   median 4.30, IQR 0.20
c) small has the widest IQR — it is the only size that includes espresso
   (£1.70) alongside small milk drinks, so the price ladder is longer;
   medium and large are almost only the milk drinks on a tight menu
d) mean ≈ 87.4, median 85.5, skew ≈ 0.24 (nearly symmetric)
```

Unit price has median a little *above* the mean because the cheap
espressos pull the mean down — the opposite tilt from revenue-per-order,
which is right-skewed by multi-cup tickets. Same shop, different column,
different shape.

</details>

<details>
<summary>One way to write it</summary>

```python
orders = load_orders(ORDERS_CSV)
p = orders["price"]
print(p.mean(), p.median(), p.std(), p.quantile(0.75) - p.quantile(0.25))

for size, g in orders.groupby("size"):
    s = g["price"]
    print(size, s.median(), s.quantile(0.75) - s.quantile(0.25))

cups = orders.set_index("timestamp")["quantity"].resample("D").sum()
cups = cups[cups > 0]
print(cups.mean(), cups.median(), cups.skew())
```

</details>

## 5. (Optional) The average that flatters

A headline says: "Average order value up to £5.46!" after a month where a
local office started placing one £80 platter order every Friday. What would
you check before celebrating, and which alternative number would you put
in the headline instead?

<details>
<summary>Check yourself</summary>

Check the **shape** and the **median**. A few huge platter orders lift the
mean while leaving most customers unchanged. Quote the **median** order
value (and maybe show both). Also look at the Friday-only slice: if the
mean jump lives entirely there, it isn't a general rise in what people pay.

</details>

---

That's the distributions lesson done. If one thing sticks: *shape first,
then pick the summary that matches the question*.
