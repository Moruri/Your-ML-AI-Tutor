# Lesson 024 - Exercises

Three quick checks, a hands-on look at payment and ticket size, and an
optional critique of a causal claim. Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-024-correlation-causation-and-the-traps-in-between/lesson.py my_lesson_024.py
python my_lesson_024.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-024-correlation-causation-and-the-traps-in-between").resolve()
```

`load_orders` and `ORDERS_CSV` are there to reuse.

---

## 1. Read the r

Without running code, roughly where does each sit: near −1, near 0, or
near +1?

- (a) Height and shoe size in adults.
- (b) Outside temperature (°C) and hot-chocolate sales in a week of mixed
  weather.
- (c) A fair coin's flip number (1st, 2nd, 3rd…) and whether it landed
  heads.

<details>
<summary>Check yourself</summary>

- **(a) Near +1** (strong positive; not perfect).
- **(b) Near −1** (hotter → fewer hot chocolates).
- **(c) Near 0** (independent).

</details>

## 2. Name the trap

For each claim, name the trap (confounder, restricted range, time-twins,
reverse arrow, or aggregation):

- (a) "Our app's daily downloads correlate 0.97 with the FTSE 100 this
  quarter — the market must be driving installs."
- (b) "Among our large drinks only, size and price are uncorrelated, so
  size doesn't affect price."
- (c) "Staff count correlates with queue length; clearly the queues make
  us hire."

<details>
<summary>Check yourself</summary>

- **(a) Time-twins** (both series drift; or a shared calendar effect).
- **(b) Restricted range** (large-only removes size variance).
- **(c) Reverse arrow** (hiring for busy times is at least as plausible).

</details>

## 3. Grain check

`hour.corr(revenue)` on **orders** is about −0.03. On **totals per
hour-of-day** it is about −0.77. Which one answers "does an order at 4pm
tend to be cheaper than an order at 8am?" Which answers "do later hours
take less money in total?"

<details>
<summary>Check yourself</summary>

Per-order r ≈ 0 answers the first (ticket size barely depends on hour).
Per-hour-of-day r ≈ −0.77 answers the second (later hours have fewer
orders, so less total revenue). Quote the grain with the number.

</details>

## 4. Hands-on: card, cash and ticket size

Using the September orders:

**a)** Make a 0/1 column `is_card` (`payment == "card"`). What is
`is_card.corr(revenue)`?

**b)** Split by `drink`. Inside each drink, what is
`is_card.corr(revenue)`? (Some may be near zero.)

**c)** Compare mean revenue for card vs cash overall, and then *within*
lattes only. Does the gap shrink?

**d)** In one sentence: is "card payers spend more" a safe causal claim
from this alone? What else might be going on?

Hints:

- `orders["is_card"] = (orders["payment"] == "card").astype(int)`
- For (b), `orders.groupby("drink").apply(...)` or a small loop.
- For (c), `orders.groupby("payment")["revenue"].mean()` and the same
  after `orders[orders["drink"] == "latte"]`.

<details>
<summary>Expected results</summary>

```
a) r(is_card, revenue) ≈ -0.02  (essentially zero)
b) inside drinks, each r is near 0 (roughly -0.15 to 0.03)
c) overall means: cash ≈ £5.60, card ≈ £5.42 (cash slightly higher);
   inside lattes: card ≈ £6.16, cash ≈ £5.90 (gap flips direction)
d) not a safe causal claim — the association is tiny, flips when you
   hold drink fixed, and payment method may track time of day or
   regulars vs tourists anyway
```

The point is the association is negligible, and even the small overall
gap can flip inside a single drink. A headline "card customers spend
more" (or less) would be overselling noise.

</details>

<details>
<summary>One way to write it</summary>

```python
orders = load_orders(ORDERS_CSV)
orders["is_card"] = (orders["payment"] == "card").astype(int)
print(orders["is_card"].corr(orders["revenue"]))

for drink, g in orders.groupby("drink"):
    print(drink, round(g["is_card"].corr(g["revenue"]), 3))

print(orders.groupby("payment")["revenue"].mean())
latte = orders[orders["drink"] == "latte"]
print(latte.groupby("payment")["revenue"].mean())
```

</details>

## 5. (Optional) Rewrite the claim

A slide says: "Correlation between morning-shift staffing and revenue is
0.82, so hiring more morning staff will raise revenue." Rewrite it as two
sentences a careful analyst would accept — one about the association, one
about what would be needed to support the causal claim.

<details>
<summary>Check yourself</summary>

Something like: "Morning staffing and morning revenue move together
(r = 0.82) across the days we observed." And: "To claim that *adding*
staff would raise revenue we'd need a comparison that changes staffing
while holding demand fixed — a trial, a natural experiment, or at least
a model that accounts for the busy mornings that cause both."

</details>

---

That's the correlation lesson done. If one thing sticks: *r is a clue,
not a cause — and the grain of the table is part of the claim*.
