# Lesson 025 - Exercises

Three quick checks, a hands-on look at conditioning in the orders, and an
optional Bayes rewrite. Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-025-probability-intuition-for-ml/lesson.py my_lesson_025.py
python my_lesson_025.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-025-probability-intuition-for-ml").resolve()
```

`load_orders` and `ORDERS_CSV` are there to reuse.

---

## 1. Read the share

Without running code, roughly where does each sit: near 0, near 0.5, or
near 1?

- (a) `P(a fair coin lands heads)`.
- (b) `P(order revenue > £50)` in this coffee shop.
- (c) `P(card | payment is card)` — yes, really.

<details>
<summary>Check yourself</summary>

- **(a) Near 0.5** (exactly 0.5 if the coin is fair).
- **(b) Near 0** — the max order in September is £19.20.
- **(c) Exactly 1** — conditioning on the event itself.

</details>

## 2. Base-rate trap

A fraud flag is 95% accurate on fraudsters and fires on 2% of honest
users. Fraud is 1 in 1,000. Of the flagged users, are most fraudsters or
most honest?

<details>
<summary>Check yourself</summary>

**Mostly honest.** In 100,000 users: ~100 fraudsters → ~95 flags; ~99,900
honest → ~2,000 flags. So roughly 95 / 2,095 ≈ 4.5% of flags are real
fraud. Same shape as the loyalty-ping example: rare event + imperfect
test ⇒ most positives are false.

</details>

## 3. Expected value sanity

A game: pay £1, win £5 with probability 0.1, else nothing. Is the EV
positive, negative, or zero? Would you play it once for fun? Would you
play it 10,000 times with money you need?

<details>
<summary>Check yourself</summary>

`E[net] = 0.1 × £5 − £1 = −£0.50` — **negative**. Playing once for fun is
a taste question; playing 10,000 times is a reliable way to lose about
£5,000. EV guides the long run, not a single ticket.

</details>

## 4. Hands-on: condition on morning and payment

Using `orders_september.csv`:

**a)** What is `P(espresso)`? `P(espresso | morning)`? `P(espresso | afternoon)`?

**b)** What is `P(quantity >= 3)`? `P(quantity >= 3 | card)`?

**c)** Expected revenue for morning orders vs afternoon orders (just the
two means). Which is higher?

**d)** In one sentence: did conditioning on morning change the espresso
share much? What about the multi-cup share when you condition on card?

Hints:

- Morning: `orders["timestamp"].dt.hour < 12`.
- `(orders["drink"] == "espresso").mean()` is `P(espresso)`.
- Filter first, then take `.mean()` for the conditional.

<details>
<summary>Expected results</summary>

```
a) P(espresso) ≈ 0.129; morning ≈ 0.124; afternoon ≈ 0.137
   (slightly *fewer* espressos in the morning under this hour cut)
b) P(qty ≥ 3) ≈ 0.142; given card ≈ 0.135 (essentially unchanged)
c) morning mean ≈ £5.51, afternoon ≈ £5.36 — mornings a touch higher
d) conditioning on morning moves espresso only a little; card barely
   moves the multi-cup share — payment and cup count look nearly
   independent here
```

Re-run section 2 of `lesson.py` if you want the exact latte numbers as a
template for the same style of printout.

</details>

<details>
<summary>One way to write it</summary>

```python
orders = load_orders(ORDERS_CSV)
morning = orders[orders["is_morning"]]
afternoon = orders[~orders["is_morning"]]

print((orders["drink"] == "espresso").mean())
print((morning["drink"] == "espresso").mean())
print((afternoon["drink"] == "espresso").mean())

print((orders["quantity"] >= 3).mean())
print((orders.loc[orders["is_card"], "quantity"] >= 3).mean())

print(morning["revenue"].mean(), afternoon["revenue"].mean())
```

</details>

## 5. (Optional) Rewrite with Bayes

A slide says: "Our churn model is 90% sensitive, so a flagged customer is
90% likely to churn." Rewrite it as two careful sentences — one about
what 90% actually is, one about what else you need for the posterior.

<details>
<summary>Check yourself</summary>

Something like: "90% is `P(flag | will churn)` — the hit rate among
people who do churn, not the chance a flagged person churns." And: "To
get `P(will churn | flag)` we also need the base rate of churn and the
false-positive rate among stayers; without those the posterior can be
far below 90%."

</details>

---

That's the probability lesson done. If one thing sticks: *counts before
formulas, and a rare event plus a noisy test means most alarms are
false*.
