# Lesson 025 - Probability intuition for ML

**Phase 1 - Data science basics** | Week 4, Day 2 | Tuesday 2026-09-29

> **Goal:** Understand conditional probability, Bayes' rule and expected
> value through small simulations — so "given that…" and "on average…"
> become tools you can use, not formulas you memorise and forget.

Time: about 50 minutes. Needs the venv from lesson 013 (`pandas`, `numpy`).

---

Yesterday's lesson asked whether two columns move together. Today asks a
more basic betting question: *how likely is this, and how does that change
when I learn something new?*

Probability is the language ML models speak when they spit out a score,
a calibrated chance, or an expected payoff. You do not need measure theory
for that. You need counts, a honest "given that", and the habit of
simulating when the algebra gets sticky.

## How to follow along

Venv active, `python` in the repo root. Same `orders_september.csv` as
lessons 017–024. Full script:

```bash
python lessons/phase-1-data-science/lesson-025-probability-intuition-for-ml/lesson.py
```

It writes no files. The Monte Carlo uses a seeded generator (lesson 016),
so your numbers will match.

## 1. Probability is a count with a denominator

```
orders, count                                  -> 1214
P(latte)  = latte / all                        -> 0.311
P(card)   = card / all                         -> 0.780
P(morning)= morning / all                      -> 0.652
```

A probability is a share: favourable outcomes over possible ones. On this
month of tickets, roughly three in ten orders are lattes, about four in
five are card, and mornings take a bit over half. Nothing mystical — just
a careful fraction.

## 2. Conditional probability: "given that…"

```
P(latte)                                       -> 0.311
P(latte | morning)                             -> 0.300
P(latte | afternoon)                           -> 0.331
```

The `|` means **given that**. Restrict the denominator to morning orders
and the latte share shifts a little; restrict to afternoon and it shifts
the other way. Conditioning on drink changes `P(card | drink)` too —
tea buyers tip slightly more toward card than cappuccino buyers.

New information updates the denominator. That is the whole idea. In ML,
every feature a model conditions on is doing a version of this: "given
these inputs, what's the chance of churn / spam / a click?"

## 3. Bayes with counts, not formulas

A loyalty-ping thought experiment. 1,000 customers; 5% are regulars. The
ping fires for 90% of regulars and 8% of visitors:

```
true regulars (5%)                             -> 50
pings among regulars (90%)                     -> 45
pings among visitors (8%)                      -> 76
total pings                                    -> 121
of those pings, how many are regulars?         -> 45/121 = 0.372
```

The ping fires 121 times. Only 45 of those are real regulars. So
**P(regular | ping) ≈ 0.37 — not 0.90**. The test is sensitive, but
regulars are rare, so most pings are false alarms. Counts first; the
formula just compresses the same arithmetic.

This is the classic rare-event / imperfect-test trap. Medical screening,
fraud flags and "this email looks like spam" all live here.

## 4. Bayes as an update (same story, formula form)

```
prior P(regular)                               -> 0.05
likelihood P(ping | regular)                   -> 0.90
P(ping)  (law of total probability)            -> 0.121
posterior P(regular | ping)                    -> 0.372
```

Same answer as the count table. Bayes' rule in one line: **posterior ∝
likelihood × prior**, then renormalise by `P(data)`. You start with a
belief (5% regulars), see evidence (a ping), and leave with an updated
belief (~37%). In ML you'll meet this as updating when new evidence
arrives — spam filters, medical models, A/B readouts.

## 5. Expected value: the long-run average payoff

```
E[order revenue]  (= mean)                     -> 5.459
P(customer tips)                               -> 0.30
tip size when they do (10% of bill)            -> 0.10
E[tip per order]                               -> 0.164
raffle: E[net] = 0.01×£4.20 − £0.20            -> -0.158
```

**Expected value** is the probability-weighted average outcome. The mean
order revenue *is* an expected value estimated from data. A tip that
lands 30% of the time at 10% of the bill is worth about 16p per order in
the long run. The raffle has negative EV — the house wins if you play
enough. One draw can still win; EV is about many repeats.

## 6. A tiny Monte Carlo: simulate, don't just derive

```
simulated orders                               -> 10000
mean bill in the sim                           -> 4.861
mean tip in the sim                            -> 0.147
theory E[tip] ≈ 0.30 × 0.10 × E[bill]          -> 0.146
```

Theory and simulation agree within noise. When a closed-form answer is
awkward, draw many random outcomes and average — that's **Monte Carlo**.
Seed the generator (lesson 016) so the demo is reproducible. A huge
chunk of modern ML evaluation (bootstrap CIs tomorrow, permutation tests
later, dropout as approximate Bayesian inference) is this same habit:
simulate the uncertainty you cannot write down cleanly.

## What you can do now

- Read a probability as a count over a denominator.
- Compute and interpret `P(A | B)` by restricting the table.
- Work Bayes with a count table before reaching for the formula.
- Update a prior to a posterior when evidence arrives.
- Compute a simple expected value and spot a negative-EV game.
- Run a small seeded Monte Carlo and check it against theory.

## What to do now

1. Run `lesson.py`. Change the prior in section 4 from 0.05 to 0.20 and
   watch the posterior jump — same test, different base rate.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   conditions on payment and time of day in the real orders.
3. Next, lesson 026: sampling, confidence and the bootstrap. Probability
   told you how to bet; sampling asks how sure the bet is. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Counts before formulas. Condition before you update. Simulate when the
algebra gets sticky.
