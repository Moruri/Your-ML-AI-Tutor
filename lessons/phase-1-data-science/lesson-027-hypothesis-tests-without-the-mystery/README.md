# Lesson 027 - Hypothesis tests without the mystery

**Phase 1 - Data science basics** | Week 5, Day 1 | Monday 2026-10-05

> **Goal:** Run and interpret a t-test and a permutation test, and know
> what a p-value does and doesn't say, so "statistically significant"
> stops being a magic phrase and becomes a question you can answer.

Time: about 55 minutes. Needs the venv from lesson 013 (`pandas`, `numpy`,
`scipy`).

---

Lesson 026 asked *where does a number probably sit?* Today asks the
question that usually comes next in a meeting: *these two numbers are
different. Is that real?*

Cash orders average a bit more than card orders. Large cups earn more than
small ones. Mornings look slightly better than afternoons. Some of those
gaps are real and some are noise. A **hypothesis test** is just a careful
way to ask how easily chance alone could have produced a gap that big.
You'll build one yourself by shuffling labels, then check it against the
standard t-test, so you know what the formula is doing for you.

## How to follow along

Venv active, `python` in the repo root. Same `orders_september.csv` as
lessons 017-026. Full script:

```bash
python lessons/phase-1-data-science/lesson-027-hypothesis-tests-without-the-mystery/lesson.py
```

It writes no files. Shuffles use a seeded generator (lesson 016), so your
numbers will match.

## 1. Two numbers differ. So what?

```
order revenue by payment method:
             count   mean    std
    payment
    card       947  5.421  3.154
    cash       267  5.596  3.310
cash minus card, mean revenue (£)              -> 0.175
```

Cash orders are 17p bigger on average. Before anyone redesigns the till,
remember lesson 026: sample means wobble. The question isn't "are the
numbers different?" (they always are). It's "is this gap bigger than the
wobble?"

## 2. A permutation test: let chance have a go

```
observed gap, cash - card (£)                  -> 0.175
shuffles of the payment labels                 -> 5000
middle 95% of shuffled gaps (£)                -> (-0.437, 0.431)
share of shuffles with a gap this big          -> 0.438
```

Start by assuming the boring explanation: payment method makes **no
difference** to how much people spend. That's the **null hypothesis**.

If it's true, the "cash" and "card" labels are arbitrary stickers. So peel
them off, shuffle them, stick them back on at random and recompute the gap.
Do that 5,000 times and you get a picture of what gaps chance alone
produces. Then compare the real gap with that picture.

About 44% of shuffles produced a gap at least as big as 17p (in either
direction). That share is the **p-value**. A gap chance makes 44% of the
time is nothing to write home about.

The whole test fits in a few lines. This is `permutation_test` in
`lesson.py`:

```python
pooled = np.concatenate([a, b])
for i in range(n_perm):
    shuffled = rng.permutation(pooled)
    null[i] = shuffled[:n_a].mean() - shuffled[n_a:].mean()
p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
```

The `+ 1` counts the real labelling as one of the possible shuffles, which
stops you from ever claiming p = 0 from a finite number of shuffles.

## 3. The same question with a t-test

```
Welch t statistic                              -> 0.77
Welch p-value                                  -> 0.442
permutation p-value (section 2)                -> 0.438
```

`scipy.stats.ttest_ind(a, b, equal_var=False)` is **Welch's t-test**. It
gets to the same place with a formula. The **t statistic** is the gap
divided by its standard error (lesson 026), so it says "how many wobbles
wide is this gap?" Here it's under one, which is nothing unusual.

`equal_var=False` means the test doesn't assume both groups are equally
spread out. Leave it on; it's the safer default and costs you almost
nothing. With hundreds of rows per group the shuffle and the formula agree
closely. With small or very skewed samples, trust the shuffle more.

Verdict: **no good evidence** that cash orders are bigger. Notice the
wording. It doesn't say "cash and card are the same". It says this data
doesn't give us a reason to believe they differ.

## 4. A difference that survives the shuffle

```
large-cup orders / small-cup orders            -> (308, 409)
observed gap, large - small (£)                -> 2.333
largest shuffled gap seen (£)                  -> 0.93
permutation p-value                            -> 0.0002
Welch p-value                                  -> 7.6e-21
95% bootstrap CI for the gap (£)               -> (1.874, 2.802)
```

Now a gap that's obviously real (large cups cost more, after all). Across
5,000 shuffles the biggest gap chance produced was 93p, and the real gap
is £2.33. The permutation p-value is 1 in 5,001, as small as that many
shuffles can measure. The t-test, which isn't limited by a shuffle count,
puts it at about 10⁻²⁰.

Then comes the part people skip: **how big is the effect?** The bootstrap
interval from lesson 026 says the gap is probably between £1.87 and £2.80
per order. Report both. The test says "probably real"; the interval says
"and roughly this large".

## 5. What a p-value is, and what it is not

```
random splits where nothing is going on        -> 200
splits with p < 0.05 anyway                    -> 8
share of false alarms                          -> 0.04
chance of >=1 'hit' when you run 20 null tests -> 0.642
```

Here the script splits orders into two groups **completely at random**, 200
times. The null is true by construction, because there's nothing to find.
Even so, 8 of those tests came out "significant" at the usual 0.05 line.
That's the deal you sign up for: at 0.05, about 1 in 20 pure-noise tests
passes.

So, in plain words, a p-value **is**:

> If there were really no difference, how often would chance give a gap
> at least this big?

A p-value is **not**:

- the probability that the null hypothesis is true,
- the probability that your result is a fluke,
- a measure of how big or important the effect is.

The last line of the output is why people get burned. Run 20 tests on
noise and the chance that at least one "hits" is about 64%. Slice your data
by drink, size, milk, hour and weekday, test everything, and a
"significant" finding is close to guaranteed. Decide what you're testing
*before* you look, and be suspicious of the one hit among many tries.

## 6. Significant is not the same as important

```
morning orders / later orders                  -> (724, 490)
gap, morning - later (£ per order)             -> 0.275
permutation p-value                            -> 0.139
same gap with 10x the orders, p-value          -> 3e-06
gap x later orders (£ per month)               -> 134.68
```

Morning orders run about 28p bigger, with p = 0.14: not convincing on one
month of data. Now the script pretends we had ten times as many orders
with exactly the same gap and spread (it just repeats the data ten times).
The gap hasn't moved, but the p-value drops to 0.000003.

That's the trap. **p-values shrink as data grows**, even when the effect
stays the same. With a big enough dataset almost any gap becomes
"significant", including ones too small to care about. The last line
translates the gap into money: about £135 a month if later orders caught
up. Whether that matters is a business question, not a statistics one.

So the useful question is never just "is p small?" It's "is the effect big
enough to matter, and how sure are we of its size?" That's why section 4
reported an interval, and why you'll keep doing that in Phase 2 when you
compare models.

## What you can do now

- State a null hypothesis in plain words before testing anything.
- Build a permutation test by shuffling labels, and read its p-value.
- Run Welch's t-test with `scipy.stats.ttest_ind(..., equal_var=False)`.
- Say what a p-value means, and list three things it doesn't mean.
- Spot the multiple-testing trap when many slices are tested at once.
- Pair every test with an effect size and an interval.

## What to do now

1. Run `lesson.py`. In section 5 change the `0.05` line to `0.01` and see
   how the false-alarm count drops.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one is
   a borderline case, which is where judgement actually matters.
3. Next, lesson 028: a mini-project that takes a fresh dataset from raw CSV
   to a short written analysis. Everything from Phase 1 gets used. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Shuffle the labels to see what chance can do. Then ask how big the effect
is, because that's the part anyone will actually care about.
