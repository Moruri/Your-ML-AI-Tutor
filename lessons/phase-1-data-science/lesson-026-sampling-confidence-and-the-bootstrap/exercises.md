# Lesson 026 - Exercises

Three quick checks, a hands-on bootstrap of a median, and an optional
wording fix. Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-026-sampling-confidence-and-the-bootstrap/lesson.py my_lesson_026.py
python my_lesson_026.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-026-sampling-confidence-and-the-bootstrap").resolve()
```

`load_orders`, `ORDERS_CSV` and `RNG` are there to reuse.

---

## 1. What wobbles?

You draw ten different samples of size 30 from the same population and
compute the mean each time. The ten means are not identical. Is that a
bug in your code, bias in the data, or expected sampling variability?

<details>
<summary>Check yourself</summary>

**Expected sampling variability.** Different samples → different means.
That scatter is the point of section 2. A bug would be something else
(wrong column, `replace` messed up, seed ignored when you expected a
match). Bias is a systematic miss, not scatter around the truth.

</details>

## 2. Read the interval

A 95% CI for mean order revenue is (£4.30, £6.10). Which readings are
fair?

- (a) "There's a 95% chance the true mean is between £4.30 and £6.10."
- (b) "If we repeated this process many times, about 95% of such
  intervals would cover the true mean."
- (c) "95% of orders fall between £4.30 and £6.10."

<details>
<summary>Check yourself</summary>

- **(a) Sloppy** — common, but the true mean is fixed; the interval is
  the random thing.
- **(b) Fair** — the process-coverage reading from section 5.
- **(c) Wrong** — that would be a prediction interval / percentile of
  the raw data, not a CI for the mean.

</details>

## 3. Wider or narrower?

All else equal, does each change make a bootstrap CI **wider** or
**narrower**?

- (a) Sample size from 40 → 200.
- (b) The data gets much noisier (std doubles).
- (c) You switch from a 95% interval to a 80% interval.

<details>
<summary>Check yourself</summary>

- **(a) Narrower** — more data, less sampling noise.
- **(b) Wider** — noisier data, bigger SE.
- **(c) Narrower** — you are demanding less coverage, so the band
  shrinks (2.5/97.5 → 10/90 percentiles).

</details>

## 4. Hands-on: bootstrap a median

Using the September orders as the population:

**a)** Draw one sample of `n=50` order revenues (without replacement).
What is the sample median?

**b)** Bootstrap that sample 2,000 times (`replace=True`, same `n`).
For each replicate, store the **median** (not the mean).

**c)** Report the 95% percentile CI for the median.

**d)** Is the true population median (£4.20) inside your interval?

Hints:

- `sample = RNG.choice(rev, size=50, replace=False)`
- `np.median(...)` for each replicate
- `np.percentile(boot, [2.5, 97.5])`

<details>
<summary>Expected results</summary>

```
a) sample median will sit near £4.20 (often exactly 4.2 on this menu)
b) 2000 bootstrap medians, clustered tightly around the sample median
c) a 95% CI something like (£3.40, £5.10) — exact bounds depend on the
   sample; medians on this discrete price ladder jump between menu prices
d) usually yes for n=50; if your seed draws an unlucky sample, maybe not
   — that is sampling variability, not a failed lesson
```

With seed `RNG = np.random.default_rng(26)` and the pattern below, one
run gave sample median 4.2 and CI about (3.4, 5.5), with £4.20 inside.

</details>

<details>
<summary>One way to write it</summary>

```python
orders = load_orders(ORDERS_CSV)
rev = orders["revenue"].to_numpy()
sample = RNG.choice(rev, size=50, replace=False)
print("sample median", float(np.median(sample)))

boot = np.array([
    np.median(RNG.choice(sample, size=50, replace=True))
    for _ in range(2000)
])
lo, hi = np.percentile(boot, [2.5, 97.5])
print("95% CI", round(float(lo), 2), round(float(hi), 2))
print("true median inside?", lo <= 4.2 <= hi)
```

</details>

## 5. (Optional) Fix the slide

A slide says: "We are 95% confident the mean is £5.15." Rewrite it so a
careful analyst would accept the wording — include an interval, and say
what the 95% refers to.

<details>
<summary>Check yourself</summary>

Something like: "A 95% bootstrap CI for mean order revenue is
(£4.32, £6.07); the sample mean is £5.15. If we repeated the sampling
and interval-building process many times, about 95% of such intervals
would cover the true mean."

</details>

---

That's the sampling lesson done. If one thing sticks: *resample your
sample, read the percentiles, and say what "95%" covers — the process,
not a mystical chance that this one band is right*.
