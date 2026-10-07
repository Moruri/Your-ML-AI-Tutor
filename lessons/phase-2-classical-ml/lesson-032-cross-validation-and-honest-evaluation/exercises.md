# Lesson 032 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-032-cross-validation-and-honest-evaluation/lesson.py my_lesson_032.py
python my_lesson_032.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-032-cross-validation-and-honest-evaluation").resolve()
```

`load_rides`, `cv_mae`, `rounded`, `FEATURES`, `CANDIDATES` and `TARGET`
are there to reuse.

---

## 1. The minus sign

A friend runs this and proudly reports an MAE of -2.21 minutes. What
happened, and is their model good or bad?

```python
cross_val_score(LinearRegression(), X, y, cv=folds,
                scoring="neg_mean_absolute_error").mean()
```

<details>
<summary>Check yourself</summary>

scikit-learn returns the **negative** MAE so that bigger is always
better. Their model's MAE is really 2.21 minutes, which is the same
five-feature model from the lesson: decent, and far better than the 7.26
baseline. Put a minus in front of `cross_val_score` to get the error back.

</details>

## 2. More folds, more wobble?

With the five-feature model, 3 folds give a spread (standard deviation)
of 0.05 minutes and 10 folds give 0.17. The mean is 2.21 either way. Is
10-fold worse?

<details>
<summary>Check yourself</summary>

No. The spread is the spread of the *individual fold scores*. With 10
folds each validation fold has only about 118 rides, so each score is
noisier, but you're averaging ten of them, and each model trained on 90%
of the data instead of 67%. The mean is what you use, and it's the same.
The per-fold spread just tells you how much one small split could have
fooled you.

</details>

## 3. Which number goes in the report?

After the honest routine, you have a CV estimate of 2.05 minutes and a
test score of 1.81. Your manager asks "how accurate is it?". What do you
say?

<details>
<summary>Check yourself</summary>

Something like: "On rides it had never seen, it was off by about 1.8
minutes on average; cross-validation suggests 2 minutes is a safe thing
to expect." Quote the test score as the final, untouched check, and use
the CV estimate to stop anyone reading 1.81 as a promise. What you don't
do is quote the training error, or go back and tweak the model to make
the test number even better.

</details>

## 4. Hands-on: cross-validate through time

The rides in `rides_clean.csv` are in order of `started_at`. Shuffled
k-fold mixes the days up. `TimeSeriesSplit` doesn't: each round trains on
everything up to a point in time and validates on the slice right after.

**a)** Build `ts = TimeSeriesSplit(n_splits=5)` (from
`sklearn.model_selection`) and loop over `ts.split(rides)`. For each
round, print how many rides it trains on and the first and last date of
the validation rides.

**b)** Cross-validate the five-feature model and the e-bike slope model
with `cv=ts`. Per-fold MAE and mean for each?

**c)** Compare with the shuffled 5-fold results in section 4 (2.21 and
2.00). Does the winner change?

Hints:

- `rides["started_at"].iloc[val_idx[0]].date()` gives the first date.
- `cv_mae(LinearRegression(), rides[cols], rides[TARGET], ts)` works with
  any splitter.

<details>
<summary>Expected results</summary>

```
a) train 196 rides, validate 2026-09-04 to 2026-09-08
   train 392 rides, validate 2026-09-08 to 2026-09-13
   train 588 rides, validate 2026-09-13 to 2026-09-18
   train 784 rides, validate 2026-09-18 to 2026-09-24
   train 980 rides, validate 2026-09-24 to 2026-09-30
   (196 rides in each validation slice)
b) five features: [2.07, 2.49, 2.09, 2.35, 2.29], mean about 2.26
   e-bike slope:  [2.04, 2.33, 1.86, 1.98, 2.04], mean about 2.05
c) Both are a little worse than shuffled CV, which is normal: the early
   rounds train on only a few days, and every round predicts days it
   never saw. The e-bike slope model still wins in all five rounds.
```

</details>

<details>
<summary>One way to write it</summary>

```python
from sklearn.model_selection import TimeSeriesSplit

rides = load_rides(RIDES_CSV)
ts = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in ts.split(rides):
    first = rides["started_at"].iloc[val_idx[0]].date()
    last = rides["started_at"].iloc[val_idx[-1]].date()
    print(len(train_idx), "rides, validate", first, "to", last)

for name in ["five features", "e-bike slope"]:
    cols = CANDIDATES[name]
    scores = cv_mae(LinearRegression(), rides[cols], rides[TARGET], ts)
    print(name, rounded(scores), round(float(scores.mean()), 2))
```

`TimeSeriesSplit` relies on the rows already being in time order. If
your data isn't, sort it by the date column first, or the "future" folds
won't be the future at all.

</details>

## 5. (Optional) The sneaky leak

Someone fills missing temperatures with the **average temperature of the
whole dataset**, then runs 5-fold cross-validation. Why is the CV score
now very slightly dishonest, and how would you fix it?

<details>
<summary>Check yourself</summary>

The average was computed using every row, including the rows that later
sit in each validation fold. So each model got a tiny peek at its
validation data through that average. The fix is to compute the fill
value from the training folds only, inside each round. Doing that by
hand is fiddly, which is exactly what `Pipeline` solves in lesson 036:
it re-learns every preprocessing step inside each fold, so nothing
leaks.

</details>

---

That's cross-validation. If one thing sticks: *choose with many folds,
confirm with one untouched test set, and never let the test set help you
choose.*
