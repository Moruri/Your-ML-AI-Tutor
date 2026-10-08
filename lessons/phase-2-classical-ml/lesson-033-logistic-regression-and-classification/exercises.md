# Lesson 033 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-033-logistic-regression-and-classification/lesson.py my_lesson_033.py
python my_lesson_033.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-033-logistic-regression-and-classification").resolve()
```

`load_rides`, `split`, `sigmoid`, `FEATURES` and `TARGET` are there to
reuse.

---

## 1. Two answers to one question

For one ride, `model.predict_proba(ride)` returns `[[0.38, 0.62]]`. What
does `model.predict(ride)` return, and what does the 0.38 mean?

<details>
<summary>Check yourself</summary>

`predict` returns `[1]`, casual, because P(casual) = 0.62 is above 0.5.
The 0.38 is P(member). The columns follow `model.classes_`, which is
`[0, 1]`, so the first column is member and the second is casual. The two
always add up to 1.

</details>

## 2. Odds are not probabilities

A weekday ride gets P(casual) = 0.20. The same ride on a Saturday has its
odds multiplied by 3.33 (the `weekend` "odds x" from section 5). Does its
probability become 0.20 × 3.33 = 0.67?

<details>
<summary>Check yourself</summary>

No. Convert to odds first: 0.20 / 0.80 = 0.25. Multiply: 0.25 × 3.33 ≈
0.83. Convert back: 0.83 / (1 + 0.83) ≈ 0.45. So the Saturday version is
about 45% likely to be casual, not 67%. Multiplying odds always keeps
the probability between 0 and 1; multiplying probabilities wouldn't.
(Try it on a ride already at 0.40: odds 0.67 × 3.33 ≈ 2.2, probability
about 0.69, while 0.40 × 3.33 would be an impossible 1.33.)

</details>

## 3. Why stratify?

Only about a third of rides are casual. What could go wrong with
`train_test_split` if you left out `stratify=`, and would it matter more
or less if casual riders were 2% of rides?

<details>
<summary>Check yourself</summary>

Without it, the test set's casual share is left to chance. With 236 test
rides it might come out at 28% or 36%, which shifts every score you
compute. With a rare class it's worse: 2% of 236 is about 5 rides, and an
unlucky split could leave 1 or 9, or even none. Stratifying fixes the
mix to match the whole dataset. Use it by default for classification;
`StratifiedKFold` does the same for cross-validation, and scikit-learn
uses it automatically when you pass `cv=5` to a classifier.

</details>

## 4. Hands-on: does rush hour help?

Members commute, so rides at commuting times should lean member. Test
that idea.

**a)** Add an `hour` column (`rides["started_at"].dt.hour`) and a `peak`
column that is 1 when the hour is 7, 8, 17 or 18. What share of peak
rides and of off-peak rides are casual?

**b)** Fit `LogisticRegression(max_iter=1000)` on `FEATURES + ["peak"]`
using the same `split`. What is the `peak` coefficient, and its odds
multiplier?

**c)** What's the test accuracy now, compared with 0.792? What happened?

Hints:

- `rides["hour"].isin([7, 8, 17, 18]).astype(int)`
- `rides.groupby("peak")["casual"].mean()`

<details>
<summary>Expected results</summary>

```
a) off-peak 0.362 casual, peak 0.284 casual (about 49% of rides are peak)
b) peak coef about -0.02, odds x about 0.98
c) accuracy 0.788, a whisker below 0.792
```

On its own, peak time does lean member: 28% casual against 36%. But once
the model already knows the ride's length, duration and whether it's a
weekend, `peak` adds almost nothing. Most of what it knew was "weekday",
and the model already had that. The coefficient sits next to zero and the
accuracy drops by one ride. A feature can be true and still not useful,
and the only way to know is to test it, honestly, on data the model
didn't train on.

</details>

<details>
<summary>One way to write it</summary>

```python
rides = load_rides(RIDES_CSV)
rides["hour"] = rides["started_at"].dt.hour
rides["peak"] = rides["hour"].isin([7, 8, 17, 18]).astype(int)
print(rides.groupby("peak")["casual"].mean().round(3))

train, test = split(rides)
cols = FEATURES + ["peak"]
model = LogisticRegression(max_iter=1000).fit(train[cols], train[TARGET])
print(dict(zip(cols, model.coef_[0].round(2))))
print(np.exp(model.coef_[0]).round(2))
print(round(accuracy_score(test[TARGET], model.predict(test[cols])), 3))
```

One test split is still partly luck (lesson 031), so a one-ride
difference means "no real change". To be sure, compare both feature sets
with cross-validation on the training rides.

</details>

## 5. (Optional) Why "regression"?

It predicts a category, so why is it called logistic *regression*?

<details>
<summary>Check yourself</summary>

Because what it actually fits is a number: the probability, or more
precisely the log of the odds, which is the straight-line score `z`. That
part is ordinary regression. The classification only happens at the end,
when you cut the probability at a threshold. That last step being
separate is useful: lesson 034 moves the threshold away from 0.5 without
retraining anything.

</details>

---

That's logistic regression. If one thing sticks: *it's a line turned into
a probability, and the class is just that probability cut at a
threshold.*
