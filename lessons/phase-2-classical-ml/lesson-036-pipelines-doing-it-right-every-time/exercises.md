# Lesson 036 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-036-pipelines-doing-it-right-every-time/lesson.py my_lesson_036.py
python my_lesson_036.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-036-pipelines-doing-it-right-every-time").resolve()
```

`load_rides`, `split`, `make_preprocess`, `make_model`, `FOLDS` and the
column lists are there to reuse.

---

## 1. Inside or outside?

Which of these belong inside the pipeline?

**a)** Filling missing temperatures with the mean temperature.

**b)** Dropping rides with a negative duration (a logging bug).

**c)** Keeping only the 10 most useful columns.

**d)** Converting `started_at` from text to a datetime.

<details>
<summary>Check yourself</summary>

- **a) Inside.** The mean is learned from data, so it must come from
  the training part only. `SimpleImputer` is the pipeline step for it.
- **b) Outside is fine.** It's a fixed rule about bad records, not
  something learned, and it applies the same to every row. (Just don't
  drop test rows because the *answer* looks odd.)
- **c) Inside.** It learns from the answers, which is exactly section
  4's leak.
- **d) Outside is fine.** Parsing a date learns nothing.

The test: does the step look at the data to decide what to do? Then
it's inside.

</details>

## 2. What does predict skip?

When you call `model.predict(test)` on a fitted pipeline, which methods
run on the `StandardScaler`, and why does it matter?

<details>
<summary>Check yourself</summary>

Only `transform`. The scaler uses the mean and spread it learned during
`fit` on the training rides, and never re-learns from the test rides.
If it ran `fit` again, the test rides would be measured with their own
ruler, which is the leak we've been avoiding since lesson 031.

</details>

## 3. Where did ride_id go?

Your colleague's ColumnTransformer lists only the numeric columns and
sets `remainder="passthrough"` "so nothing gets lost". Fitting crashes
with `could not convert string to float: 'R00919'`. What happened, and
what would have been worse than the crash?

<details>
<summary>Check yourself</summary>

`passthrough` sent every unlisted column straight to the model,
including `ride_id`, `started_at` and the text columns. The model can't
do maths on "R00919", so it crashed. Worse would have been an ID that
*was* numeric: no crash, and the model would happily learn from a
meaningless label, or from a leaked column like the answer itself.
Listing the columns you want, and letting the rest drop, is the safer
default.

</details>

## 4. Hands-on: pick k the honest way

Using `make_model()` (the k-NN pipeline), cross-validate on the training
rides with `n_neighbors` set to 5, 15, 31, 61 and 121. Print the mean
and spread for each. Which `k` would you pick, and why might you not
trust the winner too much?

Hints:

- `model.set_params(knn__n_neighbors=k)` changes it in place and
  returns the pipeline, so you can pass it straight to
  `cross_val_score`.
- Use `FOLDS` so your numbers match.

<details>
<summary>Expected results</summary>

```
k=5    0.761 (spread 0.022)
k=15   0.781 (spread 0.043)
k=31   0.787 (spread 0.028)
k=61   0.769 (spread 0.016)
k=121  0.757 (spread 0.017)
```

31 wins, with 15 close behind. Small `k` listens to a handful of
neighbours and gets noisy; huge `k` averages over so many rides it
blurs everything towards "member". The sweet spot is in between.

Don't trust the winner too much: 0.787 and 0.781 are well inside each
other's spread, so 15 and 31 are really a tie. And because you chose
`k` by looking at these scores, the winning CV number is a little
optimistic. That's why the test set stays untouched until the very end,
and why lesson 042 does this search with `GridSearchCV`.

</details>

<details>
<summary>One way to write it</summary>

```python
rides = load_rides(RIDES_CSV)
train, test = split(rides)
model = make_model()
for k in [5, 15, 31, 61, 121]:
    scores = cross_val_score(model.set_params(knn__n_neighbors=k),
                             train, train[TARGET], cv=FOLDS)
    print(k, round(scores.mean(), 3), round(scores.std(), 3))
```

</details>

## 5. (Optional) How much did the weekend flag buy?

Section 3's logistic pipeline scores 0.790 in CV, against 0.772 in
lesson 035 without the `weekend` column. Remove `"weekend"` from
`FLAGS` (set it to an empty list and drop the `flags` entry from
`make_preprocess`) and re-run the logistic CV. Before you run it: will
you land exactly on 0.772?

<details>
<summary>Check yourself</summary>

Yes, 0.772, the same as lesson 035's "numbers + station + bike" row.
That's a nice check in itself: for logistic regression the small leak
in lesson 035 happened to change nothing you can see at three decimals,
and the whole 0.018 gain comes from the weekend flag. Leaks aren't
always big. You fix them anyway, because you can't tell in advance
which kind you've got, and section 4 showed what the big kind looks
like.

</details>

---

That's pipelines. If one thing sticks: *if a step learns from data, it
lives inside the pipeline.*
