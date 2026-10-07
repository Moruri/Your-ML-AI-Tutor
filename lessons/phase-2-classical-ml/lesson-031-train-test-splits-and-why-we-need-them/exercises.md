# Lesson 031 - Exercises

Three quick checks, a hands-on experiment, and an optional judgement call.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-031-train-test-splits-and-why-we-need-them/lesson.py my_lesson_031.py
python my_lesson_031.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-031-train-test-splits-and-why-we-need-them").resolve()
```

`load_rides`, `add_junk_columns`, `mae`, `FEATURES` and `TARGET` are there
to reuse.

---

## 1. Spot the leak

What's wrong with this, even though it uses `train_test_split`?

```python
model = LinearRegression().fit(rides[FEATURES], rides[TARGET])
X_train, X_test, y_train, y_test = train_test_split(
    rides[FEATURES], rides[TARGET], test_size=0.25, random_state=42
)
print(mean_absolute_error(y_test, model.predict(X_test)))
```

<details>
<summary>Check yourself</summary>

The model was fitted on **all** the rides before the split, so the "test"
rides were part of its training data. The score is a training score
wearing a test score's name tag. Split first, then fit on `X_train`,
`y_train` only.

</details>

## 2. Test beat train. Bug?

In section 2 the test MAE (2.18) came out slightly lower than the training
MAE (2.22). Is something broken?

<details>
<summary>Check yourself</summary>

No. With a simple model the two should be close, and which one is lower
is down to which rides landed where. This test set happened to get
slightly easier rides. Section 4 shows test scores moving by more than
that just from reshuffling. What *would* worry you is a test error far
above the training error.

</details>

## 3. Which split?

You're building a model to predict how many rides each station will have
**next week**, from two years of daily station counts. Random split or
time split, and why?

<details>
<summary>Check yourself</summary>

Time split: train on the older data, test on the most recent weeks. That
matches how the model will be used. A random split would let it learn
from days on both sides of each test day, including holidays and
seasonal patterns it couldn't know about in advance, so its score would
flatter it.

</details>

## 4. Hands-on: how much data tames the junk?

Section 3 showed 25 junk columns wrecking a model trained on 30 rides.
Keep those 25 junk columns (plus `distance_km`) and grow the training set.

**a)** Build `noisy = add_junk_columns(load_rides(RIDES_CSV), 30)` and the
column list `cols = ["distance_km"] + [f"junk_{i:02d}" for i in range(25)]`.

**b)** For `train_size` of 30, 100, 300 and 882, split with
`train_test_split(noisy, train_size=n, random_state=1)`, fit, and record
the train and test MAE.

**c)** What happens to the gap between them as the training set grows?

Hints:

- With one DataFrame, `train_test_split` returns just two pieces:
  `train, test = train_test_split(noisy, train_size=n, random_state=1)`.
- `mae(model, train[cols], train[TARGET])` does the scoring.

<details>
<summary>Expected results</summary>

```
train_size   train MAE   test MAE
30           1.19        8.50
100          1.94        3.14
300          2.52        2.92
882          2.76        2.70
```

c) The gap shrinks from more than 7 minutes to almost nothing. With 30
rides, 26 features are enough to memorise noise. With 882 rides, the junk
columns can't line up with that many random accidents at once, so the
model mostly ignores them. Training error *rises* as you add data,
because memorising gets harder; test error falls. Both converge on the
honest answer.

</details>

<details>
<summary>One way to write it</summary>

```python
noisy = add_junk_columns(load_rides(RIDES_CSV), 30)
cols = ["distance_km"] + [f"junk_{i:02d}" for i in range(25)]

for n in [30, 100, 300, 882]:
    train, test = train_test_split(noisy, train_size=n, random_state=1)
    model = LinearRegression().fit(train[cols], train[TARGET])
    print(n, mae(model, train[cols], train[TARGET]), mae(model, test[cols], test[TARGET]))
```

Moral: overfitting is a ratio problem. More freedom needs more data.
When you can't get more data, you take freedom away instead, which is
what regularisation (lesson 041) does.

</details>

## 5. (Optional) The tempting shortcut

A teammate tries 40 different feature combinations, scores each on the
same test set, and reports the best one: "test MAE 1.91". Why might the
model do worse than 1.91 next month?

<details>
<summary>Check yourself</summary>

Picking the winner of 40 tries on the test set means the test set helped
choose the model. Some of that 1.91 is real skill and some is the winner
happening to suit those particular test rides (section 4 showed how much
luck moves a score). The test set is no longer unseen. The fix is to
choose using the training data only, with a validation set or the
cross-validation in lesson 032, and score the final pick on the test set
once.

</details>

---

That's train/test splits. If one thing sticks: *a model is only as good
as its score on data it never trained on, so split first and keep the
test set for the end.*
