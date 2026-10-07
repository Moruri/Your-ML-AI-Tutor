# Lesson 031 - Train/test splits and why we need them

**Phase 2 - Classical machine learning** | Week 5, Day 3 | Wednesday 2026-10-07

> **Goal:** Split data properly, see overfitting happen on purpose, and stop
> trusting training accuracy.

Time: about 45 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Lesson 030 ended with a confession: every error we measured was on the
same rides the model learned from. That's marking your own homework with
the answer sheet open. Today we close the answer sheet.

The idea is simple. Hide some of the data before training, train on the
rest, then score the model on the hidden part. The hidden rides stand in
for the future: rides the model will meet once it's out in the world. The
whole of honest machine learning grows from this one habit.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` as lessons
029 and 030. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-031-train-test-splits-and-why-we-need-them/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match.

## 1. Marking your own homework

```
MAE on the rides it learned from (min)         -> 2.2
```

That's lesson 030's five-feature model, scored on its own training data.
It answers "how well does this line fit September?". Useful, but nobody
needs September predicted; they already know how long those rides took.
The question that matters is how the model does on rides it hasn't seen,
and the training score can't answer it.

## 2. Holding rides back with `train_test_split`

```
training rows, test rows                       -> (882, 294)
MAE on training rides (min)                    -> 2.22
MAE on test rides, never seen (min)            -> 2.18
```

scikit-learn does the shuffling and slicing for you:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
model = LinearRegression().fit(X_train, y_train)   # train rows only
mean_absolute_error(y_test, model.predict(X_test))  # judged on test rows
```

Three details worth knowing:

- It returns **four** things, always in that order: train X, test X,
  train y, test y. Mixing up the order is the classic bug.
- `test_size=0.25` keeps a quarter of the rows back. Somewhere between 20%
  and 30% is normal.
- `random_state` fixes the shuffle, so you get the same split every run.
  Without it, your score changes each time and you can't tell whether a
  change to the model helped.

Here the test error is about the same as the training error (a hair lower,
by luck of the draw). That's good news. A straight line with five features
doesn't have enough freedom to memorise individual rides, so what it
learned carries over. Not every model is that modest.

## 3. Overfitting on purpose

```
30 training rides, distance plus random junk:
                  train MAE  test MAE
    junk columns
    0                  2.67      2.77
    5                  2.57      2.96
    10                 2.25      3.27
    15                 2.10      3.47
    20                 1.84      6.16
    25                 1.19      8.50
    28                 0.00   1721.84
```

To see the problem clearly, the script trains on only 30 rides and adds
columns of **pure random noise**, `junk_00`, `junk_01` and so on. They
contain no information about ride times. None.

Read the table top to bottom:

- **Training error keeps falling.** Every junk column gives the model one
  more knob, and with enough knobs it can twist itself to match the
  accidents in 30 particular rides.
- **Test error keeps rising.** Those accidents don't repeat in new rides,
  so the twists make things worse.
- **At 28 junk columns, training error is exactly zero.** Distance, 28
  junk columns and the intercept make 30 numbers to fit 30 rides, so the
  model can pass through every point. It has memorised the training set,
  and its predictions for new rides are off by more than a day.

That is **overfitting**: learning the noise in the training data instead
of the pattern behind it. Its signature is always the same, a training
score that looks great and a test score that doesn't. If you only ever
looked at the left column, the bottom row would look like your best model.

Real overfitting is rarely this cartoonish, but it comes from the same
place: a model with a lot of freedom and not much data. The flexible
models later in Phase 2 (trees, boosting) can do this with no junk at all.

## 4. The split itself is luck

```
test MAE for random_state 0..9 (min)           -> [2.3, 2.44, 2.24, 2.07, 2.28, 2.15, 2.11, 2.07, 1.98, 2.34]
lowest, highest                                -> (1.98, 2.44)
```

Same model, same data. The only thing that changes is which rides the
coin toss puts in the test set, and the score moves between 1.98 and 2.44
minutes. Some splits hand the model easy test rides, some hand it the
long scenic ones.

So a single test score has an error bar you can't see. If you compare two
models on one split and one wins by 0.1 minutes, that might just be the
coin. Lesson 032 fixes this with cross-validation: many splits, one
average.

## 5. When order matters: split by time

```
rides before / from 24 September               -> (956, 220)
MAE on the first 23 days (train, min)          -> 2.17
MAE on the last 7 days (test, min)             -> 2.25
```

A random split mixes days: the model might train on Tuesday's 8am rides
and be tested on Tuesday's 9am rides, in the same weather. Real use isn't
like that. You train on the past and predict the future.

For anything with dates in it, a fairer test is to **train on the earlier
part and test on the latest part**, with a plain date filter:

```python
past = rides[rides["started_at"] < "2026-09-24"]
future = rides[rides["started_at"] >= "2026-09-24"]
```

Our model holds up: 2.25 minutes on the last week, close to its training
score. Bike rides in September don't change much week to week. Sales,
prices or anything with trends can look fine on a random split and fall
apart on a time split, which is exactly why you check.

## Three rules for test sets

1. **Split first.** Before you fit anything, before you tune anything,
   ideally before you stare too hard at the data. Anything the model, or
   you, learn from the test rows makes the test score optimistic.
2. **Touch it once.** If you try 50 versions and keep the one with the
   best test score, you've trained on the test set by hand. Use the test
   set for the final check; lesson 032 shows how to choose between
   versions without it.
3. **Make the split look like real use.** Random for independent rows,
   by time when the future is what you're predicting.

## What you can do now

- Split data with `train_test_split` and unpack its four results in the
  right order.
- Fit on the training rows only and report the error on the test rows.
- Spot overfitting: falling training error, rising test error.
- Explain why more freedom plus less data means more overfitting.
- Say why one test score is partly luck.
- Split by time when the task is predicting the future.

## What to do now

1. Run `lesson.py`. Change `TRAIN_RIDES = 30` to `300` (just above
   section 3) and watch the junk columns lose most of their power to fool the model.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   measures how much data it takes to tame overfitting.
3. Next, lesson 032: cross-validation, for a score you can actually
   trust. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

A training score tells you how well the model remembers. A test score
tells you how well it will do. Only one of those is your job.
