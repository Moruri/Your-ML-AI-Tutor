# Lesson 032 - Cross-validation and honest evaluation

**Phase 2 - Classical machine learning** | Week 5, Day 3 | Wednesday 2026-10-07

> **Goal:** Use k-fold cross-validation to get a stable estimate of how a
> model will really do, and use it to choose between models without
> spending your test set.

Time: about 50 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Lesson 031 left two problems on the table. One test split is partly luck:
the same model scored anywhere from 1.98 to 2.44 minutes depending on the
shuffle. And if you pick between models using the test set, it stops
being a test.

**Cross-validation** fixes both. Instead of one split, you make several,
train a model on each, and average the scores. Every row gets a turn
being unseen data. You get a steadier number, a sense of how much it
wobbles, and a fair way to compare models, all without touching the test
set.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` as lessons
029 to 031. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-032-cross-validation-and-honest-evaluation/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match.

## 1. k-fold by hand

```
fold 1: train on 940 rides, score on 236  -> MAE 2.32
fold 2: train on 941 rides, score on 235  -> MAE 2.20
fold 3: train on 941 rides, score on 235  -> MAE 2.26
fold 4: train on 941 rides, score on 235  -> MAE 2.27
fold 5: train on 941 rides, score on 235  -> MAE 2.00
mean MAE across 5 folds (min)                  -> 2.21
spread, standard deviation (min)               -> 0.11
```

**k-fold cross-validation** with k = 5:

1. Shuffle the rides and cut them into 5 equal piles, called **folds**.
2. Hold out fold 1, train on folds 2 to 5, score on fold 1.
3. Hold out fold 2, train on the rest, score on fold 2. And so on.
4. Average the 5 scores.

Every ride is scored exactly once, always by a model that never saw it.
The held-out fold is usually called the **validation** fold, to keep the
word "test" for the set you lock away at the end (section 5).

`KFold` hands you the row positions for each round, and the loop is
ordinary Python:

```python
from sklearn.model_selection import KFold

folds = KFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, val_idx in folds.split(X):
    model = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
    score = mean_absolute_error(y.iloc[val_idx], model.predict(X.iloc[val_idx]))
```

Look at the spread: one fold says 2.00, another 2.32. Any one of those
could have been "the" test score in lesson 031. The mean, 2.21, is a much
steadier guess, and the standard deviation, 0.11, tells you how far a
single split could have pulled you off.

## 2. `cross_val_score` does the loop

```
per-fold MAE from cross_val_score              -> [2.32, 2.2, 2.26, 2.27, 2.0]
same as the hand-written loop?                 -> True
```

You'll rarely write that loop again:

```python
from sklearn.model_selection import cross_val_score

scores = -cross_val_score(LinearRegression(), X, y, cv=folds,
                          scoring="neg_mean_absolute_error")
```

Two things to know:

- Pass the same `folds` object to everything you compare, so every model
  gets exactly the same splits.
- The scoring name starts with `neg_` because scikit-learn's rule is
  "bigger is better" for every score. An error is better when it's
  smaller, so it hands you the negative. The minus sign in front flips it
  back to minutes. Forget it and you'll wonder why your MAE is -2.21.

## 3. `cross_validate`: more than one number

```
five features, averaged over the folds (min):
          train  validation
    MAE    2.20        2.21
    RMSE   3.21        3.22
```

`cross_validate` is the bigger sibling. It takes a list of scores and,
with `return_train_score=True`, also scores each model on the rows it
trained on. It returns a dictionary of arrays with keys like
`test_neg_mean_absolute_error` and `train_neg_mean_absolute_error` ("test"
there means the validation fold).

Putting train and validation side by side is lesson 031's overfitting
check, done five times. A big gap means the model is memorising; here
it's 0.01 minutes, so the five-feature line is learning a real pattern.

## 4. Comparing models fairly

```
5-fold MAE per model (min):
                    mean MAE   std
    guess the mean      7.26  0.29
    distance only       2.75  0.17
    five features       2.21  0.11
    e-bike slope        2.00  0.09
five features minus e-bike slope, per fold     -> [0.31, 0.12, 0.18, 0.24, 0.18]
```

Four candidates, the same five folds:

- **guess the mean** is `DummyRegressor`, which ignores the features and
  predicts the training average. Every comparison needs a baseline, and
  scikit-learn has one that fits the same `fit`/`predict` pattern.
- **distance only** and **five features** are lessons 029 and 030.
- **e-bike slope** is the model from lesson 030's hands-on exercise: an
  `e_km` column that lets e-bikes have their own minutes-per-km.

The e-bike slope model is best on average, but the stronger evidence is
the last line. Because every model saw identical folds, you can compare
them **fold by fold**, and it wins in all five, by 0.12 to 0.31 minutes.
A gap that shows up in every fold isn't a lucky split.

When two models' means are closer than their spreads, and the per-fold
winner flips back and forth, call it a tie and keep the simpler one.

## 5. The honest routine: choose with CV, test once

```
development rides, locked-away test rides      -> (940, 236)
CV MAE on development rides only:
    distance only    2.79
    five features    2.27
    e-bike slope     2.05
chosen model                                   -> 'e-bike slope'
MAE on the test rides, looked at once (min)    -> 1.81
```

Here's the full routine, the one to use from now on:

1. **Lock away a test set first.** `train_test_split` with 20% held back.
   Don't look at it.
2. **Choose using cross-validation on the rest** (the development set).
   Compare features, models and settings here as many times as you like.
3. **Refit the winner on the whole development set.** The folds were for
   choosing; the final model gets all 940 rides.
4. **Score it on the test set once**, and report that.

The test score, 1.81, is honest because the test rides played no part
in any decision. It's lower than the CV estimate of 2.05 because 236
rides is still one sample, and this one happened to be kind. That's
lesson 031's luck again, which is why you quote it with care ("about 1.8
to 2.1 minutes") rather than as a promise. What you must not do is go
back and re-choose now that you've seen it. The test set has done its
one job.

## Choosing k, and when not to shuffle

- **5 or 10 folds** is the usual choice. More folds means each model
  trains on more data, but each validation fold is smaller and noisier,
  and you train more models. 5 is a good default.
- **Shuffle** (`shuffle=True`) unless order means something. For time
  data, use `TimeSeriesSplit`, which always validates on rows that come
  after the training rows. The hands-on exercise tries it.
- Cross-validation trains k models, so it costs k times as long. With a
  straight line on 1,176 rides that's nothing. With a big model it's
  something to budget for.

## What you can do now

- Explain k-fold cross-validation in your own words.
- Run it by hand with `KFold`, or in one line with `cross_val_score`.
- Read `neg_` scores and flip them back.
- Use `cross_validate` to see train and validation scores together.
- Compare models on the same folds, fold by fold, against a
  `DummyRegressor` baseline.
- Follow the honest routine: lock away a test set, choose with CV, test
  once.

## What to do now

1. Run `lesson.py`. In `main`, change `n_splits=5` to `10` and see what
   happens to the mean and the spread.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   cross-validates through time.
3. Next, lesson 033: logistic regression, and predicting categories
   instead of numbers. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

One split is an opinion. Five folds are a conversation. And the test set
is the final word, so you only ask it once.
