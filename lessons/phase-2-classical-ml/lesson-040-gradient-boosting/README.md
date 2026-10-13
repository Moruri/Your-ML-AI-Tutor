# Lesson 040 - Gradient boosting

**Phase 2 - Classical machine learning** | Week 6, Day 2 | Tuesday 2026-10-13

> **Goal:** Understand boosting as "fix the last model's mistakes" and
> tune a `HistGradientBoosting` model.

Time: about 45 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

A random forest grows lots of deep trees side by side and lets them
vote. **Boosting** flips that around. It grows *small* trees *one after
another*, and each new tree has a single job: fix whatever the trees
before it still get wrong. No tree is impressive on its own. The team
is.

Gradient boosting is the model that wins a huge share of real-world
contests on tables of data like ours, so it's worth understanding the
idea properly before we lean on the library. We'll build it by hand
first, then hand it to scikit-learn's `HistGradientBoostingClassifier`
and learn the few dials that matter.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` and the
same stratified train/test split as lessons 033 to 039. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-040-gradient-boosting/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match. It takes about half a minute.

## 1. Boosting by hand: fix the last model's mistakes

```
learning rate                                  -> 0.3
each tree                                      -> 'DecisionTreeRegressor(max_depth=2)'
errors shrink, round by round (minutes):
                  train RMSE  test RMSE
  start (mean)         10.28       9.01
  after 1 trees         7.98       7.08
  after 2 trees         6.45       5.85
  after 5 trees         4.36       4.08
  after 10 trees        3.77       3.66
  after 25 trees        3.49       3.67
  after 50 trees        3.27       3.65
```

To see the idea clearly we switch, for this section only, to a
regression task: predict a ride's duration in minutes from its distance
and the weekend flag. The loop is short:

1. Start with a dumb guess: the average duration for every ride.
2. Work out the **residuals**, which are what we still get wrong
   (`actual - prediction`).
3. Fit a tiny tree (depth 2) to the residuals, not to the durations.
4. Add a fraction of its answer to our prediction. That fraction is the
   **learning rate**.
5. Repeat.

The error falls fast at first, from about 9 minutes to under 4 in ten
rounds, then flattens. Look at the last rows: train RMSE keeps
dropping, test RMSE stops improving. That's the familiar overfitting
gap opening up, and it's why the number of rounds is a dial to choose,
not "as many as possible".

Why "gradient"? For squared error, the residual is exactly the
direction that reduces the loss fastest (its negative gradient). For
classification the library fits trees to the gradient of log loss
instead, but the habit is the same: *find what's still wrong, take a
small step to fix it.*

## 2. HistGradientBoosting on our classification task

```
forest vs boosting:
                              train  CV mean  test
  random forest (lesson 039)  0.850    0.794  0.75
  boosting, defaults          0.995    0.755  0.72
default learning_rate / max_iter               -> (0.1, 100)
default max_leaf_nodes / min_samples_leaf      -> (31, 20)
```

Back to casual vs member. `HistGradientBoostingClassifier` is
scikit-learn's fast boosting model. "Hist" means it first sorts every
numeric column into at most 255 buckets, which makes it quick even on
millions of rows. Two handy extras: it handles missing values on its
own, and it can split on categories directly. We give it categories as
integer codes with `OrdinalEncoder` and point `categorical_features`
at those columns, so there's no one-hot explosion.

Now the honest surprise: with default settings it *loses* to
yesterday's forest. The train score of 0.995 gives the game away. The
defaults (100 rounds of trees with up to 31 leaves each) are built for
big datasets, and on 940 training rides they memorise. Boosting is
powerful, and powerful models need a leash. The next three sections
are the leash.

## 3. Learning rate vs number of trees

```
CV mean, max_leaf_nodes=8:
           20 trees  100 trees  300 trees
  lr=1.0      0.762      0.737      0.737
  lr=0.3      0.773      0.741      0.732
  lr=0.1      0.781      0.763      0.751
  lr=0.03     0.768      0.774      0.769
```

Read across a row to see what more trees do; read down a column to see
what a smaller learning rate does.

- **Big learning rate (1.0, 0.3):** each tree takes a big step. It's
  good early, then overshoots and starts fitting noise, so more trees
  make it *worse*.
- **Small learning rate (0.03):** careful steps. Twenty trees aren't
  enough to get anywhere, but it stays steady even at 300.

The two dials are tied together: halve the learning rate and you need
roughly twice the trees. A common recipe is to pick a smallish rate
(0.05 to 0.1) and let the next trick choose the number of trees.

## 4. Early stopping: let validation pick the number of trees

```
trees allowed / trees actually built           -> (1000, 57)
best validation round                          -> 37
test accuracy                                  -> 0.758
CV mean with early stopping                    -> 0.786
```

Instead of guessing the number of trees, set a generous ceiling
(`max_iter=1000`) and turn on `early_stopping=True`. The model quietly
holds back 20% of the *training* rides (`validation_fraction=0.2`),
checks its loss on them after every tree, and stops once 20 trees in a
row fail to improve it (`n_iter_no_change=20`).

Here it peaked after 37 trees and gave up at 57, nowhere near 1,000.
CV jumps from 0.755 with the defaults to 0.786. The test set played no
part in that choice, which is the whole point: it stays a fair final
exam.

## 5. Tuning the shape of each tree

```
lr=0.05, 200 trees, CV on training rides:
                         CV mean  CV std
  defaults                 0.755   0.019
  leaves=8                 0.764   0.021
  leaves=4                 0.784   0.018
  leaves=8, min_leaf=40    0.778   0.013
  leaves=8, l2=1.0         0.771   0.017
picked by CV                                   -> 'leaves=4'
test accuracy, looked at once                  -> 0.78
'always member' baseline                       -> 0.678
```

The other leash is how big each correction can be:

- **`max_leaf_nodes`**: how many leaves each tree may have. Fewer
  leaves means simpler fixes.
- **`min_samples_leaf`**: refuse to make a leaf from too few rides.
- **`l2_regularization`**: a penalty that shrinks each leaf's
  correction towards zero.

All three help on our small dataset, and tiny 4-leaf trees win on CV.
We pick that one using only the training rides, then look at the test
set once: 0.780, ahead of the forest's 0.758 and right up with the
best models from earlier lessons. Your mileage will vary by dataset; on a few hundred
rides a forest and a tuned booster are often neck and neck. The habit
that transfers is the workflow: small learning rate, early stopping or
a sensible number of trees, small trees, choose by CV, test once.

Trying settings one by one gets tedious fast. Lesson 042 automates the
search with `GridSearchCV` and `RandomizedSearchCV`.

## What you can do now

- Explain boosting as fitting each new tree to the mistakes of the
  ones before it, and build it by hand in a loop.
- Train a `HistGradientBoostingClassifier` with native categorical
  features.
- Trade off `learning_rate` against `max_iter`, and let early stopping
  pick the number of trees.
- Tame overfitting with `max_leaf_nodes`, `min_samples_leaf` and
  `l2_regularization`, chosen by cross-validation.

## What to do now

1. Run `lesson.py`. In section 1, try `lr = 1.0` and watch train and
   test RMSE drift apart.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   combines early stopping with tiny trees.
3. Next, lesson 041: regularisation with Ridge, Lasso and friends. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

A forest is a crowd voting at once. Boosting is a relay: each tree
fixes what the last one missed. Small steps, small trees, and stop when
validation says you're done.
