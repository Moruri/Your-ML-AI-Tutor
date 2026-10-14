# Lesson 042 - Hyperparameter search

**Phase 2 - Classical machine learning** | Week 6, Day 3 | Wednesday 2026-10-14

> **Goal:** Use `GridSearchCV` and `RandomizedSearchCV` without
> accidentally tuning on the test set.

Time: about 45 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Every model so far has come with dials: `k` in k-NN, depth in trees,
the learning rate in boosting, `alpha` and `C` in this morning's
regularised models. These are **hyperparameters**: settings *you*
choose before training, as opposed to **parameters** like coefficients
and split points, which the model learns from the data.

Turning them one at a time by hand gets old fast. Worse, it tempts you
into the most common quiet mistake in machine learning: trying each
setting against the test set and keeping the best one. Today we see
exactly why that lies to you, then hand the search to scikit-learn and
do it properly.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` and the
same stratified train/test split as lessons 033 to 041. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-042-hyperparameter-search/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match. It takes about a minute, because a search means
training a lot of models.

## 1. Why you never tune on the test set

```
labels shuffled, so the true skill is          -> 0.5
settings tried against the test set            -> 40
best k, picked by test ROC-AUC                 -> 79
its test ROC-AUC                               -> 0.548
the same k, CV on training rides               -> 0.469
average test ROC-AUC over all 40               -> 0.512
worst test ROC-AUC of the 40                   -> 0.474
```

A rigged experiment, so we know the truth. We shuffle the
casual/member labels at random. Now there is *nothing* to learn: no
model and no setting can genuinely do better than a ROC-AUC of 0.5,
which is coin-flipping (lesson 034).

Then we make the mistake on purpose. We try 40 values of `k` for our
k-NN pipeline, score each one on the test rides, and keep the best. The
winner, `k=79`, scores 0.548, which *looks* like a small real signal.
It isn't. The same `k` scores 0.469 under cross-validation on the
training rides, and the 40 test scores average 0.512, scattered around
0.5 just as luck would have it.

That's the trap. Each extra setting you check against the test set is
another lottery ticket. Pick the luckiest ticket and the test score
stops being an honest estimate, because the test set helped make the
choice. On real data the effect is the same, just harder to spot,
because the luck sits on top of genuine skill.

The fix is a rule, not a trick: **choose settings with
cross-validation on the training rides, and look at the test set once,
at the end.** Everything below is that rule, automated.

## 2. GridSearchCV: every combination, cross-validated

```
combinations x folds = models trained          -> (8, 5, 40)
top five by CV mean:
       n_neighbors   weights   mean    std  rank
    0           31  distance  0.787  0.024     1
    1           31   uniform  0.787  0.028     1
    2           15   uniform  0.781  0.043     3
    3           15  distance  0.779  0.037     4
    4           51  distance  0.778  0.018     5
best_params_                                   -> {'model__n_neighbors': 31, 'model__weights': 'uniform'}
best_score_ (CV mean of the winner)            -> 0.787
refit on all training rides, test once         -> 0.733
```

`GridSearchCV` takes a pipeline and a dictionary of values to try. It
cross-validates *every* combination (here 4 values of `k` times 2
weighting schemes, times 5 folds: 40 fits) and ranks them by CV mean.

Three things worth knowing:

- **The names.** `model__n_neighbors` means "the `n_neighbors`
  argument of the step called `model`". Step name, two underscores,
  argument. It's the same naming `set_params` used in lesson 036, and
  it works for preprocessing steps too.
- **The results.** `cv_results_` holds every score; turn it into a
  DataFrame and read it. Here two settings tie at 0.787, and the top
  five are all within one CV std of each other. That's normal, and a
  hint not to fuss over the third decimal.
- **The refit.** With `refit=True` (the default), the winner is
  retrained on all 940 training rides. The search object then behaves
  like a fitted model: `search.predict(...)`, `search.score(...)`.

The single look at the test set gives 0.733, a fair bit below the CV
mean of 0.787. Some of that is the test set's own noise on 236 rides;
some is that a winner's CV score is a little flattering. Section 5
measures that second part.

## 3. RandomizedSearchCV: when the grid gets too big

```
full grid for boosting: combinations           -> 1024
models to train with 5 folds                   -> 5120
random draws tried / models trained            -> (25, 125)
top five by CV mean:
       learning_rate  max_iter  max_leaf_nodes  min_samples_leaf  l2_regularization   mean    std  rank
    0          0.011       293               9                52              0.002  0.787  0.019     1
    1          0.015       338               5                14              1.325  0.783  0.014     2
    2          0.012       341              10                16              0.617  0.780  0.023     3
    3          0.025       243              10                44              0.002  0.778  0.015     4
    4          0.078       251               4                15              8.073  0.778  0.014     4
best_score_ (CV mean of the winner)            -> 0.787
test accuracy, looked at once                  -> 0.771
lesson 040's hand-tuned CV mean, for reference -> 0.784
```

Grids multiply. Lesson 040's booster has five dials worth tuning; four
values each makes 1,024 combinations and 5,120 fits. That's a long
coffee for a small dataset and impossible for a big one.

`RandomizedSearchCV` samples a fixed number of combinations instead.
You describe each dial as a *range* (a distribution) rather than a list:

- `loguniform(0.01, 0.3)` for the learning rate and the L2 penalty.
  These dials matter in ratios (0.01 to 0.03 is as big a change as 0.1
  to 0.3), so we sample evenly on a log scale.
- `randint(50, 401)` for counts like the number of trees or leaves.

Twenty-five draws, 125 fits, and the best one scores 0.787 on CV, level
with lesson 040's careful hand search (0.784). Its single test score is
0.771. Notice the pattern in the top rows: small learning rates with a
few hundred trees, which is exactly what lesson 040 taught us by hand.

Why does random beat a grid of the same size? Usually only one or two
dials really matter. A grid wastes most of its fits repeating the same
few values of the important dial; random draws try a new value of it
every time.

## 4. Searching over whole models

```
one search, three kinds of model:
                                                 CV mean  CV std
    candidate
    RandomForestClassifier, min_samples_leaf=3     0.797   0.030
    RandomForestClassifier, min_samples_leaf=10    0.794   0.024
    LogisticRegression, C=10                       0.793   0.015
    LogisticRegression, C=1                        0.790   0.017
    LogisticRegression, C=0.1                      0.787   0.021
    KNeighborsClassifier, n_neighbors=31           0.787   0.028
    KNeighborsClassifier, n_neighbors=15           0.781   0.043
winner                                         -> 'RandomForestClassifier, min_samples_leaf=3'
test accuracy, looked at once                  -> 0.746
```

The grid can be a *list* of grids, and the `model` step itself can be
one of the things you search over. Each grid swaps in a different
model with its own dials, and they all share the same preprocessing
and the same five folds, so the contest is fair.

The forest comes top on CV, but look at the spread: seven candidates
from three model families, all within about one CV std of each other.
On this data they're about equally good. The search picks the highest
number mechanically. You're allowed to overrule it in favour of the
simpler, faster, easier-to-explain model, *as long as you decide using
CV*, not the test score.

That test score is a good lesson in itself: 0.746 for the forest, while
logistic regression scored 0.797 on the same test rides in lesson 036.
It's tempting to switch now. Don't. That's section 1's trap in a
different coat. With 236 test rides, a few rides one way or the other
moves the score by a couple of points.

## 5. Nested cross-validation: an honest score for the whole search

```
best_score_ from section 2's search            -> 0.787
nested CV: mean over 5 outer folds             -> 0.781
outer fold scores                              -> (0.787, 0.771, 0.803, 0.777, 0.766)
setting each outer fold picked                 -> ((31, 'distance'), (31, 'uniform'), (31, 'distance'), (31, 'distance'), (31, 'distance'))
```

`best_score_` has a small honesty problem: it's the score of the winner
on the same folds that *chose* the winner. A smaller version of
section 1's lottery.

**Nested cross-validation** fixes it with two loops:

- The **outer loop** splits the training rides into 5 folds.
- For each one, the **inner loop** runs the entire grid search on the
  other four folds only, then scores its winner on the held-out fold,
  which the search never saw.

The nested estimate, 0.781, sits a little below `best_score_` (0.787).
That gap is the flattery. It's small here because we only tried 8
settings; search over hundreds and it grows. It's also reassuring that
every outer fold picked `k=31`: the recipe is stable.

Be clear on what nested CV answers. It scores the **tuning recipe**
("k-NN, with `k` and `weights` chosen by a 5-fold grid search"), not
one particular `k`. You still use the ordinary search from section 2 to
pick the final settings.

## What you can do now

- Explain why picking settings by test score gives an optimistic
  result, and avoid it.
- Run `GridSearchCV` on a pipeline using `step__argument` names, read
  `cv_results_`, and use the refitted best model.
- Use `RandomizedSearchCV` with `loguniform` and `randint` when a grid
  is too big.
- Search across different models in one fair contest, and judge the
  whole search with nested cross-validation.

## What to do now

1. Run `lesson.py`. In section 1, change `range(1, 81, 2)` to try more
   settings and see whether the luckiest test score climbs.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   runs a random search for k-NN and changes the scoring metric.
3. Next, lesson 043: imbalanced data and threshold tuning. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Search as much as you like on the training rides. Judge with
cross-validation. Open the test set once, at the very end, and live
with what it says.
