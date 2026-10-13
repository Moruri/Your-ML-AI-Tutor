# Lesson 039 - Random forests and bagging

**Phase 2 - Classical machine learning** | Week 6, Day 2 | Tuesday 2026-10-13

> **Goal:** Combine many trees to cut variance, and read feature
> importances with healthy suspicion.

Time: about 45 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Yesterday's tree was a checklist you could read aloud. The catch is
that it's a nervous checklist: give it a slightly different set of
rides and it may ask different questions and give different answers.
Today we measure that nervousness, then fix it the way people fix a
shaky opinion: ask lots of independent voices and take the vote.

That idea has two ingredients. **Bagging** trains each tree on its own
random resample of the rows. A **random forest** adds a second shake by
letting each split look at only a few random columns. Together they
turn a jumpy model into one of the most dependable defaults in all of
machine learning.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` and the
same stratified train/test split as lessons 033 to 038. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-039-random-forests-and-bagging/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match. It takes about 20 seconds, because it trains a lot
of trees.

## 1. One tree is jumpy

```
test accuracy of 5 unlimited trees             -> (0.691, 0.716, 0.716, 0.669, 0.712)
spread (max - min)                             -> 0.047
first splits                                   -> ('num__duration_min <= 16.25', 'num__duration_min <= 16.30', 'num__duration_min <= 15.95', 'num__distance_km <= 3.78')
test rides where the 5 trees split votes       -> 0.479
```

We train five unlimited trees, each on a random 80% of the training
rides, and test them on the same held-out rides. Their scores wander
by almost five points, one of them even picks a different first
question, and on nearly half the test rides at least one tree
disagrees with the others.

That sensitivity to the exact training rows is called **variance**.
Deep trees have a lot of it: they're low on bias (they can fit almost
anything) and high on variance (what they fit depends on luck). The
good news is that variance is exactly what averaging is good at
removing.

## 2. Bagging: bootstrap, train, vote

```
bootstrap sample size                          -> 940
share of rides that made it in at least once   -> 0.638
single unlimited tree, test accuracy           -> 0.699
25 bagged trees by hand, test accuracy         -> 0.754
CV mean: one tree / BaggingClassifier(100)     -> (0.695, 0.76)
```

You met the **bootstrap** in lesson 026: draw `n` rows *with
replacement* from `n` rows. Some rides get picked twice, some not at
all. On average about 63% of the rides make it in.

**Bagging** (short for *bootstrap aggregating*) does three things:

1. Draw a bootstrap sample.
2. Train a full, unpruned tree on it.
3. Repeat many times, then let the trees vote.

Our hand-rolled version with 25 trees jumps from 0.699 to 0.754 on the
test rides. scikit-learn's `BaggingClassifier` does the same with 100
trees, and cross-validation agrees: 0.695 for one tree, 0.760 for the
bag. Each tree still overfits its own sample, but they overfit in
*different* ways, and the vote cancels much of that out.

## 3. Random forests: bagging plus random columns

```
max_features at each split:
                          CV mean   test
  all features (bagging)    0.761  0.729
  sqrt (forest default)     0.779  0.750
  a third                   0.782  0.737
more trees, min_samples_leaf=5:
             CV mean  CV std
  1 trees      0.737   0.038
  10 trees     0.790   0.038
  50 trees     0.788   0.025
  200 trees    0.793   0.027
  500 trees    0.794   0.025
```

Bagged trees still look alike. Duration is so useful that almost every
tree asks about it first, so their mistakes are correlated, and
correlated mistakes don't cancel. A **random forest** fixes that: at
every split, each tree may only choose from a random subset of the
columns (`max_features`). Sometimes duration isn't on the menu, so the
tree has to find another good question.

Trees that differ more make a better crowd. CV goes from 0.761 with
all features to about 0.78 with the default `"sqrt"` (the square root
of the number of columns).

The second table answers the usual question, "how many trees?". One
tree is a weak forest. By ten the score is close to its ceiling, and
after that more trees mostly make it *steadier* (smaller CV std) and
slower. Extra trees don't cause overfitting; they just stop helping.
We also set `min_samples_leaf=5` here, which nudges CV up a little by
stopping each tree from carving leaves out of single rides.

## 4. The out-of-bag score: a free validation set

```
out-of-bag accuracy                            -> 0.803
5-fold CV mean                                 -> 0.795
test accuracy                                  -> 0.758
'always member' baseline                       -> 0.678
```

Remember that each bootstrap leaves out about a third of the rides?
For any one ride, there's a group of trees that never saw it. Ask only
those trees to vote on it, do that for every ride, and you get the
**out-of-bag (OOB) score**: an honest estimate from a single fit, with
no extra split. Turn it on with `oob_score=True`.

Here OOB says 0.803 and five-fold CV says 0.795. Close enough to use
OOB as a quick check while you experiment, and it's five times cheaper.
The test score is a little lower (0.758), which is the normal wobble of
a 236-ride test set. Either way, the forest beats the "always member"
baseline by a comfortable margin, and beats every single tree from
lesson 038's section 3 on CV.

## 5. Feature importances, with suspicion

```
impurity importance (top 6):
  num__duration_min    0.288
  num__distance_km     0.276
  num__noise           0.146
  num__temp_c          0.093
  flags__weekend       0.044
  num__rain_mm         0.042
permutation importance on test:
  duration_min     0.062
  weekend          0.012
  rain_mm          0.004
  noise            0.003
  temp_c           0.002
  distance_km      0.001
  start_station   -0.003
  bike_type       -0.006
noise rank: impurity / permutation             -> (3, 4)
```

Forests are less readable than one tree, so people lean on
`feature_importances_`. To test it, we sneak in a column called
`noise` that is literally random numbers. Watch what happens.

**Impurity importance** (the built-in one) ranks noise *third*, above
temperature and the weekend flag. That's not a bug in your code. It's
measured on the training rides, and unpruned trees happily split on a
column with hundreds of distinct values to purify tiny leaves, so
noise collects credit.

**Permutation importance** asks a fairer question: if I shuffle this
one column on the *held-out* rides, how much accuracy do I lose?
Shuffling noise costs almost nothing (0.003). Shuffling duration costs
six points.

Notice distance, though. It looks nearly useless by permutation, even
though it clearly matters. That's because duration and distance carry
much the same information: shuffle one and the forest leans on the
other. **Correlated columns share credit**, under any importance
method. So treat importances as a conversation starter, never as proof
that a column causes anything.

## What you can do now

- Explain variance, and why a single deep tree has a lot of it.
- Describe bagging (bootstrap, train, vote) and build it by hand.
- Train a `RandomForestClassifier` and explain what `max_features`,
  `n_estimators` and `min_samples_leaf` do.
- Use `oob_score=True` as a cheap honest estimate.
- Prefer permutation importance on held-out data, and remember that
  correlated columns split the credit.

## What to do now

1. Run `lesson.py`. In section 3, try `max_features=1` and see what a
   *too* random forest does.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   removes duration and watches the importances reshuffle.
3. Next, lesson 040: gradient boosting. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

One tree has opinions. A forest has a vote. Shake up the rows, shake
up the columns, and let the crowd smooth out the noise.
