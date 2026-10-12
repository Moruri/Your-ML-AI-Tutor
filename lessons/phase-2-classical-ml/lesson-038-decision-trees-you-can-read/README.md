# Lesson 038 - Decision trees you can read

**Phase 2 - Classical machine learning** | Week 6, Day 1 | Monday 2026-10-12

> **Goal:** Train, visualise and prune a decision tree; understand
> impurity and depth.

Time: about 45 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

k-NN votes by distance. A decision tree asks a sequence of yes/no
questions about one column at a time — "is duration over 16 minutes?",
"is it the weekend?" — and you can literally read the questions. That
makes trees a great model to learn after neighbours: same rides, same
target, but an explanation you can say out loud.

Trees also ignore units. They split one column at a time, so scaling
doesn't matter (lesson 035's table said so). Categories still need
encoding, depth still needs a leash, and an unlimited tree will
memorise the training set. Today we train, read, and prune.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` and the
same stratified train/test split as lessons 033 to 037. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-038-decision-trees-you-can-read/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match. We print the tree with `export_text`, which works
fine headless — no matplotlib window required.

## 1. A tree of yes/no questions

```
test accuracy, depth 3                         -> 0.767
'always member' baseline                       -> 0.678
number of leaves                               -> 8
depth                                          -> 3
readable tree (depth 3):
    |--- num__duration_min <= 16.25
    |   |--- num__duration_min <= 9.95
    |   |   |--- flags__weekend <= 0.50
    |   |   |   |--- class: 0
    ...
    |--- num__duration_min >  16.25
    |   |--- num__distance_km <= 5.35
    |   |   |--- flags__weekend <= 0.50
    |   |   |   |--- class: 0
    |   |   |--- flags__weekend >  0.50
    |   |   |   |--- class: 1
    ...
```

We still predict casual (1) vs member (0). The pipeline one-hots the
categories and passes numbers and the weekend flag through untouched —
no `StandardScaler`, because trees don't need one. `max_depth=3` keeps
the checklist short enough to read.

Follow any path and you have a story: a short weekday hop is called a
member; a longer weekend ride is called casual. That's the whole model.

## 2. Impurity: why this split?

```
share casual in training rides                 -> 0.324
Gini impurity at the root                      -> 0.438
entropy at the root                            -> 0.909
first split feature                            -> num__duration_min
first split threshold                          -> 16.25
Gini if we split on weekend == 0               -> 0.407
impurity drop (information gained)             -> 0.031
share casual on weekdays / weekends            -> (0.261, 0.57)
```

A node is **pure** when every ride in it has the same label. **Gini
impurity** is 0 for a pure node and highest for a 50/50 mix:

```
Gini = 1 - (p² + (1 - p)²)
```

With 32.4% casual at the root, Gini is about 0.44. The tree tries every
feature and every threshold, and picks the split that drops the
*weighted* impurity of the two child nodes the most. Duration at 16.25
minutes wins that contest here. Weekend alone would drop impurity by
about 0.03 — useful, but duration does better as the first question.

**Entropy** tells the same story on a log scale (`-p log₂ p - …`).
scikit-learn's default is Gini; set `criterion="entropy"` if you prefer.
For reading trees, the difference rarely matters.

## 3. Depth, leaves and overfitting

```
deeper trees remember more:
             leaves  train  CV mean   test
    depth=1       2  0.768    0.760  0.746
    depth=2       4  0.768    0.755  0.746
    depth=3       8  0.796    0.776  0.767
    depth=5      27  0.827    0.761  0.742
    depth=8      88  0.894    0.726  0.729
    unlimited   199  1.000    0.695  0.699
```

A stump (depth 1) underfits — two leaves, not much to say. An unlimited
tree grows until every leaf is pure, nails the training rides (1.000)
and then fails on new ones. Cross-validation peaks around depth 3 on
this data. The gap between train score and CV score *is* overfitting,
growing with depth.

## 4. Pruning without the drama

```
prune with max_depth and min_samples_leaf:
                        leaves  CV mean   test
    depth=3                  8    0.776  0.767
    depth=5                 27    0.761  0.742
    min_leaf=20             31    0.756  0.763
    min_leaf=50             16    0.781  0.775
    depth=5, min_leaf=20    21    0.771  0.763
chosen: depth=3, min_leaf=20, test accuracy    -> 0.767
```

Two dials do most of the everyday pruning:

- **`max_depth`** — how many questions you're allowed to ask.
- **`min_samples_leaf`** — refuse to carve a leaf from a handful of
  rides.

Either one (or both) stops the tree from memorising quirks. On these
rides, `min_samples_leaf=50` even edges the CV mean up a touch. We'll
stick with a depth-3 tree for readability in the next section.
Cost-complexity pruning (`ccp_alpha`) is another dial; forests and
boosting will want it more than we do today.

## 5. Reading the pruned tree

```
top feature importances:
    num__duration_min    0.760
    num__distance_km     0.142
    flags__weekend       0.099
Saturday long ride / Tuesday hop -> class      -> (1, 0)
chance each is casual                          -> (1.0, 0.04)
```

`feature_importances_` credits each column with how much impurity it
dropped across the whole tree. Duration does most of the work here;
weekend and distance chip in. Treat importances as a hint, not a proof —
correlated columns share credit awkwardly, and a column that helps only
deep in a rare branch can look small.

The two new rides behave as the checklist suggests: a long Saturday
ride is casual with high confidence; a short Tuesday hop is a member.
Next lesson grows many trees and averages them, which usually beats one
tree and softens the overfit.

## What you can do now

- Train a `DecisionTreeClassifier` in a pipeline and read it with
  `export_text`.
- Explain Gini impurity (and entropy) as "how mixed is this node?".
- Show how depth drives overfitting with train vs CV scores.
- Prune with `max_depth` and `min_samples_leaf`, and skim feature
  importances with healthy suspicion.

## What to do now

1. Run `lesson.py`. In section 3, try `max_depth=4` and compare CV.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   builds a stump and a pruned tree side by side.
3. Next, lesson 039: random forests and bagging. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

A tree is a checklist you can read. Cap its depth, insist on decent
leaf sizes, and you'll trust the checklist on new rides.
