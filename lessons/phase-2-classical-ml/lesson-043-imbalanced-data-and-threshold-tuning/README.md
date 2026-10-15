# Lesson 043 - Imbalanced data and threshold tuning

**Phase 2 - Classical machine learning** | Week 6, Day 4 | Thursday 2026-10-15

> **Goal:** Handle rare classes with reweighting, resampling and choosing a
> threshold on purpose.

Time: about 45 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Most classifiers quietly assume both classes show up about as often. When
one class is rare, that assumption breaks. A model that always predicts
the common class gets a shiny accuracy and still never finds the rare
one.

Today we make a deliberately rare label: **long rides** of 30 minutes or
more, about one ride in thirteen. We leave `distance_km` out of the
features on purpose — distance would give the game away — so the rare
class is genuinely hard to spot. Then we fix the mess three ways:
reweight the loss, resample the training mix, and move the 0.5 threshold
on purpose. Lesson 034's metrics (precision, recall, F1) are the
scoreboard; accuracy sits on the bench.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` as
lessons 029 to 042, with a stratified train/test split on the new rare
label. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-043-imbalanced-data-and-threshold-tuning/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match. It takes a second or two.

## 1. A rare class, and an accuracy that lies

```
long rides (>= 30 min) in the full data        -> '91/1176 (0.077)'
train / test prevalence                        -> (0.078, 0.076)
always-predict-common on the test set          -> {'accuracy': 0.924, 'precision': 0.0, 'recall': 0.0, 'F1': 0.0, 'found / total long': '0/18'}
default logistic regression at threshold 0.5   -> {'accuracy': 0.924, 'precision': 0.0, 'recall': 0.0, 'F1': 0.0, 'found / total long': '0/18'}
confusion matrix [[TN FP] [FN TP]]             -> array([[218,   0],
     [ 18,   0]])
```

Ninety-one long rides out of 1,176. An always-predict-common rule is
right 92.4% of the time and finds **none** of them. A default logistic
regression, with equal class weights and the usual 0.5 threshold, does
exactly the same: accuracy 0.924, recall 0.0. The confusion matrix is a
giveaway — eighteen long rides, eighteen misses, zero true positives.

That is the trap. When the rare class is the one you care about
(fraud, a broken sensor, a long leisure ride that needs a water
nudge), accuracy is the wrong scoreboard. Precision, recall and F1
from lesson 034 are the ones that tell the truth.

## 2. Reweighting: make the rare class cost more

```
same model, two weight schemes:
                            accuracy precision recall     F1 found / total long
    default (equal weight)     0.924       0.0    0.0    0.0               0/18
    class_weight="balanced"    0.725     0.195  0.833  0.316              15/18
weights it used (common / rare)                -> (0.542, 6.438)
```

`class_weight="balanced"` multiplies each class's loss by the inverse of
how often it appears. On our training rides that makes a miss on a long
ride hurt about **12×** as much as a miss on a short one. The decision
boundary moves. Accuracy drops (we now raise more false alarms), but
recall jumps from 0 to 0.833 — fifteen of the eighteen long rides are
found. F1 goes from 0 to 0.316.

You did not invent or throw away any rows. You only changed what a
mistake *costs*.

## 3. Resampling: change the training mix

```
training rows after oversample (common / rare) -> (867, 867)
training rows after undersample (common / rare) -> (73, 73)
four ways to train, same test set:
                            accuracy precision recall     F1 found / total long
    default on original        0.924       0.0    0.0    0.0               0/18
    oversample then fit        0.725     0.195  0.833  0.316              15/18
    undersample then fit       0.725     0.195  0.833  0.316              15/18
    class_weight="balanced"    0.725     0.195  0.833  0.316              15/18
```

**Oversampling** duplicates rare rows (with replacement) until the
classes match. **Undersampling** throws away common rows until they
match. Both change the *training mix*; the test set stays untouched —
otherwise you would be cheating.

On this data the three honest fixes land on the same numbers. That is
not a law: with a bigger rare class or a non-linear model they can
diverge. Prefer reweighting first — it needs no extra rows and no
discarded ones — and reach for resampling when weights are not enough
or your model has no `class_weight` knob. (Libraries like `imbalanced-learn`
add fancier samplers; everything today is plain scikit-learn and NumPy.)

## 4. Threshold tuning: 0.5 is not sacred

```
threshold sweep on the tuning slice:
             precision  recall     F1
    thr=0.3      0.170   0.944  0.288
    thr=0.4      0.215   0.944  0.351
    thr=0.5      0.215   0.944  0.351
    thr=0.6      0.200   0.722  0.313
    thr=0.7      0.156   0.389  0.222
threshold that maximises F1 on the tuning slice -> 0.552
precision / recall / F1 at that threshold (tune) -> (0.227, 0.944, 0.366)
same threshold held out on the test set        -> {'accuracy': 0.72, 'precision': 0.176, 'recall': 0.722, 'F1': 0.283, 'found / total long': '13/18'}
confusion matrix on the test set               -> array([[157,  61],
     [  5,  13]])
```

`predict_proba` gives a score between 0 and 1, not a decision. The
threshold that turns the score into "long" or "not" is yours. Lower it
to catch more of the rare class (higher recall, more false alarms);
raise it when false alarms are costly (higher precision, more misses).

We fit on three-quarters of the training rides, sweep thresholds on the
held-out quarter, and pick the one that maximises F1 (0.552 here). Then
we lock that threshold and score **once** on the test set. That is the
same honesty rule as lesson 034: never tune on the test set.

On the test rides, recall of 0.722 at a precision of 0.176 means we find
thirteen of eighteen long rides and raise sixty-one false alarms. Whether
that trade is acceptable depends on the cost of a miss versus a false
alarm — exactly the cost story from lesson 034.

## What you can do now

- Spot when accuracy is lying because one class is rare.
- Use `class_weight="balanced"` (or a manual weight dict) so the rare
  class costs more.
- Oversample or undersample the training mix when weights are not enough,
  without touching the test set.
- Sweep thresholds on `predict_proba`, pick one on a tuning slice, and
  report precision/recall/F1 — not accuracy alone.

## What to do now

1. Run `lesson.py`. In section 1, add `distance_km` back into `NUMERIC`
   and watch how much easier the rare class becomes — then take it out
   again so the rest of the lesson still makes sense.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   picks a threshold for a different cost ratio.
3. Later today, lesson 044: unsupervised learning with k-means, where
   there is no label at all. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

When one class is rare, accuracy will flatter you. Reweight or resample
so the rare class matters, then move the threshold to match what a miss
actually costs.
