# Lesson 034 - Classification metrics that matter

**Phase 2 - Classical machine learning** | Week 5, Day 4 | Thursday 2026-10-08

> **Goal:** Choose between accuracy, precision, recall, F1 and ROC-AUC
> based on what a mistake costs.

Time: about 50 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Lesson 033 built a model that guesses whether a ride was taken by a
casual rider or a member, and it's 79% accurate. Is that good?

You can't answer that without asking what the model is *for*. Ours
decides who sees a "become a member" offer at the end of a ride. It can
go wrong in two ways:

- It shows the offer to someone who's already a member. Mildly annoying.
- It misses a casual rider, who never sees the offer. A lost sign-up.

Those two mistakes don't cost the same, and accuracy counts them the
same. Today is about the numbers that tell them apart, and how to pick
the one that matches what you care about.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv`, same four
features and the same stratified train/test split as lesson 033. Full
script:

```bash
python lessons/phase-2-classical-ml/lesson-034-classification-metrics-that-matter/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match.

## 1. Accuracy hides the mistakes

```
model accuracy on test rides                   -> 0.792
'always member' accuracy                       -> 0.678
casual riders in the test set                  -> 76
of those, how many the model caught            -> 41
```

79% beats the do-nothing baseline. But of the 76 casual riders in the
test set, the very people the offer is for, the model found 41. Almost
half slipped through, and the accuracy number gave no hint of it.

That's the general problem. When one class is more common, a model can
score well by leaning towards it. The rarer the thing you're hunting
(fraud, disease, a faulty part), the worse it gets: with 1% fraud,
"never fraud" is 99% accurate and completely useless.

## 2. The confusion matrix

```
rows are the truth, columns are the guess:
                   said member  said casual
    actual member          146           14
    actual casual           35           41
TN, FP, FN, TP                                 -> (146, 14, 35, 41)
```

The **confusion matrix** counts every combination of truth and guess.
With "casual" as the positive class, the four cells have names:

| | said member | said casual |
|---|---|---|
| **actual member** | true negative (TN) | false positive (FP) |
| **actual casual** | false negative (FN) | true positive (TP) |

"True" or "false" says whether the guess was right; "positive" or
"negative" says what the guess was. So a **false positive** is a member
we wrongly flagged (a false alarm), and a **false negative** is a casual
rider we missed.

```python
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(test["casual"], pred).ravel()
```

`.ravel()` flattens the 2 × 2 grid in that order, TN, FP, FN, TP. Every
metric below is built from these four numbers.

## 3. Precision, recall and F1

```
precision = TP / (TP + FP), by hand            -> 0.745
recall    = TP / (TP + FN), by hand            -> 0.539
F1, the harmonic mean, by hand                 -> 0.626
same as scikit-learn?                          -> True
```

Two questions, two metrics:

- **Precision**, TP / (TP + FP): when the model says casual, how often
  is it right? 41 out of 55, so 74.5%. High precision means few false
  alarms.
- **Recall**, TP / (TP + FN): of all the casual riders, how many did it
  find? 41 out of 76, so 53.9%. High recall means few misses. (You'll
  also see it called sensitivity or the true positive rate.)

They pull against each other. Flag everyone and recall is perfect but
precision collapses. Flag only the surest cases and precision is high
but recall drops.

**F1** combines them into one number, `2 * P * R / (P + R)`. It's a
**harmonic mean**, which stays low if either one is low: a model with
precision 1.0 and recall 0.1 gets F1 0.18, not the 0.55 a plain average
would give. Use F1 when you want both to be decent and don't have a
clear cost for each mistake.

scikit-learn has all three: `precision_score`, `recall_score` and
`f1_score`, each taking `(y_true, y_pred)`.

## 4. Moving the threshold, honestly

```
training folds, a miss costs 4, a false alarm 1:
               precision  recall  cost
    threshold
    0.10            0.40    0.97   485
    0.15            0.46    0.90   440
    0.20            0.52    0.85   420
    0.25            0.57    0.77   451
    0.30            0.62    0.71   482
    0.40            0.69    0.63   542
    0.50            0.77    0.54   610
threshold with the lowest cost                 -> 0.2
rule of thumb: 1 / (1 + 4)                     -> 0.2
test at 0.5: precision, recall, cost           -> (0.75, 0.54, 154)
test at 0.2: precision, recall, cost           -> (0.54, 0.87, 96)
```

`predict` cuts the probability at 0.5, but nothing says it has to. You
can pick your own **threshold**:

```python
p = model.predict_proba(test[FEATURES])[:, 1]
pred = (p >= 0.2).astype(int)
```

Lower it and the model flags more rides as casual: recall goes up,
precision goes down. Which threshold is right depends on the costs, so
put numbers on them. Say a missed casual rider costs 4 (a lost sign-up)
and a false alarm costs 1 (one annoyed member). Then the total cost is
`1 * FP + 4 * FN`, and you pick the threshold that makes it smallest.

The honest way to choose, following lesson 032, keeps the test set out
of it:

1. `cross_val_predict(..., method="predict_proba")` on the training
   rides. It's cross-validation that hands back predictions instead of
   scores: every training ride gets a probability from a model that
   never saw it.
2. Try thresholds on those probabilities, and pick the cheapest: 0.2.
3. Check it once on the test set.

On the test rides, moving from 0.5 to 0.2 catches 87% of casual riders
instead of 54%, and the total cost drops from 154 to 96. Precision falls
to 0.54, so nearly half the offers go to members, and that's fine:
we decided a false alarm was cheap.

There's a neat shortcut. If the probabilities are trustworthy, the
cheapest threshold is `cost of false alarm / (cost of false alarm + cost
of miss)`, here 1 / (1 + 4) = 0.2, which is exactly what the search
found. With equal costs it's 0.5, which is why that's the default.

## 5. ROC curve and AUC

```
a few points on the ROC curve:
               false pos rate  true pos rate
    threshold
    0.71                 0.04           0.32
    0.50                 0.09           0.54
    0.30                 0.22           0.75
    0.20                 0.37           0.87
    0.11                 0.62           0.97
ROC-AUC on the test rides                      -> 0.838
share of (casual, member) pairs ranked right   -> 0.838
```

Every threshold gives a different trade-off, and the **ROC curve** shows
them all at once. For each threshold it plots two rates:

- **True positive rate**, which is just recall: the share of casual riders
  caught.
- **False positive rate**, FP / (FP + TN): the share of members wrongly
  flagged.

Sliding the threshold down from 1 to 0 walks the curve from (0, 0), flag
nobody, to (1, 1), flag everyone. A good model climbs steeply: lots of
casual riders caught before many members get flagged. `roc_curve` gives
you the points; plot `tpr` against `fpr` with matplotlib (lesson 022) to
see the shape.

**ROC-AUC** is the area under that curve, one number between 0 and 1.
It has a lovely plain meaning, and the script checks it: pick a random
casual rider and a random member, and AUC is the chance the model gives
the casual rider the higher probability. Ours does that 83.8% of the
time. 0.5 is a coin toss; 1.0 is perfect.

AUC measures how well the model **ranks** rides, before any threshold is
chosen. That makes it good for comparing models, and useless for telling
you which threshold to use.

## 6. The one-line report

```
              precision    recall  f1-score   support

      member       0.81      0.91      0.86       160
      casual       0.75      0.54      0.63        76

    accuracy                           0.79       236
   macro avg       0.78      0.73      0.74       236
weighted avg       0.79      0.79      0.78       236
```

`classification_report` prints precision, recall and F1 for *each*
class, plus **support** (how many test rides were in it). The casual row
is the one we've been computing. The member row treats "member" as the
positive class instead. **Macro avg** averages the two rows equally;
**weighted avg** weights them by support. Read the row for the class you
care about first.

## Which metric, when

| If... | Look at |
|---|---|
| Classes are roughly balanced and mistakes cost about the same | accuracy |
| A false alarm is expensive (flagging an honest customer as fraud) | precision |
| A miss is expensive (a missed diagnosis, a missed sign-up) | recall |
| You want both decent and have no costs to go on | F1 |
| You're comparing models before picking a threshold | ROC-AUC |
| You can put a number on each mistake | total cost, and choose the threshold with it |

And always: compare against a baseline, and choose thresholds on the
training folds, never on the test set.

## What you can do now

- Explain why accuracy can look good while the model fails at its job.
- Read a confusion matrix and name TN, FP, FN and TP.
- Compute precision, recall and F1, by hand and with scikit-learn.
- Move the threshold and say what it trades.
- Choose a threshold from costs with `cross_val_predict`, without
  touching the test set.
- Read a ROC curve and explain ROC-AUC in plain words.

## What to do now

1. Run `lesson.py`. Change `COST_MISS` to 2 and see where the cheapest
   threshold moves.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   flips the costs around.
3. Next, lesson 035: feature scaling and encoding, and why some models
   care. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

A metric is a statement about what a mistake costs. Pick it before you
build the model, and you'll know what "good" means when you get there.
