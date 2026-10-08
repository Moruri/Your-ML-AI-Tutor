# Lesson 034 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-034-classification-metrics-that-matter/lesson.py my_lesson_034.py
python my_lesson_034.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-034-classification-metrics-that-matter").resolve()
```

`load_rides`, `split`, `FEATURES` and `TARGET` are there to reuse.

---

## 1. Spam or cancer?

For each, which matters more, precision or recall?

**a)** A spam filter that moves mail to a junk folder.

**b)** A first-round screening test that sends people for a proper
check-up.

<details>
<summary>Check yourself</summary>

**a) Precision.** A false positive is a real email hidden in junk, maybe
a job offer. A bit of spam in the inbox (a miss) is only annoying.

**b) Recall.** A miss sends a sick person home. A false alarm costs one
extra appointment, and the proper check-up catches it. That's why
screening tests are tuned to flag generously.

</details>

## 2. Four numbers by hand

A model's confusion matrix on 120 cases: TN = 90, FP = 10, FN = 5,
TP = 15. Work out accuracy, precision, recall and F1.

<details>
<summary>Check yourself</summary>

- accuracy = (90 + 15) / 120 = 0.875
- precision = 15 / (15 + 10) = 0.60
- recall = 15 / (15 + 5) = 0.75
- F1 = 2 × 0.60 × 0.75 / (0.60 + 0.75) ≈ 0.667

87.5% accurate, yet 4 in 10 of its alarms are wrong. Same story as the
lesson.

</details>

## 3. Why didn't AUC move?

In section 4 we moved the threshold from 0.5 to 0.2 and precision,
recall and cost all changed. Would ROC-AUC change too?

<details>
<summary>Check yourself</summary>

No. AUC is computed from the probabilities, across every possible
threshold, so picking one threshold doesn't touch it. It only changes if
the model's ranking of the rides changes, which needs a different model
or different features. That's the point of it: AUC says how good the
ranking is, and the threshold decides where to cut it.

</details>

## 4. Hands-on: new costs, new threshold

The business changes its mind twice.

**a)** "Both mistakes cost the same." Set `COST_FALSE_ALARM = 1` and
`COST_MISS = 1`. Which threshold is cheapest on the training folds?

**b)** "Members are furious about the pop-ups. A false alarm costs 3, a
miss costs 1." What does the rule of thumb predict, and what does the
search find? Search a wider grid, from 0.1 to 0.9 in steps of 0.05.

**c)** For (b), compare the test cost at your chosen threshold with the
test cost at 0.5, and give the test precision and recall.

Hints:

- `np.arange(0.1, 0.95, 0.05)` gives the wider grid. Round each value
  with `round(t, 2)` before printing.
- Reuse the body of `section_threshold`: the out-of-fold probabilities
  don't depend on the costs, only the cost column does.

<details>
<summary>Expected results</summary>

```
a) cheapest threshold 0.5 (training cost 190), as the rule of thumb
   1 / (1 + 1) = 0.5 says. On the test rides: cost 49 at 0.5.
b) rule of thumb: 3 / (3 + 1) = 0.75
   search: cheapest at 0.85 (cost 242), with 0.70 to 0.85 all
   close together (258, 250, 249, 242)
c) test cost 70 at 0.85 against 77 at 0.5,
   precision about 0.86, recall about 0.16
```

When false alarms are the expensive mistake, the threshold climbs and
the model only flags rides it's very sure about. Precision goes up and
recall collapses: only about 1 in 6 casual riders now sees the offer.
That's the right call under those costs, even if it feels strange.

The search landed at 0.85 rather than exactly 0.75 because the cost
curve is nearly flat up there, and on 940 rides a few rides either way
moves the minimum. When the cheapest spot sits on a flat stretch, any
threshold along it is a defensible choice.

</details>

<details>
<summary>One way to write it</summary>

```python
rides = load_rides(RIDES_CSV)
train, test = split(rides)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
oof = cross_val_predict(LogisticRegression(max_iter=1000), train[FEATURES],
                        train[TARGET], cv=folds, method="predict_proba")[:, 1]

def total_cost(y_true, y_pred, false_alarm, miss):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return false_alarm * fp + miss * fn

grid = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]
costs = {t: total_cost(train[TARGET], (oof >= t).astype(int), 3, 1) for t in grid}
best = min(costs, key=costs.get)
print(best, costs[best])

model = LogisticRegression(max_iter=1000).fit(train[FEATURES], train[TARGET])
p = model.predict_proba(test[FEATURES])[:, 1]
for t in [0.5, best]:
    guess = (p >= t).astype(int)
    print(t, total_cost(test[TARGET], guess, 3, 1),
          round(precision_score(test[TARGET], guess), 2),
          round(recall_score(test[TARGET], guess), 2))
```

</details>

## 5. (Optional) The 99% model

A colleague shows you a fraud model that is 99.2% accurate. Fraud is
0.8% of transactions. What one question do you ask, and which numbers do
you want to see instead?

<details>
<summary>Check yourself</summary>

"How accurate is a model that always says *not fraud*?" The answer is
99.2%, exactly the same, so the accuracy tells you nothing. Ask for the
confusion matrix, recall on the fraud class (how much fraud it catches),
precision on it (how many flagged customers are innocent), and ROC-AUC
to see whether it ranks fraud above normal transactions at all. For a
class that rare, the precision-recall curve
(`sklearn.metrics.precision_recall_curve`) is often more telling than
ROC.

</details>

---

That's classification metrics. If one thing sticks: *decide what each
mistake costs first, then let that pick your metric and your threshold.*
