# Lesson 043 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-043-imbalanced-data-and-threshold-tuning/lesson.py my_lesson_043.py
python my_lesson_043.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-043-imbalanced-data-and-threshold-tuning").resolve()
```

`load_rides`, `split`, `make_pipe`, `metrics`, `oversample`, `undersample`
and `TARGET` are there to reuse.

---

## 1. Why did accuracy look fine?

A friend reports "92% accuracy on long-ride detection" and is ready to
ship. In one sentence: what should you ask them next?

<details>
<summary>Check yourself</summary>

Ask for **recall on the long class** (and ideally the full confusion
matrix). With a 7.7% rare class, 92% accuracy is what you get by
predicting "not long" every time and finding zero long rides.

</details>

## 2. Weight or resample?

You have a logistic regression and a training set of 10,000 rows where
the rare class is 2%. You want a first fix with as little fuss as
possible. Do you reach for `class_weight="balanced"`, oversampling, or
undersampling — and why?

<details>
<summary>Check yourself</summary>

**`class_weight="balanced"`.** It needs no extra rows, discards nothing,
and is one argument. Oversampling would roughly duplicate the rare rows
until you have ~20,000 training rows; undersampling would throw away
most of the common class and leave you with a tiny training set. Start
with weights; resample if you need to.

</details>

## 3. Which way does the threshold move?

You care much more about catching long rides than about false alarms
(a missed long ride costs 5× a false alarm). Starting from 0.5, do you
**lower** or **raise** the threshold on `predict_proba`?

<details>
<summary>Check yourself</summary>

**Lower it.** A lower threshold means "call it long more readily", which
raises recall and usually lowers precision. That is the right direction
when misses are expensive. (Lesson 034's cost story is the same idea.)

</details>

## 4. Hands-on: pick a threshold by cost

A missed long ride costs **5**, a false alarm costs **1** (same idea as
lesson 034). Fit `make_pipe(class_weight="balanced")` on three-quarters
of the training rides (`random_state=0`, stratified), score probabilities
on the remaining quarter, and sweep thresholds from 0.1 to 0.9 in steps
of 0.05.

**a)** Which threshold minimises `1 * FP + 5 * FN` on the tuning slice?
What are FP, FN and the total cost there?

**b)** Lock that threshold and report precision, recall, F1 and the
confusion matrix on the test set.

<details>
<summary>Expected results</summary>

```
a) thr=0.55  (FP=58, FN=1, cost=63)
b) precision 0.176, recall 0.722, F1 0.283
   confusion [[157, 61], [5, 13]]  (found 13/18 long rides)
```

With misses five times as expensive as false alarms, the sweep lands
near 0.55 — almost the same place F1 picked in the lesson. On the test
set you still pay sixty-one false alarms to catch thirteen of eighteen
long rides. Threshold tuning rearranges the trade-off; it does not
invent signal. Adding `distance_km` back (exercise 5) is the real upgrade.

</details>

<details>
<summary>One way to write it</summary>

```python
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

rides = load_rides(RIDES_CSV)
train, test = split(rides)
fit_part, tune_part = train_test_split(
    train, test_size=0.25, random_state=0, stratify=train[TARGET],
)
model = make_pipe(class_weight="balanced").fit(fit_part, fit_part[TARGET])
tune_p = model.predict_proba(tune_part)[:, 1]
test_p = model.predict_proba(test)[:, 1]

best = None
for thr in np.arange(0.1, 0.91, 0.05):
    pred = (tune_p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(tune_part[TARGET], pred).ravel()
    cost = int(fp) * 1 + int(fn) * 5
    if best is None or cost < best[0]:
        best = (cost, float(thr), int(fp), int(fn))
print(best)

pred = (test_p >= best[1]).astype(int)
print(metrics(test[TARGET], pred))
print(confusion_matrix(test[TARGET], pred))
```

</details>

## 5. (Optional) Put distance back

Add `"distance_km"` to `NUMERIC`, re-run sections 1 and 2, and compare
default recall to the balanced recall. Before you run: will the default
model still find zero long rides?

<details>
<summary>Check yourself</summary>

No — distance is a strong clue for duration, so the default model
suddenly finds most of the long rides (recall around 0.83) even without
reweighting. Balancing still pushes recall to 1.0 at the cost of more
false alarms. The lesson left distance out so the imbalance problem
would stay visible; in a real pipeline you would keep every honest
feature and *still* check class weights and the threshold.

</details>

---

That's imbalanced data. If one thing sticks: *when one class is rare,
stop trusting accuracy — reweight or resample, then pick the threshold
that matches the cost of a miss.*
