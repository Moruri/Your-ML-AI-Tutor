# Lesson 046 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-046-anomaly-detection/lesson.py my_lesson_046.py
python my_lesson_046.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-046-anomaly-detection").resolve()
```

`load_rides`, `plant_oddities`, `FEATURES`, `heading` and `show` are
there to reuse. `plant_oddities(load_rides(RIDES_CSV))` gives you the
1,184 rides with a `planted` column (1 for the eight fakes).

---

## 1. Why plant fakes?

Your manager asks why you added eight made-up rides to real data before
running a detector. In one or two sentences, what do you say?

<details>
<summary>Check yourself</summary>

Real data never tells you which rows are anomalies, so without a few
known ones you can't tell a detector that works from one that just
flags the biggest numbers. The fakes are a test set for the detector,
and they never go anywhere near a report.

</details>

## 2. Scale or not?

You're told "always scale before anomaly detection". True for which of
Isolation Forest and LOF, and why?

<details>
<summary>Check yourself</summary>

**LOF** needs scaling, because it measures distances between rows, and
unscaled minutes would swamp everything else (the k-NN lesson again).
**Isolation Forest** splits on thresholds one column at a time, like any
tree, so scaling doesn't change its answer.

</details>

## 3. Is the 71-minute rainy ride a problem?

Isolation Forest's strangest row is R01021: 12 km in 71 minutes in
16 mm of rain. Should you delete it from the data?

<details>
<summary>Check yourself</summary>

No, not on the score alone. It's unusual, not impossible: about
10 km/h, a normal bike speed, on a wet day. An anomaly score says "look
at this", not "this is wrong". Deleting real but rare rides would make
every later model worse at exactly the cases it finds hardest.

</details>

## 4. Hands-on: give LOF a clock

The 3am rides slip past LOF because "hour" is just a number to it, and 3
isn't that far from 7. Tell it what you know.

**a)** Add a column `night = (hour < 6)` as 0/1. Re-run LOF
(`n_neighbors=20`, scaled) on `FEATURES + ["night"]`. Where do the eight
planted rides rank now?

**b)** Now go the other way: drop `hour` from `FEATURES` entirely. Where
do the 3am rides rank?

<details>
<summary>Expected results</summary>

```
a) planted ranks  [5, 6, 1, 8, 7, 2, 4, 3]   (all eight in the top 8)
b) 3am ride ranks [143, 492, 879]
```

One honest feature turns "slightly early" into "a time nobody rides",
and LOF's top eight become exactly the eight fakes. Take the hour away
and the 3am rides become invisible: no algorithm can find a pattern in a
column it never sees.

</details>

<details>
<summary>One way to write it</summary>

```python
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

rides = plant_oddities(load_rides(RIDES_CSV))
rides["night"] = (rides["hour"] < 6).astype(int)

def planted_ranks(feats):
    X = StandardScaler().fit_transform(rides[feats])
    lof = LocalOutlierFactor(n_neighbors=20).fit(X)
    score = pd.Series(-lof.negative_outlier_factor_)
    ranks = score.rank(ascending=False).astype(int)
    return ranks[rides["planted"] == 1].tolist()

print(planted_ranks(FEATURES + ["night"]))
print(planted_ranks([f for f in FEATURES if f != "hour"])[5:])
```

</details>

## 5. (Optional) Does the clock help the forest too?

Fit `IsolationForest(n_estimators=300, random_state=0)` on
`FEATURES + ["night"]` and print the ranks of the planted rides (use
`-forest.score_samples(...)` as the score). Before you run: will the
3am rides jump into the top 10 like they did for LOF?

<details>
<summary>Check yourself</summary>

They improve but don't top the list: ranks come out as
`[8, 21, 10, 3, 7, 18, 11, 20]`. Isolation Forest still finds the
genuinely long real rides easy to cut off, so they keep crowding the
top. Different detectors answer slightly different questions, which is
why comparing two is worth the extra line of code.

</details>

---

That's anomaly detection. If one thing sticks: *plant a few known
oddities, compare two detectors, flag only what someone can review, and
remember that unusual is not the same as wrong.*
