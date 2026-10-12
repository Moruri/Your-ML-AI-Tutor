# Lesson 037 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-037-k-nearest-neighbours-and-the-curse-of-dimensionality/lesson.py my_lesson_037.py
python my_lesson_037.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-037-k-nearest-neighbours-and-the-curse-of-dimensionality").resolve()
```

`load_rides`, `split`, `make_knn`, `make_preprocess`, `FOLDS` and the
column lists are there to reuse.

---

## 1. What does k-NN store?

After `model.fit(train, train[TARGET])` on the pipeline from section 1,
what has the model actually learned? A formula? A set of weights? Something
else?

<details>
<summary>Check yourself</summary>

The training rides themselves (after preprocessing). k-NN doesn't fit
coefficients; at predict time it measures distance to those stored
points and votes. That's why people call it a **lazy** learner: almost
all the work happens at prediction, not at fit.

</details>

## 2. Odd k or even k?

You're doing two-class classification. Would you rather set `k` to 14 or
to 15, and why?

<details>
<summary>Check yourself</summary>

Prefer the odd one. With even `k` a 7–7 tie is possible, and scikit-learn
breaks ties by looking at the class order, not by anything meaningful in
the data. Odd `k` always has a majority. (Weighted voting and more than
two classes change the picture, but the odd habit is still a good
default.)

</details>

## 3. Add a useless column?

A colleague one-hots the ride ID into thousands of columns "so the model
has more information". What happens to k-NN, and why?

<details>
<summary>Check yourself</summary>

Every ride gets its own private dimension, so "closest" becomes
meaningless — each training ride is unique in ID-space, and new rides
(with new IDs) sit equally far from everyone. You've invented the curse
of dimensionality on purpose. Drop IDs; they're labels, not features.

</details>

## 4. Hands-on: regressor, k, and concentration

**a)** Swap the classifier for `KNeighborsRegressor` and predict
`duration_min` from `distance_km`, `temp_c`, `rain_mm` and `weekend`
(scale the three numbers; pass `weekend` through). Use `k=15`, report
test RMSE.

**b)** With the classification pipeline from `make_knn`, print CV mean
for `k` in `{3, 7, 15, 31, 63}`.

**c)** Reuse the distance-concentration idea from section 4: for
dimensions 2, 20 and 100, draw 300 standard-normal points and print the
ratio of nearest to farthest distance from the first point.

Hints:

- `from sklearn.neighbors import KNeighborsRegressor`
- RMSE: `mean_squared_error(y_test, pred, squared=False)` on older
  sklearn, or `root_mean_squared_error` on newer ones; or
  `np.sqrt(((y_test - pred) ** 2).mean())`.

<details>
<summary>Expected results</summary>

```
a) test RMSE around 8.3 to 8.6 minutes (seed-dependent wrapper; with
   random_state=0 split and StandardScaler on the three numbers, expect
   roughly 8.4)
b) k=3  ~0.74;  k=7  ~0.76;  k=15  0.781;  k=31  0.787;  k=63  ~0.77
c) nearest/farthest rises toward 1 as dimension grows (around 0.02, 0.4,
   0.7 — exact draws vary with the RNG seed)
```

(a) A regressor averages the neighbours' durations instead of voting.
Same distance machinery, different aggregation.

(b) Same sweet spot as section 2; 15 and 31 are close.

(c) The ratio climbing toward 1 *is* the curse: "nearest" stops being
special.

</details>

<details>
<summary>One way to write it</summary>

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

rides = load_rides(RIDES_CSV)
train, test = split(rides)

prep = ColumnTransformer([
    ("num", StandardScaler(), ["distance_km", "temp_c", "rain_mm"]),
    ("flags", "passthrough", ["weekend"]),
])
reg = Pipeline([("prep", prep), ("knn", KNeighborsRegressor(n_neighbors=15))])
reg.fit(train, train["duration_min"])
pred = reg.predict(test)
print("rmse", round(float(np.sqrt(((test["duration_min"] - pred) ** 2).mean())), 2))

model = make_knn()
for k in [3, 7, 15, 31, 63]:
    scores = cross_val_score(model.set_params(knn__n_neighbors=k),
                             train, train[TARGET], cv=FOLDS)
    print(k, round(scores.mean(), 3))

rng = np.random.default_rng(0)
for d in [2, 20, 100]:
    pts = rng.normal(size=(300, d))
    dist = np.linalg.norm(pts[1:] - pts[0], axis=1)
    print(d, round(float(dist.min() / dist.max()), 3))
```

</details>

## 5. (Optional) Weighted neighbours

`KNeighborsClassifier(n_neighbors=15, weights="distance")` lets closer
neighbours vote louder. Cross-validate it against the uniform default on
the full pipeline. Before you run: do you expect a big lift on this
data?

<details>
<summary>Check yourself</summary>

Usually a small change here, not a breakthrough. With `k=15` on well
scaled rides the nearest few already agree most of the time, so
re-weighting them rarely flips the vote. Weighted voting helps more
when neighbours disagree a lot, or when `k` is large. Try it, note the
CV means, and keep the simpler default unless the gap is real.

</details>

---

That's k-NN. If one thing sticks: *closest only counts when the axes are
fair and few.*
