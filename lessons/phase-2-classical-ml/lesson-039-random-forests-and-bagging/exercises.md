# Lesson 039 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-039-random-forests-and-bagging/lesson.py my_lesson_039.py
python my_lesson_039.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-039-random-forests-and-bagging").resolve()
```

`load_rides`, `split`, `make_model`, `make_preprocess`, `FOLDS` and the
column lists are there to reuse. `make_model` takes an optional
`numeric=` list, which the hands-on task needs.

---

## 1. Why not prune the trees?

Inside a bagged ensemble or a forest, the individual trees are usually
grown deep, with little or no pruning. Yesterday we pruned hard. Why
the change of heart?

<details>
<summary>Check yourself</summary>

Pruning trades variance for bias. A deep tree has low bias and high
variance, and averaging many of them removes much of the variance
while keeping the low bias. Prune each tree too hard and you average
lots of similar, simple trees, which can't recover the detail you cut
away. A light floor like `min_samples_leaf=5` is fine; a depth-3 cap
usually isn't.

</details>

## 2. Bootstrap arithmetic

You draw a bootstrap sample of 1,000 rows from 1,000 rows. Roughly how
many distinct rows end up in it? Where does the rest of the sample come
from?

<details>
<summary>Check yourself</summary>

About 632 distinct rows. Each row has a `(1 - 1/n)^n` chance of being
missed every time, which is about `1/e`, or 37%. The sample still has
1,000 rows in total because the picked rows show up more than once.
The 368 or so rows left out are that tree's out-of-bag rows.

</details>

## 3. Bagging vs forest

What's the one difference between `BaggingClassifier` around a
decision tree and a `RandomForestClassifier`? Why does it help?

<details>
<summary>Check yourself</summary>

The forest also picks a random subset of columns at every split
(`max_features`). That stops every tree from asking the same strong
first question, so their errors are less correlated, and less
correlated errors cancel better when you average. Section 3 shows CV
going from about 0.76 to about 0.78.

</details>

## 4. Hands-on: forest dials and a missing column

**a)** With 300 trees, compare CV mean and test accuracy for
`RandomForestClassifier` with default settings and with
`min_samples_leaf=5`.

**b)** Swap in `ExtraTreesClassifier(n_estimators=300,
min_samples_leaf=5, random_state=0)` from `sklearn.ensemble`. It also
picks the split *thresholds* at random. How does it compare?

**c)** Drop `duration_min` from the numeric columns, refit the
`min_samples_leaf=5` forest, and compute permutation importance on the
test rides. What happens to `distance_km`? What happens to test
accuracy?

Hints:

- `make_model(est, numeric=["distance_km", "temp_c", "rain_mm"])`
- `permutation_importance(model, test[cols], test[TARGET], n_repeats=10, random_state=0)`,
  where `cols` lists the raw columns the model uses.

<details>
<summary>Expected results</summary>

```
a) default:      CV 0.781, test 0.750
   min_leaf=5:   CV 0.795, test 0.758
b) extra trees:  CV 0.795, test 0.750
c) distance_km permutation importance jumps to about 0.13 (top by far);
   test accuracy about 0.763
```

(a) A small leaf floor helps a little, as in section 3.

(b) Extra trees land in the same place here. They're faster to train
and sometimes a touch better on noisy data; worth trying, not a magic
upgrade.

(c) With duration gone, distance stops sharing credit and takes all of
it. Accuracy barely changes, which tells you the two columns were
largely saying the same thing. That's the section 5 warning in action.

</details>

<details>
<summary>One way to write it</summary>

```python
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

rides = load_rides(RIDES_CSV)
train, test = split(rides)

for name, est in [
    ("default", RandomForestClassifier(n_estimators=300, random_state=0)),
    ("min_leaf=5", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0)),
    ("extra trees", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, random_state=0)),
]:
    model = make_model(est)
    cv = cross_val_score(model, train, train[TARGET], cv=FOLDS).mean()
    model.fit(train, train[TARGET])
    print(name, round(cv, 3), round(model.score(test, test[TARGET]), 3))

numeric = ["distance_km", "temp_c", "rain_mm"]
model = make_model(RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                          random_state=0), numeric=numeric)
model.fit(train, train[TARGET])
cols = numeric + CATEGORICAL + FLAGS
perm = permutation_importance(model, test[cols], test[TARGET], n_repeats=10, random_state=0)
print(pd.Series(perm.importances_mean, index=cols).sort_values(ascending=False).round(3))
print("test", round(model.score(test, test[TARGET]), 3))
```

</details>

## 5. (Optional) A forest that regresses

`RandomForestRegressor` averages the trees' numbers instead of counting
votes. Predict `duration_min` (drop it from the numeric columns first,
or you'll leak the answer) and compare test RMSE against a single
`DecisionTreeRegressor(max_depth=3)` from lesson 038's optional
exercise. Before you run: which do you expect to win, and by roughly
how much?

<details>
<summary>Check yourself</summary>

The forest should win, usually clearly, because averaging smooths the
staircase of predictions a single shallow tree makes. Distance will do
most of the work in both models. If the forest *loses*, check that you
really did remove `duration_min` from the inputs and that you're
computing RMSE on the test rides.

</details>

---

That's random forests. If one thing sticks: *many different, slightly
overfit trees beat one careful tree, and importances are hints, not
proof.*
