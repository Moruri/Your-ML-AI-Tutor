# Lesson 038 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-038-decision-trees-you-can-read/lesson.py my_lesson_038.py
python my_lesson_038.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-038-decision-trees-you-can-read").resolve()
```

`load_rides`, `split`, `make_tree`, `make_preprocess`, `FOLDS` and the
column lists are there to reuse.

---

## 1. Scale a tree?

Would you put a `StandardScaler` in front of a decision tree? Why or why
not?

<details>
<summary>Check yourself</summary>

No need. A tree asks one column at a time ("duration > 16.25?"), so the
units of different columns never meet. Scaling won't change which splits
it finds (thresholds just move with the units). Categories still need
encoding; numbers can pass through as they are.

</details>

## 2. Gini by hand

A node has 8 casual rides and 2 member rides. What is its Gini impurity?
What would it be if all 10 were casual?

<details>
<summary>Check yourself</summary>

p = 0.8 casual, so Gini = 1 - (0.8² + 0.2²) = 1 - (0.64 + 0.04) = **0.32**.

If all 10 are casual, p = 1, Gini = 1 - (1 + 0) = **0**. Pure.

</details>

## 3. Train score of 1.0

Your unlimited tree scores 1.000 on the training rides and 0.70 on the
test rides. Is that a good model? What would you change first?

<details>
<summary>Check yourself</summary>

No — it memorised the training set (overfitting). Cap `max_depth` or
raise `min_samples_leaf`, and pick the setting with cross-validation,
not by watching the test score. Section 3's table is exactly this
pattern.

</details>

## 4. Hands-on: stump vs pruned tree

**a)** Fit a depth-1 tree (`make_tree(max_depth=1)`). Print `export_text`
and the test accuracy. What single question does it ask?

**b)** Fit `make_tree(max_depth=4, min_samples_leaf=30)`. Report leaves,
CV mean, and test accuracy.

**c)** Compare `criterion="gini"` vs `criterion="entropy"` at
`max_depth=3`. Do the CV means differ much?

Hints:

- `export_text(model.named_steps["tree"], feature_names=list(model.named_steps["prep"].get_feature_names_out()))`
- Use `FOLDS` with `cross_val_score` so numbers match the lesson.

<details>
<summary>Expected results</summary>

```
a) first split is duration_min <= 16.25; test accuracy 0.746
b) around 9–15 leaves, CV mean near 0.77, test near 0.76
c) Gini and entropy CV means within about 0.01 of each other
```

(a) The stump's one question is the same first split section 2 found.

(b) Depth 4 with a leaf floor sits near the pruned trees in section 4.

(c) On this data the two criteria barely disagree. Pick one and stay
consistent; Gini is the default for a reason.

</details>

<details>
<summary>One way to write it</summary>

```python
from sklearn.tree import export_text
from sklearn.model_selection import cross_val_score

rides = load_rides(RIDES_CSV)
train, test = split(rides)

stump = make_tree(max_depth=1).fit(train, train[TARGET])
names = list(stump.named_steps["prep"].get_feature_names_out())
print(export_text(stump.named_steps["tree"], feature_names=names))
print("stump test", round(stump.score(test, test[TARGET]), 3))

pruned = make_tree(max_depth=4, min_samples_leaf=30)
cv = cross_val_score(pruned, train, train[TARGET], cv=FOLDS)
fitted = pruned.fit(train, train[TARGET])
print("leaves", fitted.named_steps["tree"].get_n_leaves(),
      "cv", round(cv.mean(), 3),
      "test", round(fitted.score(test, test[TARGET]), 3))

for crit in ["gini", "entropy"]:
    m = make_tree(max_depth=3, criterion=crit)
    print(crit, round(cross_val_score(m, train, train[TARGET], cv=FOLDS).mean(), 3))
```

</details>

## 5. (Optional) A tree that regresses

`DecisionTreeRegressor` uses squared-error impurity instead of Gini.
Predict `duration_min` from the same features (drop it from `NUMERIC`
first, or you'll leak the answer). Cap depth at 3, print `export_text`,
and check test RMSE. Before you run: do you expect the first split to
still be about distance or weekend?

<details>
<summary>Check yourself</summary>

The first split will usually involve `distance_km` or something tightly
tied to how long a ride takes — not the casual/member weekend pattern
from the classifier. Same tree machinery, different target, different
questions. RMSE on this split lands roughly in the mid-to-high single
digits of minutes with depth 3; compare it to the k-NN regressor from
lesson 037's exercises if you did that one.

</details>

---

That's decision trees. If one thing sticks: *a tree is a checklist —
cap its depth or it will memorise the answers.*
