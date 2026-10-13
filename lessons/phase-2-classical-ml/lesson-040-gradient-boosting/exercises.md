# Lesson 040 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-040-gradient-boosting/lesson.py my_lesson_040.py
python my_lesson_040.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-040-gradient-boosting").resolve()
```

`load_rides`, `split`, `make_boost`, `rmse`, `FOLDS` and the column
lists are there to reuse. `make_boost(**kwargs)` passes anything you
give it straight to `HistGradientBoostingClassifier`.

---

## 1. Forest or relay?

In one sentence each: how does a random forest combine its trees, and
how does gradient boosting combine its trees?

<details>
<summary>Check yourself</summary>

A forest trains deep trees independently on bootstrap samples and
averages their votes, which mostly cuts variance. Boosting trains
small trees one after another, each fitted to what the ensemble still
gets wrong, and adds them up, which mostly cuts bias.

</details>

## 2. One round by hand

Three rides last 10, 20 and 30 minutes. Round 0 predicts the mean for
all of them. A perfect little tree then predicts the residuals exactly.
With a learning rate of 0.5, what are the predictions after round 1?

<details>
<summary>Check yourself</summary>

The mean is 20, so the residuals are -10, 0 and +10. Adding half of
each gives **15, 20 and 25**. Closer, not there yet. That's what a
learning rate below 1 does on purpose: small steps that later trees
can keep correcting.

</details>

## 3. Which number is honest?

With early stopping on, the model scores itself on a validation slice
after every tree. Can you report that validation score as your final
result?

<details>
<summary>Check yourself</summary>

Not really. That slice was used to *choose* when to stop, so it's
slightly flattering, just like picking a setting by the test score
would be. Report CV on the training rides while you're choosing, and
the untouched test set once at the end.

</details>

## 4. Hands-on: step size and early stopping

**a)** Rerun section 1's hand-made boosting loop for 50 trees with
`lr = 1.0` and then `lr = 0.05`. Record train and test RMSE for each.

**b)** Fit `make_boost(learning_rate=0.05, max_iter=1000,
max_leaf_nodes=4, early_stopping=True, validation_fraction=0.2,
n_iter_no_change=20, scoring="loss")`. How many trees did it build
(`model.named_steps["model"].n_iter_`), and what's the test accuracy?

**c)** What CV mean does `make_boost(learning_rate=0.1, max_iter=50,
max_leaf_nodes=4)` get? Compare it with section 5's winner.

<details>
<summary>Expected results</summary>

```
a) lr=1.0:   train RMSE 2.74, test RMSE 4.08
   lr=0.05:  train RMSE 3.92, test RMSE 3.77
b) 99 trees, test accuracy 0.792
c) CV mean 0.780, close to section 5's 0.784
```

(a) The big step fits the training rides much better and the test
rides worse: overfitting in two numbers. The small step hasn't quite
converged after 50 trees, but it's already better on new rides.

(b) Tiny trees learn slowly, so early stopping lets it run longer than
section 4's 57. Combining the two leashes gives the best test score in
this lesson. Don't get attached to the third decimal on 236 test
rides, though.

(c) Doubling the learning rate while using a quarter of the trees
lands in nearly the same place. The two dials really do trade off.

</details>

<details>
<summary>One way to write it</summary>

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

rides = load_rides(RIDES_CSV)
train, test = split(rides)

cols = ["distance_km", "weekend"]
X_tr, X_te = train[cols].to_numpy(), test[cols].to_numpy()
y_tr, y_te = train["duration_min"].to_numpy(), test["duration_min"].to_numpy()
for lr in [1.0, 0.05]:
    p_tr = np.full(len(y_tr), y_tr.mean())
    p_te = np.full(len(y_te), y_tr.mean())
    for _ in range(50):
        tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X_tr, y_tr - p_tr)
        p_tr += lr * tree.predict(X_tr)
        p_te += lr * tree.predict(X_te)
    print(lr, round(rmse(y_tr, p_tr), 2), round(rmse(y_te, p_te), 2))

model = make_boost(learning_rate=0.05, max_iter=1000, max_leaf_nodes=4,
                   early_stopping=True, validation_fraction=0.2,
                   n_iter_no_change=20, scoring="loss").fit(train, train[TARGET])
print(model.named_steps["model"].n_iter_, round(model.score(test, test[TARGET]), 3))

quick = make_boost(learning_rate=0.1, max_iter=50, max_leaf_nodes=4)
print(round(cross_val_score(quick, train, train[TARGET], cv=FOLDS).mean(), 3))
```

</details>

## 5. (Optional) Boosting for durations

`HistGradientBoostingRegressor` is the regression sibling. Predict
`duration_min` from every other column (remove it from `NUMERIC` and
from the preprocessing first, or you'll leak the answer). Use
`learning_rate=0.05`, `max_leaf_nodes=8` and early stopping. Before
you run: will it beat the 3.65-minute test RMSE of section 1's
hand-made 50-tree version, which only saw distance and the weekend
flag?

<details>
<summary>Check yourself</summary>

Probably only a little, if at all. Distance explains most of a ride's
duration, and section 1 already used it. Extra columns like
temperature and station add a small amount at best and can add noise
on a dataset this size. More features don't guarantee a better model;
the test score decides. If yours is far *better*, check that
`duration_min` really isn't sneaking in as an input.

</details>

---

That's gradient boosting. If one thing sticks: *small trees, small
steps, and stop when validation stops improving.*
