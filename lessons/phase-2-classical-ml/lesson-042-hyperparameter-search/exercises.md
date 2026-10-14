# Lesson 042 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-042-hyperparameter-search/lesson.py my_lesson_042.py
python my_lesson_042.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-042-hyperparameter-search").resolve()
```

`load_rides`, `split`, `make_knn`, `make_boost`, `FOLDS` and `TARGET`
are there to reuse. Both `make_` functions return an unfitted pipeline
whose last step is called `model`.

---

## 1. Parameter or hyperparameter?

Sort these into "learned from the data" and "chosen by you": a linear
regression's coefficients, `k` in k-NN, Lasso's `alpha`, the split
points inside a decision tree, a boosting model's learning rate.

<details>
<summary>Check yourself</summary>

**Learned (parameters):** the coefficients and the split points.

**Chosen (hyperparameters):** `k`, `alpha` and the learning rate.

A quick test: if `.fit()` works it out, it's a parameter. If you pass
it in when you create the model, it's a hyperparameter.

</details>

## 2. Count the fits

A grid has 3 values of one dial, 4 of another and 5 of a third. You run
`GridSearchCV` with 5-fold CV and the default `refit=True`. How many
times is a model trained?

<details>
<summary>Check yourself</summary>

3 x 4 x 5 = 60 combinations, times 5 folds = 300 fits, plus **1** more
to refit the winner on all the training rows: **301**. Add one dial
with 4 values and it's 1,201. That's why section 3 exists.

</details>

## 3. Is `best_score_` the number to report?

Your grid search tried 200 settings and `best_score_` is 0.83. Your
manager asks how well the model will do on new customers. Why is 0.83
probably a bit high, and what are two honest numbers you could give
instead?

<details>
<summary>Check yourself</summary>

The 0.83 is the best of 200 tries, measured on the same folds that
picked it, so some of it is luck (section 1, in miniature). Honest
options: a **nested CV** score of the whole search recipe (section 5),
or the score on an untouched **test set**, looked at once after all
choices are made. Ideally both, with a reminder of how many rows the
test set had.

</details>

## 4. Hands-on: search k-NN at random, then change the metric

**a)** Run `RandomizedSearchCV(make_knn(), {"model__n_neighbors": randint(1, 101), "model__weights": ["uniform", "distance"]}, n_iter=20, cv=FOLDS, random_state=0)`
on the training rides. What are `best_params_`, `best_score_` and the
test accuracy?

**b)** Rerun section 3's boosting search with `random_state=1`
instead of 0. What's the new `best_score_` and test accuracy? Did it
find the same settings?

**c)** Rerun section 2's grid with `scoring="roc_auc"`. Does the
winner change?

<details>
<summary>Expected results</summary>

```
a) {'model__n_neighbors': 37, 'model__weights': 'distance'}
   best_score_ 0.783, test accuracy 0.729
b) best_score_ 0.788, test accuracy 0.754
   learning_rate 0.022, max_iter 179, max_leaf_nodes 16,
   min_samples_leaf 76, l2_regularization 9.912
c) {'model__n_neighbors': 51, 'model__weights': 'uniform'}
   best_score_ 0.812, test ROC-AUC 0.812
```

(a) The random search lands on `k=37`, close to the grid's 31, with
about the same CV score. Random search doesn't need you to guess good
grid values up front.

(b) Completely different settings, nearly identical CV score (0.788
vs 0.787). That's a flat "good region": lots of combinations work
about equally well. Don't read too much into any one winning recipe.

(c) Yes. The metric you search with is part of the question you're
asking. ROC-AUC rewards good *ranking* of probabilities, and a
smoother vote from more neighbours ranks better. Note that
`search.score` now reports ROC-AUC too, not accuracy. Choose the
metric before you search, based on what you care about (lesson 034).

</details>

<details>
<summary>One way to write it</summary>

```python
from scipy.stats import loguniform, randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

rides = load_rides(RIDES_CSV)
train, test = split(rides)

knn_space = {"model__n_neighbors": randint(1, 101),
             "model__weights": ["uniform", "distance"]}
s = RandomizedSearchCV(make_knn(), knn_space, n_iter=20, cv=FOLDS,
                       random_state=0).fit(train, train[TARGET])
print(s.best_params_, round(s.best_score_, 3), round(s.score(test, test[TARGET]), 3))

boost_space = {
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_iter": randint(50, 401),
    "model__max_leaf_nodes": randint(4, 32),
    "model__min_samples_leaf": randint(5, 81),
    "model__l2_regularization": loguniform(1e-3, 10),
}
s = RandomizedSearchCV(make_boost(), boost_space, n_iter=25, cv=FOLDS,
                       random_state=1).fit(train, train[TARGET])
print(s.best_params_, round(s.best_score_, 3), round(s.score(test, test[TARGET]), 3))

grid = {"model__n_neighbors": [5, 15, 31, 51],
        "model__weights": ["uniform", "distance"]}
s = GridSearchCV(make_knn(), grid, cv=FOLDS, scoring="roc_auc").fit(train, train[TARGET])
print(s.best_params_, round(s.best_score_, 3), round(s.score(test, test[TARGET]), 3))
```

</details>

## 5. (Optional) Search the preprocessing too

Search parameters can reach any step, not just the model. In
`make_knn()`, the numeric columns are handled by a step you can reach
as `prep__num`. Run a grid over `{"prep__num": [StandardScaler(),
"passthrough"], "model__n_neighbors": [15, 31]}`. Before you run: will
the unscaled version ever win?

<details>
<summary>Check yourself</summary>

```
                    k=15   k=31
StandardScaler()   0.781  0.787
passthrough        0.772  0.773
```

Scaling wins at both values of `k`, which is lesson 035's point
confirmed by a search. The surprise is that the gap is only about one
point. Unscaled, duration (spread over tens of minutes) drowns out the
other columns, but duration also happens to be the column the trees in
lessons 038 and 039 split on first, so a k-NN that mostly looks at
duration still does reasonably well. On data where the loud column is
the useless one, the gap would be far bigger. Searching over preprocessing is a fair way to
settle "does this step help?" arguments, as long as the test set stays
out of it.

</details>

---

That's hyperparameter search. If one thing sticks: *let
cross-validation choose, and keep the test set for one honest look at
the end.*
