# Lesson 036 - Pipelines: doing it right every time

**Phase 2 - Classical machine learning** | Week 5, Day 5 | Friday 2026-10-09

> **Goal:** Chain preprocessing and models with `Pipeline` and
> `ColumnTransformer` so nothing leaks.

Time: about 50 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Lesson 035 ended with a confession. To cross-validate the scaled and
encoded rides, we fitted the scaler and encoder on all the training
rides first, so every validation fold had quietly helped set its own
ruler. A small leak, and doing it properly by hand would have meant
re-fitting both inside a loop over folds. Fiddly, easy to get wrong, and
exactly the kind of thing you forget at 11pm.

So we stop doing it by hand. Today you'll build the whole thing, from
raw columns to prediction, as one object that can't get the order wrong.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` and the
same train/test split as lessons 033 to 035. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-036-pipelines-doing-it-right-every-time/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match.

## 1. A two-step pipeline

```
by hand: scaler, then k-NN, test accuracy      -> 0.767
pipeline, test accuracy                        -> 0.767
step names                                     -> ('standardscaler', 'kneighborsclassifier')
the scaler inside learned these means          -> (3.2, 15.76, 17.64, 2.41)
```

A **pipeline** is a list of steps: any number of transformers, then a
model at the end. It behaves like a model itself.

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
pipe.fit(train[NUMERIC], train[TARGET])
pipe.score(test[NUMERIC], test[TARGET])
```

- `fit` runs `fit_transform` on each transformer in turn, then `fit` on
  the model.
- `predict` and `score` run only `transform` on each transformer (using
  what was learned during `fit`), then the model.

That's lesson 035's recipe exactly, which is why the two scores match.
The difference is you can't forget a step, or accidentally fit the
scaler on the test rides. `make_pipeline` names the steps after their
classes; `pipe[0]` or `pipe.named_steps["standardscaler"]` gets one
back, fitted, if you want to look inside.

## 2. Different columns, different treatment

```
columns in                                     -> 7
columns out                                    -> 13
first few names out  -> ('num__distance_km', 'num__duration_min', 'num__temp_c',
                         'num__rain_mm', 'cat__start_station_Market Square', ...)
last few names out   -> ('cat__start_station_University', 'cat__bike_type_classic',
                         'cat__bike_type_electric', 'flags__weekend')
first training ride, transformed  -> (-0.69, -0.54, -0.9, 1.4, 0.0, 0.0, 0.0, 0.0,
                                      1.0, 0.0, 1.0, 0.0, 0.0)
```

Real tables mix numbers and categories, and they need different
treatment. A **ColumnTransformer** routes each group of columns to its
own step and glues the results side by side:

```python
from sklearn.compose import ColumnTransformer

prep = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("flags", "passthrough", FLAGS),
])
```

Each entry is a name, a transformer, and the columns it gets. Four
numbers stay four standardised numbers; two categories become eight
one-hot columns (six stations, two bike types); the `weekend` flag,
already 0 or 1, passes through untouched. 7 in, 13 out.

Anything you don't list, like `ride_id` and `started_at`, is **dropped**
by default. That's a feature: the ride ID is just a label, and you don't
want a model learning from it. `get_feature_names_out()` tells you which
step made each output column, which is a lifesaver when you come back
to read the weights.

Then the full model is a pipeline whose first step is the whole
ColumnTransformer:

```python
model = Pipeline([
    ("prep", prep),
    ("knn", KNeighborsClassifier(n_neighbors=15)),
])
model.fit(train, train[TARGET])      # the raw DataFrame goes straight in
```

## 3. Cross-validating the whole thing

```
5-fold CV accuracy, preprocessing inside each fold:
               mean  spread
    logistic  0.790   0.017
    k-NN      0.781   0.043
logistic pipeline on the test rides, once      -> 0.797
k-NN pipeline on the test rides, once          -> 0.737
```

Hand the pipeline to `cross_val_score` and the leak is gone:

```python
cross_val_score(model, train, train[TARGET], cv=folds)
```

For every fold, scikit-learn makes a fresh copy of the *whole*
pipeline and fits it on that fold's training part only, scaler and
encoder included. The validation part is then treated exactly like new
rides. That's lesson 035's confession fixed, in one line.

Two other things worth noticing. Logistic regression jumped from 0.773
in lesson 035 to 0.790, and the reason isn't the pipeline: it's the
`weekend` flag we brought back (57% of weekend rides are casual,
against 26% on weekdays). And k-NN's test score, 0.737, sits a good way
under its CV mean, but its folds spread by 0.043, so that's within what
the folds already warned us about. Logistic regression is both better
and steadier here, and it's the one we'd pick.

## 4. A leak that looks like a breakthrough

```
rides x columns of pure random noise           -> (940, 1000)
select on all rides, then CV (ROC-AUC)         -> 0.7
select inside the pipeline, CV (ROC-AUC)       -> 0.522
```

Lesson 035's leak was small. This one isn't, and it's a mistake people
make in real research.

We invent 1,000 columns of random numbers. They know nothing about
riders, so any honest score should be a coin toss: ROC-AUC about 0.5
(lesson 034). Then we do something that sounds sensible: keep the 20
columns most related to the answer (`SelectKBest`), and cross-validate a
model on those.

Done outside the pipeline, the selection saw every training ride's
answer, including the rides that later act as validation folds. With
1,000 columns, some will match those answers by pure luck, and CV then
rewards the luck: AUC 0.70, which looks like a real model. Put the
selection *inside* the pipeline and each fold has to pick its own 20
from its own training part. The luck doesn't carry over, and the score
falls to 0.52, what noise deserves.

The rule this teaches: **anything that learns from the data is part of
the model**. Scaling, encoding, filling missing values, choosing
features. If it has a `fit`, it goes in the pipeline.

## 5. One object, raw rides in, answers out

```
two raw rides: a Saturday 38-minute ride and a Monday 7-minute hop
chance each new ride is casual                 -> (0.93, 0.0)
k before                                       -> 15
k after set_params(knn__n_neighbors=31)        -> 31
```

Here's the everyday payoff. The fitted pipeline takes raw rides, as a
DataFrame with the same column names, and does every step itself. The
long Saturday ride looks very casual; the quick Monday hop looks like a
member on the way to work. The hop starts at "Harbour", a station the
encoder never saw, and nothing breaks, thanks to
`handle_unknown="ignore"`.

Settings of any step are reachable through `step__setting` names, with
a double underscore:

```python
model.set_params(knn__n_neighbors=31)
```

That looks like a small convenience. It's actually how lesson 042 will
search over settings safely: the whole pipeline is re-fitted for every
combination, inside every fold.

And when it's time to ship, the one fitted pipeline is the thing you
save and load (`joblib.dump(model, "model.joblib")`), so the app can't
forget a preprocessing step either. More on that in Phase 5.

## What you can do now

- Build a `Pipeline` from transformers and a model, and say what `fit`
  and `predict` do at each step.
- Route numeric and categorical columns with a `ColumnTransformer`, and
  read its output names.
- Cross-validate a whole pipeline so preprocessing never leaks.
- Explain why feature selection outside the pipeline can make noise look
  like signal.
- Predict on raw new data and change a step's settings with
  `step__setting`.

## What to do now

1. Run `lesson.py`. In section 4, change the 20 to 5 and to 100 and
   watch both scores.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   picks `k` the honest way.
3. Next, lesson 037: k-nearest neighbours properly, and why more
   features aren't always better. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

From here on, every model in this course is a pipeline. It's the habit
that keeps your scores honest without you having to be careful.
