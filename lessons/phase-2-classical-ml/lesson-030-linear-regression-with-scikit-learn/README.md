# Lesson 030 - Linear regression with scikit-learn

**Phase 2 - Classical machine learning** | Week 5, Day 2 | Tuesday 2026-10-06

> **Goal:** Use the `fit`/`predict` API, read coefficients, and measure
> error with MAE and RMSE, so you can train a real model in three lines
> and still know exactly what it's doing.

Time: about 50 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Lesson 029 built a line by hand: pick a loss, find the slope and intercept
that minimise it, look at the errors. It worked, and it showed the model
was missing a feature: it couldn't tell an e-bike from a classic bike.

Today **scikit-learn** does the fitting. It's the standard Python library
for classical machine learning, and the reason it's worth learning
properly is that almost every model in it works the same way. Learn the
pattern once with linear regression and you already know how to use
decision trees, random forests and most of Phase 2.

## How to follow along

Venv active, `python` in the repo root. If you haven't yet, install the
Phase 2 requirement:

```bash
pip install -r requirements.txt
python -c "import sklearn; print(sklearn.__version__)"   # 1.4 or newer
```

Same `rides_clean.csv` as lesson 029. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-030-linear-regression-with-scikit-learn/lesson.py
```

It writes no files, and nothing is random, so your numbers will match.

## 1. X and y: the shapes scikit-learn expects

```
X shape (rows, features)                       -> (1176, 1)
y shape (rows,)                                -> (1176,)
```

scikit-learn is strict about shapes, and this trips everyone up once:

- `X` is **2-D**: one row per example, one column per feature. Even with
  a single feature, it's a table with one column: `rides[["distance_km"]]`,
  with double brackets.
- `y` is **1-D**: one target value per row: `rides["duration_min"]`.

Pass `rides["distance_km"]` (single brackets) as `X` and you'll get an
error saying it expected a 2D array. Now you know what it means.

## 2. fit, then predict

```
slope, model.coef_[0]                          -> 4.41
intercept, model.intercept_                    -> 1.667
predictions for 1, 3, 8 km                     -> [6.1, 14.9, 36.9]
```

The whole pattern:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()   # 1. build it (choose the kind of model)
model.fit(X, y)              # 2. train it (find the parameters)
model.predict(new_X)         # 3. use it
```

The slope and intercept match lesson 029's least-squares formula to the
last decimal. `LinearRegression` *is* least squares; it just handles any
number of features.

Notice the trailing underscores on `coef_` and `intercept_`. That's a
scikit-learn convention: anything ending in `_` was learned by `fit()`
and doesn't exist before you call it. `new_X` must have the same columns
as the `X` you trained on, so the script builds a small DataFrame with a
`distance_km` column for the three new rides.

## 3. Measuring error: MAE and RMSE

```
error in minutes:
                     MAE   RMSE
    guess the mean  7.25  10.04
    distance only   2.74   3.90
share of rides missed by more than 10 min      -> 0.029
```

`sklearn.metrics` has the scores ready-made: `mean_absolute_error(y,
y_pred)` and `root_mean_squared_error(y, y_pred)`.

**RMSE** is the square root of lesson 029's MSE. Taking the root brings
it back into minutes, so you can put it next to MAE. The two tell you
different things:

- **MAE** is the typical miss: "usually off by about 2.7 minutes".
- **RMSE** leans towards the big misses, because they were squared before
  averaging. It's always at least as big as MAE.

When RMSE sits well above MAE, as here, a handful of rides are being
predicted badly rather than every ride a bit badly. Under 3% of rides are
missed by more than 10 minutes, but those few are what push RMSE up.
Report both, with the baseline row next to them so the numbers mean
something.

## 4. More features, same three lines of code

```
features                                       -> ['distance_km', 'electric', 'casual', 'temp_c', 'rain_mm']
MAE / RMSE, distance only (min)                -> (2.74, 3.9)
MAE / RMSE, all five features (min)            -> (2.2, 3.21)
```

Linear models need numbers, so `load_rides` turns the two yes/no columns
into 0/1 flags:

```python
rides["electric"] = (rides["bike_type"] == "electric").astype(int)
rides["casual"] = (rides["rider_type"] == "casual").astype(int)
```

(Lesson 035 does this properly for columns with more than two values.)

Then the model code doesn't change at all. Only the list of columns in
`X` does, and the typical miss drops from 2.7 to 2.2 minutes. The model
has no idea what "electric" means; it just sees a column of 0s and 1s
that helps explain the minutes.

## 5. Reading the coefficients

```
coefficient per feature (minutes per unit):
    distance_km    4.36
    electric      -4.46
    casual         1.13
    temp_c        -0.01
    rain_mm       -0.04
intercept (min)                                -> 3.32
```

With several features, the model is:

*minutes = 4.36 x km - 4.46 x electric + 1.13 x casual - 0.01 x temp -
0.04 x rain + 3.32*

Read each coefficient as **"how much this adds, holding everything else
fixed"**:

- Each kilometre adds about 4.4 minutes.
- An e-bike takes about 4.5 minutes off a ride of the same distance.
- A casual rider adds about a minute compared with a member.
- Temperature and rain barely move it, at least in September's data.

Two cautions. First, **units matter**: `temp_c` is per degree and
`rain_mm` is per millimetre, so you can't compare the raw sizes to rank
features by importance. (Lesson 035 scales features so you can, and lesson
047 covers better ways to measure importance.) Second, a coefficient
describes a pattern in this data, **not cause and effect** (lesson 024).
"Casual riders take a minute longer" doesn't mean signing up makes you
faster.

## 6. Checking the leftovers

```
average error (actual - predicted) by group:
    bike_type  rider_type
    classic    casual        0.63
               member       -0.30
    electric   casual       -1.10
               member        0.53
three biggest misses:
         distance_km bike_type rider_type  duration_min  error
    451         10.3   classic     casual          76.4   27.1
    464          7.0   classic     casual          55.6   20.9
    574          6.7   classic     casual          48.8   15.5
```

Same check as lesson 029: group the errors and look for a pattern. The
3-minute biases are down to about a minute or less. The biggest misses
are long, slow casual rides, someone taking the scenic route or stopping
for lunch. No feature we have can predict that, and that's fine. Some
error is just life.

Now the honest warning. **Every number today was measured on the same
rides the model learned from.** That's like marking your own homework
with the answer sheet open. A model can look great on its training data
and do worse on rides it has never seen. Lesson 031 splits the data so
you can measure that properly.

## What you can do now

- Shape `X` (2-D) and `y` (1-D) the way scikit-learn expects.
- Build, `fit` and `predict` with `LinearRegression`.
- Read `coef_` and `intercept_`, and explain each in plain words.
- Measure error with `mean_absolute_error` and `root_mean_squared_error`,
  next to a baseline.
- Say what it means when RMSE is much bigger than MAE.
- Turn yes/no columns into 0/1 features and add them to a model.

## What to do now

1. Run `lesson.py`. In section 4 drop `temp_c` and `rain_mm` from
   `FEATURES` and see whether the error changes at all.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   lets e-bikes have their own slope.
3. Next, lesson 031: train/test splits, and watching overfitting happen
   on purpose. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

Build it, fit it, predict with it, then measure how wrong it is next to a
baseline. That loop is most of scikit-learn.
