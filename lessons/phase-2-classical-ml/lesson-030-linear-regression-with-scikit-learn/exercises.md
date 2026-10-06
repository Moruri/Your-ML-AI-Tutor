# Lesson 030 - Exercises

Three quick checks, a hands-on feature, and an optional prediction.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-030-linear-regression-with-scikit-learn/lesson.py my_lesson_030.py
python my_lesson_030.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-030-linear-regression-with-scikit-learn").resolve()
```

`load_rides`, `FEATURES` and `RIDES_CSV` are there to reuse.

---

## 1. The shape error

What goes wrong here, and what's the one-character family of fixes?

```python
model = LinearRegression().fit(rides["distance_km"], rides["duration_min"])
```

<details>
<summary>Check yourself</summary>

`rides["distance_km"]` is a 1-D Series, and scikit-learn wants `X` to be
2-D (rows by features), so it raises a `ValueError` about expecting a 2D
array. Use double brackets, `rides[["distance_km"]]`, to get a one-column
DataFrame. `y` stays single-bracketed.

</details>

## 2. Read a coefficient

The five-feature model has `electric` = -4.46. Finish the sentence:
"Compared with a classic bike, ..."

<details>
<summary>Check yourself</summary>

"... an e-bike ride of the same distance, by the same kind of rider, in
the same weather, takes about 4.5 minutes less on average in this data."
The "same distance, same rider, same weather" part is what "holding the
other features fixed" means, and it's what makes a coefficient different
from a simple group average.

</details>

## 3. MAE or RMSE?

Two models predict delivery times. Model A: MAE 4.0, RMSE 4.5. Model B:
MAE 3.5, RMSE 9.0. Which would you rather ship, and what would you check?

<details>
<summary>Check yourself</summary>

B is better on a typical order, but its RMSE is far above its MAE, so it
makes some very big misses. A spreads its errors more evenly. If a badly
late delivery is what upsets customers, A may be the better choice. Check
by looking at B's biggest misses (like section 6) and asking whether
they share a pattern you could fix with a feature.

</details>

## 4. Hands-on: let e-bikes have their own slope

In lesson 029's exercise, e-bikes had a smaller slope (minutes per km)
than classic bikes, not just a smaller intercept. The five-feature model
can only shift e-bikes down by a fixed 4.46 minutes. Fix that.

**a)** Add a column `e_km = electric * distance_km`. It's 0 for classic
bikes and the distance for e-bikes.

**b)** Fit `LinearRegression` on `distance_km`, `electric`, `casual` and
`e_km`. What are the coefficients for `distance_km` and `e_km`?

**c)** What is the slope for an e-bike, in minutes per km?

**d)** Compare MAE and RMSE with the five-feature model (2.20 and 3.21).

Hints:

- `rides["e_km"] = rides["electric"] * rides["distance_km"]`
- `pd.Series(model.coef_, index=cols)` makes the coefficients readable.

<details>
<summary>Expected results</summary>

```
b) distance_km about 4.83, e_km about -1.12
c) about 4.83 - 1.12 = 3.71 minutes per km for an e-bike
d) MAE about 1.99, RMSE about 3.00 (better than 2.20 and 3.21,
   with one fewer column than the five-feature model)
```

</details>

<details>
<summary>One way to write it</summary>

```python
rides = load_rides(RIDES_CSV)
rides["e_km"] = rides["electric"] * rides["distance_km"]
cols = ["distance_km", "electric", "casual", "e_km"]
y = rides["duration_min"]

model = LinearRegression().fit(rides[cols], y)
pred = model.predict(rides[cols])
print(pd.Series(model.coef_, index=cols).round(2))
print("MAE", round(mean_absolute_error(y, pred), 2))
print("RMSE", round(root_mean_squared_error(y, pred), 2))
```

A column built by multiplying two others is called an **interaction**: it
lets the effect of one feature (distance) depend on another (bike type).
A linear model can only draw straight lines, but you get to choose which
columns it draws them through. The rain and temperature columns, by the
way, were adding nothing, which is why this does better with fewer.

</details>

## 5. (Optional) Price a ride

Using the five-feature model from section 4, predict the minutes for a
5 km ride by a member on an e-bike, at 15°C with no rain.

<details>
<summary>Check yourself</summary>

About **20.5 minutes**. By hand: 4.36 x 5 - 4.46 + 0 - 0.01 x 15 - 0 +
3.32 is roughly 20.5. In code, build a one-row DataFrame with the same
five columns, in the same order as `FEATURES`, and call `model.predict`
on it. Getting the columns right is most of the work.

</details>

---

That's linear regression in scikit-learn. If one thing sticks: *build,
fit, predict, then measure the error next to a baseline, and read the
coefficients as "holding everything else fixed".*
