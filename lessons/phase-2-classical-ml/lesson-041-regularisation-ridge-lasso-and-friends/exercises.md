# Lesson 041 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-041-regularisation-ridge-lasso-and-friends/lesson.py my_lesson_041.py
python my_lesson_041.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-041-regularisation-ridge-lasso-and-friends").resolve()
```

`load_rides`, `split`, `make_reg`, `fit_report`, `coefs`, `NOISE` and
`DURATION` are there to reuse. `make_reg(model)` wraps any regressor in
the scaling and encoding pipeline, `fit_report(model, rides, test)`
fits it and returns `(train RMSE, test RMSE)`, and `coefs(model)` gives
the fitted coefficients by column name.

---

## 1. Ridge or Lasso?

In one sentence each: what does Ridge do to the coefficients, and what
does Lasso do to them?

<details>
<summary>Check yourself</summary>

Ridge (L2) shrinks every coefficient towards zero but leaves them all
in the model. Lasso (L1) shrinks them too, and pushes the least useful
ones to exactly zero, which removes those columns.

</details>

## 2. Which way does the dial turn?

You raise Ridge's `alpha` from 1 to 100. What happens to the training
error? And in logistic regression, does `C=100` mean more or less
regularisation than `C=1`?

<details>
<summary>Check yourself</summary>

Training error goes **up**: the model is no longer allowed to bend as
far to fit the training rows (section 2 went from 1.35 to 4.03
minutes). Test error may go up or down depending on which side of the
sweet spot you land.

`C` is the inverse of alpha, so `C=100` means **less**
regularisation. Small `C`, strong penalty.

</details>

## 3. Why scale first?

A friend fits a Lasso on unscaled data where income is in shillings
and age is in years. Why might the result be unfair to one of them?

<details>
<summary>Check yourself</summary>

The penalty is on coefficient size, and size depends on units. Income
in shillings has huge values, so its coefficient is tiny and the
penalty barely touches it. Age has small values and a bigger
coefficient, so it gets penalised much harder and may be dropped for
no good reason. Scaling puts every column on the same footing first.

</details>

## 4. Hands-on: give Lasso more data

**a)** Make `mid = train.sample(240, random_state=0)`, four times as
many rides as the lesson used. Fit `make_reg(Lasso(alpha=0.3,
max_iter=50_000))` with `fit_report`. What are the train and test RMSE,
and how many real and noise columns are kept?

**b)** Same 240 rides, `make_reg(Ridge(alpha=10))`. Train and test
RMSE?

**c)** Now use all 940 training rides with `make_reg(LassoCV(alphas=np.logspace(-2, 1, 30), cv=5, max_iter=50_000, random_state=0))`.
Which `alpha` does it pick, what's the test RMSE, and which real
columns survive?

<details>
<summary>Expected results</summary>

```
a) train RMSE 3.76, test RMSE 2.91; 2 real columns kept, 9 noise
b) train RMSE 3.47, test RMSE 3.19
c) alpha 0.137, test RMSE 2.90; 9 noise columns kept
   real: distance_km 9.36, bike_type_electric -3.83,
         casual 0.58, rain_mm -0.13
```

(a) With four times the rides, Lasso beats the lesson's best (3.44)
and even the "real columns only" model on 60 rides (3.21). More data
is the best regulariser there is.

(b) Ridge also improves a lot with more rides, but Lasso still edges
it on this data, because only a few columns truly matter.

(c) With all 940 rides, CV picks a lighter penalty (0.137 instead of
0.356): there's less variance to fight, so less shrinking is needed.
Casual riders come out about half a minute slower and rain a touch
quicker. Nine noise columns still sneak in with small weights, which
is a good reminder to read Lasso's choices with suspicion.

</details>

<details>
<summary>One way to write it</summary>

```python
import numpy as np
from sklearn.linear_model import Lasso, LassoCV, Ridge

rides = load_rides(RIDES_CSV)
train, test = split(rides)
mid = train.sample(240, random_state=0)

lasso = make_reg(Lasso(alpha=0.3, max_iter=50_000))
print(fit_report(lasso, mid, test))
c = coefs(lasso)
print((c.drop(NOISE) != 0).sum(), (c[NOISE] != 0).sum())

print(fit_report(make_reg(Ridge(alpha=10)), mid, test))

cv = make_reg(LassoCV(alphas=np.logspace(-2, 1, 30), cv=5, max_iter=50_000,
                      random_state=0))
print(fit_report(cv, train, test), round(cv.named_steps["model"].alpha_, 3))
c = coefs(cv)
real = c.drop(NOISE)
print(real[real != 0].round(2), (c[NOISE] != 0).sum())
```

</details>

## 5. (Optional) How stable is Lasso's choice?

Fit `make_reg(Lasso(alpha=0.3, max_iter=50_000))` on five different
60-ride samples, `train.sample(60, random_state=rs)` for `rs` from 1
to 5. Each time, list the real columns it keeps. Before you run: will
the list be the same every time?

<details>
<summary>Check yourself</summary>

No. Distance and the e-bike flag survive every time, because they
really drive duration. The rest wobble: `casual` in some samples,
`temp_c` or `rain_mm` in others, and between 9 and 17 noise columns
each time. With few rows, *which* weak columns Lasso keeps is partly
luck. If you need to know which features matter, refit on resamples
like this and trust only the ones that keep showing up.

</details>

---

That's regularisation. If one thing sticks: *a penalty on big weights
buys you less variance for a little bias, and Lasso turns that penalty
into feature selection.*
