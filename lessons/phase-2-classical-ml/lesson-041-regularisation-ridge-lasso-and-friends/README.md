# Lesson 041 - Regularisation: Ridge, Lasso and friends

**Phase 2 - Classical machine learning** | Week 6, Day 3 | Wednesday 2026-10-14

> **Goal:** Trade a little bias for a lot less variance, and use Lasso
> for feature selection.

Time: about 45 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

A plain linear regression is a very trusting model. Hand it a column
and it will use it, bending its coefficients as far as it likes to fit
the training rows. With lots of rows and a few sensible columns, that's
fine. With few rows and lots of columns, it memorises noise.

**Regularisation** puts a price on big coefficients. The model now has
two jobs: fit the data *and* keep its weights modest. A weight only gets
big if it really earns it. That's the whole idea, and it shows up all
over machine learning: in Ridge and Lasso today, in the `C` of
logistic regression you've been using since lesson 033, in the
`l2_regularization` of yesterday's boosting, and later in neural
networks as "weight decay".

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` and the
same stratified train/test split as lessons 033 to 040. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-041-regularisation-ridge-lasso-and-friends/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match. It takes a couple of seconds.

For sections 1 to 4 we go back to lesson 030's regression task:
predict a ride's duration in minutes. To make overfitting easy to see,
we make life hard on purpose. We train on only **60** of the training
rides, and we add **40 columns of pure random noise**, standing in for
"extra data someone handed us" that has nothing to do with duration.
The test set is the usual 236 rides.

## 1. Too many columns, too few rides

```
training rides / columns after encoding        -> (60, 52)
plain linear regression, RMSE in minutes:
                             train  test
    real columns only         2.23  3.21
    real + 40 noise columns   1.26  5.80
largest |coefficient| on a noise column        -> 1.73
coefficient on distance_km (scaled)            -> 5.93
```

After one-hot encoding there are 52 columns (12 real, 40 noise) and
only 60 rides. With almost as many knobs as rows, the model can fit
nearly anything. Adding the noise columns *lowers* the training error,
from 2.23 to 1.26 minutes, because random columns always line up with
the training rows a little by luck. On new rides the error almost
doubles, to 5.80. One noise column even got a coefficient of 1.73
minutes per standard deviation, which is pure invention.

That's high variance in its plainest form: the answer depends on which
60 rides we happened to get.

## 2. Ridge: shrink every coefficient a little

```
Ridge on 60 rides, real + noise columns:
                train RMSE  test RMSE  size of weights  distance coef
    alpha=0           1.26       5.80             9.94           5.93
    alpha=0.1         1.26       5.59             9.40           5.84
    alpha=1           1.35       5.15             7.52           5.34
    alpha=10          1.89       5.53             5.26           4.02
    alpha=100         4.03       7.08             2.59           1.88
    alpha=1000        6.52       8.63             0.50           0.35
```

**Ridge** regression minimises the usual squared error *plus*
`alpha` times the sum of the squared coefficients. That second part is
called an **L2 penalty**. `alpha` is the price: zero means plain linear
regression, and bigger means weights cost more.

Read the table top to bottom:

- **Size of weights** (the length of the coefficient vector) falls
  steadily. Every coefficient shrinks towards zero, but none of them
  reaches it.
- **Train RMSE** rises. We're stopping the model from fitting the
  training rows perfectly. That's a little added **bias**.
- **Test RMSE** falls, from 5.80 to 5.15 at `alpha=1`. That's
  **variance** going away. Then, past the sweet spot, shrinking starts
  to crush the real signal too (look at the distance coefficient
  collapsing at `alpha=1000`), and both errors climb.

That U-shape in the test column is the bias-variance trade-off from
lesson 031, now with a dial you can turn. Notice that Ridge helps but
never gets near the 3.21 of the "real columns only" model. It shrinks
the 40 noise columns but keeps all of them in play.

## 3. Lasso: shrink some coefficients all the way to zero

```
Lasso on 60 rides (12 real columns, 40 noise):
                train RMSE  test RMSE  real kept  noise kept
    alpha=0.01        1.30       5.00         12          35
    alpha=0.1         1.74       3.64          6          28
    alpha=0.3         2.36       3.44          2          11
    alpha=1           3.29       4.18          1           1
    alpha=3           4.37       5.64          1           0
alpha=0.3, the columns that survived:
    distance_km           6.09
    bike_type_electric   -2.04
    noise_23              0.52
    noise_11              0.42
    noise_14              0.39
    ...
```

**Lasso** swaps the squared coefficients for their absolute values, an
**L1 penalty**. That small change has a big effect: instead of making
every weight a bit smaller, it pushes many weights to *exactly* zero.
A zero weight means the column is switched off, so Lasso does
**feature selection** as it fits.

At `alpha=0.3`, Lasso keeps distance and bike type (e-bikes are about
two minutes quicker), drops most of the noise, and gets the test error
down to 3.44, far better than Ridge managed and close to the model that
was only ever shown the real columns. The noise columns it kept have
tiny weights.

Two honest cautions:

- **It isn't a perfect detector.** Eleven noise columns survived at
  `alpha=0.3`. With 60 rides, some random columns really do look useful.
- **"Kept" means "useful for predicting", not "causes it".** When two
  columns carry the same information, Lasso tends to keep one and drop
  the other more or less at random. Lesson 024's warning about
  correlation still applies.

When does each win? If only a few columns matter (like here), Lasso
shines. If lots of columns each matter a little, Ridge usually does
better.

## 4. Scale first, then let cross-validation pick alpha

```
Lasso alpha=1, distance in km vs metres:
                        minutes per km  columns kept
    km, not scaled                3.71             2
    km, scaled                    3.43             2
    metres, not scaled            4.12             2
    metres, scaled                3.43             2
RidgeCV picked alpha                           -> 12.69
LassoCV picked alpha                           -> 0.356
test RMSE: RidgeCV / LassoCV                   -> (5.63, 3.47)
LassoCV noise columns kept (of 40)             -> 10
```

The penalty looks at the *size* of each coefficient, and size depends on
units. Measure distance in metres and its coefficient becomes a thousand
times smaller, so the penalty barely touches it. Same rides, same
`alpha`, different answer (3.71 vs 4.12 minutes per km). Put a
`StandardScaler` in the pipeline and the units stop mattering: both
versions agree on 3.43. Rule of thumb: **always scale before Ridge or
Lasso**, inside the pipeline so it's learned from training rows only
(lesson 036).

Choosing `alpha` by hand is the same chore as choosing `k` was.
`RidgeCV` and `LassoCV` take a list of alphas, cross-validate each on
the training rides, and refit with the winner. LassoCV lands on 0.356,
right next to our hand-picked 0.3, with a test RMSE of 3.47. The test
set played no part in choosing it.

## 5. Friends: Elastic Net, and the C in logistic regression

```
ElasticNet alpha=0.3 on 60 rides:
                  test RMSE  columns kept
    l1_ratio=0.1       5.63            46
    l1_ratio=0.5       4.74            35
    l1_ratio=0.9       3.67            15
casual vs member, all 940 training rides:
             L2 CV mean  L1 CV mean  L1 columns kept
    C=0.001       0.703       0.676                0
    C=0.01        0.765       0.779                2
    C=0.1         0.787       0.784                4
    C=1           0.790       0.789               10
    C=100         0.793       0.793               13
```

**Elastic Net** mixes the two penalties. `l1_ratio` is the share that's
Lasso-style: near 0 it behaves like Ridge, near 1 like Lasso. It's a
good choice when columns come in correlated groups, because it tends to
keep or drop a group together instead of picking one member at random.
On our noise-heavy data, the more Lasso in the mix, the better.

**Logistic regression has been regularised all along.** Its `C` is
the *inverse* of alpha: small `C` means a strong penalty, big `C` a
weak one. The default is `C=1` with an L2 penalty. With an L1 penalty
you get Lasso-style selection for classification. At `C=0.001` it
switches off every column and just predicts "member" for everyone
(0.676, which is simply the share of members among the training
rides).
At `C=0.01` it keeps two columns and is already within a point of the
best.

Here there's little to rescue: 940 rides and 13 columns is a
comfortable ratio, so anything from `C=0.1` up scores about the same.
Regularisation matters most when columns are many and rows are few.

> **A note on versions:** scikit-learn 1.8 renamed the switch for an L1
> logistic regression from `penalty="l1"` to `l1_ratio=1`. The
> `l1_logistic` helper in `lesson.py` uses whichever your version
> understands, so the script runs on anything the course's
> `requirements.txt` allows.

## What you can do now

- Explain regularisation as a price on big coefficients that trades a
  little bias for less variance.
- Use `Ridge` to shrink every weight, and `Lasso` to switch useless
  columns off.
- Scale inside a pipeline before penalising, and let `RidgeCV` or
  `LassoCV` choose `alpha` with cross-validation.
- Recognise the same idea in `ElasticNet` and in logistic regression's
  `C`.

## What to do now

1. Run `lesson.py`. Near the top, change `SMALL_N` from 60 to 400 and
   watch how much less regularisation matters with more rides.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   gives Lasso more data and then all of it.
3. Next, lesson 042: hyperparameter search, so you stop picking `alpha`,
   `k` and friends by hand. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

A model that pays nothing for confidence will be confident about
noise. Put a price on big weights, let cross-validation set the price,
and keep only what earns its place.
