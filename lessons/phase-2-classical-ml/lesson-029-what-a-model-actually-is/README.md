# Lesson 029 - What a model actually is

**Phase 2 - Classical machine learning** | Week 5, Day 2 | Tuesday 2026-10-06

> **Goal:** Define features, targets, parameters and loss, then fit a line
> by hand before using a library, so "training a model" stops being a
> black box and becomes something you could do with a pencil.

Time: about 50 minutes. Needs the venv from lesson 013 (`pandas`, `numpy`).
No scikit-learn yet; that's tomorrow's half of today.

---

Welcome to Phase 2. Everything so far has been about *describing* data:
what's in it, how it's spread, what goes with what. Machine learning asks
for one more step: *given what I know about a new example, predict
something I don't know yet.*

The bike-share scheme from lesson 028 has a practical version of that
question. When someone unlocks a bike and types in where they're going,
can we tell them roughly how long it'll take? We know the distance. We want
the minutes.

The word "model" gets used as if it were mysterious. It isn't. By the end
of this lesson you'll have built one from nothing but a formula, a score
and a loop, and you'll see that every fancy model later in the course is
the same four parts in a bigger costume.

## How to follow along

Venv active, `python` in the repo root. The data is `rides_clean.csv`, the
September rides after lesson 028's cleaning (spellings fixed, broken rides
dropped, weather joined on). Full script:

```bash
python lessons/phase-2-classical-ml/lesson-029-what-a-model-actually-is/lesson.py
```

It writes no files, and nothing is random, so your numbers will match.

## 1. Features, target, rows

```
rides (rows) / columns                         -> (1176, 9)
feature x = distance_km, mean                  -> 3.16
target  y = duration_min, mean                 -> 15.58
```

Three words you'll use every day from now on:

- **Target** (often `y`): the thing you want to predict. Here,
  `duration_min`.
- **Features** (often `X`): what you know *before* you need the
  prediction. Here, just `distance_km` for now.
- **Rows** (also called examples or samples): the past cases the model
  learns from. We have 1,176 rides.

That "before" matters. A feature has to be something you'd actually have
at prediction time. Ride rating is known only after the ride, so it could
never be a feature for predicting duration, however well it correlates.
Using information you wouldn't really have is called **leakage**, and it's
the most common way models look brilliant in a notebook and fail in
real use.

## 2. The laziest model: always guess the average

```
prediction for every ride (min)                -> 15.58
baseline MAE (min)                             -> 7.25
```

Our first model ignores the features entirely and predicts 15.6 minutes
for every ride. On average it's off by about 7 minutes.

It looks silly, but it's the most useful number in the lesson. It's the
**baseline**: the score you get without trying. A model that can't beat
"guess the average" hasn't learned anything, and you'd be surprised how
often a complicated one doesn't. Compute the baseline first, every time.

**MAE**, mean absolute error, is the average size of a miss: take each
`actual - predicted`, drop the sign, average. It's in the target's own
units (minutes), which makes it easy to explain to anyone.

## 3. A model is a function with knobs

```
slope=3.0, intercept=0.0: MAE / MSE            -> (6.12, 61.4)
slope=5.0, intercept=0.0: MAE / MSE            -> (2.84, 16.8)
slope=4.0, intercept=2.0: MAE / MSE            -> (2.72, 16.9)
```

Here's an actual model:

```python
def predict(x, slope, intercept):
    return slope * x + intercept
```

*Minutes = slope x kilometres + intercept.* The shape is fixed; the
**parameters** (`slope` and `intercept`) are the knobs. Different knob
settings give different predictions, and a **loss** function scores how
wrong each setting is.

Two losses appear here. MAE you've met. **MSE**, mean squared error,
squares each miss before averaging. A 10-minute miss counts 100, a
1-minute miss counts 1, so MSE cares a lot about big mistakes.

Look closely: the third setting has the better MAE but the slightly worse
MSE. The two losses disagree about which line is "best", because they
care about different things. Choosing a loss is choosing what kind of
mistake hurts. For the rest of the lesson we use MSE, because it's the
one with a neat exact answer (section 5).

**Training** a model means one thing: find the parameter values that make
the loss as small as possible on the data you have.

## 4. Training by brute force: try lots of knob settings

```
settings tried                                 -> 10201
best slope (min per km)                        -> 4.4
best intercept (min)                           -> 1.7
best MSE                                       -> 15.2
```

The most honest training method there is: try every slope from 2 to 7 in
steps of 0.05, every intercept from -5 to 5 in steps of 0.1, and keep the
pair with the lowest MSE. Ten thousand tries, a fraction of a second.

The answer reads naturally: about 4.4 minutes per kilometre, plus about
1.7 minutes that every ride costs regardless of distance (unlocking,
finding a dock).

Brute force breaks down fast, though. Two knobs with 100 values each is
10,000 tries. Twenty knobs is 100²⁰, which no computer will ever finish.
So real training uses either maths (next section) or a smarter search
that follows the slope of the loss downhill. You'll write that search,
**gradient descent**, yourself in Phase 3.

## 5. The exact answer: least squares in two lines

```
least-squares slope                            -> 4.41
least-squares intercept                        -> 1.667
grid search got                                -> (4.4, 1.7)
np.polyfit(x, y, 1) agrees                     -> (4.41, 1.667)
```

For a straight line with MSE as the loss, someone worked out the minimum
with calculus a couple of centuries ago:

```python
slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
intercept = y.mean() - slope * x.mean()
```

You don't need to derive it. Read it as "how much x and y move together,
divided by how much x moves on its own", which is a close cousin of the
correlation from lesson 024. The intercept just makes the line pass
through the point (average distance, average duration).

This is **least squares**, and it lands exactly where the grid search
found its best pair. `np.polyfit` agrees too. When you call a library
tomorrow, this is what it's doing.

## 6. Using the model, and what it gets wrong

```
predicted minutes for a 1 km ride              -> 6.1
predicted minutes for a 3 km ride              -> 14.9
predicted minutes for a 8 km ride              -> 36.9
model MAE (min)                                -> 2.74
baseline MAE from section 2 (min)              -> 7.25
average error (actual - predicted) by group:
    bike_type  rider_type
    classic    casual        2.96
               member        0.99
    electric   casual       -3.27
               member       -2.63
```

The line cuts the typical miss from about 7 minutes to under 3. That's a
real model, earning its keep.

Now the habit that separates careful people from everyone else: **look at
the errors, not just the score.** Grouped by bike type, the misses aren't
random. The model is about 3 minutes too slow for e-bikes and 1 to 3
minutes too quick for classic bikes. Of course it is: it has never been
told which kind of bike it's looking at.

Errors with a pattern are a signal that the model is missing a feature.
Lesson 030 hands it that feature, and lets scikit-learn turn the knobs.

## What you can do now

- Name the features, target, parameters and loss of a model.
- Explain why a feature must be known at prediction time.
- Compute a baseline and say why every model needs one.
- Calculate MAE and MSE, and say which one punishes big misses.
- Describe training as "find the parameters that minimise the loss".
- Fit a line by grid search and by least squares, and check they agree.
- Group errors to spot what a model is missing.

## What to do now

1. Run `lesson.py`. In section 4 make the grid steps ten times coarser
   (0.5 and 1.0) and see how far the "best" answer drifts.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   gives each bike type its own line.
3. Next, lesson 030: the same model in scikit-learn, with more features
   and proper error metrics. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

A model is a formula with knobs, a score for how wrong it is, and a way to
turn the knobs. Everything else is detail.
