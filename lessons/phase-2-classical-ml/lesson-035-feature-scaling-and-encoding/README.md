# Lesson 035 - Feature scaling and encoding

**Phase 2 - Classical machine learning** | Week 5, Day 5 | Friday 2026-10-09

> **Goal:** Standardise numbers and one-hot encode categories, and see
> which models care.

Time: about 45 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Every model so far has been handed columns that were already numbers,
and we never stopped to ask what those numbers were measured in. A ride's
duration runs into the tens of minutes, its distance a few kilometres,
the rain a few millimetres. To you those are different things. To a
model that compares rides by distance, they're just numbers, and the
biggest numbers win the argument.

And then there are the columns that aren't numbers at all, like the
station a ride started from. Today we fix both: put numbers on a common
ruler, and turn categories into something a model can use, without
inventing anything that isn't in the data.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` and the
same stratified train/test split as lessons 033 and 034. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-035-feature-scaling-and-encoding/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match.

## 1. Same rides, very different scales

```
the four numeric columns, training rides:
                   mean    std   min    max
    distance_km    3.20   2.17  0.41  16.52
    duration_min  15.76  10.29  2.60  76.50
    temp_c        17.64   3.39  9.10  22.80
    rain_mm        2.41   4.13  0.00  16.30
share of a typical squared distance:
    distance_km     0.03
    duration_min    0.76
    temp_c          0.08
    rain_mm         0.12
```

Picture each ride as a point with four coordinates. The usual way to say
how far apart two rides are is the one you learned at school: take the
gap in each column, square it, add them up, take the square root.

The catch is in the second table. Averaged over every pair of rides,
duration supplies 76% of that squared distance, and distance in km only
3%. Not because duration matters more, just because it's counted in
minutes and spreads widely. Measure it in hours and it would almost
vanish. The units, which you chose, are deciding what the model listens
to.

## 2. A model that measures distance gets fooled

```
k-NN, raw numbers                              -> 0.763
k-NN, temperature in milli-degrees             -> 0.725
k-NN, standardised                             -> 0.767
'always member' baseline                       -> 0.678
```

**k-nearest neighbours** (k-NN) is the model that cares most, so it's
our test subject. It's as simple as models get: to classify a ride, find
the 15 training rides closest to it and take a vote. Lesson 037 gives it
a proper look; today it's our canary.

Look at the middle line. We changed nothing about the rides, only wrote
temperature in thousandths of a degree, and accuracy fell four points.
Temperature now dwarfs everything else, so "closest" just means "same
weather". A model whose answers depend on your choice of units is a
model you can't trust.

## 3. Standardising by hand, then with StandardScaler

```
learned means (scaler.mean_)                   -> (3.2, 15.76, 17.64, 2.41)
learned spreads (scaler.scale_)                -> (2.17, 10.28, 3.39, 4.12)
by hand matches StandardScaler?                -> True
training means after scaling                   -> (0.0, 0.0, 0.0, 0.0)
training spreads after scaling                 -> (1.0, 1.0, 1.0, 1.0)
test means after scaling (not exactly 0)       -> (-0.1, -0.09, 0.01, -0.04)
```

**Standardising** rewrites every value as "how many standard deviations
from the mean":

```
z = (x - mean) / std
```

A 36-minute ride becomes about 2.0 (two spreads above a typical ride),
whatever units you started in. Every column ends up centred on 0 with a
spread of 1, and no column can shout over the others.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train[NUMERIC])   # learn mean and std
X_train = scaler.transform(train[NUMERIC])
X_test = scaler.transform(test[NUMERIC])        # same mean and std
```

This is the same `fit` / `transform` shape as a model's `fit` /
`predict`, and the same rule applies: **fit on the training rides
only**. The mean and spread are things the scaler *learned*, and
learning from the test set is peeking (lesson 031). That's why the test
means come out near 0 but not exactly: they're new rides, measured with
the training ruler, which is exactly what will happen to real rides
next month.

(`scale_` reads 10.28 where pandas said 10.29: pandas divides by n - 1,
scikit-learn by n. It doesn't matter here.)

If you'd rather squeeze each column into 0 to 1, `MinMaxScaler` does
that. It's touchier about outliers, since one 76-minute ride sets the
top of the range. Standardising is the usual default.

## 4. Turning categories into columns

```
label codes, 0 to 5  -> {'Market Square': 0, 'Old Mill': 1, 'Park Gate': 2,
                         'Riverside': 3, 'Station Road': 4, 'University': 5}
one-hot, first three test rides:
        start_station  Market Square  Old Mill  Park Gate  Riverside  Station Road  University
    34   Station Road              0         0          0          0             1           0
    383    University              0         0          0          0             0           1
    237    University              0         0          0          0             0           1
a station the encoder never saw                -> (0, 0, 0, 0, 0, 0)
pd.get_dummies gives columns                   -> ('Station Road', 'University')
```

A model needs numbers, so the tempting move is to number the stations 0
to 5. Don't. That tells the model Park Gate sits halfway between Old
Mill and Riverside, and that University is "five times" Old Mill.
Alphabetical order isn't a fact about stations.

**One-hot encoding** gives each category its own 0/1 column instead. A
ride from Station Road has a 1 in the Station Road column and 0
everywhere else. No order, no fake distances.

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
enc.fit(train[["start_station"]])
enc.transform(test[["start_station"]])
```

Two details make `OneHotEncoder` the right tool for models:

- It **remembers the training categories**, so the test set always gets
  the same six columns in the same order. `pd.get_dummies` only looks at
  the rows you give it: hand it three rides and you get two columns,
  which won't line up with anything. Great for exploring, risky for
  models.
- With `handle_unknown="ignore"`, a station it never saw (a new one
  opens, "Harbour") becomes all zeros instead of crashing your app.

Bike type, with only two values, is one-hot too. Lessons 033 and 034
already did it by hand with the `electric` column.

## 5. Do the new columns help?

```
share casual, by bike type:
    classic     0.32
    electric    0.33
5-fold CV accuracy on the training rides:
                                        mean  spread
    numbers only             logistic  0.773   0.028
                             k-NN      0.759   0.017
    numbers + station + bike logistic  0.772   0.017
                             k-NN      0.784   0.030
```

Not much, and that's a useful lesson in itself. Casual riders pick
electric and classic bikes at almost the same rate, and the stations are
similar (you'll check in the exercises), so logistic regression gains
nothing. k-NN moves up about two and a half points, roughly one
fold-spread: a hint worth following, not proof.

Encoding doesn't make a column useful. It only lets the model find out
whether it is.

One honest confession: this section fitted the scaler and encoder on
*all* the training rides before cross-validating, so each validation
fold had a hand in its own scaling. That's a small leak. Doing it
properly by hand is fiddly, which is exactly what lesson 036 is for.

## 6. Which models care?

```
test accuracy, raw vs standardised:
                           raw  standardised
    logistic regression  0.754         0.754
    k-NN                 0.763         0.767
```

| Model | Needs scaling? | Why |
|---|---|---|
| k-NN, and anything built on distances | Yes | Big-number columns dominate the distance. |
| Logistic and linear regression | Not for accuracy, yes in practice | Each weight stretches to fit its column's units, but scaling helps the solver, makes weights comparable, and regularisation (lesson 041) depends on it. |
| Decision trees, random forests, boosting | No | They split one column at a time, so units never meet. |

Here k-NN's raw units happened to be workable, but section 2 showed how
fragile that is. And every model needs categories encoded, unless it
says it handles them itself.

## What you can do now

- Explain why units can change a distance-based model's answers.
- Standardise columns by hand and with `StandardScaler`, fitting on the
  training data only.
- One-hot encode a category with `OneHotEncoder`, and say why it beats
  numbering the categories or `pd.get_dummies` for models.
- Say which models need scaling and which don't care.

## What to do now

1. Run `lesson.py`. In section 2, put distance in metres instead of
   temperature in milli-degrees and see what happens.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   checks the stations and tries a different scaler.
3. Next, lesson 036: pipelines, so the scaler and encoder are fitted in
   the right place every time without you thinking about it. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

A model only sees numbers. Your job is to make sure the numbers mean
what you think they mean.
