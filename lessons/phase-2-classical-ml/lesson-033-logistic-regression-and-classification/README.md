# Lesson 033 - Logistic regression and classification

**Phase 2 - Classical machine learning** | Week 5, Day 4 | Thursday 2026-10-08

> **Goal:** Predict categories, read probabilities, and understand the
> decision boundary.

Time: about 50 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Every model so far has answered "how many minutes?". Plenty of real
questions have a different shape: spam or not, fraud or not, will this
customer leave. The answer is a **category**, and predicting one is called
**classification**.

Today's question: given a finished ride, was the rider a **casual** rider
or a **member**? It's a useful one. If the app can tell, it can show
casual riders a "become a member" offer at the end of their trip.

The workhorse for this is **logistic regression**. Despite the name, it
classifies. It's a straight line from lesson 030 with one extra step that
turns the line's output into a probability.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` as lessons
029 to 032. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-033-logistic-regression-and-classification/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match.

## 1. From numbers to categories

```
rides that were casual riders (share)          -> 0.324
train rides, test rides                        -> (940, 236)
casual share in train / test                   -> (0.324, 0.322)
baseline: always say 'member', accuracy        -> 0.678
```

We make a target column, `casual`, that is 1 for a casual rider and 0 for
a member. The 1 is called the **positive class**: the thing we're looking
for. It's not a judgement, just a label.

Two habits carry over from lessons 031 and 032:

- **Lock away a test set.** Same 80/20 split, but with
  `stratify=rides["casual"]`. That keeps the casual share the same in
  both halves (32.4% and 32.2%), so a lucky split can't hand the test set
  an odd mix.
- **Start with a baseline.** `DummyClassifier(strategy="most_frequent")`
  always says "member". Because about two thirds of rides are members, it
  scores 67.8% **accuracy** (the share of guesses that are right) without
  looking at a single feature. That's the bar to clear.

## 2. Why not a straight line?

```
straight line on 0/1 labels, by distance (km):
    0.2    -0.01
    0.5     0.02
    2.0     0.19
    4.0     0.41
    8.0     0.86
    12.0    1.30
```

You could fit `LinearRegression` to the 0s and 1s and read the output as
a probability. It even sort of works in the middle. But a 12 km ride gets
a "probability" of 1.30, and the shortest rides go negative. A line keeps
going forever in both directions, and a probability has to stay between
0 and 1.

## 3. The sigmoid squash

```
sigmoid of:
    -6    0.002
    -2    0.119
    -1    0.269
     0    0.500
     1    0.731
     2    0.881
     6    0.998
```

The fix is to keep the line but squash its output. The **sigmoid**
function does exactly that:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Feed it any number and you get something between 0 and 1. Big negative
numbers end up near 0, big positive ones near 1, and 0 lands exactly on
0.5. So logistic regression is two steps:

1. Compute a score with a straight line, `z = w * x + b`, just like
   lesson 030.
2. Squash it: `probability = sigmoid(z)`.

Training finds the `w` and `b` that make the probabilities match the
real 0s and 1s as closely as possible. (The details of "as closely as
possible" use a measure called log loss; you'll meet it properly in
Phase 3. For now, scikit-learn handles it.)

## 4. One feature: distance

```
weight w, intercept b                          -> (0.731, -3.132)
P(casual) by distance (km):
    1    0.083
    2    0.159
    3    0.281
    4    0.449
    6    0.778
    8    0.938
same as sigmoid(w*km + b) by hand?             -> True
distance where P(casual) = 0.5 (km)            -> 4.28
```

Same `fit` and `predict` pattern as every scikit-learn model:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(train[["distance_km"]], train["casual"])
model.predict_proba(km)[:, 1]   # P(casual) for each row
model.predict(km)               # 0 or 1
```

Two methods, two kinds of answer:

- `predict_proba` gives one column per class, member first then casual
  (the order of `model.classes_`). `[:, 1]` picks P(casual).
- `predict` gives the class: 1 if P(casual) is at least 0.5, else 0.

The probabilities curve the way you'd hope: an 8% chance for a 1 km ride,
94% for 8 km, and they never escape 0 to 1. The script also checks that
they really are `sigmoid(w * km + b)`. No magic.

Where does the answer flip? Exactly where the line's score is zero, so
`w * km + b = 0`, which gives `km = -b / w`, about 4.28 km. Shorter rides
get "member", longer ones "casual".

## 5. More features, and reading the coefficients

```
coefficients (one per feature):
                  coef  odds x
    distance_km   0.13    1.14
    duration_min  0.14    1.15
    electric      0.40    1.48
    weekend       1.20    3.33
five test rides:
         distance_km  duration_min  electric  weekend  p_casual  predicted  actual
    34          3.80          20.4         0        0      0.38          0       1
    383         1.87           8.4         1        0      0.12          0       0
    237         2.84          12.4         0        1      0.37          0       1
    192         2.80          11.9         1        0      0.20          0       1
    177         0.83           4.0         1        0      0.06          0       0
accuracy on the test rides                     -> 0.792
```

Now four features: distance, duration, e-bike or not, and a new
`weekend` column (1 for Saturday and Sunday, from `started_at`). We pass
`max_iter=1000` so the fitting has room to finish; with unscaled
features it sometimes needs more than the default steps.

Reading the coefficients:

- **The sign is the direction.** Positive pushes towards casual. All
  four are positive: longer, slower, e-bike and weekend rides all lean
  casual.
- **`odds x` is `exp(coef)`.** The **odds** are P(casual) / P(member).
  One more unit of a feature multiplies the odds by this number. A
  weekend ride has 3.3 times the odds of being casual, everything else
  equal. That's multiplying the odds, not the probability, which
  exercise 2 untangles.
- **Don't rank them by size yet.** One extra km and one extra minute are
  different-sized steps, so 0.13 and 0.14 aren't directly comparable.
  Lesson 035 puts features on the same scale.

The test accuracy is 79.2%, well clear of the 67.8% baseline. But look at
those five rides: three were casual riders, and the model called all
three "member". Hold that thought. It's lesson 034.

## 6. The decision boundary

```
weekday: casual beyond (km)                    -> 4.68
weekend: casual beyond (km)                    -> 3.02
```

The **decision boundary** is where the model is exactly on the fence:
the line's score is zero and P(casual) is 0.5. With one feature it's a
single point (4.28 km). With two features, distance and weekend, it's a
line through the space of features, and every ride on one side gets
"casual".

Because `weekend` is just 0 or 1, the line shows up as two cut-offs: a
weekday ride needs to be longer than 4.68 km before the model calls it
casual, a weekend ride only 3.02 km. That matches what you'd guess about
who's riding on a Saturday.

The boundary is always straight (flat, in more dimensions), which is
what makes logistic regression a **linear classifier**. If the real
boundary curves, it can't follow. Lessons 037 to 040 bring models that
can.

## What you can do now

- Say what classification is, and why the positive class is just a label.
- Explain why a straight line makes a bad probability, and what the
  sigmoid fixes.
- Fit `LogisticRegression` and use `predict_proba` and `predict`.
- Read a coefficient's sign, and turn it into an odds multiplier with
  `exp`.
- Find the decision boundary and say why it's straight.
- Beat a `DummyClassifier` baseline, on a stratified test set.

## What to do now

1. Run `lesson.py`. In section 6, swap `weekend` for `electric` and see
   how the cut-offs change for e-bikes.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   tests a feature that sounds useful.
3. Next, lesson 034: what "79% accurate" actually hides, and the metrics
   that tell you. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

A line, squashed into a probability, cut at 0.5. That's all logistic
regression is, and it's still one of the most used models in the world.
