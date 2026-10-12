# Lesson 037 - k-Nearest Neighbours and the curse of dimensionality

**Phase 2 - Classical machine learning** | Week 6, Day 1 | Monday 2026-10-12

> **Goal:** Build intuition for distance-based models and why more
> features aren't always better.

Time: about 45 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Lessons 035 and 036 used k-nearest neighbours as a canary for scaling
and as the last step of a pipeline. Today it gets the proper look. The
idea is almost embarrassingly simple: to classify a new ride, find the
training rides closest to it and take a vote. No formula is fitted; the
data itself is the model.

That simplicity is also the trap. "Closest" only means something when
the axes are fair and few. Scale the columns wrong, or pile on noisy
ones, and every neighbour starts to look equally far away.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` and the
same stratified train/test split as lessons 033 to 036. Full script:

```bash
python lessons/phase-2-classical-ml/lesson-037-k-nearest-neighbours-and-the-curse-of-dimensionality/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match.

## 1. Vote of the nearest rides

```
one test ride  -> casual, 20.4 min, 3.8 km, weekday, Station Road
its 5 nearest training rides: four members, one casual, all from
    Station Road, durations within about two minutes
votes among the five                           -> {'member': 4, 'casual': 1}
k-NN (k=15) prediction for that ride           -> 0  (member)
'always member' baseline on the test set       -> 0.678
k-NN (k=15) test accuracy                      -> 0.737
```

We keep predicting casual vs member, the same target as lessons 033 to
036, because it's a clear two-class question and the weekend pattern
is strong. k-NN stores every training ride. For a new one, it measures
distance in the preprocessed feature space (standardised numbers,
one-hot stations and bike types, the weekend flag untouched — the
pipeline from lesson 036), picks the `k` closest, and majority-votes.

The test ride above really is casual, but four of its five nearest
neighbours are members with almost the same duration and station. With
`k=15` the model still calls it a member. Neighbours can be wrong; the
hope is that across hundreds of rides the votes average out.

## 2. How k changes the answer

```
accuracy as k grows:
           CV mean  CV spread   test
    k=1      0.702      0.017  0.720
    k=5      0.761      0.022  0.695
    k=15     0.781      0.043  0.737
    k=31     0.787      0.028  0.733
    k=61     0.769      0.016  0.742
    k=121    0.757      0.017  0.720
```

`k=1` copies the single nearest neighbour and overfits the noise.
`k=121` averages over more than a tenth of the training set and blurs
toward "member", the majority class. Cross-validation likes 15 to 31
here — the same range lesson 036's exercise found. The test column
wiggles; trust the CV mean, and leave the test set alone until you've
picked `k`.

## 3. Scaling still matters (a pipeline reminder)

```
k-NN on raw numbers only                       -> 0.763
k-NN on standardised numbers only              -> 0.767
full pipeline (scale + one-hot + weekend)      -> 0.737
```

Same story as lesson 035: duration in minutes still shouts over
distance in km when the columns are raw. Standardising levels the
field a little. The full pipeline's *test* score here is the bounce
you already saw in lesson 036 (its CV mean was 0.781). The habit that
matters is unchanged: put the scaler inside the pipeline so every CV
fold re-fits it on its own training part.

## 4. The curse: more features, less meaningful neighbours

```
CV accuracy, 2 useful columns (duration + weekend)  -> 0.766
add pure noise columns:
              features  CV mean   test
    +0 noise         2    0.766  0.737
    +5 noise         7    0.765  0.737
    +20 noise       22    0.733  0.720
    +50 noise       52    0.716  0.686
    +100 noise     102    0.700  0.686

distances in pure noise (400 points):
         mean distance  nearest / farthest
    d=2            1.34               0.01
    d=10           3.78               0.20
    d=50          10.40               0.64
    d=200         20.14               0.78
```

Start with two columns that actually help, then glue on columns of
random noise. CV accuracy falls as the noise piles up: the useful
signal is still there, but "closest" is increasingly decided by
nonsense dimensions.

The second table is the geometric version of that idea. In two
dimensions, the nearest of 400 random points is tiny compared with the
farthest. By 200 dimensions, nearest and farthest are almost the same
ratio. When every point is roughly equally far, a distance-based vote
stops meaning much. That's the **curse of dimensionality** in one
sentence: more features are not always better for neighbours.

## 5. When k-NN is a good idea

```
test accuracy (k=15 pipeline)                  -> 0.737
training rides remembered                      -> 940
```

Reach for k-NN when similar inputs really do have similar labels, the
feature space is low-dimensional and well scaled, and you can afford
to store the training set. It struggles with many noisy columns, with
huge datasets (every prediction has to find neighbours), and when you
need a compact explanation rather than a vote. Tomorrow's decision
trees give you that readable explanation.

## What you can do now

- Explain k-NN as a vote of the nearest training examples.
- Say how small and large `k` behave, and pick `k` with CV.
- Put a scaler in a pipeline before k-NN, and say why.
- Demonstrate the curse of dimensionality with noisy features or a
  distance-concentration sketch.

## What to do now

1. Run `lesson.py`. In section 2, try `k=3` and `k=200` and watch CV.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   swaps in a regressor and checks distance concentration.
3. Next, lesson 038: decision trees you can read. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Closest only means something when the axes are fair and few.
