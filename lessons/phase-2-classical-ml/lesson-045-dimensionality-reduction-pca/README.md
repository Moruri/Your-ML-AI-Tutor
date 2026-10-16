# Lesson 045 - Dimensionality reduction: PCA

**Phase 2 - Classical machine learning** | Week 6, Day 5 | Friday 2026-10-16

> **Goal:** Compress features, plot high-dimensional data in 2D, and read
> explained variance.

Time: about 40 minutes. Needs the venv from lesson 013 with scikit-learn
and matplotlib installed (`pip install -r requirements.txt` covers both).

---

Yesterday k-means grouped rides using six columns. Six is already more
than a screen can show: a scatter plot has two axes, and our eyes give up
somewhere around three. Yet we often want to *see* the data, or hand a
model fewer, tidier inputs.

**Principal Component Analysis (PCA)** finds new axes, called
**components**, that line up with the directions where the data spreads
out most. The first component carries the most spread, the second the
most of what's left, and so on. Keep the first two and you can draw
every ride on one flat picture. Keep a few more and you have a compressed
version of the table that loses surprisingly little.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv`. Full
script:

```bash
python lessons/phase-2-classical-ml/lesson-045-dimensionality-reduction-pca/lesson.py
```

It writes one picture, `output/pca_rides.png`, next to the script. Every
random step has a fixed seed, so your numbers will match. It takes a
couple of seconds.

## 1. Seven columns, two eyes

```
rides / features                               -> (1176, 7)
feature columns                                -> ['distance_km', 'duration_min', 'temp_c', 'rain_mm', 'weekend', 'hour', 'electric']
correlations between the features:
                  distance_km  duration_min  temp_c  rain_mm  weekend  hour  electric
    distance_km          1.00          0.92   -0.01     0.02     0.14  0.03      0.08
    duration_min         0.92          1.00   -0.01     0.00     0.13  0.03     -0.14
    temp_c              -0.01         -0.01    1.00    -0.52    -0.06  0.01      0.04
    rain_mm              0.02          0.00   -0.52     1.00     0.05 -0.03     -0.01
    ...
```

Seven traits of each ride: the four numbers you know, plus a weekend
flag, the start hour and an electric-bike flag. Rider type stays out of
the inputs; we'll use it later to colour the picture and as a label.

Look at the correlations. Distance and duration move together at 0.92:
two columns telling almost one story. Temperature and rain pull against
each other at -0.52. That overlap is exactly what PCA notices and folds
into single new axes.

## 2. Scale, fit PCA, read explained variance

```
explained variance ratio by component:
         share  cumulative
    PC1   0.28        0.28
    PC2   0.22        0.50
    PC3   0.15        0.65
    PC4   0.14        0.79
    PC5   0.14        0.92
    PC6   0.07        0.99
    PC7   0.01        1.00
components needed for 80% of the spread        -> 5
PC1 share WITHOUT scaling                      -> 0.704
```

The **explained variance ratio** says how much of the total spread each
component carries. PC1 holds 28%, the first two together half, and five
of seven get you past 80%. Look at PC7: just 1%. That's the
distance-versus-duration overlap being squeezed out. Seven columns really
hold about six columns' worth of information.

The usual way to pick how many to keep: read the cumulative column and
stop where it crosses a level you're happy with (80%, 90%, 95%), or where
the share drops off a cliff.

Now the trap. Without scaling, PC1 claims 70% of the spread, but only
because duration is measured in minutes with a standard deviation of
about 10, which dwarfs a 0/1 flag. PCA chases variance, and unscaled
variance is mostly a story about units. Same habit as k-means: **scale
first**.

## 3. What the new axes mean

```
loadings (how much each feature feeds each component):
                   PC1   PC2   PC3
    distance_km   0.69  0.07  0.15
    duration_min  0.69  0.07 -0.07
    temp_c       -0.05  0.70 -0.00
    rain_mm       0.05 -0.70  0.06
    weekend       0.20 -0.11  0.00
    hour          0.05  0.06 -0.17
    electric     -0.05  0.05  0.97
```

Each component is a weighted blend of the original features, and these
weights are called **loadings**. Read them down a column:

- **PC1** is "how big a ride is": distance and duration together, with a
  nudge from the weekend.
- **PC2** is "what the weather was like": warm and dry at one end, cool
  and wet at the other.
- **PC3** is almost entirely "electric bike or not".

That's a lovely result: PCA rediscovered ride size and weather without
being told either existed. One caution: the sign of a whole component is
arbitrary. Another library could flip PC2 so that wet is positive. Read
which features move *together*, not whether a number is above zero.

## 4. Look at 1,176 rides in 2D

```
2D shape of the projected data                 -> (1176, 2)
average position by rider type:
             PC1   PC2
    member -0.53 -0.02
    casual  1.10  0.05
picture written to                             -> 'output/pca_rides.png'
```

`pca.transform(X)` gives every ride a coordinate on each component. Keep
the first two and you have a scatter plot of the whole table. Open
`output/pca_rides.png`: members and casual riders in two colours, PC1
across, PC2 up.

Rider type was never an input, yet casual riders sit well to the right
on PC1. They take bigger rides, which matches everything since lesson
033. On PC2 the two groups are level: weather doesn't separate them. And
the colours still overlap a lot. Two axes show half the spread, not all
of it.

## 5. Compress, then model: what does it cost?

```
keep k components, then rebuild the 7 columns:
             kept  recon_error
    1 comp.  0.28         0.72
    2 comp.  0.50         0.50
    3 comp.  0.65         0.35
    5 comp.  0.92         0.08
    7 comp.  1.00         0.00
casual-vs-member ROC AUC, 5-fold CV:
    PCA(1)           0.83
    PCA(2)           0.83
    PCA(3)           0.83
    PCA(5)           0.84
    all 7 features   0.84
```

PCA is also a compressor. Keep `k` components, then
`inverse_transform` back to seven columns, and the **reconstruction
error** (average squared miss, in scaled units) is exactly the share you
threw away. Five components rebuild the table with only 8% lost.

Does compression hurt a model? We predict casual versus member with
logistic regression, PCA sitting *inside* the pipeline so each fold
learns its axes from training rows only (lesson 036). A single component
scores 0.83 ROC AUC against 0.84 for all seven features. That's because
the thing that separates casual riders, ride size, is also the biggest
spread in the data, so PC1 kept it.

That is luck, not a law. **PCA never looks at the label.** It keeps the
directions of biggest spread, and sometimes the signal you need lives in
a small one.

## 6. When PCA helps, and when it doesn't

- **Great for looking.** A 2D picture of many columns is the fastest way
  to spot groups, outliers and data mistakes.
- **Useful for compressing** many correlated columns (sensor readings,
  pixels, word counts) before a simple model, or to speed things up.
- **Spread is not signal.** A small component can still carry what your
  label depends on, so always compare against the full feature set, as
  we did.
- **You lose plain meanings.** "PC1 went up 0.4" is harder to explain
  than "the ride was 2 km longer".
- **It's linear.** Curved structure needs other tools such as t-SNE or
  UMAP, which are for pictures only, not for feeding models.

## What you can do now

- Explain PCA as finding new axes along the directions of biggest spread.
- Scale before PCA, and read the explained variance table to choose how
  many components to keep.
- Read loadings to name what each component means.
- Project data to 2D for a plot, and put PCA inside a pipeline when you
  use it before a model.

## What to do now

1. Run `lesson.py` and open `output/pca_rides.png`. Find the long casual
   rides on the right and the wet days at one end of PC2.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   removes duration and watches a new PC1 take over.
3. Next weekday, lesson 046: anomaly detection, flagging the rides that
   don't look like the rest. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

PCA folds overlapping columns into a few new axes. Scale first, read the
explained variance to decide how many to keep, read the loadings to see
what each axis means, and remember that spread is not the same as signal.
