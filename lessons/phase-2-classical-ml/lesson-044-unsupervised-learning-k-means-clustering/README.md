# Lesson 044 - Unsupervised learning: k-means clustering

**Phase 2 - Classical machine learning** | Week 6, Day 4 | Thursday 2026-10-15

> **Goal:** Find groups in unlabelled data, pick `k` sensibly, and
> interpret the clusters.

Time: about 40 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Every lesson since 029 handed the model a label to predict: duration,
casual-or-member, long-or-not. Today we put the labels away.
**Clustering** asks a different question: given only the features, are
there natural groups in the rides?

**k-means** is the simplest answer. Pick `k` centres, assign every ride
to the nearest one, move each centre to the mean of its rides, repeat
until the assignments stop changing. It is a good place to learn the
habits of unsupervised work: scale first, choose `k` with care, read the
centres in human units, and never treat a cluster id as ground truth.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv`. No
train/test split today — there is no label to hold out against. Full
script:

```bash
python lessons/phase-2-classical-ml/lesson-044-unsupervised-learning-k-means-clustering/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match. It takes a couple of seconds.

## 1. Put the labels away

```
rides / features used for clustering           -> (1176, 6)
feature columns                                -> ['distance_km', 'duration_min', 'temp_c', 'rain_mm', 'weekend', 'casual']
feature means (raw units):
    distance_km     3.16
    duration_min   15.58
    temp_c         17.64
    rain_mm         2.37
    weekend         0.21
    casual          0.32
feature stds (raw units):
    distance_km     2.10
    duration_min   10.04
    temp_c          3.36
    rain_mm         4.11
    weekend         0.41
    casual          0.47
```

No duration target, no casual label as the *thing to predict*. We keep
`casual` and `weekend` as **features** — traits of the ride — not as
answers. Clustering only sees the six numbers above. Categories like
station name would need encoding; we leave them out so the centres stay
readable in human units.

## 2. Scale first, then ask k-means for groups

```
cluster sizes without scaling (k=3)            -> {0: 396, 1: 692, 2: 88}
cluster sizes with StandardScaler (k=3)        -> {0: 645, 1: 282, 2: 249}
centres in raw units, AFTER scaling was used:
       distance_km  duration_min  temp_c  rain_mm  weekend  casual
    0         2.34         11.71   19.05     0.44     0.12    0.10
    1         5.61         27.12   17.97     1.45     0.46    0.94
    2         2.48         12.54   13.64     8.42     0.14    0.21
```

k-means uses Euclidean distance. Without scaling, a 10-minute gap in
duration looks ten times bigger than a 1 km gap in distance, and the
0/1 flags barely register. The unscaled run dumps most rides into one
big blob (692) and a tiny leftover (88). Scale with `StandardScaler`
first, fit, then **inverse-transform** the centres back to kilometres
and minutes so you can read them. Same habit as Ridge and Lasso in
lesson 041: the algorithm sees scaled values; you read raw ones.

## 3. Choosing k: elbow and silhouette

```
inertia (lower) and silhouette (higher) by k:
         inertia  silhouette
    k=2  5124.83        0.33
    k=3  4098.56        0.31
    k=4  3374.70        0.34
    k=5  2845.16        0.35
    k=6  2484.76        0.34
    k=7  2228.06        0.36
    k=8  2049.19        0.33
k with the best silhouette                     -> 7
biggest inertia drop happens into k            -> 3
```

**Inertia** is the total squared distance from each ride to its centre.
It always falls as `k` rises — more centres fit the points better by
construction. Look for the **elbow**, where the next centre helps much
less. Here the biggest drop is into `k=3`.

**Silhouette** asks, roughly, "how much closer am I to my own cluster
than to the next?" It peaks at a sensible `k` (here 7). The two often
disagree a little. Neither is a law. For a first readable sketch we
read the `k=3` centres next; you can always try 4 or 7 afterwards and
see whether the story gets clearer or just noisier.

## 4. Read the clusters in human terms

```
centres for k=3, sorted by duration:
               distance_km  duration_min  temp_c  rain_mm  weekend  casual  size
    cluster 0         2.34         11.71   19.05     0.44     0.12    0.10   645
    cluster 1         2.48         12.54   13.64     8.42     0.14    0.21   249
    cluster 2         5.61         27.12   17.97     1.45     0.46    0.94   282
Reading the centres (your numbers, plain English):
  cluster 0: short city hop (n=645, 2.3 km, 12 min)
  cluster 1: medium everyday ride, wetter days (n=249, 2.5 km, 13 min)
  cluster 2: long leisure ride, often weekend, often casual (n=282, 5.6 km, 27 min)
clusters vs the long-ride flag from lesson 043:
    long>=30    0   1
    cluster
    0         645   0
    1         244   5
    2         196  86
```

Three readable groups fall out:

- **Cluster 0** — short, dry, mostly member weekday hops. The bulk of
  the data.
- **Cluster 1** — similar length, but on wetter, cooler days.
- **Cluster 2** — longer, farther, mostly casual, often weekend.
  Leisure rides.

As a sanity check we cross-tabulate against lesson 043's long-ride flag
(`duration_min >= 30`). We never told k-means about "long", yet cluster
2 soaks up 86 of the 91 long rides. That is a *check*, not a proof.
Clusters are a useful sketch of the data, not ground-truth labels.

## 5. What unsupervised is not

- Cluster ids are arbitrary. Cluster 0 today might be cluster 2
  tomorrow after a shuffle; only the *grouping* matters.
- k-means assumes round-ish blobs of similar size. Elongated or nested
  groups need different tools (later lessons).
- Scaling, feature choice and `k` all change the answer. Two honest
  analysts can get different sketches of the same CSV.
- Never score a clustering with accuracy against a label you hid, then
  claim you "discovered" that label. Use labels only as a sanity check,
  the way we did with long rides above.

## What you can do now

- Explain clustering as finding groups without labels.
- Scale features before k-means, then read centres in raw units.
- Use inertia (elbow) and silhouette together to pick a sensible `k`.
- Interpret cluster centres in plain language, and treat cluster ids as
  a sketch — not a fact.

## What to do now

1. Run `lesson.py`. Change the interpretation `k` from 3 to 4 (or to
   silhouette's 7) and see whether the story gets clearer or just
   noisier.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   drops a feature and watches the centres move.
3. Tomorrow, lesson 045: dimensionality reduction with PCA, so you can
   plot high-dimensional data in 2D. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Unsupervised learning finds structure without answers. Scale the
features, pick `k` with elbow and silhouette, read the centres in human
units, and treat every cluster id as a sketch — not a fact.
