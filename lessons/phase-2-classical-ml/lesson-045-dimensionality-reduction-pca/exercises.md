# Lesson 045 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-045-dimensionality-reduction-pca/lesson.py my_lesson_045.py
python my_lesson_045.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-045-dimensionality-reduction-pca").resolve()
```

`load_rides`, `FEATURES`, `heading` and `show` are there to reuse.

---

## 1. Why did PC7 get so little?

The last component carries only 1% of the spread. Which pair of features
is it mostly made of, and why is there so little left for it?

<details>
<summary>Check yourself</summary>

**Distance and duration.** They correlate at 0.92, so PC1 already
captures what they share. What's left is the small difference between
them (rides that were unusually fast or slow for their length), and
that's a tiny amount of spread. Strongly correlated columns always leave
a near-empty component behind.

</details>

## 2. Scale or not?

A colleague skips `StandardScaler` and reports "PC1 explains 70% of the
variance, so one number summarises our rides". What's wrong with that?

<details>
<summary>Check yourself</summary>

Without scaling, variance depends on units. Duration in minutes has a
standard deviation near 10, while the flags sit between 0 and 1, so PC1
is almost just duration (loading about 0.98). The 70% is a fact about
minutes, not about rides. Scale first, and PC1 honestly carries 28%.

</details>

## 3. Does PCA know about casual riders?

PC1 separates casual riders from members quite well. Did PCA use the
rider type to build PC1?

<details>
<summary>Check yourself</summary>

No. PCA never sees a label. PC1 is "ride size", and casual riders happen
to take bigger rides. If the label had depended on a low-spread
direction, PC1 would have been useless for it. That's why section 5
compares against the full feature set.

</details>

## 4. Hands-on: take duration away

**a)** Fit PCA on the six features left after dropping `duration_min`
(scaled). Print the explained variance ratios. How much does PC1 carry
now, and what is it made of?

**b)** Fit PCA on just `distance_km` and `duration_min` (scaled). Before
you run: roughly what share will PC1 get?

<details>
<summary>Expected results</summary>

```
a) shares   0.256 0.193 0.169 0.163 0.138 0.080
   PC1 loadings: temp_c 0.69, rain_mm -0.69 (weather is now PC1)
   PC2 loadings: distance_km 0.71, weekend 0.58
b) shares   0.961 0.039
```

With duration gone, distance has no twin, so the ride-size direction
shrinks and the weather pair takes over as the biggest spread. Notice
there's no near-empty component any more: the smallest still holds 8%.
In (b), two columns at 0.92 correlation compress into one axis with 96%
of the spread.

</details>

<details>
<summary>One way to write it</summary>

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rides = load_rides(RIDES_CSV)

feats = [f for f in FEATURES if f != "duration_min"]
X = StandardScaler().fit_transform(rides[feats])
pca = PCA().fit(X)
print(pca.explained_variance_ratio_.round(3))
print(pd.DataFrame(pca.components_[:2].T, index=feats).round(2))

X2 = StandardScaler().fit_transform(rides[["distance_km", "duration_min"]])
print(PCA().fit(X2).explained_variance_ratio_.round(3))
```

</details>

## 5. (Optional) Colour by cluster

Fit `KMeans(n_clusters=3, n_init=10, random_state=0)` on the same scaled
seven features, then redraw the 2D PCA scatter coloured by cluster
instead of rider type. Before you run: which cluster will sit far to the
right on PC1?

<details>
<summary>Check yourself</summary>

The long-ride cluster, because PC1 is ride size. You'll likely see the
other two clusters split along PC2 (weather) or PC3 (electric), and some
clusters will overlap in 2D even though they're well apart in seven
dimensions. A 2D picture is a shadow of the data, and shadows can hide
gaps.

</details>

---

That's PCA. If one thing sticks: *scale first, keep components by
explained variance, read the loadings, and remember that spread is not
the same as signal.*
