# Lesson 044 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-044-unsupervised-learning-k-means-clustering/lesson.py my_lesson_044.py
python my_lesson_044.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-044-unsupervised-learning-k-means-clustering").resolve()
```

`load_rides`, `scale`, `FEATURES`, `heading` and `show` are there to
reuse. `scale(rides)` returns `(X, scaler)` with `X` already
standardised.

---

## 1. Why scale?

A friend runs k-means on `distance_km` and `duration_min` without
scaling. In one sentence: whose units quietly dominate the grouping?

<details>
<summary>Check yourself</summary>

**Duration**, because it is measured in minutes (~2 to 76) while
distance is in kilometres (~0.4 to 16). Euclidean distance treats a
"10" the same in both columns, so a 10-minute gap looks as big as a
10 km gap. Scaling puts both on a shared ruler first.

</details>

## 2. Elbow or silhouette?

Inertia falls every time you add a centre. So why not always pick the
biggest `k` you can afford?

<details>
<summary>Check yourself</summary>

Because inertia *has* to fall — more centres fit the points better by
construction, even if the extra centres are splitting noise. The elbow
asks where the next centre stops helping much; silhouette asks whether
points sit closer to their own cluster than to the next. Both are
guides, not laws, and they can disagree (here: elbow at 3, silhouette
at 7).

</details>

## 3. Are cluster ids labels?

You fit k-means with `random_state=0` and get cluster 2 as "long leisure
rides". A colleague fits the same data with `random_state=1` and their
"long leisure" group is cluster 0. Did one of you get it wrong?

<details>
<summary>Check yourself</summary>

No. Cluster ids are arbitrary tags. Only the *grouping* matters. Sort
centres by a readable column (duration, in the lesson) before you
compare runs, and never treat a cluster id as ground truth.

</details>

## 4. Hands-on: drop rain and re-cluster

**a)** Build a feature list without `rain_mm`, scale, and fit
`KMeans(n_clusters=3, n_init=10, random_state=0)`. Print the centres in
raw units (inverse-transform) and the cluster sizes.

**b)** Same as (a) but `n_clusters=4`. Which new split appears that
`k=3` had merged?

<details>
<summary>Expected results</summary>

```
a) sizes {0: 772, 1: 200, 2: 204}; centres sorted by duration:
       2.35 km / 11.8 min / weekday / mostly member
       2.88 km / 14.3 min / weekend / half casual
       6.54 km / 31.4 min / often casual (long leisure)
b) sizes {0: 679, 1: 193, 2: 207, 3: 97}; the long group splits into a
   medium casual-weekday slice (~3.6 km, 18 min) and a truly long
   leisure slice (~8.4 km, 40 min).
```

Without rain, the wet-weather blob from section 2 dissolves into the
big weekday-member group. Feature choice *is* the model when there is
no label to correct you.

</details>

<details>
<summary>One way to write it</summary>

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

rides = load_rides(RIDES_CSV)
feats = [f for f in FEATURES if f != "rain_mm"]
scaler = StandardScaler()
X = scaler.fit_transform(rides[feats])

for k in (3, 4):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    centres = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_), columns=feats,
    )
    centres["size"] = pd.Series(km.labels_).value_counts().sort_index().values
    centres = centres.sort_values("duration_min").reset_index(drop=True)
    print(k, centres.round(2), sep="\n")
```

</details>

## 5. (Optional) Silhouette's favourite

Fit k-means for `k` from 2 to 8 on the full scaled features (with rain),
record silhouette each time, and confirm the peak is at 7. Then print
the `k=7` centres sorted by duration. Before you run: will every centre
still be easy to name in plain English?

<details>
<summary>Check yourself</summary>

Silhouette does peak at 7 (0.36). The centres are still readable, but
several of them are small variations on "short member hop" split by
weekend or weather. Past a point you are carving finer slices, not
finding new kinds of ride. That is why the lesson read `k=3` even
though silhouette liked 7: a sketch you can explain beats a slightly
higher score you cannot.

</details>

---

That's k-means. If one thing sticks: *scale first, pick `k` with elbow
and silhouette, read the centres in human units, and treat every
cluster id as a sketch — not a fact.*
