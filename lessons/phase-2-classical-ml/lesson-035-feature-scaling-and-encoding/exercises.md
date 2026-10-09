# Lesson 035 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-035-feature-scaling-and-encoding/lesson.py my_lesson_035.py
python my_lesson_035.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-035-feature-scaling-and-encoding").resolve()
```

`load_rides`, `split`, `acc`, `NUMERIC` and `TARGET` are there to reuse.

---

## 1. Scale or not?

For each, would you standardise the numeric columns first?

**a)** k-NN to recommend similar flats from size (m²), price (KES) and
number of rooms.

**b)** A decision tree on the same columns.

**c)** Logistic regression where you want to say which feature matters
most by comparing the weights.

<details>
<summary>Check yourself</summary>

**a) Yes, definitely.** Price is in the millions, rooms are 1 to 5. Raw,
"similar" would just mean "similar price".

**b) No need.** A tree asks one column at a time ("price above
4,000,000?"), so units never get compared with each other.

**c) Yes.** Raw weights are per unit: a weight per shilling looks tiny
next to a weight per room even if price matters more. Standardised, each
weight is "per one spread", so they're comparable.

</details>

## 2. One ride by hand

The scaler learned a mean of 15.76 and a spread of 10.28 for
`duration_min`. What does a 36.3-minute ride become? And a 5-minute
one?

<details>
<summary>Check yourself</summary>

- (36.3 - 15.76) / 10.28 ≈ **2.0**, two spreads longer than average.
- (5 - 15.76) / 10.28 ≈ **-1.05**, about one spread shorter.

Negative just means below the training mean. Nothing is wrong with it.

</details>

## 3. Fit on everything?

A friend says: "Simpler to fit the scaler on all 1,176 rides before
splitting. It's only a mean and a spread." What's wrong with that, and
how big a deal is it here?

<details>
<summary>Check yourself</summary>

The test rides then help set the ruler they're measured with, so the
test score is no longer a fair stand-in for brand-new rides. It's the
same peeking problem as lesson 031, just quieter.

On this data the damage is small (k-NN scores 0.771 on the test rides
that way, against 0.767 done properly), because a mean and spread from
940 rides barely move when you add 236 more. But "small here" is luck,
not a rule. With a small dataset, a strong outlier in the test set, or
fancier preprocessing, it grows. Fit on training data, every time, and
you never have to wonder.

</details>

## 4. Hands-on: stations, units and a second scaler

**a)** Print the share of casual riders from each start station in the
training rides. Do any stand out?

**b)** In turn, multiply each of the four numeric columns by 1000 (train
and test both) and record k-NN's test accuracy, everything else raw.
Which column hurts most when it's blown up?

**c)** Swap `StandardScaler` for `MinMaxScaler` and compare k-NN's test
accuracy.

Hints:

- `train.groupby("start_station")[TARGET].mean()` does (a).
- `from sklearn.preprocessing import MinMaxScaler`. It has the same
  `fit` / `transform` as `StandardScaler`.

<details>
<summary>Expected results</summary>

```
a) Market Square 0.33, Old Mill 0.30, Park Gate 0.30,
   Riverside 0.34, Station Road 0.36, University 0.33
b) distance_km x1000  -> 0.729
   duration_min x1000 -> 0.750
   temp_c x1000       -> 0.725
   rain_mm x1000      -> 0.771
c) MinMaxScaler 0.780, StandardScaler 0.767
```

(a) All six sit between 0.30 and 0.36, so the station on its own says
little, which matches section 5.

(b) Blowing up temperature or distance hurts most: k-NN ends up matching
rides on one weak column. Rain at 0.771 is a surprise worth thinking
about. Most rides have zero rain, so huge rain numbers mostly split
"dry" from "wet", and among the dry rides the other columns still get to
vote. Units change *which* question the model asks, sometimes for
better, mostly for worse, and never on purpose.

(c) MinMax edges ahead on this test set. With 236 test rides that's
about two rides, so don't read much into it. Both are fair choices; the
point is to pick one deliberately and fit it on the training rides.

</details>

<details>
<summary>One way to write it</summary>

```python
from sklearn.preprocessing import MinMaxScaler

rides = load_rides(RIDES_CSV)
train, test = split(rides)
print(train.groupby("start_station")[TARGET].mean().round(2))

knn = KNeighborsClassifier(n_neighbors=15)
for col in NUMERIC:
    a, b = train[NUMERIC].copy(), test[NUMERIC].copy()
    a[col] *= 1000
    b[col] *= 1000
    print(col, acc(knn, a, train[TARGET], b, test[TARGET]))

mm = MinMaxScaler().fit(train[NUMERIC])
print("minmax", acc(knn, mm.transform(train[NUMERIC]), train[TARGET],
                    mm.transform(test[NUMERIC]), test[TARGET]))
```

</details>

## 5. (Optional) Is Tuesday a number?

You want to add the day of the week. You could use `dayofweek` (0 for
Monday to 6 for Sunday) as a number, or one-hot it into seven columns.
What does each choice tell the model, and is there a third option?

<details>
<summary>Check yourself</summary>

As a number, the model assumes days are evenly spaced and in order, and
that Sunday (6) is as far from Monday (0) as possible, when really
they're neighbours. One-hot treats each day as its own thing, with no
order at all, which is safe but uses seven columns.

The third option is to keep only what you think matters: lessons 033
and 034 did exactly that with a single `weekend` column. A fourth, for
things that wrap around like hours and days, is to encode them as a
point on a circle with `sin` and `cos`, so Sunday sits next to Monday.
Choosing an encoding is choosing what the model is allowed to notice.

</details>

---

That's scaling and encoding. If one thing sticks: *learn the ruler from
the training data, then measure everything with it.*
