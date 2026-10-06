# Lesson 029 - Exercises

Three quick checks, a hands-on fit, and an optional thinking question.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-029-what-a-model-actually-is/lesson.py my_lesson_029.py
python my_lesson_029.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-029-what-a-model-actually-is").resolve()
```

`predict`, `mae`, `mse` and `RIDES_CSV` are there to reuse.

---

## 1. Name the parts

A shop wants to predict tomorrow's croissant sales from the weather
forecast and the day of the week, using a formula
`sales = a * temperature + b * is_weekend + c`.

What are the target, the features, the parameters and a sensible loss?

<details>
<summary>Check yourself</summary>

- **Target:** tomorrow's croissant sales.
- **Features:** forecast temperature and `is_weekend`. Both are known
  the day before, so they're fair to use.
- **Parameters:** `a`, `b` and `c`, the knobs that training sets.
- **Loss:** MAE (average croissants off) is easy to explain; MSE if a
  big miss, like running out on a Saturday, hurts much more than a
  small one.

</details>

## 2. MAE and MSE by hand

Three rides took 10, 12 and 20 minutes. A model predicted 11, 11 and 14.
Work out MAE and MSE without code.

<details>
<summary>Check yourself</summary>

Misses are 1, 1 and 6.

- **MAE** = (1 + 1 + 6) / 3 = **2.67** minutes.
- **MSE** = (1 + 1 + 36) / 3 = **12.67** square minutes.

The single 6-minute miss is three-quarters of the MAE but almost all of
the MSE. That's what "MSE punishes big misses" means.

</details>

## 3. Spot the leak

Which of these could be features for predicting ride duration at the
moment someone unlocks a bike?

- (a) distance to the destination they typed in
- (b) the rating they give after the ride
- (c) bike type
- (d) the station they end at
- (e) today's rainfall so far

<details>
<summary>Check yourself</summary>

**(a), (c) and (e)** are known at unlock time. **(b)** happens after the
ride, so it's leakage. **(d)** is a trap: if they typed a destination it's
fine, but if you only learn the end station when they dock, it's leakage
too. Always ask "would I have this value at the moment I need the
prediction?"

</details>

## 4. Hands-on: one line per bike type

Section 6 showed the single line is too slow for e-bikes and too quick for
classic bikes. Give each bike type its own line.

**a)** Using the least-squares formula (or `np.polyfit(x, y, 1)`), fit a
line to classic-bike rides only, then to electric rides only. What slope
and intercept do you get for each?

**b)** Which is faster per kilometre, and by roughly how much?

**c)** Predict every ride with its own bike type's line. What is the
overall MAE now? Compare it with the single line's 2.74 minutes.

Hints:

- `rides.groupby("bike_type")` and a loop, or two boolean masks.
- Store predictions in an array the same length as `rides` and fill it
  with `pred[g.index] = ...` inside the loop.

<details>
<summary>Expected results</summary>

```
a) classic:  slope about 4.96 min/km, intercept about 1.63 min
   electric: slope about 3.84 min/km, intercept about 0.73 min
b) e-bikes are about 1.1 minutes faster per km
c) overall MAE about 2.02 minutes, down from 2.74
```

</details>

<details>
<summary>One way to write it</summary>

```python
rides = pd.read_csv(RIDES_CSV)
pred = np.empty(len(rides))
for bike, g in rides.groupby("bike_type"):
    slope, intercept = np.polyfit(g["distance_km"], g["duration_min"], 1)
    print(bike, round(slope, 2), round(intercept, 2))
    pred[g.index] = predict(g["distance_km"], slope, intercept)

print("MAE", round(mae(rides["duration_min"].to_numpy(), pred), 2))
```

Two lines means four parameters instead of two, and the error drops by a
quarter. Adding information the model was missing usually pays off like
that. Tomorrow you'll do the same thing more neatly, by giving one model a
bike-type feature.

</details>

## 5. (Optional) Why not zero?

The best line has an intercept of about 1.7 minutes, so a 0 km "ride"
is predicted to take 1.7 minutes. Is that nonsense, or does it mean
something?

<details>
<summary>Check yourself</summary>

It means something. Every ride has a fixed cost that doesn't depend on
distance: unlocking, adjusting the seat, finding a free dock at the end.
But be careful about trusting a model far outside the data it saw. There
are no 0 km rides in the data, so the intercept is mostly a by-product of
fitting the line where the data actually is.

</details>

---

That's what a model is. If one thing sticks: *features in, prediction out,
a loss to say how wrong, and training to turn the knobs until the loss is
as small as it gets.*
