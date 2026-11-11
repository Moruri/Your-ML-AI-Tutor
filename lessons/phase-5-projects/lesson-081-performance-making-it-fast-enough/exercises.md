# Lesson 081 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere of
your own and edit the copy (it writes an `output/` folder next to
wherever it sits). From the repo root:

```bash
cd lessons/phase-5-projects/lesson-081-performance-making-it-fast-enough
python lesson.py
cp lesson.py ../../../my_speed.py
cd ../../..
python my_speed.py
```

Remember that your timings will differ from anyone else's. Compare
ratios, not milliseconds.

---

## 1. Median or p95?

A service answers 19 requests in 1 ms and the 20th in 200 ms, over and
over. What are its median, its average and its p95, roughly? Which one
would you put in a budget, and why?

<details>
<summary>Check yourself</summary>

The median is 1 ms. The average is (19 x 1 + 200) / 20, about 11 ms.
The p95 sits right at the edge: 95% of requests are 1 ms, so depending
on how it's computed it lands between 1 and 200 ms; the p99 is clearly
200 ms. The median says "all fine", the average describes no request
that ever happened, and only the high percentile shows that one caller
in twenty waits 200 times longer. Budgets use a high percentile (p95 or
p99) because that's the waiting people actually notice, and because a
page that makes 20 calls will almost always hit the slow one.

</details>

## 2. Read the profile

Open `output/profile_naive.txt`. What's the difference between `tottime`
and `cumtime`? Why does `predict_naive` have the largest `cumtime` but a
tiny `tottime`?

<details>
<summary>Check yourself</summary>

`tottime` is time spent in the function's *own* lines, not counting the
functions it calls. `cumtime` includes everything it calls. Ours does
almost nothing itself; it calls `read_csv`, `build_features` and
`predict`, which do the work. So its `cumtime` is the whole run, and its
`tottime` is close to zero. Read `cumtime` from the top to find the
expensive *branch*, then `tottime` to find the expensive *line* at the
bottom of it.

</details>

## 3. When is a cache wrong?

Which of these is safe to wrap in `lru_cache`, and which isn't?
(a) `forecast(station, day, data_version)` from the lesson;
(b) `current_bikes_at(station)`, which asks the docks how many bikes are
there right now; (c) `tokenise(text)`; (d) `predict(model, features)`,
where `features` is a pandas DataFrame.

<details>
<summary>Check yourself</summary>

(a) Safe: same inputs, same answer, and new data changes the key.
(b) Not safe: the answer changes every minute while the input stays the
same, so the cache serves a stale count. If you must cache it, give it
an expiry time of a few seconds. (c) Safe and often worthwhile, since
it depends only on the text. (d) It doesn't even work: `lru_cache`
needs hashable arguments, and DataFrames aren't hashable. Cache on
something small and stable instead, like `(station, day)`.

</details>

## 4. Hands-on: fast without hand-written maths

Section 3 got under a microsecond with hand-written numpy, and the
README said not to do that in real life. Find a middle way in
`my_speed.py`.

1. Time two things separately: building the one-row DataFrame, and
   `LOADED.predict` on it. Which costs more?
2. Train a second model on numpy arrays instead of a DataFrame:
   `Ridge().fit(feats[FEATURES].to_numpy(), feats["rides"])`, and
   predict with `np.array([[temp_c, rain_mm, lag_1, avg_7, 0]])`.
3. How fast is it now, and is the answer still 41.9?

<details>
<summary>Check yourself</summary>

Something like:

```python
ARR_MODEL = Ridge().fit(feats[FEATURES].to_numpy(), feats["rides"])

def predict_array(station, temp_c, rain_mm):
    lag_1, avg_7 = LATEST[station]
    return float(ARR_MODEL.predict(np.array([[temp_c, rain_mm, lag_1, avg_7, 0]]))[0])

show("array model", ms(timed(lambda: predict_array("Market", 21.0, 0.0), 2000)))
show("same answer?", round(predict_array("Market", 21.0, 0.0), 6) == round(answer_naive, 6))
```

On a typical laptop, building the DataFrame is about a third of the
0.6 ms and scikit-learn's checks on it (column names, types) are most of
the rest. The array version lands around 0.05 ms, roughly ten times
faster than `predict_warm`, with the same answer and with scikit-learn
still doing the maths. The trade: the model no longer knows its column
names, so *you* must keep the order of `FEATURES` right. Writing that
order in one place (as `FEATURES` does) is what makes it safe.

</details>

## 5. Optional puzzle: does int8 always survive?

Our int8 network lost almost nothing. Quantisation uses one scale per
layer: the largest absolute weight divided by 127. What happens to the
small weights in a layer if one weight is 100 times larger than all the
others? How would you fix it?

<details>
<summary>Check yourself</summary>

The scale is set by the giant weight, so one step of the int8 grid is
huge compared with the ordinary weights. Most of them round to 0, 1 or
-1, and the layer loses nearly all its detail. Accuracy can fall off a
cliff. Fixes, in the order people reach for them: one scale per output
column (per-channel quantisation) instead of per layer, so one outlier
only spoils its own column; keep outlier weights in float and quantise
the rest (what several LLM quantisation methods do); or clip extreme
weights before choosing the scale. Whatever you do, re-check accuracy
on the test set afterwards, as the lesson did.

</details>
