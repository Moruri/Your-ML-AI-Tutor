# Lesson 080 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere of
your own and edit the copy (it writes an `output/` folder next to
wherever it sits). From the repo root:

```bash
cd lessons/phase-5-projects/lesson-080-monitoring-and-drift
python lesson.py
cp lesson.py ../../../my_monitor.py
cd ../../..
python my_monitor.py
```

---

## 1. Name the drift

For each story, say whether it's data drift, concept drift or a data
problem, and which of our three monitors would notice first: (a) a
university moves its exams a month earlier, so the student rush at Uni
Square comes earlier; (b) a city-wide heatwave; (c) the app starts
sending rain in centimetres; (d) ride prices double overnight.

<details>
<summary>Check yourself</summary>

(a) Concept drift: the same weather and weekday now mean a different
number of riders, and only live error sees it (the model's inputs don't
include "exam season"). (b) Data drift: the inputs move together across
stations; the input monitor fires first, and if the model learned the
temperature effect well, error may stay fine. (c) A data problem: rain
values become ten times smaller, so the input monitor catches it the
first week. (d) Concept drift again: same inputs, fewer riders, caught
by live error. In real life, (d) is also something your colleagues
*know about in advance*. Monitoring is the backstop, not a substitute for
asking "is anything changing next week?"

</details>

## 2. Why two weeks in a row?

Our alert needs the error above 1.5 times training two weeks running.
What do you gain and lose compared with (a) one week, (b) four weeks in
a row? Use the table in section 4.

<details>
<summary>Check yourself</summary>

(a) One week: Market would alert in week 3 instead of 4, but Lakeside
would also alert in week 8, by chance: 5.2 is just over 1.5 times its
3.2, a week *before* the Fahrenheit started. That's a false alarm, and
too many of those and people stop reading alerts. (b) Four weeks: hardly
any false alarms, but Market alerts in week 6 and Lakeside in week 12,
so you'd have served a month of nonsense temperatures. The rule should
fit the cost of each mistake. For Lakeside's kind of failure, you'd
rather lean on the input monitor, which fires in the first week.

</details>

## 3. Why did the noise-free PSI say 3.76?

The reference compared with a sample of *itself* scored 3.76 with seven
rows. Explain where such a big number comes from, using what PSI does
with buckets.

<details>
<summary>Check yourself</summary>

Ten buckets, each holding 10% of the reference. Seven values can fill
at most seven buckets, so at least three are empty. An empty bucket's
share is 0, which we clip to 0.0001. For that bucket, `(c - r)` is
about -0.1 and `log(c / r)` is about log(0.001), roughly -6.9, so it adds
about 0.69 on its own. The buckets that got one or two values add more.
Nothing changed; there just aren't enough values. Fixes: fewer buckets,
larger windows (a month instead of a week), or a simpler score like the
mean shift, as in the lesson.

</details>

## 4. Hands-on: make rain drift

Edit `my_monitor.py`.

1. In `make_world`, make rain more frequent in the live period from
   week 6: rain on 75% of days instead of 25%.
2. Run it. Does the `rain shift` column flag it? Does live error change?
3. The shift score looks only at the **mean**. Add a second rain check,
   "share of rainy days this week minus the reference share", and flag
   it when that moves by more than 0.35. Which check fires?

<details>
<summary>Check yourself</summary>

Something like:

```python
live_week = np.where(dates >= LIVE_START, (dates - LIVE_START).days // 7 + 1, 0)
rain = np.where(rng.random(n) < np.where(live_week >= 6, 0.75, 0.25), rng.exponential(6, n), 0.0)
...
ref_share = (ref["rain_mm"] > 0).mean()
row["rainy_days"] = (g["rain_mm"] > 0).mean() - ref_share
```

(`week` is computed a few lines further down in `make_world`, so either
move that line up or compute it again, as here.)

With this lesson's seed, the rain shift score creeps up to about 0.9 and
never reaches 2, so it never flags. The rainy-day share jumps by about
0.4 to 0.6 and flags from week 6 on. Rain is mostly zeros with an
occasional big value, so its standard deviation is large and its mean
is a noisy, weak summary; the share of rainy days is a steadier signal.
Live error barely changes, because the model knows what rain does: this
is data drift without harm, like autumn. And notice week 5's rainy share
was already +0.3 by pure chance, which is why the threshold is 0.35 and
not 0.25. The general lesson: pick a check that suits the shape of each
column, then look at a few normal weeks before choosing its threshold.

</details>

## 5. Optional puzzle: lags hide the rival

Lessons 078 and 079 used `lag_1` and `avg_7`, yesterday's rides and last
week's average. If this lesson's model had them, what would happen to
Market's error after the rival opens, and to the input monitor? Is that
good news?

<details>
<summary>Check yourself</summary>

After a week, `lag_1` and `avg_7` at Market would drop, because
yesterday really did have fewer riders, so the predictions would follow
the new level and the error would mostly recover by itself. The input
monitor would now flag Market (its lag features shift down), so the
rival becomes *visible* as input drift. Mostly good news: lag features
make a model adapt to level changes. The catch: for about a week the
error is still high, and the lags also copy *bad* data forward (if a
broken counter records zero rides for three days, the model predicts
low for the next week). Lags make a model more adaptive, not more
trustworthy, so you still watch all three monitors.

</details>
