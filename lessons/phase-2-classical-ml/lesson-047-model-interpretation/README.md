# Lesson 047 - Model interpretation

**Phase 2 - Classical machine learning** | Week 7, Day 1 | Monday 2026-10-19

> **Goal:** Explain predictions with permutation importance and partial
> dependence, and know their limits.

Time: about 40 minutes. Needs the venv from lesson 013 with scikit-learn
and matplotlib installed (`pip install -r requirements.txt` covers both).

---

Since lesson 029 we've been asking models *what* they predict. Today we
ask them *why*. Sooner or later someone will look at your model and say
"fine, but what is it actually using?" You want a better answer than a
shrug, and a better answer than a pretty chart that happens to be wrong.

Two tools cover most of it. **Permutation importance** says how much a
model leans on each column. **Partial dependence** says which way the
prediction moves when a column goes up or down. Both are easy to run and
both have traps, so we'll spring every trap on purpose.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv`. Full
script:

```bash
python lessons/phase-2-classical-ml/lesson-047-model-interpretation/lesson.py
```

It writes one picture, `output/partial_dependence.png`, next to the
script. Every random step has a fixed seed, so your numbers will match.
It takes about ten seconds.

## 1. A model worth explaining

```
train rows / test rows                         -> (882, 294)
test ROC AUC                                   -> 0.837
first three rows of the inputs:
          distance_km  duration_min  temp_c  rain_mm  weekend  hour  electric   coin
    382         4.660        23.000  22.600    0.000        0     8         0  0.038
    1112        1.660         9.500  14.500    0.000        0    12         0 -0.742
    890         3.120        13.300  12.600    3.200        0    17         0 -1.201
```

The job is lesson 033's: tell casual riders from members. The model is
lesson 039's random forest, 200 trees, with at least three rides per
leaf. A test AUC of 0.84 means it ranks a random casual rider above a
random member about 84% of the time. Worth explaining.

There's one odd column. `coin` is pure random noise, drawn with a fixed
seed. It has nothing to do with riders, and we *know* that. It's our
control: any tool that says the coin matters is a tool that can fool you.

## 2. The built-in importance, and the coin that fooled it

```
forest.feature_importances_ (sums to 1):
    distance_km    0.331
    duration_min   0.315
    coin           0.115
    temp_c         0.075
    weekend        0.058
    hour           0.057
    rain_mm        0.030
    electric       0.019
```

Every tree model hands you `feature_importances_` for free. It adds up
how much each column's splits tidied the training data. It's quick, and
it's the first chart most people make.

And it ranks the coin **third**, ahead of weekend, which we know matters
a lot (57% of weekend rides are casual, 26% of weekday ones). Two
reasons. A column with many distinct values offers the trees hundreds of
split points, and deep trees use them to memorise individual training
rows. And this score is computed on the training data only, so
memorising looks exactly like learning. Treat built-in importances as a
rough hint, never as evidence.

## 3. Permutation importance: shuffle a column, watch the score drop

```
test AUC as it is                              -> 0.837
test AUC with duration_min shuffled            -> 0.693
AUC drop when each column is shuffled (20 repeats, test set):
                  mean_drop   std
    duration_min      0.145 0.027
    distance_km       0.066 0.018
    weekend           0.020 0.007
    temp_c            0.009 0.006
    coin              0.005 0.005
    hour              0.003 0.003
    electric          0.002 0.004
    rain_mm           0.001 0.002
```

Here's a better question: if this column suddenly turned to nonsense,
how much worse would the model get? To find out, **shuffle** the column:
mix its values up across the rows, so every ride gets someone else's
duration. The column keeps its spread, but its link to the label is
gone. Everything else stays put. Then score the model again.

Shuffling duration takes the test AUC from 0.84 to 0.69. That drop,
0.145, *is* duration's importance. `permutation_importance` does it for
every column, 20 shuffles each, and reports the mean drop and its
spread.

Now the coin scores 0.005, with a spread just as big: nothing. Weekend
moves up to third. Because it's measured on **test data** with a score
you care about, memorised noise earns no credit. Notice that `hour`
barely registers either. Midday rides *are* more often casual, but 43%
of them are weekend rides, so the model already gets that from the
weekend flag.

## 4. Twins share the credit

```
correlation of distance_km and duration_min    -> 0.921
drop for distance alone + duration alone       -> 0.211
drop when both are shuffled together           -> 0.288
test AUC after dropping duration_min           -> 0.814
distance_km importance, with duration in       -> 0.066
distance_km importance, duration dropped       -> 0.263
```

Permutation importance has its own trap. Distance and duration move
together at 0.92: a long ride is a far ride. Shuffle distance alone and
the model still has duration, which tells almost the same story, so the
score barely suffers and distance looks minor.

Shuffle both together and the drop (0.288) is bigger than the two single
drops added up (0.211). And retrain without duration: the AUC hardly
moves, 0.84 to 0.81, but distance's importance quadruples to 0.263.
Same world, same riders, a different-looking chart.

So read importance as "what this model would lose", not "how much this
fact matters". With correlated columns, the credit is **shared**, and
how it's split depends on which twin the model happened to grab. When
two columns are close, shuffle them as a group.

## 5. Partial dependence: which way does the prediction move?

```
avg P(casual) if every ride lasted ... min     -> {5: 0.172, 10: 0.232, 15: 0.347, 20: 0.435, 30: 0.529, 45: 0.576, 60: 0.577}
avg P(casual) if every ride were weekend=...   -> {0: 0.281, 1: 0.417}
avg P(casual) if every ride started at ...h    -> {8: 0.333, 12: 0.313, 17: 0.293}
picture written to                             -> 'output/partial_dependence.png'
```

Importance says *how much*. It doesn't say *which way*. For that,
**partial dependence** runs a little thought experiment: take every test
ride, pretend it lasted exactly 5 minutes, predict, and average. Then
pretend 10 minutes, and so on. The script does it by hand in five lines
so you can see there's no magic.

The curve climbs steadily: a 5-minute ride averages a 17% chance of
being casual, a 30-minute ride 53%, and then it flattens out. Weekend
adds about 14 points. Hour is nearly flat, which matches its tiny
importance.

Open `output/partial_dependence.png`. The thick dashed line is that
average; each thin line is one ride's own curve, called **ICE**
(individual conditional expectation). They mostly rise together, which
tells you the average isn't hiding riders who go the other way. The
steps are the forest's split points; trees make staircases, not slopes.

## 6. What these tools can't tell you

```
share of rides that become < 4 km/h at 60 min  -> 0.755
```

- **Partial dependence imagines rides that never happen.** Set every ride
  to 60 minutes and three in four become trips slower than walking pace,
  because their distance didn't change. With correlated columns, the far
  end of the curve is the model guessing about an empty part of the map.
  Trust the middle of the curve, where real rides live (the ticks along
  the bottom of the plot show where they are).
- **Importance describes the model, not the world.** Electric bikes score
  near zero because they don't separate riders in *this* data, not
  because bike type can never matter.
- **None of it is causation.** Making members ride longer won't turn them
  into casual riders. Lesson 024 still applies.
- **Use test data and a score you care about.** On training rows, a model
  that memorised noise will cheerfully tell you the noise was useful.

## What you can do now

- Explain why built-in tree importances can rank pure noise highly.
- Compute permutation importance on test data and read its spread.
- Spot correlated columns sharing credit, and shuffle them as a group.
- Compute partial dependence by hand and with scikit-learn, and read an
  ICE plot.
- Say what these tools describe (the model) and what they don't (cause
  and effect).

## What to do now

1. Run `lesson.py` and open `output/partial_dependence.png`. Find a thin
   line that starts high and stays high: what kind of ride might it be?
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   runs permutation importance on the *training* set to see the coin
   come back to life.
3. Next weekday, lesson 048: the Phase 2 mini-project, from raw CSV to a
   cross-validated, tuned, explained model with a short write-up. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Permutation importance on test data tells you how much a model leans on
each column; partial dependence tells you which way it pushes. Both
describe the model, both share credit between correlated columns, and
neither proves cause and effect.
