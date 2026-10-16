# Lesson 046 - Anomaly detection

**Phase 2 - Classical machine learning** | Week 6, Day 5 | Friday 2026-10-16

> **Goal:** Flag unusual points with Isolation Forest and density-based
> methods.

Time: about 40 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

In lesson 045's PCA picture a few dots sat far out on their own. Today
we go looking for those on purpose. **Anomaly detection** asks one
question: which rows don't look like the rest?

Sometimes the answer is a broken GPS, a typo or fraud. Sometimes it's
the most interesting ride of the month. The detector can't tell you
which. It can only point, and rank rows from "most out of place" down to
"perfectly ordinary". Deciding what an anomaly *means* stays with you.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv`. Full
script:

```bash
python lessons/phase-2-classical-ml/lesson-046-anomaly-detection/lesson.py
```

It writes no files, and every random step has a fixed seed, so your
numbers will match. It takes a couple of seconds.

## 1. What counts as unusual?

```
real rides                                     -> 1176
speed_kmh in real rides:
    mean      12.34
    std        2.69
    min        4.63
    max       20.38
rides with planted oddities                    -> 1184
the eight planted rides:
         ride_id  distance_km  duration_min  speed_kmh  rain_mm  hour
    1176  X00001        11.72          7.90      89.01     0.00     9
    1177  X00002         9.23          8.80      62.93     0.00     8
    1178  X00003         9.82          6.30      93.52     0.00    17
    1179  X00004         0.81        175.60       0.28     8.20     9
    1180  X00005         0.89        187.40       0.28     0.00     9
    1181  X00006         1.67         10.20       9.82    11.60     3
    1182  X00007         4.49         22.70      11.87    10.50     3
    1183  X00008         2.37          9.40      15.13    11.40     3
```

The awkward thing about anomalies is that real data never comes with a
list of them. So to test a detector honestly, we **plant** eight fake
rides (copied from real rows and bent) and see whether it finds them.
There are three kinds:

- **X00001-3:** far too fast for a bike, 60 to 90 km/h. A GPS jump, or a
  bike thrown in a van.
- **X00004-5:** under a kilometre in about three hours. The bike was
  never docked and the clock kept running.
- **X00006-8:** the sneaky ones. No single column is extreme, but they
  start at 3am, when nobody in September rode at all, and in steady rain.

We also add `speed_kmh` (distance over time). Real rides sit between
about 5 and 20 km/h. A feature like that, built from what you know about
bikes, often does more for anomaly detection than any algorithm.

## 2. The one-column baseline: z-scores

```
rows with any |z| > 3                          -> {'flagged': 53, 'planted caught': '5 / 8'}
which columns triggered most:
    distance_km     30
    duration_min    17
    speed_kmh        3
    temp_c           0
    rain_mm         18
    hour             0
```

Start simple. A **z-score** is how many standard deviations a value sits
from its column's mean. Flag any row with a value more than 3 away.
It catches the five physically impossible rides. But it also flags 48
real ones: every long weekend ride and every wet day. And it misses all
three 3am rides, because an hour of 3 is only about 2.6 standard
deviations from the average start time. Each value is fine on its own;
only the *combination* is odd, and one-column rules can't see
combinations.

## 3. Isolation Forest: how quickly can a row be cut off?

```
contamination=1%                               -> {'flagged': 12, 'planted caught': '3 / 8'}
ten strangest rows by Isolation Forest:
         ride_id  distance_km  duration_min  speed_kmh  rain_mm  hour  planted  score
    1008  R01021        12.02         71.40      10.10    16.30    12        0   0.72
    967   R00979        13.96         51.10      16.39    12.30     9        0   0.71
    1060  R01074        16.52         76.50      12.96     5.60     8        0   0.69
    1179  X00004         0.81        175.60       0.28     8.20     9        1   0.68
    ...
    1176  X00001        11.72          7.90      89.01     0.00     9        1   0.65
    ...
```

**Isolation Forest** is a forest of random trees with a twist. Each tree
picks a random column and a random split point, again and again, until
every row is alone in its own leaf. An ordinary row is buried in the
crowd and takes many splits to isolate. An odd row gets cut off in just
a few. Average that over 300 trees and you have a score: higher means
stranger. Like any tree it splits on thresholds, so **no scaling is
needed**.

`contamination=0.01` tells it to flag the strangest 1% (12 rows). Only
three of them are planted. The rest are real rides that are genuinely
rare: 12 to 16 km and an hour or more, often in the rain. That's not a
bug. Those rides *are* unusual in this data, just not wrong. Isolation
Forest is good at "extreme in some column", and it doesn't single out
the 89 km/h ride the way you might hope, because lots of real rides are
also easy to cut off on distance or duration.

## 4. Local Outlier Factor: density, neighbour by neighbour

```
contamination=1%                               -> {'flagged': 12, 'planted caught': '5 / 8'}
eight strangest rows by LOF:
         ride_id  distance_km  duration_min  speed_kmh  rain_mm  hour  planted  score
    1178  X00003         9.82          6.30      93.52     0.00    17        1  12.68
    1176  X00001        11.72          7.90      89.01     0.00     9        1  11.01
    1177  X00002         9.23          8.80      62.93     0.00     8        1   8.28
    1180  X00005         0.89        187.40       0.28     0.00     9        1   5.26
    1179  X00004         0.81        175.60       0.28     8.20     9        1   4.99
    1060  R01074        16.52         76.50      12.96     5.60     8        0   1.79
    ...
LOF rank of the three 3am rides                -> [33, 20, 87]
rows in BOTH top-12 lists                      -> 6
```

**Local Outlier Factor (LOF)** is a *density* method. For each row it
looks at its 20 nearest neighbours and asks: is my neighbourhood much
emptier than my neighbours' neighbourhoods? A score near 1 means "about
as crowded as everyone around me". Well above 1 means "I'm out here on
my own". It works on distances, so **scale first**, just like k-NN in
lesson 037.

The top five are exactly the five impossible rides, with scores from 5
to 12, and then a clear drop to 1.8 for the first real ride. That gap is
the kind of thing you want to see. Long real rides have other long rides
nearby, so LOF doesn't panic about them. The impossible ones sit in empty
space.

The 3am rides still slip through, ranked 20th, 33rd and 87th. In scaled
space they sit fairly close to plenty of ordinary rainy morning rides.
The exercises fix that with one extra feature.

Six rows appear in both detectors' top 12. When two methods built on
different ideas agree on a row, that row is worth a look first.

## 5. Choosing how many to flag

```
review the top n rows (out of 8 planted):
            IF planted found  LOF planted found
    top 5                  1                  5
    top 8                  1                  5
    top 12                 3                  5
    top 20                 4                  6
    top 40                 5                  7
```

`contamination` sounds scientific but it's just your guess at the share
of odd rows, and all it does is move the cut-off. The useful output is
the **score**, which ranks every row. So the real question is the same
one as lesson 043's threshold: how many flagged rows can a person
actually check? If that's five a day, LOF's top five are all real
problems. If it's forty, you'd catch seven of the eight, but you'd also
read 33 rides that were fine.

## 6. What an anomaly score is not

- **Unusual is not wrong.** A 70-minute ride in the rain may be someone's
  favourite ride of the month. A human decides what to do with a flag.
- **The detector only knows the features you gave it.** Drop the hour,
  and a 3am ride looks perfectly normal. Adding `speed_kmh` mattered
  more than picking an algorithm.
- **Fit on mostly clean data.** If a quarter of the rows are broken,
  broken becomes "normal".
- **If you have labelled examples of the bad thing, use them.** Then it's
  a (very imbalanced) classification problem, which is lesson 043's
  territory, and usually works better.

## What you can do now

- Explain anomaly detection as ranking rows by how out of place they look.
- Test a detector by planting known oddities before trusting it.
- Use z-scores as a baseline, and say why they miss odd combinations.
- Fit Isolation Forest (no scaling) and Local Outlier Factor (scale
  first), read their scores, and compare their top lists.
- Choose how many rows to flag by review capacity, not by a magic
  `contamination` value.

## What to do now

1. Run `lesson.py`. Change `n_neighbors` in LOF from 20 to 10 and watch
   the 3am rides move.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   adds a single feature that lets LOF catch all eight planted rides.
3. Next weekday, lesson 047: model interpretation, asking a trained model
   *why* it predicts what it does. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Anomaly detection ranks rows by how out of place they look. Plant a few
known oddities to test it, compare two detectors, flag only as many as
someone can review, and let a human decide what an anomaly means.
