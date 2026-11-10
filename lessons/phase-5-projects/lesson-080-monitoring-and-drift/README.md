# Lesson 080 - Monitoring and drift

**Phase 5 - Projects and production intuition** | Week 10, Day 2 | Tuesday 2026-11-10

> **Goal:** Detect when the world changes and your model quietly gets
> worse.

Time: about 45 minutes. Same venv as lesson 013; nothing new to install.

---

Here's the uncomfortable truth about a deployed model: it never tells
you when it's wrong. Yesterday's lesson put a model behind an API.
Suppose it's still there in three months. The code hasn't changed and
nothing has crashed, so the model answers every request in a few
milliseconds. Meanwhile, the season has turned, a competitor has opened
down the road and somebody upstream changed a unit. Each of those
quietly bends the world away from the data the model learned from.
This is called **drift**, and the job of noticing it is **monitoring**.

Today we play out twelve weeks of live traffic in which three things go
wrong, and build three monitors to catch them:

1. **Input drift:** are the inputs different from what we trained on?
2. **Prediction drift:** is the model saying different things?
3. **Live error:** once the real answers arrive, how wrong were we?

You'll see that none of the three is enough on its own, which is the
main thing to take away.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-5-projects/lesson-080-monitoring-and-drift/lesson.py
```

It takes a second or two. No downloads and no data file: the rides are
simulated, so we know exactly what changed and when. The model is
trained on 13 weeks from 2026-05-04, then "goes live" on 2026-08-03 for
12 weeks, during which:

- **from week 3**, a rival scheme opens near **Market** and takes 30% of
  its riders;
- **from week 5**, **autumn** arrives and temperatures slide about 12
  degrees by week 12;
- **from week 9**, **Lakeside**'s weather feed starts sending Fahrenheit
  in the `temp_c` column.

To keep the effects easy to see, the model is a simple ridge on weather,
weekend and station (no lag features this time; exercise 5 asks what
lags would change). A small report goes to an `output/` folder next to
the script, wiped every run. The numbers below should match yours.

## 1. A model that never complains

```
  training days, live days                       -> (91, 84)
  MAE on the training period                     -> 3.02
  live MAE, week by week                         -> [3.2, 2.72, 6.86, 6.44, 6.76, 5.61, 6.48, 6.67, 24.12, 24.52, 23.23, 21.77]
  errors or warnings raised by the model         -> 0
```

In training the model was off by about 3 rides a day. Live, it's fine
for two weeks, then twice as bad from week 3, then eight times as bad
from week 9. Not one error or warning was raised. Without the third line
nobody would know, and in real life the third line doesn't exist unless
you build it. That's what the rest of the lesson does.

## 2. Input drift: compare this week with the reference

The first monitor needs no answers at all, only the inputs, so it works
from day one. Keep a **reference**, the training data, and ask each week
whether the new inputs look like it.

A popular score for this is the **population stability index (PSI)**.
Cut the reference into ten buckets with equal counts, see what share of
the new data falls in each bucket, and add up how far the shares moved.
Rules of thumb: under 0.1 is steady, over 0.25 is a big move. But watch:

```
  PSI of the reference vs itself, 7 rows         -> 3.76
  PSI of the reference vs itself, 300 rows       -> 0.03
  PSI vs itself shifted up 3 degrees, 300 rows   -> 1.85
```

Seven values (one station, one week) drawn from the *reference itself*
give a PSI of 3.76, which the rule of thumb calls an enormous move. Ten
buckets can't be filled sensibly with seven values. With 300 values it
behaves: 0.03 for no change, 1.85 for a three-degree shift. **A drift
score is only as good as the amount of data under it.** PSI (and its
cousin, the Kolmogorov-Smirnov test from `scipy.stats.ks_2samp`) is
excellent for busy services with thousands of requests a day.

We have seven rows per station per week, so we use something humbler:
the **shift score**, how many reference standard deviations this week's
mean has moved. Over 2 either way is flagged.

```
  week  temp shift: Uni Sq  Market  Lakeside   rain shift
     1               -0.4    -0.0       0.3         -0.2
     2               -1.1    -1.4      -1.2         -0.1
     3               -1.0    -1.0      -1.2         -0.2
     4                0.2    -0.1      -0.0          0.1
     5                0.8     0.9       0.7          0.3
     6               -0.4    -0.1      -0.9          0.1
     7               -2.2    -2.4      -2.8         -0.1  <- drift: Uni Square, Market, Lakeside
     8               -3.3    -2.8      -3.5         -0.2  <- drift: Uni Square, Market, Lakeside
     9               -3.2    -3.4      15.9         -0.1  <- drift: Uni Square, Market, Lakeside
    10               -3.3    -2.6      16.3         -0.1  <- drift: Uni Square, Market, Lakeside
    11               -3.6    -3.6      15.1          0.3  <- drift: Uni Square, Market, Lakeside
    12               -5.8    -5.2      11.6         -0.1  <- drift: Uni Square, Market, Lakeside

  first week each input drifted                  -> {'Uni Square': 7, 'Market': 7, 'Lakeside': 7, 'rain': None}
  Lakeside temp_c mean, weeks 8 and 9            -> (14.4, 59.6)
```

Read the table the way you'd read it at work:

- **Weeks 1-6:** wobbles, nothing flagged. Normal weather.
- **From week 7**, all three stations cool together, by about 2 to 6
  standard deviations. Several stations moving the *same way at the same
  time*: that's the world changing (autumn), not a broken sensor.
- **From week 9**, Lakeside jumps to +16 while its neighbours stay at
  -3. One station moving alone, by a huge amount, overnight: that's
  almost always a **data problem**. Its mean "Celsius" went from 14.4 to
  59.6. That's Fahrenheit.
- **Rain** never moves, and **Market** never looks any different from
  the others. Keep that in mind.

The *size*, *direction* and *timing* of a shift tell you more than the
flag itself. Lesson 078's `compare` did the same job for whole
deliveries; this is the same idea on a schedule.

## 3. Prediction drift: watch what the model says

```
  mean prediction in training, by station        -> {'Uni Square': 63.9, 'Market': 47.4, 'Lakeside': 32.8}
  change in mean prediction, week 2              -> {'Uni Square': -3.1, 'Market': -4.6, 'Lakeside': -2.8}
  change in mean prediction, week 12             -> {'Uni Square': -17.6, 'Market': -17.4, 'Lakeside': 37.7}
  Market, mean prediction weeks 1-2 vs 3-4       -> (45.4, 45.7)
```

The second monitor also needs no answers: watch the model's own
outputs, which your request log from lesson 079 already has. By week 12,
predictions are down about 17 rides at the two healthy stations (cold
days, fewer riders, which is right) and **up 38** at Lakeside, where the
model believes it's 60-degree weather. A prediction that moves that far
from anything seen in training deserves a look, even before anyone knows
the real number.

But look at the last line. When the rival opened in week 3, Market's
predictions didn't move at all: 45.4 before, 45.7 after. Of course they
didn't. The weather didn't change, so the inputs didn't change, so the
predictions didn't change. Only the *riders* changed.

## 4. Live error, once the real numbers arrive

```
  week   MAE: Uni Sq  Market  Lakeside
     1          3.1     2.5       4.0
     2          2.7     2.7       2.7
     3          4.3    13.0       3.3
     4          1.8    13.0       4.5  <- alert: Market
     5          3.6    14.1       2.6  <- alert: Market
     6          2.4    12.3       2.2  <- alert: Market
     7          3.8    12.5       3.1  <- alert: Market
     8          3.4    11.3       5.2  <- alert: Market
     9          2.8    11.3      58.3  <- alert: Market, Lakeside
    10          1.9    12.0      59.7  <- alert: Market, Lakeside
    11          2.7     9.0      58.0  <- alert: Market, Lakeside
    12          3.6     7.1      54.6  <- alert: Market, Lakeside

  training MAE by station                        -> {'Uni Square': 2.9, 'Market': 3.0, 'Lakeside': 3.2}
  first alert week by station                    -> {'Uni Square': None, 'Market': 4, 'Lakeside': 9}
  Market, weeks 3-8: mean actual vs predicted    -> (31.5, 44.3)
```

The third monitor is the one that matters most and arrives last: real
ride counts come in at the end of each day, so we can compute the error
per station per week. The alert rule is deliberately simple: **error
above 1.5 times the training error, two weeks running**. Two in a row
because a single bad week happens by chance (Uni Square's 4.3 in week 3
is one); asking for two trades a week of delay for far fewer false
alarms.

- **Market** alerts in week 4. The model expects 44 riders and gets 31.
  This is the drift no input monitor could see: the relationship
  between weather and riders changed, not the weather. It's called
  **concept drift**, and only labels catch it.
- **Lakeside** alerts in week 9, the first week of Fahrenheit.
- **Uni Square** never alerts, through twelve degrees of autumn.

A sneaky detail: Market's error *improves* in weeks 11 and 12. Nothing
was fixed. Autumn pushed predictions down towards the lower ride counts
the rival caused, so two mistakes partly cancel. An improving metric is
not proof of a healthy model; it's proof that you should look.

## 5. Which monitor caught what

```
  autumn (cooler days)     inputs True   predictions True   error False
  rival scheme at Market   inputs False  predictions False  error True
  Fahrenheit at Lakeside   inputs True   predictions True   error True
```

This little table is the lesson:

- **Autumn** set off the input and prediction monitors, yet the model
  stayed accurate. The model had learned "colder means fewer riders",
  and that kept being true. That's **data drift without harm**: worth
  knowing about, not worth waking anyone at night.
- **The rival** was invisible to everything except the error. Pure
  concept drift.
- **Fahrenheit** set off everything. Of the three it was the easiest to
  catch and the most damaging, and the input monitor caught it *in its
  first week*, without needing a single ride count.

So you want all three. Input checks are fast and cheap, and catch broken
pipes early. Live error is slow, because it waits for labels (for some
problems that's weeks or months, like whether a loan is repaid), but
it's the only one that measures what you care about. Prediction drift
sits in between.

## 6. Responding: fix the feed, then retrain on recent data

```
  weeks 10-12, old model, raw feed               -> {'Uni Square': 2.7, 'Market': 9.4, 'Lakeside': 57.4}
  weeks 10-12, old model, feed fixed             -> {'Uni Square': 2.7, 'Market': 9.4, 'Lakeside': 3.2}
  weeks 10-12, retrained, feed fixed             -> {'Uni Square': 3.4, 'Market': 3.6, 'Lakeside': 3.1}
  weeks 10-12, trained on weeks 3-8 only         -> {'Uni Square': 3.0, 'Market': 3.4, 'Lakeside': 3.1}
```

Different drifts need different fixes, and the order matters:

- **A broken feed is fixed at the door, not by retraining.** Converting
  Lakeside's temperatures back to Celsius on arrival takes its error
  from 57 to 3.2 with the *same* model. Retraining on Fahrenheit data
  would teach the model a bug. (A good door would also refuse values
  outside a sane range, as the API did in lesson 079.)
- **Concept drift is fixed by retraining on data that shows the new
  world.** We retrain on the training period plus weeks 3-8 (after the
  rival, before the bad feed), counting the recent weeks three times.
  Market's error drops from 9.4 to 3.6.
- **How much history to keep is a trade-off.** Training on weeks 3-8
  only does a little better here (Uni Square 3.0, Market 3.4), because
  those weeks are most like weeks 10-12. With a messier model, or a
  feature that never varied in six weeks (no rain, no holidays), a
  short window can forget things you'll need again. There's no
  universal answer; you try a few windows and evaluate on the most
  recent weeks, as here.

Each retrained model should go through lesson 077's tracker and get
lesson 078's data and feature versions, so "why did Market's numbers
change in September?" has an answer.

## What you can do now

- Explain data drift, concept drift and a broken pipeline, and why a
  model reports none of them.
- Compare live inputs with a training reference, and know when a score
  like PSI is trustworthy.
- Read size, direction and timing of a shift to tell the world changing
  from a data bug.
- Track live error per segment with an alert rule that resists noise.
- Pick the right response: fix the input, retrain, or just note it.

## What to do now

1. Run `lesson.py`, then open `output/monitoring_report.json`.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   adds a drift check for `rain_mm` and makes it fire.
3. Next weekday, lesson 081: performance, making it fast enough.

The model won't tell you it's wrong. Your monitors have to.
