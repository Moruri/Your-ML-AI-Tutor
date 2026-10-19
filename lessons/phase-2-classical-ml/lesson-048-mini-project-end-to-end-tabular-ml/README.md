# Lesson 048 - Mini-project: end-to-end tabular ML

**Phase 2 - Classical machine learning** | Week 7, Day 1 | Monday 2026-10-19

> **Goal:** Go from raw CSV to a cross-validated, tuned, explained model
> with a short write-up.

Time: about 45 minutes. Needs the venv from lesson 013 with scikit-learn
installed (`pip install -r requirements.txt` covers it).

---

Phase 2 ends the way Phase 1 did: with one project that pulls everything
together. Nothing in it is new. Splits (031), cross-validation (032),
pipelines (036), boosting (040), tuning (042), thresholds (043) and
interpretation (047) all make an appearance. The new skill is putting
them in the **right order**, without peeking and without cutting corners,
and then explaining the result to someone who wasn't in the room.

Here's the brief. When a ride ends, the app can show a "become a member"
offer. Showing it to members is pointless, so: **can we tell from the
ride alone whether the rider was casual or a member?**

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv`. Full
script:

```bash
python lessons/phase-2-classical-ml/lesson-048-mini-project-end-to-end-tabular-ml/lesson.py
```

It writes one file, `output/report.md`, next to the script. Every random
step has a fixed seed, so your numbers will match. It takes about ten
seconds.

## 1. The question, and the file as it arrives

```
rows / columns                                 -> (1176, 9)
columns                                        -> ['ride_id', 'started_at', 'start_station', 'bike_type', 'rider_type', 'distance_km', 'temp_c', 'rain_mm', 'duration_min']
missing values (all columns)                   -> 0
duplicate ride_ids                             -> 0
rider_type counts:
    rider_type
    member    795
    casual    381
```

Before any model, look at the file exactly as it is on disk. How big?
Any gaps? Any ride counted twice? This one was cleaned back in lesson
018, but checking costs three lines, and one day the file you're handed
won't be clean.

The label split matters most. About a third of rides are casual, so a
model that always says "member" is right 68% of the time. That's why
accuracy alone won't do today; we'll use ROC AUC and balanced accuracy
(lesson 034).

## 2. Features, and a test set locked away

```
numeric features                               -> ['distance_km', 'duration_min', 'speed_kmh', 'temp_c', 'rain_mm', 'hour', 'weekend']
categorical features                           -> ['start_station', 'bike_type']
train rows / test rows                         -> (882, 294)
casual share, train / test                     -> (0.324, 0.323)
```

Three new columns: start hour, a weekend flag and average speed. All of
them are known the moment a ride ends, which is when the offer would
show. That's the leakage check to run on every feature: could the app
actually know this value at prediction time? `rider_type` is the answer,
so it never goes in, and nor does anything computed from it.

Then a quarter of the rows go **in a drawer**. Stratified, so both sides
keep a third casual. Every decision from here to section 5 (which model,
which settings, which threshold) uses the other 882 rows only.

## 3. A baseline and two honest contenders

```
5-fold CV on the training rows, default settings:
                         auc_mean  auc_std  bal_acc
    always 'member'         0.500    0.000    0.500
    logistic regression     0.840    0.027    0.733
    gradient boosting       0.811    0.030    0.693
```

Always start with a model that knows nothing. The dummy always says
"member", and sets the floor at 0.5 on both scores. Anything that can't
clear it isn't a model.

Then two real contenders, each a full **pipeline**: logistic regression
behind a scaler and a one-hot encoder for the station and bike type, and
gradient boosting behind just the encoder (trees don't need scaling).
Because the preprocessing lives inside the pipeline, it's refit inside
every fold, so no fold ever learns from rows it's scored on. Out of the
box, the plain linear model is ahead.

## 4. Tune, choose, and set a threshold

```
best of each grid (5-fold CV AUC):
                                                   best_settings auc_mean auc_std
    logistic regression                              {'C': 0.01}    0.841   0.021
    gradient boosting    {'learning_rate': 0.03, 'max_depth': 2}    0.844   0.019
CV leader / gap to logistic / leader's std     -> ('gradient boosting', 0.003, 0.019)
chosen model                                   -> 'logistic regression'
out-of-fold scores by threshold:
          bal_acc  precision  recall
    0.25    0.741      0.526   0.850
    0.30    0.747      0.591   0.741
    0.35    0.751      0.673   0.654
    0.40    0.744      0.746   0.584
    0.45    0.718      0.783   0.503
    0.50    0.685      0.810   0.416
threshold with the best balanced accuracy      -> 0.35
```

Small grids, on purpose: four values of `C` for the logistic model, six
combinations for boosting. Tuned, boosting edges ahead, 0.844 to 0.841.

So which one? Decide the rule **before** you look, so you can't talk
yourself into the answer you hoped for. Ours: if the simpler model is
within one standard deviation of the best, take the simpler one. The gap
is 0.003; the fold-to-fold wobble is 0.019. That's not a difference,
that's noise. Logistic regression wins: fewer moving parts, faster, and
its coefficients can be read out loud.

Last decision: the threshold. `C=0.01` is strong regularisation, which
pulls every probability towards the base rate of a third, so few rides
ever clear 0.5. `cross_val_predict` gives every training ride a score
from a model that never saw it (**out-of-fold** predictions), and we
pick the threshold on those. It's lesson 043's tuning slice, except every
training ride gets a vote. 0.35 gives the best balanced accuracy. The
test set is still in the drawer.

## 5. One look at the test set

```
test ROC AUC                                   -> 0.839
test scores, default vs chosen threshold:
                    bal_acc  precision  recall
    threshold 0.50    0.654      0.745   0.368
    threshold 0.35    0.764      0.688   0.674
confusion matrix at 0.35:
               said member  said casual
    is member          170           29
    is casual           31           64
```

Now, once, open the drawer. Test AUC 0.839 against a CV estimate of
0.841. When those two agree, your process was honest. If the test score
had come out far below CV, something leaked or the test set is
different; far above, and you got lucky.

The threshold earns its keep. At 0.5 the model would miss nearly two in
three casual riders. At 0.35 it finds 64 of the 95, and 69% of its
"casual" calls are right, just as the out-of-fold table promised. Which
threshold is *right* depends on what a wasted offer costs compared with
a missed one. That's a business question, and the write-up should say
so.

## 6. Explain the model

```
permutation importance on test (AUC drop):
                   mean_drop   std
    duration_min       0.089 0.016
    distance_km        0.077 0.015
    weekend            0.040 0.013
    speed_kmh          0.008 0.002
    hour               0.002 0.001
    temp_c             0.001 0.002
    rain_mm            0.000 0.002
    start_station     -0.000 0.001
    bike_type         -0.001 0.001
logistic coefficients, numeric (per 1 std):
    duration_min    0.512
    distance_km     0.481
    weekend         0.329
    temp_c          0.067
    rain_mm         0.035
    hour           -0.022
    speed_kmh      -0.078
```

Lesson 047's tools, on the chosen model. Permutation importance shuffles
*raw* columns, even through the pipeline, so `start_station` counts as
one column rather than six one-hot ones. That's usually the question
people are really asking: does the station matter? Here, no.

Duration and distance carry the model, weekend is a clear third, and
everything else barely moves the score. Because the inputs were scaled,
the logistic coefficients are "per one standard deviation" and can be
compared: longer rides and weekends push towards casual. Distance and
duration are twins, so the 0.51 versus 0.48 split between them means
little. Together they're the story.

## 7. The write-up

```
report written to                              -> 'output/report.md'
```

Open `output/report.md`. The script builds it from the numbers it just
computed, so it can never drift out of date. It's a page, not a novel,
and it answers what a reader actually wants:

- **Question.** What were we trying to decide?
- **Data.** What went in, and how clean was it?
- **Method.** How was it tested, so I can trust the numbers?
- **Choice and result.** Which model, and how good, in words someone
  can act on ("finds 67% of casual riders, and 69% of its calls are
  right").
- **What drives it.** The top columns, and the direction they push.
- **Limits and next steps.** One month, one city, a random split. Say it
  before someone else does.

If you can't fill in the limits section, you haven't finished the
project.

## What you can do now

- Run a tabular ML project in the right order: check, split, baseline,
  cross-validate, tune, choose, threshold, test once, explain.
- Keep preprocessing inside pipelines so cross-validation stays honest.
- Choose between close models with a rule set in advance.
- Pick a threshold on out-of-fold predictions, not on the test set.
- Write a one-page report a non-specialist can act on.

## What to do now

1. Run `lesson.py` and read `output/report.md` as if a colleague had
   sent it to you. What would you ask them first?
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   asks what's left when you can only use what's known as a ride
   *starts*.
3. That's Phase 2 done: twenty lessons of classical machine learning.
   Next weekday, Phase 3 begins with lesson 049: a single neuron, built
   by hand in NumPy. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

Check the data, lock away a test set, beat a baseline with
cross-validation, tune a little, choose by a rule you set in advance,
test once, explain, and write it down for someone else. That's the whole
job, and now you've done all of it.
