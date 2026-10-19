# Lesson 048 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-048-mini-project-end-to-end-tabular-ml/lesson.py my_lesson_048.py
python my_lesson_048.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-048-mini-project-end-to-end-tabular-ml").resolve()
```

`build_features`, `make_logistic`, `make_boosting`, `CV`, `NUMERIC`,
`CATEGORICAL`, `FEATURES`, `heading` and `show` are there to reuse.

---

## 1. Why only one look?

Suppose you checked the test score after every change: new feature, new
grid, new threshold, keeping whatever scored best. After twenty rounds
the test AUC is 0.87. Why should nobody believe that number?

<details>
<summary>Check yourself</summary>

Because the test set has quietly become part of training. Each choice
was nudged towards what happens to work on those 294 rows, so the score
is tuned to them, the same way an overfit model is tuned to its training
rows. Cross-validation on the training rows is where the choosing
happens; the test set is opened once, to report.

</details>

## 2. Boosting scored higher. Why not use it?

Tuned gradient boosting beat logistic regression in cross-validation,
0.844 to 0.841. Give two reasons the lesson's choice is still sensible.

<details>
<summary>Check yourself</summary>

The gap (0.003) is far smaller than the fold-to-fold spread (about
0.02), so the data can't really tell the two apart; a different seed
could flip the order. And when they're tied, the simpler model wins on
everything else: it's faster, has fewer settings to get wrong, and its
coefficients explain it directly. The rule was set before looking, so
it's not a choice made to suit a favourite.

</details>

## 3. Where does the threshold come from?

Why pick 0.35 using out-of-fold predictions on the training rows, rather
than trying thresholds on the test set and keeping the best?

<details>
<summary>Check yourself</summary>

A threshold is a setting like any other, so choosing it on the test set
leaks the test set into the model (question 1 again). Out-of-fold
predictions are the honest stand-in: every training ride is scored by a
model that never saw it, so they behave like new data. And they did: the
test numbers at 0.35 matched the out-of-fold ones closely.

</details>

## 4. Hands-on: decide at the dock

Change the brief. Now the offer has to show **when the ride starts**,
so distance, duration and speed aren't known yet.

**a)** In your copy, set
`NUMERIC = ["temp_c", "rain_mm", "hour", "weekend"]` and run sections 1
to 3 only. What CV AUC do the two contenders get now?

**b)** Is that model worth shipping? What would you write in the
report's limits section?

<details>
<summary>Expected results</summary>

```
5-fold CV AUC, start-of-ride features only:
    logistic regression     0.618  (std 0.024)
    gradient boosting       0.592  (std 0.014)
```

The same pipelines, the same data, and the score drops from 0.84 to
0.62, because the information that mattered (how long and how far) only
exists after the ride. That's not a modelling failure; it's the honest
answer to a harder question. It still beats the 0.5 floor, so it might
be good enough for a cheap, low-stakes nudge, but the report should say
plainly that at the dock the model can only lean on weekend and time of
day, and is wrong a lot.

</details>

<details>
<summary>One way to write it</summary>

Near the top of `my_lesson_048.py`:

```python
NUMERIC = ["temp_c", "rain_mm", "hour", "weekend"]
```

And at the bottom of `main()`, comment out everything after
`section_contenders(X_train, y_train)`.

</details>

## 5. (Optional) A split that respects time

The report admits the random split "may flatter the model". Test that.
Train the chosen model (`make_logistic().set_params(model__C=0.01)`) on
rides that started before 24 September, and score AUC on rides from the
24th onwards. Before you run: higher or lower than 0.839?

<details>
<summary>Check yourself</summary>

```
train rows / test rows -> (956, 220)
test AUC on the last week -> 0.847
```

About the same, slightly higher. Casual and member riding didn't change
much within September, so the random split wasn't flattering this model.
Now you know rather than hope, and the limits section can say so. With a
whole year of data, seasons would make the time-based split matter a lot
more.

```python
rides = build_features(pd.read_csv(RIDES_CSV))
early = pd.to_datetime(rides["started_at"]) < "2026-09-24"
model = make_logistic().set_params(model__C=0.01)
model.fit(rides.loc[early, FEATURES], rides.loc[early, "casual"])
proba = model.predict_proba(rides.loc[~early, FEATURES])[:, 1]
show("train rows / test rows", (int(early.sum()), int((~early).sum())))
show("test AUC on the last week", r3(roc_auc_score(rides.loc[~early, "casual"], proba)))
```

</details>

---

That's the Phase 2 mini-project, and the end of classical ML. If one
thing sticks: *decide everything on the training rows, open the test
set once, choose by rules set in advance, and write down the limits
before anyone has to ask.*
