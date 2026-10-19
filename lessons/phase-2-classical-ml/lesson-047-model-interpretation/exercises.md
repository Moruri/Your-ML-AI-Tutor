# Lesson 047 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-2-classical-ml/lesson-047-model-interpretation/lesson.py my_lesson_047.py
python my_lesson_047.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-2-classical-ml/lesson-047-model-interpretation").resolve()
```

`load_rides`, `make_forest`, `auc`, `FEATURES`, `heading` and `show` are
there to reuse. This gives you the same split as the lesson:

```python
rides = load_rides(RIDES_CSV)
X, y = rides[FEATURES], rides["casual"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)
```

---

## 1. Why add a column of noise?

The lesson adds `coin`, a column of random numbers. Isn't that just
making the model worse? What is it for?

<details>
<summary>Check yourself</summary>

It's a control. We know for certain the coin has nothing to do with
riders, so any importance tool that ranks it highly is showing you its
own bias. It's a cheap sanity check you can add to any real project, and
it costs at most a sliver of accuracy.

</details>

## 2. "Distance barely matters, let's drop it"

A teammate reads section 3, sees distance's importance at 0.066 next to
duration's 0.145, and suggests removing distance from the data
collection to save effort. What do you tell them?

<details>
<summary>Check yourself</summary>

Distance looks small only because duration is standing in for it (they
correlate at 0.92). Drop *duration* instead and distance's importance
jumps to 0.263 with almost no loss in AUC. Importance is shared between
twins, so it can't tell you which one to keep. Shuffle them as a group,
or retrain with each removed, before deciding.

</details>

## 3. Read the partial dependence

The duration curve goes from 0.17 at 5 minutes to 0.58 at 60 minutes.
Does that mean "a 60-minute ride has a 58% chance of being casual"?

<details>
<summary>Check yourself</summary>

Not quite. It's the *average* prediction over all test rides after each
one was forced to 60 minutes, keeping its own distance, weather and day.
Most of those imagined rides never happen (three in four become slower
than walking pace). The curve tells you the direction and rough size of
the model's response in the range where real rides are; at the far ends
it's the model extrapolating.

</details>

## 4. Hands-on: bring the coin back to life

**a)** Fit `make_forest()` on the training rows and run
`permutation_importance` on the **training** rows instead of the test
rows (`scoring="roc_auc"`, `n_repeats=20`, `random_state=0`). Where does
`coin` rank, and what's the training AUC?

**b)** Now refit the forest without `coin`. What's the test AUC? Is the
difference from 0.837 worth worrying about?

<details>
<summary>Expected results</summary>

```
a) training importance (top four):
       duration_min   0.144
       distance_km    0.086
       coin           0.040
       temp_c         0.038
   training AUC -> 0.981
b) test AUC without coin -> 0.828
```

On training rows the forest has memorised the noise, so shuffling it
hurts, and the coin climbs back to third. A training AUC of 0.98 against
0.84 on test is the same overfitting you met in lesson 031. In (b) the
AUC moves by 0.009, well inside the run-to-run wobble of a forest on 294
test rows: the coin never helped, and removing it doesn't really hurt.

</details>

<details>
<summary>One way to write it</summary>

```python
forest = make_forest().fit(X_train, y_train)
res = permutation_importance(forest, X_train, y_train, scoring="roc_auc",
                             n_repeats=20, random_state=0, n_jobs=-1)
show("training importance",
     pd.Series(res.importances_mean, index=FEATURES).sort_values(ascending=False))
show("training AUC", auc(forest, X_train, y_train))

no_coin = [f for f in FEATURES if f != "coin"]
forest2 = make_forest().fit(X_train[no_coin], y_train)
show("test AUC without coin", auc(forest2, X_test[no_coin], y_test))
```

</details>

## 5. (Optional) Make the trees humbler

Built-in importances fell for the coin because deep trees memorise.
Fit `RandomForestClassifier(n_estimators=200, min_samples_leaf=30,
random_state=0)` and print its `feature_importances_`. Before you run:
does the coin still rank third?

<details>
<summary>Check yourself</summary>

No. With at least 30 rides per leaf the trees can't chase single noisy
rows, and the coin drops to fourth with 0.033, behind weekend at 0.092.
Test AUC barely changes (0.829). Built-in importance gets more honest as
the model gets less overfit, but permutation importance on test data was
honest all along, which is why it's the one to report.

</details>

---

That's model interpretation. If one thing sticks: *measure importance by
shuffling on test data, shuffle twins together, read partial dependence
in the middle of the data, and never mistake any of it for cause and
effect.*
