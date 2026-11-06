# Lesson 076 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work on a copy of the
project the lesson wrote (running `lesson.py` again rebuilds
`output/bike_demand`, so don't edit it in place). From the repo root:

```bash
cd lessons/phase-5-projects/lesson-076-project-structure-tests-and-reproducibility
python lesson.py
cp -r output/bike_demand ../../../my_bike_demand
cd ../../../my_bike_demand
PYTHONPATH=src python -m unittest discover -s tests    # Windows PowerShell: $env:PYTHONPATH="src"
```

---

## 1. What belongs in `src/`?

You have a notebook with 40 cells: loading, cleaning, three plots, a
feature function, a model fit, and a final chart. Which pieces would you
move into `src/`, and which stay in the notebook?

<details>
<summary>Check yourself</summary>

Move anything you'd want to **run twice or test**: loading, cleaning,
the feature function, the fitting and scoring code. Keep the
exploration (plots, "what does this column look like?") in the notebook,
and have it import from `src/`. A good sign you've split it well: the
notebook gets shorter, and the functions in `src/` take their inputs as
arguments instead of reading global variables that were set in some
earlier cell.

</details>

## 2. Which test would have caught it?

A teammate changes `make_rides` so that Uni Square's rides are recorded
in the wrong order by date. Nothing crashes. Which existing tests might
notice, and what new test would you add?

<details>
<summary>Check yourself</summary>

`add_features` sorts by station and date first, so the lag test may
still pass: the features would be right even if the raw table is not.
Whether it matters then depends on the rest of the code, which is a
reason to test the raw data too. A new test in `test_data.py`: for each
station, `df.groupby("station")["date"].is_monotonic_increasing` is
`True` for every group, and the dates have no gaps (`diff()` is exactly
one day). Validation checks like these belong in `validate`, so they
run on real data as well as in tests.

</details>

## 3. Does a seed make it identical everywhere?

You set `random_state=76` and get an MAE of 4.411. A colleague on
another laptop, with the same code, gets 4.409. List three possible
reasons.

<details>
<summary>Check yourself</summary>

(1) Different library versions: a scikit-learn upgrade can change how
trees are grown. That's what exact pins and the fingerprint are for.
(2) A different operating system, CPU, or number of threads: floating
point sums can be added up in a different order, and the last digits
differ. (3) Different data: a CSV that was re-exported, or rows in a
different order. The data hash would show that. Tiny differences like
this are normal; a big one means something real changed. Compare the
fingerprints first.

</details>

## 4. Hands-on: write the test first

Add a rule: `rain_mm` can't be negative. Do it test-first. In
`tests/test_data.py`, write a test that sets a negative rain value and
expects a `ValueError` mentioning "rain". Run the tests and watch
exactly one fail. Then add the rule to `validate` in `data.py` and run
them again. Finally, as an experiment, remove `random_state` from the
model, then remove the `set_seeds` call, and run the training command
three times each time. Predict what you'll see.

<details>
<summary>Check yourself</summary>

The new test, inside the `DataTests` class:

```python
    def test_negative_rain_rejected(self):
        df = make_rides(1)
        df.loc[5, "rain_mm"] = -1.0
        with self.assertRaisesRegex(ValueError, "rain"):
            validate(df)
```

First run: `Ran 10 tests`, `FAILED (failures=1)`, with
`AssertionError: ValueError not raised`. That red result matters: it
proves the test can fail, so a pass later means something. Then add to
`validate`, before the duplicates check:

```python
    if (df["rain_mm"] < 0).any():
        raise ValueError("negative rain")
```

Second run: `Ran 10 tests`, `OK`.

For the seed experiment, run:

```bash
PYTHONPATH=src python -m bike_demand_pkg.train config.json
```

With `random_state` removed but `set_seeds` kept, all three runs print
the same MAE (4.411). Why? A model with no `random_state` uses numpy's
global generator, and `set_seeds` seeds it, so the seed still reaches
it. With `set_seeds` removed too, the three runs differ: about 4.359,
4.397 and 4.439 for us. Your numbers will vary. That spread, from the
model alone, is bigger than the gap between seeds 76 and 77 in the
lesson, so one lucky run proves nothing.

</details>

## 5. Optional puzzle: a smarter leak test

`test_today_never_leaks_into_todays_features` checks one row. Why is
that enough to catch the missing `shift(1)`, and what kind of
mistake might it still miss?

<details>
<summary>Check yourself</summary>

The bug puts today's value inside today's average, so changing today's
rides to 10,000 changes the average at that row, and the test sees it.
Any row from day 8 onwards would show the problem. It could miss a
leak that only happens somewhere else: say, a feature that's right
for the middle of a station's history but uses the *next* station's
rows at the boundary, or one that only breaks when a day is missing. A
stronger version loops over many rows, or uses a property-style check
("for any row, setting rides to anything doesn't change that row's
features"). Cheap tests catch the common mistakes; add more when you
find a new kind of bug, so that mistake can only happen once.

</details>
