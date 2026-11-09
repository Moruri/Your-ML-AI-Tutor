# Lesson 077 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere
of your own (it writes an `output/` folder next to wherever it sits),
and edit the copy. From the repo root:

```bash
cd lessons/phase-5-projects/lesson-077-experiment-tracking
python lesson.py
cp lesson.py ../../../my_tracker.py
cd ../../..
python my_tracker.py
```

---

## 1. What belongs in a run record?

You trained a model, and a week later you want to rerun it exactly.
List the things you'd need to have written down.

<details>
<summary>Check yourself</summary>

The settings of the model, the random seed, a version of the data (a
hash or a file name plus date), the code version (a git commit), the
library versions (lesson 076's fingerprint), and how the train/test
split was made. Without any one of these, "the same run" may give a
different number. The scores come after: they're what you compare, not
what you need to reproduce.

</details>

## 2. Same settings, new id?

In the tracker, run ids come from a hash of the settings and the data
hash. If you add one row to the data and rerun, will the run get the
same id? Is that what you want?

<details>
<summary>Check yourself</summary>

No: the data hash changes, so the id changes and the run is treated as
new. That is what you want: the old result belongs to the old data, and
reusing it would be silently wrong. If you left the data hash out of
the id, then "skip runs that already finished" would hand back stale
scores. Anything that changes the outcome belongs in the id.

</details>

## 3. Is the winner real?

A colleague logs two runs: model A has validation MAE 4.52 and model B
has 4.49. They say B is better. Without running anything, what would
you ask for?

<details>
<summary>Check yourself</summary>

How much does each score move if you only change the seed (or the
split)? If that spread is about 0.05, a 0.03 difference is noise. Ask
for several seeds of each, then compare the averages and spreads. Also
ask whether B was picked from many tries on the same validation set,
since the best of many tries is partly luck. If both are fine and B
still wins, great. A tiny edge is also worth weighing against cost: a
model that's slightly better but ten times slower to run may not be.

</details>

## 4. Hands-on: log the git commit and a duration

Edit your copy, `my_tracker.py`:

1. In `Run.__init__`, add `"git_commit"` to `self.config`, taken from
   `subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)`.
   If git isn't available or you're not in a repo, store `"unknown"`
   instead of crashing.
2. Should the commit be part of the run **id**? Decide, and explain it
   in a comment.
3. Log the fit time as a metric, `fit_seconds`, using `time.time()`
   around `model.fit`.
4. Re-run and open one `params.json` and `metrics.json`.

<details>
<summary>Check yourself</summary>

Wrapping `subprocess.run` in `try/except (FileNotFoundError, ...)` and
checking `returncode` gives the "unknown" fallback. On the id question
there are two defensible answers. Including the commit means any code
change makes a new id (very safe, but a harmless comment edit repeats
all runs). Leaving it out means the id says "same settings, same data"
and the commit is a note you check when something looks odd. Many
teams keep it out of the id and *in* the record. Whichever you pick,
write the reason down. `fit_seconds` will vary slightly from run to run
(that's why it belongs in `metrics.json`, not in the id).

</details>

## 5. Optional puzzle: two runs, one folder

Two processes start the *same* run at the same moment (same settings, so
the same id). What could go wrong with this tracker, and how would you
fix it?

<details>
<summary>Check yourself</summary>

Both write into the same folder: `params.json` is identical, but they
overwrite each other's `metrics.json`, `status.json` and model file,
and one may read half a file. Fixes: create the folder with
`mkdir(exist_ok=False)` and bail out if it exists (the first process
"claims" the run); or add a timestamp or a counter to the folder name
and keep the settings hash as a separate field; or write each file to a
temporary name and rename it, since a rename is all-or-nothing. Real
tools give every run a fresh random id for exactly this reason, and
keep a hash of the settings as a field for finding duplicates.

</details>
