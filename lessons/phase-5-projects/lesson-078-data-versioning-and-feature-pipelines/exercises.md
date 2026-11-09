# Lesson 078 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere of
your own and edit the copy (it writes an `output/` folder next to
wherever it sits). From the repo root:

```bash
cd lessons/phase-5-projects/lesson-078-data-versioning-and-feature-pipelines
python lesson.py
cp lesson.py ../../../my_versions.py
cd ../../..
python my_versions.py
```

---

## 1. Does the hash notice?

For each change, say whether `table_hash` changes: (a) rows are shuffled;
(b) the columns are put in a different order; (c) a station name goes
from `"Market"` to `"market"`; (d) one temperature goes from 21.0 to
21.00.

<details>
<summary>Check yourself</summary>

(a) No: we sort the rows first. (b) Yes: we sort by the columns in
their current order and hash the values by position, so reordering the
columns changes the result. If you want it to be insensitive, sort the
column names first. (c) Yes: different text, different data. If you'd
call these "the same", normalise the text *before* hashing, as part of
the cleaning step. (d) No: `21.0` and `21.00` are the same float once
read in. Whatever you decide, hashing after cleaning and parsing
answers "is the *meaning* the same?", while hashing raw bytes answers
"is the *file* the same?". Both are legitimate, but pick on purpose.

</details>

## 2. Which failure is the unkind one?

A new delivery (a) has a missing column, (b) has a column renamed, (c)
has all temperatures in Fahrenheit, and (d) has 40% missing rain
values. Rank them from most to least dangerous, and say why.

<details>
<summary>Check yourself</summary>

(c) is the most dangerous: nothing crashes, the model runs, and every
prediction is quietly wrong. (d) comes next, since missing values may
be filled in or ignored silently depending on the code. (a) and (b)
are the same sort of failure and are the *easy* ones: the code
crashes loudly on the first line that uses the column, so you notice the
same day. The general lesson is to spend your checking effort on
changes that don't crash. Cheap checks (the mean, the range, the
missing-value rate, the set of categories) catch most of them.

</details>

## 3. Why both halves of the cache key?

Our cache file is called `<data hash>-<feature version>.csv`. Suppose
you used only the data hash. Describe a day when this goes wrong. Then
do the same for only the feature version.

<details>
<summary>Check yourself</summary>

Data hash only: you change `rolling(7)` to `rolling(14)`, rerun, and
get the *old* features, because a file for that data already exists.
Your "new" experiment is the old one, and you wonder why nothing
improved. Feature version only: a corrected delivery arrives, the
features are unchanged in code, so you reuse a cache built from the
*old* data, and your model trains on stale rows. In both cases nothing
errors; you just get quietly wrong numbers, which is why the key must
contain everything the output depends on.

</details>

## 4. Hands-on: two more checks for `compare`

Edit your copy, `my_versions.py`. Extend `compare(old, new)` so that it
also reports:

1. **A changed row count**, if the new delivery has fewer rows than the
   old one, or more than 20% more.
2. **A value outside the old range**: for each numeric column, any new
   value below the old minimum or above the old maximum, with a count.
   (For `rides`, delivery 2's corrected rows should *not* trip it, and
   a negative ride count should.)

Test it by changing `d2`, then by adding a row with `rides = -3`.

<details>
<summary>Check yourself</summary>

Something like:

```python
if len(new) < len(old) or len(new) > 1.2 * len(old):
    problems.append(f"row count moved from {len(old)} to {len(new)}")
...
out = ((new[col] < old[col].min()) | (new[col] > old[col].max())).sum()
if out:
    problems.append(f"{col}: {out} values outside the old range")
```

Delivery 2 adds 7 to three rides, which could still push one past the
old maximum, so look at the output rather than assuming. If it does, ask
yourself whether "outside the old range" is a *problem* or just *news*.
Good checks give a human something to look at; they don't need to
decide. A negative ride count, on the other hand, should be rejected
outright by `validate`, as in lesson 076.

</details>

## 5. Optional puzzle: the leak in the cache

`get_features` builds features from the whole delivery, then the caller
splits into train and test. A teammate suggests building features for
train and test separately, "to be safe". Which choice is safer for
`lag_1` and `avg_7`, and when would it matter?

<details>
<summary>Check yourself</summary>

For these two features, building from the whole table is fine, because
they only look *backwards* (`shift(1)` then `rolling`), so no test row
leaks into training. Building separately would actually *hurt*: the
first week of the test period would lose its history and get dropped or
filled with NaN, though in real life yesterday's rides are known.
It matters for features that use **statistics of the whole table**,
like "ride count divided by the station's average" or scaling to the
mean. Those must be computed from the training part only, then applied
to the test part (lesson 035's scalers do exactly that). The rule isn't
"always split first" or "never": ask of each feature, "could this value
be known on the day we'd need it?"

</details>
