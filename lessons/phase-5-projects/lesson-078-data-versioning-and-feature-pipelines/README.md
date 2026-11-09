# Lesson 078 - Data versioning and feature pipelines

**Phase 5 - Projects and production intuition** | Week 10, Day 1 | Monday 2026-11-09

> **Goal:** Keep data and features reproducible as they change under you.

Time: about 40 minutes. Same venv as lesson 013; nothing new to install.

---

Lesson 077 logged *which run* produced a score. Today's lesson is about
the thing that sneaks around the log: the data. Code has git, so you can
always ask "what did this file look like last Tuesday?" Data usually has
`rides.csv`, then `rides_new.csv`, then `rides_final_v2_REAL.csv`, and
sometimes somebody quietly fixes three rows in the original.

Your model didn't change. Your code didn't change. Your score did. Where
did the difference come from? If you can't answer that, you can't trust
any comparison you've made, which makes this one of the least glamorous
and most useful lessons in the course.

We'll fix it in three small steps: give each dataset a **fingerprint**,
keep **snapshots** under that fingerprint, and make features with **one
versioned function** whose results are cached by what went in.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-5-projects/lesson-078-data-versioning-and-feature-pipelines/lesson.py
```

It takes a second or two. No downloads and no data file: the ride data
(three stations, 90 days) is simulated, in the same form as lessons 076
and 077. The script writes an `output/` folder next to itself and wipes
it at the start of every run. The numbers below should match yours,
with one exception: the **hash strings** come from your pandas version,
so yours may look different. What matters is what they do, for example
"same" and "changed", not the letters.

## 1. The problem: same name, different data

```
  file name                                      -> 'rides.csv'
  MAE on Monday                                  -> 8.826
  MAE after somebody fixed three rows            -> 8.888
  file name changed?                             -> False
  can you get Monday's file back?                -> 'no'
```

We train a small ridge model on `rides.csv`: error 8.83. Then somebody
"just fixes" three rows in the same file, and the same code gives 8.89.
The gap is small here, but nothing in the file name, the code or the
log says the data moved. On a real project it could be a lot bigger, and
the old file is gone for good. A score is only meaningful *together
with* the data that produced it.

## 2. Fingerprints: a hash of the contents

```
  hash of delivery 1                             -> '000b712ac338'
  hash of delivery 2                             -> '4b354d581967'
  three rows changed -> hash changed             -> True
  same data, shuffled rows -> same hash          -> True
  one value off by 0.1 -> hash changed           -> True
```

A **hash** is a short code computed from contents, as in lessons 076
and 077. Change anything, even one temperature by 0.1, and it changes.
Leave the contents alone and it stays the same. It gives you a name for
a dataset that doesn't depend on the file name, the date or who sent it.

One choice to make is *what* you hash. Hashing the file's raw bytes
treats two CSVs with the same rows in a different order as different.
Our `table_hash` sorts the rows first and hashes the values, so the
shuffled copy gets the **same** fingerprint. That fits our data, where
row order means nothing. If row order matters for yours (a time series
stored in order, say), hash it in order. For files too big to load, the
usual choice is hashing the bytes in chunks.

## 3. Snapshots, stored under their hash

```
  snapshots in the store                         -> True
  number of snapshots                            -> 2
  re-sent delivery 1 stored again?               -> False
  files in a snapshot                            -> ['data.csv', 'manifest.json']
  manifest rows, columns                         -> (270, ['date', 'station', 'temp_c', 'rain_mm', 'rides'])
  score on snapshot 1 (Monday's number)          -> 8.826
  score on snapshot 2                            -> 8.888
  a model can now say which data it used         -> {'data_version': '000b712ac338'}
```

`save_snapshot` puts each new dataset in `store/<hash>/`, next to a
small `manifest.json` that records the row count, the column names and
types, and a note about where it came from. Two things follow
automatically:

- **Nothing is overwritten.** A new delivery gets a new folder. Old
  ones stay put.
- **Duplicates collapse.** When delivery 1 arrives again with its rows
  shuffled, it hashes to a folder we already have, and nothing new is
  stored. That's called *content addressing*, and it's how git stores
  files too.

Now Monday's number is reproducible: load snapshot `000b712ac338` and
you get 8.826 again, however many corrections arrived since. The last
line shows the habit: store the data version with the model, as we
stored the run settings in lesson 077.

This is also what tools like **DVC**, **lakeFS** and Delta Lake's "time
travel" do at scale: keep big data outside git, and put a small pointer
(a hash) *inside* git. For a small project, our store plus a
`data_version` in the config is a perfectly respectable version of it.
Large files shouldn't go into a plain git repo, because every change
keeps a full copy of the old one forever.

## 4. Checking a new delivery before you trust it

```
  delivery 2 problems                            -> []
  delivery 3 problems found                      -> 3
    - columns changed: missing ['rain_mm'], new ['rainfall']
    - station: new values ['Airport']
    - temp_c: mean moved from 22.1 to 71.7
```

Deliveries arrive with surprises. `compare(old, new)` lists them in
plain English, without us reading every row. Delivery 3 has three:

- a column was **renamed** (`rain_mm` became `rainfall`), so code that
  asks for `rain_mm` will crash, which is the *kind* failure;
- a **new station** appeared, so a model that never saw "Airport" meets
  it with no history;
- the temperature column **mean jumped** from 22 to 72. The numbers
  still look like numbers, and nothing crashes. The data team switched
  to Fahrenheit. This is the *unkind* failure: the model would run and
  give nonsense.

Note delivery 2, our "three corrected rows". `compare` finds nothing
wrong, which is right: the corrections are plausible. The hash still
tells you it's different data. So the two tools do different jobs: the
**hash** tells you *that* something changed; the **checks** help you
decide whether it's *safe*. Lesson 076's `validate` is the sibling:
validation asks "is this table legal?", `compare` asks "is it like the
last one?".

## 5. A feature pipeline with a version

```
  feature version                                -> '4257441b'
  same code again -> same version                -> True
  rows in, rows out                              -> (270, 249)
  rows lost to warm-up (7 days x 3 stations)     -> 21
  first row has no unknown past                  -> True
```

A **feature pipeline** is the code that turns raw data into the columns
a model sees. The rule of thumb: there should be **one** function for
it (`build_features`), and everything (training, evaluation, serving)
calls that function. Copies drift apart; one of them gets fixed and
the others don't.

Our function sorts by station and date, builds `lag_1` (yesterday's
rides), `avg_7` (the average of the previous seven days, with the
`shift(1)` that keeps today out of it, as lesson 076 insisted) and a
weekend flag. Seven days of history are needed, so the first week of
each station can't have features: 21 rows are dropped, and 249 remain.

`feature_version()` hashes the *source code* of `build_features` plus
the list of feature columns. We never have to remember to bump a number:
edit the function or the list, and the version changes by itself. That's
the same instinct as lesson 077's run id: let a hash name it rather than
trusting a human to update a label.

## 6. Caching by (data version, feature version)

```
  calls: built, reused                           -> (2, 1)
  cached files                                   -> True
  edited feature code -> new version             -> True
  old cache file would be hit by new code        -> False
```

Feature building can be slow in real projects (joins, windows over
millions of rows). So `get_features` saves each result to a file called
`<data hash>-<feature version>.csv` and reuses it when it can. We call it
three times: twice for snapshot 1 and once for snapshot 2. It builds
twice and reuses once.

The cache key has both halves on purpose. With the data hash alone,
changing the features would hand back **stale** features. With the
feature version alone, new data would. A cache that can be wrong without
telling you is worse than no cache. Then the script "improves" the
window from 7 days to 14: the version changes, and the old cache file
isn't picked up. (Old files stay around for old models; deleting them
is a separate decision.)

## 7. Training and serving use the same function

```
  serving row date                               -> '2026-10-30'
  lag_1 is yesterday's rides                     -> True
  avg_7 is the mean of the last 7 days           -> True
  features used by the model                     -> ['temp_c', 'rain_mm', 'lag_1', 'avg_7', 'is_weekend']
  columns in the serving row                     -> True
  what we'd store with the model                 -> {'data_version': '000b712ac338', 'feature_version': '4257441b', 'features': ['temp_c', 'rain_mm', 'lag_1', 'avg_7', 'is_weekend']}
```

At serving time you'll have one new row: "Market, today, 24 degrees, no
rain." The features that need history (`lag_1`, `avg_7`) have to come
from the last few days of stored data. We build them by running the
**same** `build_features` over ten days of history plus today's row, and
taking the last row. We then check that `lag_1` really is yesterday's
rides and `avg_7` really is the seven-day mean before today.

If training used one version of "average of the last 7 days" and
serving used a hand-written copy that counted today too, the model would
meet inputs it never saw in training. That's **training-serving skew**,
and it's a common reason a model looks great offline and poor in
production. One function removes the whole class of bug. (Lesson 079
will serve a model, so this is the groundwork.)

The final line is the **record** to keep next to the model:
`data_version`, `feature_version` and the column list. Together with
lesson 077's run record, "which data, which features, which settings, which
score" now has an answer for every model you ship.

Big organisations package this idea as a **feature store**: a shared
place where features are defined once, computed for training and
looked up for serving. You don't need one yet. The habit it encodes, one
definition and one version, you can start with a single function today.

## What you can do now

- Fingerprint a dataset, and choose what to hash.
- Keep immutable snapshots under their hash, with a manifest.
- Compare a new delivery with the last and catch renamed columns, new
  categories and silent unit changes.
- Build features in one function, and version it with a hash of its code.
- Cache features by (data version, feature version), and say why both.
- Record data and feature versions with every model.

## What to do now

1. Run `lesson.py`, then open `output/store/` and look at a
   `manifest.json`, and `output/features/` for the cache names.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   adds two more checks to `compare`.
3. Next weekday, lesson 079: serving a model as an API, where today's
   feature function and version record get used for real.

If you can't name the data, you can't trust the number.
