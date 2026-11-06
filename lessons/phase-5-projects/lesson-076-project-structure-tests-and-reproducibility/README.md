# Lesson 076 - Project structure, tests and reproducibility

**Phase 5 - Projects and production intuition** | Week 9, Day 5 | Friday 2026-11-06

> **Goal:** Lay out a real project, pin dependencies, seed everything,
> and test the data code.

Time: about 45 minutes. Same venv as lesson 013; nothing new to install.

---

Think of the last notebook you made. It runs from top to bottom today.
In a month, will it? Did you run cell 14 before cell 9? Which version of
pandas was it? Which random numbers did it see? A result you can't
reproduce is a rumour, and the person most likely to be fooled by it is
you, three weeks from now.

Lesson 075 decided what to build. This lesson is about building it so
that it still works when somebody else (or you, later) comes back to it.
We do that with a small project that has a clear layout, tests, fixed
seeds and pinned packages. The setting is still the bike-share scheme:
predicting daily rides at three stations.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-5-projects/lesson-076-project-structure-tests-and-reproducibility/lesson.py
```

The script *writes the project for you*, into an `output/` folder next
to it, then runs it and breaks it on purpose. It starts Python several
times, so give it 10 to 20 seconds. No downloads and no data file: the
demand data is simulated by the project. The numbers below should match
yours. The one thing that will differ is the version numbers, since
those are your own.

## 1. A layout you can find things in

```
    README.md
    config.json
    requirements.txt
    src/bike_demand_pkg/__init__.py
    src/bike_demand_pkg/data.py
    src/bike_demand_pkg/features.py
    src/bike_demand_pkg/train.py
    tests/test_data.py
    tests/test_features.py
  files written                                  -> 9
  lines of code in src/                          -> 97
```

Open `output/bike_demand/` after the run and look around. Each piece has
one job:

- `src/bike_demand_pkg/` is the code, split by purpose: `data.py` makes
  and checks the table, `features.py` turns rows into model inputs,
  `train.py` fits and scores. It's the same split as lesson 036's
  pipeline, as files. (It's a *package*: a folder with an
  `__init__.py`, as in lesson 009.)
- `tests/` holds checks that run in seconds and say "still correct" or
  "something broke".
- `config.json` has the settings (seed, test-window length), so they
  aren't buried in the code.
- `requirements.txt` says which package versions the project was run
  with.
- `README.md` says how to run it in two commands.

Notebooks are good for looking around. Once something works, move the
logic into functions in `src/` and call them from the notebook, so
there's one copy of the logic and it can be tested.

## 2. Tests that guard the data code

```
  exit code                                      -> 0
  last line                                      -> 'OK'
  tests that ran                                 -> 9
```

We use Python's built-in `unittest`, so there's nothing to install.
(The popular `pytest` finds and runs the same files, if you prefer it
later.) Run them from the project folder with:

```bash
cd lessons/phase-5-projects/lesson-076-project-structure-tests-and-reproducibility/output/bike_demand
PYTHONPATH=src python -m unittest discover -s tests       # Windows PowerShell: $env:PYTHONPATH="src"
```

`PYTHONPATH=src` tells Python where our package lives.

What's worth testing? **The code that touches your data**, because
that's where silent mistakes live. A model that's slightly worse is
visible in a metric. A feature that's slightly wrong never is. The nine
tests check:

- the same seed makes the same table, and a different seed doesn't;
- clean data passes the validator, and bad data is rejected with a clear
  message: negative ride counts, duplicate station-days, a missing
  column;
- `lag_1` really is *yesterday's* rides, for the *same station*, and the
  first day of each station has none (so values don't spill from one
  station to the next);
- changing today's rides leaves today's features alone.

Each test is small and has a name that reads like a sentence. When one
fails, the name tells you what broke before you open the code.

## 3. A test catches a quiet bug

```
  exit code with the bug                         -> 1
  failing test                                   -> 'test_today_never_leaks_into_todays_features'
  summary                                        -> 'FAILED (failures=1)'
  MAE with the leak (looks great)                -> 3.746
  MAE after fixing it (the truth)                -> 4.411
  MAE of 'same as yesterday'                     -> 8.733
```

The script now plays a colleague who "tidies up" `features.py`: the
7-day average used `shift(1).rolling(7).mean()`, and the helpful edit
drops the `shift(1)`. The average now contains *today's* number.
Everything still runs. No error, no warning.

Only one test fails, and it's the one that checks that today's rides
can't influence today's features. Meanwhile the model's error drops
from 4.41 to 3.75 rides per station per day: a great result, which is
a **lie** (it's lesson 075's leak, in code). The only thing that stops
you from reporting it is a test that you wrote earlier. The script
restores the file afterwards.

Also look at the last line: the model's 4.4 is about half the error of
"tomorrow will be like yesterday", 8.7. That's the baseline habit from
lesson 075, and the number that tells you the model is earning its
place.

## 4. Seeds, and what they don't cover

```
  unseeded, two draws equal?                     -> False
  seeded, two draws equal?                       -> True
  set order, hash seed 1                         -> "['market', 'lakeside', 'airport', 'uni', 'depot']"
  set order, hash seed 2                         -> "['market', 'depot', 'lakeside', 'airport', 'uni']"
  same hash seed twice gives same order          -> True
  two full runs give identical results           -> True
```

A **seed** is the starting point of a random number generator (lesson
016). Same seed, same "random" numbers. Everything random in a project
needs one: the data simulator, the train/test shuffle, the model
(`random_state=...`), and in PyTorch `torch.manual_seed`. In our project
`train.py` calls `set_seeds` once at the top, and passes the seed on
wherever a function takes one.

The middle lines show a sneaky one. Python randomises the order of
elements in a `set` of strings each time it starts, so a program that
loops over a set can behave differently on every run, with no random
numbers in sight. (Your exact order may differ from the one printed
here; what matters is that the two seeds disagree and the same seed
agrees.) The fix is to **sort** before looping over a set, or to use a
list or dict, which keep their order. Two other things to know: some
operations on a GPU aren't deterministic unless you ask, and
multi-threaded code can add numbers in different orders.

The last line is the proof that counts: the whole training script, run
twice in fresh Python processes, gives exactly the same output. Check
that for your own projects, because "I set a seed" and "my results
reproduce" are different statements.

## 5. Pinning the environment

```
  pinned packages                                -> ['numpy', 'pandas', 'scikit-learn']
  every pin is exact                             -> True
  environment matches pins                       -> []
  after editing a pin                            -> [('numpy', True)]
```

The course's root `requirements.txt` uses *ranges* such as
`pandas>=2.2,<3`, which are friendly to learners. A project you need to
reproduce wants **exact pins**: `numpy==x.y.z`. The script writes yours
from what's installed right now, using `importlib.metadata.version`
(you could also run `pip freeze`).

Why bother? Libraries change behaviour between versions: a default
flips, a random generator is rewritten, a function is deprecated. Two
people with different versions can run identical code and get different
numbers, and then the question "who's right?" can eat a day.

`check_env` compares what's installed with the pin file and lists the
differences, which you'd run at the start of a job. It's empty when all
is well. When we edit one pin in the file to something that isn't
installed, it reports exactly that package. A real project usually adds
Python itself (a `.python-version` file), and, for serious needs, a
lock file that also pins the packages your packages need. Tools such as
`uv` and `pip-tools` create these.

## 6. One command, and a fingerprint

```
  MAE: model                                     -> 4.411
  MAE: yesterday's rides                         -> 8.733
  rows: train, test                              -> (279, 60)
  fingerprint keys                               -> ['config_hash', 'data_hash', 'numpy', 'pandas', 'python', 'scikit-learn', 'seed']
  seed in the fingerprint                        -> 76
  data hash length                               -> 12
  seed 77: MAE model                             -> 4.27
  seed 77: data hash changed                     -> True
  seed 77: config hash changed                   -> True
```

The project runs with one command, `python -m bike_demand_pkg.train
config.json`, and prints its results as JSON. Notice the split: the
last 20 days are the test set, as lesson 021 taught for dates, so the
model never trains on the future.

Along with the scores, the output carries a **fingerprint**: the seed,
a short hash of the data, a hash of the config, and the versions of
Python and the libraries. A hash is a short code computed from the
contents: change one byte and it changes. Save the fingerprint with
every result, and "which run was that?" has an answer (which is lesson
077's subject).

The last three lines change the seed to 77. The error moves from 4.41
to 4.27, a difference that was just luck of the draw (don't read it as
"77 is better than 76"). The data hash and config hash both change, as
they should, since the data and the settings are different.

## What you can do now

- Organise a project: `src/` package, `tests/`, a config, pinned
  requirements and a README.
- Write small tests for data code, including one that proves the future
  doesn't leak into features.
- Seed every source of randomness, and know where seeds don't reach.
- Pin exact versions and check the environment against them.
- Record a fingerprint with each result.

## What to do now

1. Run `lesson.py`, then open `output/bike_demand/` and read
   `features.py` and its tests side by side.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   adds a validation rule the test-first way.
3. Next weekday, lesson 077: experiment tracking, a tidy way to log
   every run's settings and results.

If a result can't be re-run, it hasn't happened yet. Make the second
run cheap.
