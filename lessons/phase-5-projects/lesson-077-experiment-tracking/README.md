# Lesson 077 - Experiment tracking

**Phase 5 - Projects and production intuition** | Week 10, Day 1 | Monday 2026-11-09

> **Goal:** Log parameters, metrics and artifacts so you can answer
> "which run was that?"

Time: about 40 minutes. Same venv as lesson 013; nothing new to install.

---

Here's a Thursday that happens to everybody. You tried a forest with 200
trees, then 100, then a ridge model, then the forest again with a
different depth. One of them was clearly the best. Your manager asks,
"Can you rerun the good one?" and you scroll up through a notebook
looking at numbers with no labels.

**Experiment tracking** is the cure, and it's less grand than it sounds:
for every run, write down *what you ran* (settings, data, seed), *what
came out* (scores), and *what it made* (the model file, the notes). Do
it automatically, in a place you can search. That's the whole idea. The
big tools (MLflow, Weights & Biases and friends) do it with nice
dashboards; underneath, they keep the same three things.

To see how little is needed, we'll build a tracker out of the standard
library, in a few dozen lines.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-5-projects/lesson-077-experiment-tracking/lesson.py
```

It takes a few seconds. No downloads and no data file: the bike-share
demand data (three stations, daily rides) is simulated, as in lesson
076. The script writes an `output/` folder next to itself, with one
sub-folder per run, and deletes it on the next run so you start fresh.
The numbers below should match yours.

## 1. The problem: ten runs, no notes

```
  runs made                                      -> 10
  scores that scrolled past                      -> [4.618, 4.618, 4.616, 5.715, 5.092, 5.61, 4.924, 4.848, 4.864, 4.77]
  what the notebook remembers                    -> 'only the last one'
  which settings gave the best score?            -> '...it was one of those'
```

Ten quick tries: three ridge models and seven forests. The scores print
and scroll away. Looking at that list, can you say which settings gave
the 4.616? We can't either, and we just ran them. The first lesson of
tracking is that **memory is not a log**. Whatever you'd be annoyed to
lose, write to disk the moment you have it.

## 2. A tiny tracker

```
  run id (hash of the settings)                  -> '05d93d99'
  files in the run folder                        -> ['metrics.json', 'model.pkl', 'params.json', 'status.json']
  same settings give the same id                 -> True
  different settings give a new id               -> True
  data hash is part of the settings              -> True
```

The `Run` class is a context manager (the `with` blocks from lesson
008). Inside the block you train and call `r.log(...)` with scores and
`r.save(...)` with files. On the way in it writes `params.json`; on the
way out it writes `metrics.json` and a `status.json` that says
`finished` or `failed`. One folder per run:

- **params.json**: the model name, its settings, the seed, and a hash of
  the data. Everything that decides the outcome.
- **metrics.json**: the scores.
- **status.json**: did it finish, how long it took, any tags.
- **model.pkl** (or whatever you produce): the artifact.

The folder name is the **run id**, a short hash of the settings. That
has a lovely side effect: the same settings always give the same id, so
a repeated run lands on the same folder instead of creating a second
copy. Change any setting (or the data hash) and you get a new id. This
is the same trick as lesson 076's fingerprint, now used to name things.

Pickle (`model.pkl`) is Python's way of saving objects. Fine for your
own models in your own project; never unpickle a file you got from a
stranger, since it can run code.

## 3. A sweep you can question

```
  runs on disk                                   -> 10
  best run id                                    -> '9ac693af'
  best settings                                  -> ('ridge', {'alpha': 10})
  best validation MAE                            -> 4.616
  worst validation MAE                           -> 5.715
  top 3 (model, val MAE)                         -> [('ridge', 4.616), ('ridge', 4.618), ('ridge', 4.618)]
  deepest forest: train MAE vs val MAE           -> (1.615, 4.77)
```

Now we run the same ten tries again, this time inside `Run`, tagged
`sweep-1`. `load_runs()` reads every folder back into a pandas table (a
row per run: id, model, settings, status, scores). The questions that
were impossible a minute ago take one line each: sort by validation
error, take the top three, find the worst.

Two things to notice. First, the winner is the plain **ridge** model,
and the top three are all ridge. The fancy model didn't earn its place.
That is a fairly common outcome on small tabular data, and you'd never
have noticed from scrolling output. Second, the last line: the deepest
forest has a *training* error of 1.6 and a *validation* error of 4.8.
It memorised the training days (lesson 032's overfitting). Log
`train_mae` next to `val_mae` and that gap is visible at a glance.

We choose with **validation** data: the 30 days before the final 30.
The very last 30 days are the **test** set, and nothing gets decided
with it.

## 4. Comparing fairly: how big is the noise?

```
  best ridge settings                            -> {'alpha': 10}
  best forest settings                           -> {'max_depth': None, 'n_estimators': 200}
  forest val MAE over 5 new seeds                -> [4.701, 4.741, 4.754, 4.717, 4.724]
  forest spread (max - min)                      -> 0.053
  forest is worse than ridge by (mean MAE)       -> 0.111
  gap larger than the seed spread?               -> True
  verdict                                        -> 'keep the simple model'
  model in each run                              -> ('ridge', 'forest')
  val MAE change, ridge -> forest (1 seed)       -> 0.153
```

Before we crown anything, a question: is a difference of 0.15 real, or
is it the luck of one seed? A forest uses random numbers, so rerunning
it with five other seeds shows how much the score moves *by chance
alone*: from 4.70 to 4.75, a spread of 0.05. The forest averages 0.11
worse than the best ridge, which is bigger than that spread, so
the gap isn't just seed noise. That's a rough rule, and a good habit
before the next big claim: **a difference smaller than the run-to-run
noise is not a result**. (Lesson 026's bootstrap gives a more careful
version.)

The tracker makes this cheap. Five more runs cost five more lines in the
log, tagged `seeds`, and nobody has to remember which seeds were used.

One honest caveat: we also picked the best of ten settings on the same
validation set. The more things you try, the more likely the winner
looks good partly by luck (lesson 073's tuning-on-noise demo). That's
why we keep a test set.

## 5. An artifact you can reload

```
  reloaded model's validation MAE                -> 4.616
  matches the logged number                      -> True
  test MAE (looked at once, at the end)          -> 4.328
  baseline: same as yesterday, test MAE          -> 8.633
  files in the chosen run                        -> ['metrics.json', 'model.pkl', 'notes.txt', 'params.json', 'status.json']
```

A log entry nobody can reproduce is gossip. So we do the check that
makes the log trustworthy: load the saved model from the best run's
folder, score the validation set again, and compare with the number in
`metrics.json`. They match. If they hadn't, something (data, code, or
what got saved) would have been wrong, and we'd want to know now.

Only then do we touch the test set, once: MAE 4.33, against 8.63 for
the "same as yesterday" baseline (the habit from lesson 075). We save
that number into the chosen run's folder as a note. If you peek at the
test set after every run, it turns into a second validation set, and
then your final number is optimistic.

## 6. Skipping repeats, and recording failures

```
  sweep repeated, runs skipped                   -> 4
  runs on disk, before and after                 -> (15, 15)
  failed runs recorded                           -> 1
  error text kept                                -> "InvalidParameterError: The 'n_estimators' parameter of Rando"
  finished vs failed                             -> {'finished': 15, 'failed': 1}
```

Because ids come from the settings, we can ask "has this exact run
already finished?" before starting it. Repeating the first four tries
skips all four and creates nothing new, which on a 6-hour run is a
real saving. (The skip is only safe because the data hash is in the
settings; if the data changed, the id changes and the run happens
again.)

The last part is about failures. We try a forest with `n_estimators=-5`,
which crashes. `Run.__exit__` catches the error, writes `failed` with
the message, and lets the sweep carry on. A failed run in the log is
useful: next week you won't try that setting again, or you'll wonder why
there's an empty folder. Silent gaps are worse than recorded failures.

## What real tools add

You don't have to build this yourself. **MLflow** (open source, runs
on your laptop) has `mlflow.log_param`, `mlflow.log_metric` and
`mlflow.log_artifact`, plus a browser page to sort and compare runs.
**Weights & Biases** and others do the same as a hosted service. They
also record things we skipped: the code version (the git commit),
system info, and curves logged over time. Our few dozen lines show that those
tools are not magic, and that the habits are what matter:

- one record per run, written automatically;
- settings, data version and seed in it;
- scores, with the train/validation gap;
- the artifact, reloaded once to prove it works;
- failures recorded, repeats skipped.

If you do try MLflow, the lesson's vocabulary maps directly: "params",
"metrics", "artifacts" and "run id" are all the same words.

## What you can do now

- Say what an experiment record needs: settings, data version, seed,
  scores, artifacts, status.
- Write a small tracker with a context manager and one folder per run.
- Name runs from a hash of their settings, so repeats are detected.
- Query a pile of runs: best, worst, top three, train/validation gap.
- Measure seed noise before claiming that one setting beats another.
- Reload an artifact to confirm the log is true.

## What to do now

1. Run `lesson.py`, then open `output/runs/` and read a `params.json`,
   a `metrics.json` and a `status.json`.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   adds the git commit to every run's record.
3. Next, lesson 078: data versioning and feature pipelines. The data
   hash in today's run id is a small taste of it.

A run you didn't record is a run you'll have to do again.
