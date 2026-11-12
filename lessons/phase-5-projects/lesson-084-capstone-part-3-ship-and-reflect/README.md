# Lesson 084 - Capstone, part 3: ship and reflect

**Phase 5 - Projects and production intuition** | Week 10, Day 4 | Thursday 2026-11-12

> **Goal:** Deploy a minimal version, write it up, and decide what you'd
> do differently next time.

Time: about 45 minutes to read and run, then two hours or so on your own
project. Same venv as lesson 013; nothing new to install.

---

This is the last lesson of the course.

Lesson 083 ended with a model that was perfect on the sealed test set
and unsure about half of the messages written the way riders really
write. That's an ordinary place for a real project to be. The question
for today isn't "is it perfect?" but **"what can we responsibly ship,
and how will we know when it's ready for more?"**

Shipping doesn't mean a big launch. It means getting the smallest
useful version into the world with the things that make it safe to
leave there:

1. A script that rebuilds it from scratch.
2. A shipping decision that matches the evidence.
3. A small service.
4. Smoke tests.
5. A monitoring plan.
6. A model card.
7. A project README with an honest look back.

Then we'll look back at the whole course, and forward to what comes
next.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-5-projects/lesson-084-capstone-part-3-ship-and-reflect/lesson.py
```

It takes about ten seconds. Two data files sit next to the script:
`feedback.csv` (lesson 082's snapshot) and `inbox_sample.csv` (lesson
083's 48 real-style messages). The finished project is built in
`output/capstone/` (wiped every run). A small web service runs on your
own computer for a few seconds and is shut down again; nothing leaves
your machine.

## 1. One command trains it from scratch

```
  train.py, lines                                -> 59
  run 1: version, right first time, to person    -> ('router-638e94', '1.0', '0.025')
  run 2: version, right first time, to person    -> ('router-638e94', '1.0', '0.025')
  same model both times                          -> True
```

The "must" list from lesson 082 included *a script that trains from
scratch*. Not a notebook, not "run cells 3 to 17": one command. The
lesson writes `output/capstone/train.py` from four pieces of its own
code (the settings, the cleaning, the pipeline and a `train` function),
then runs it **as a separate program**, twice.

`train.py` checks the data snapshot, rebuilds lesson 082's sealed split,
fits lesson 083's chosen pipeline with the threshold chosen on dev, and
saves `model.pkl` with `model_record.json`. Both runs give the same
model version, `router-638e94`, which is a fingerprint of its test-set
predictions. Same data, same code, same seed, same model (lesson 076).
If you hand the folder to someone else, they can rebuild exactly what
you shipped, and if the data ever changes, the snapshot check stops the
script instead of quietly training on something new.

## 2. Decide how to ship it, from the evidence

```
  inbox sample: right first time, to a person    -> (0.979, 0.479)
  sealed test set (tidy): to a person per day    -> 2
  inbox sample (real-style): to a person per day -> 48
  one duty person can hand-route                 -> 40
  modes considered                               -> ['full auto', 'auto, unsure ones to a person', 'shadow: suggest, person decides']
  shipping as                                    -> 'shadow: suggest, person decides'
  switch to the next mode when                   -> '4 weeks of shadow logs show <= 15% unsure and >= 95% agreement'
```

Turn the scores into a working day. With about 100 messages a day, the
tidy test set says 2 would go to a person. The real-style sample says
48, and one duty person can hand-route about 40. Believe the sample:
it's the one that looks like Monday morning.

Three ways to ship:

- **Full auto.** Every message routed by the model. We have no honest
  idea how often it would misroute real messages, so no.
- **Auto, unsure ones to a person.** What the brief asked for. It fits
  the tidy test set and overflows the duty person on real messages.
  Not yet.
- **Shadow mode.** The duty person routes everything, as today, but sees
  the model's suggestion; both choices are logged. Nobody is worse off
  than now, the person probably gets faster, and every day produces
  real, labelled messages: exactly the data lesson 083 said was missing.

So we ship in shadow mode, **with a written rule for moving on**: four
weeks of logs with at most 15% unsure and at least 95% agreement with
the person. Writing the rule down now stops two things later: switching
on too early because it "seems fine", and never switching on at all.

This is the most common shape for a first ML release, and there's no
shame in it. A model that suggests is still shipped. It's earning its
keep and collecting its own exam papers.

## 3. A tiny routing service

```
  GET /health                                    -> (200, {'status': 'ok', 'model_version': 'router-638e94'})
  I was charged twice for one ride               -> (200, 'payment', 0.97, False, 0.286)
  Baiskeli haina breki kabisa                    -> (200, 'bike', 0.668, True, 1.0)
  POST /route with no text                       -> (400, {'error': 'send JSON like {"text": "..."}, 1 to 1000 characters'})
```

The same pattern as lesson 079, with the standard library's
`http.server`: load the model **once** at start-up, answer `GET
/health` and `POST /route`, give bad requests a clear `400` rather than
a crash, and log every request to `requests.jsonl`.

Each answer carries more than a team name: the **confidence**, whether
the model is **unsure** (below the 0.70 threshold), the share of
**unknown words**, the **mode** (`shadow`, so whoever calls it knows not
to act on it alone), and the **model version**. The Swahili message is
all unknown words, the model says it's unsure, and in shadow mode that's
fine: a person decides. Those extra fields are what make the next two
sections possible. (The unknown-word share is rough: very short words
like "I" never make it into the vocabulary, which is why even the plain
English message scores 0.29. What matters is how it moves over time.)

## 4. Smoke tests before anyone relies on it

```
  health says ok                                 -> 'pass'
  obvious payment message -> payment             -> 'pass'
  obvious station message -> station             -> 'pass'
  empty text is refused with 400                 -> 'pass'
  confidence is between 0 and 1                  -> 'pass'
  version matches the record                     -> 'pass'
  p95 latency under 50 ms                        -> 'pass'
  all smoke tests pass                           -> True
  requests logged so far                         -> 197
```

**Smoke tests** are the quick checks you run every time you deploy:
not "is the model good?" (that was lesson 083) but "is this the model I
meant, wired up the way I meant?" Seven checks, each a line of code:
the service is up, two messages nobody could misread go to the right
team, junk is refused, confidence is a probability, the version
matches the record (so you know *which* model is answering), and
95% of 192 requests come back within a 50 ms budget (lesson 081).

If any of these fail, you don't ship. They take a second to run and
catch the embarrassing problems: the old model file, a broken route, a
missing threshold.

## 5. A monitoring plan you can actually run

```
  week  real-style  unsure  unknown words  agreement  alerts
     1          0%    0.04           0.05       0.98  -
     2          5%    0.05           0.08       0.98  -
     3         10%    0.05           0.09       0.98  -
     4         20%    0.11           0.15       0.95  -
     5         35%    0.17           0.21       0.93  unsure
     6         50%    0.26           0.27       0.90  unsure, unknown_words, agreement

  first week with an alert                       -> 5
  which monitor fired first                      -> ['unsure']
```

Lesson 080 showed that a deployed model never complains; it just gets
quietly worse. So the plan names three signals, each with an alert line,
in `output/capstone/MONITORING.md`:

- **Share marked unsure**, alert above 15%: the brief's limit.
- **Share of unknown words**, alert above 25%: a sign of new language,
  a new product, a new kind of complaint. It needs no labels at all.
- **Agreement with the duty person**, alert below 90%: the only signal
  that actually measures being right. Shadow mode gives you this for
  free, because the person's choice is logged next to the suggestion.

The script tries the plan on six simulated weeks of 500 messages, in
which the share of real-style messages grows from none to half (say, as
the scheme spreads beyond the university crowd who write tidy
messages). The unsure share crosses its line first, in week 5, before
agreement has dropped far. By week 6 all three fire.

The plan also says what to **do** on an alert: read 20 flagged
messages, label them, add them to a fresh evaluation set, then retrain.
An alert with no action attached just teaches people to ignore alerts.

## 6. The model card

```
  model card sections                            -> ['What it is for', 'What it is not for', 'Data', 'How well it works', 'Limitations', 'Monitoring and retraining']
  weakest kind of message (top guess right)      -> ('no keyword', 0.5)
```

A **model card** is one page that travels with the model and answers
the questions anyone sensible will ask before relying on it. Open
`output/capstone/MODEL_CARD.md`. It has:

- **What it's for**, and just as important, **what it's not for**:
  Swahili and Sheng, messages with several problems, anything about
  safety. Lesson 074's point, written where people will see it.
- **The data**, including its biggest weakness: balanced and written
  from templates.
- **How well it works**, with *both* exams side by side, the rules'
  scores next to the model's, and accuracy for every kind of message.
  The weakest is "no keyword", right only half the time.
- **Limitations**, in plain words, including how small the 48-message
  sample is.
- **Monitoring and retraining**: a pointer to the plan and the one
  command that rebuilds it.

Every number in the card is computed by the script, not typed in.
That's what keeps a model card true after the next retrain.

## 7. The project README and the look back

```
  must-haves done                                -> '6 of 6'
  nice-to-haves done                             -> '3 of 3'
  project files                                  -> ['MODEL_CARD.md', 'MONITORING.md', 'README.md', 'data/inbox_sample.csv', 'data/raw/feedback.csv', 'model.pkl', 'model_record.json', 'requests.jsonl', 'train.py']
```

The project's own `README.md` is short: what it does, how to run it,
the results in two sentences, the checklist from lesson 082's plan
(all six musts done, and all three nice-to-haves: the service, the drift
check and the smoke tests), and a section called **Looking back**.

That last section is the one most people skip, and the one that makes
your *next* project better. Four questions, answered honestly:

- **What went well?** Sealing the test set first and logging every run.
  No number in the model card was tuned on its own exam.
- **What went badly?** The real-style sample turned up in lesson 083.
  If it had existed in lesson 082, the 15% limit would have looked
  unrealistic on day one, and the plan would have said "shadow mode
  first" from the start.
- **What surprised us?** A model that's perfect on tidy data was unsure
  about half of real messages, and its unsureness turned out to be the
  most useful thing about it.
- **What will we do differently next time?** Collect 100 real messages
  before choosing a model. Set the threshold on them. Report scores on
  the real mix of topics.

Notice that none of these is "use a bigger model". In real projects,
the lessons are almost always about data, evaluation and scope.

## Your turn: ship and write up your own capstone

About two hours:

1. Make one command that trains your project from scratch. Run it twice
   and check you get the same model.
2. Decide how to ship, from your evidence, not your hopes. Write down
   the rule for moving to the next mode.
3. Wrap it in something people can use: a small API, a command-line
   script, a scheduled job writing a CSV. Smallest useful thing.
4. Write five smoke tests and run them.
5. Name two or three monitoring signals, at least one that needs no
   labels, each with an alert line and an action.
6. Write a one-page model card, with what it's *not* for.
7. Write the README with the four look-back questions.

Then show it to one person, a friend, a classmate, a colleague, and
ask what's unclear. Explaining your project out loud is the final test.

## What you can do now

- Rebuild a model from scratch with one command, reproducibly.
- Choose a shipping mode the evidence supports, with a rule for moving on.
- Serve a model with health checks, clear errors, versions and a log.
- Write smoke tests that catch broken deployments.
- Plan monitoring with label-free signals, alert lines and actions.
- Write a model card and an honest retrospective.

## Eighty-four lessons later

```
  Phase 0: Python foundations for data work      -> '12 lessons'
  Phase 1: Data science basics                   -> '16 lessons'
  Phase 2: Classical machine learning            -> '20 lessons'
  Phase 3: Deep learning foundations             -> '12 lessons'
  Phase 4: Modern AI                             -> '14 lessons'
  Phase 5: Projects and production intuition     -> '10 lessons'
  lessons finished                               -> 84
```

Think back to lesson 001. You may not have written a line of Python.
Since then you've cleaned data with pandas, drawn plots that told the
truth, fitted and cross-validated models, built a neuron by hand and
then whole networks in PyTorch, taken attention and transformers apart,
run language and vision models, built retrieval and agents, made
digits out of noise, and today shipped a project with a model card and
a monitoring plan. That's a lot of ground. Be proud of it.

More useful than any single technique, you've picked up some habits
that will outlast every library in `requirements.txt`:

- **Start with a baseline**, and be suspicious of scores that look too
  good.
- **Seal the test set first**, and choose everything else on dev.
- **Look at the mistakes**, not just the average.
- **Ask who acts on the prediction**, and what a wrong one costs.
- **Keep it reproducible**, so future-you can trust past-you.
- **Ship small, watch it, and say plainly what it can't do.**

### Where to go next

You know enough now to teach yourself. Some good next steps, roughly in
order of how much they'll help:

1. **Finish your capstone properly**, then do a second, different one:
   a different kind of data (images, time series, tabular) and,
   ideally, a real person who'll use it. Your second project will be
   much better than your first, and twice as fast.
2. **Go deeper in one direction you enjoyed.** Time series and
   forecasting, recommender systems, computer vision, NLP and LLM
   applications, or reinforcement learning. Pick one, find a good book
   or course, and build something in it.
3. **Read the documentation.** The scikit-learn user guide and the
   PyTorch tutorials are excellent, and you can follow them now.
4. **Read papers slowly.** Start with the abstract, the figures and the
   experiments section. Ask lesson 073's questions of every result:
   what's the baseline, what's the test set, how big is the interval?
5. **Share your work.** Put your capstone on GitHub with its README and
   model card. Write a short post about what surprised you. Explaining
   is the best way to find out what you really understand.
6. **Keep the habits.** Whatever you build next, the six above apply.

Check [CURRICULUM.md](../../../curriculum/CURRICULUM.md) now and then:
optional deep-dive tracks may appear there.

## What to do now

1. Run `lesson.py`, then read `output/capstone/README.md` and
   `MODEL_CARD.md` as if you were someone new to the project.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   changes the rule for moving on and adds a smoke test.
3. Ship and write up your own capstone.
4. Then take a rest. You've earned it.

Thank you for learning with these lessons, one weekday at a time. It
has been a pleasure. Now go and build something.
