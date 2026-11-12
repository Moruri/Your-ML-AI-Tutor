# Lesson 083 - Capstone, part 2: build and evaluate

**Phase 5 - Projects and production intuition** | Week 10, Day 4 | Thursday 2026-11-12

> **Goal:** Build the full pipeline and evaluate it against a baseline.

Time: about 45 minutes to read and run, then two or three hours on your
own project. Same venv as lesson 013; nothing new to install.

---

Lesson 082 did the slow, careful part: a brief, a scope, a data
snapshot, an audit, a sealed test set and a bar to beat (keyword rules,
75.6% right first time on dev). Today is the part everyone looks forward
to: building the thing.

It's also the part where it's easiest to fool yourself. You try ten
ideas, peek at the test set "just once", nudge a threshold until the
number looks nice, and end up with a score that belongs to your
choices rather than to the model. So today's order is strict:

1. Build one pipeline that goes from raw text to a team.
2. Compare a few candidates on **dev only**, logging every run.
3. Choose the "ask a person" threshold on dev, to fit the brief.
4. Check the win is bigger than the wobble.
5. Open the sealed test set, **once**.
6. Then, the most useful hour of any project: look at what it got wrong.

The worked example is still **route each support message to the right
team**, with the same `feedback.csv`. Keep your own capstone open next
to it and do each step for yours too.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-5-projects/lesson-083-capstone-part-2-build-and-evaluate/lesson.py
```

It takes a few seconds (up to half a minute on a busy laptop). Two data
files sit next to the script: `feedback.csv`, the same snapshot as
lesson 082, and `inbox_sample.csv`, which you'll meet in section 7. The
script rebuilds the capstone folder in `output/capstone/` (wiped every
run) and adds a run log, the chosen model and its record.

## 1. Pick up exactly where lesson 082 stopped

```
  snapshot fingerprint                           -> '79a3efa5db0e'
  dev rows, test rows (test stays shut)          -> (640, 160)
  first three sealed test ids                    -> ['F0018', 'F0019', 'F0025']
```

Before building anything, prove you're standing on the same ground as
last time. The script fingerprints `feedback.csv` and **refuses to run**
if it isn't snapshot `79a3efa5db0e`. Then it rebuilds the split with
the same method and seed, so the 160 test messages are exactly the ones
sealed in lesson 082. (Compare the ids with `data/split.json` from that
lesson's output if you like. They match.)

It's a two-line check, and it saves you from the worst kind of bug: the
one where everything runs, and the numbers quietly belong to different
data.

## 2. One pipeline, from raw text to a team

```
  steps                                          -> ['features', 'model']
  features the model sees                        -> 2780
  example message                                -> 'The M-Pesa charge went through twice, can I get a refund?'
  routed to                                      -> 'payment'
```

Every candidate is a single scikit-learn `Pipeline` (lesson 045):
cleaning and features live **inside** it, next to the model. The main
one joins two kinds of TF-IDF features (lesson 061):

- **Words and word pairs**, like "refund" and "charged twice".
- **Character pieces** of 2 to 5 letters, like "efun" and "pesa". These
  give typos a fighting chance: "refnd" shares most of its pieces with
  "refund".

then a logistic regression on top. Because it's one object, the same
cleaning runs in training, in cross-validation, on the test set and,
in lesson 084, inside the service. You can't forget a step in one place
and not another, which is the bug lesson 078 warned about.

## 3. Experiments on dev only, every run logged

```
  run  candidate              C      dev accuracy  fold spread
    1  rules                    -         0.756  0.727-0.805
    2  nb_words                 -         0.967  0.953-0.984
    3  lr_words                10         0.991  0.969-1.000
    4  lr_words_chars           1         0.991  0.969-1.000
    5  lr_words_chars          10         0.992  0.977-1.000
    6  lr_words_chars         100         0.992  0.969-1.000

  chosen on dev                                  -> 'lr_words_chars C=10'
  runs in runs.jsonl                             -> 6
```

Each candidate is scored with 5-fold cross-validation **inside the 640
dev messages**, using `StratifiedGroupKFold` again so near-identical
messages never sit on both sides of a fold. Every dev message gets a
prediction from a model that never saw it (an **out-of-fold**
prediction). Each run, with its data fingerprint and seed, goes into
`runs.jsonl`, as in lesson 077.

Two things to notice:

- **The rules are in the table.** The baseline gets a row like any
  other candidate, scored the same way. 0.756, as in lesson 082.
- **The top four are the same model, give or take.** 0.991 to 0.992,
  with folds swinging from 0.969 to 1.000. The "winner" won by one
  message in 640. We take it, because we have to choose something, but
  don't read anything into the order. When candidates are inside each
  other's wobble, pick the simplest or the one you understand best.

And a third thing, which you should already be suspicious about: 99%.
Lesson 073 taught you what to think when a score looks that good. Hold
that thought until section 7.

## 4. The "ask a person" threshold, chosen to fit the brief

```
  threshold  right first time  to a person  misrouted
       0.30             0.992        0.000      0.008
       ...
       0.65             0.998        0.025      0.002
       0.70             1.000        0.034      0.000
       0.75             1.000        0.039      0.000

  threshold (highest score within 15% to a person) -> 0.7
  dev: right first time, to a person             -> (1.0, 0.034)
```

The brief said the router may pass a message to a person when it's
unsure. The model gives a confidence (its top probability), and below a
**threshold** the message goes to the duty person instead.

How do we score that? A message a person routes lands in the right
queue first time, so it counts as right, but it costs that person's
time, which is why the brief caps it at 15%. So each threshold gets
three numbers: right first time, share to a person, and **misrouted**
(sent automatically to the wrong team, the expensive mistake).

The rule from the brief is: the best right-first-time score with at
most 15% to a person. On dev's out-of-fold predictions that's 0.70,
with only 3.4% going to a person. Note we chose it on **dev**, not on
the test set. A threshold is a setting like any other.

## 5. Is the win bigger than the wobble?

```
  rules, right first time (dev)                  -> 0.756
  model + threshold, right first time (dev)      -> 1.0
  gain, 95% interval                             -> (0.244, 0.211, 0.277)
  share of resamples where the model loses       -> 0.0
```

A **paired bootstrap** (lesson 073): resample the 640 dev messages
2,000 times and, each time, score both systems on *the same* resample.
The model is ahead by 21 to 28 points in 95% of them and never behind.
On dev, this isn't luck. Lesson 082's exercise said a win should be
bigger than the wobble you get from re-splitting (a point or two). This
one is about twenty times bigger.

## 6. Open the sealed test set, once

```
  test: rules, right first time                  -> 0.781
  test: model, right first time                  -> 1.0
  test: model, to person                         -> 0.025
  test: model, misrouted                         -> 0.0
  test: model, auto accuracy                     -> 1.0
```

Now, and only now, the final model is trained on all 640 dev messages
and scored on the 160 it has never been near. Every choice was made
before this moment, so this number is honest: **right first time 100%,
2.5% to a person, nothing misrouted.** The rules get 78%. The run goes
into `runs.jsonl` marked `"test"`, so the log says when the seal was
broken.

It meets the brief. Time to celebrate?

Not quite. An honest score is only honest **about the data it was
measured on**. These 160 messages were written from the same templates
as the 640 training ones. Lesson 073 showed that a test set like that
mirrors training. It tells you the code has no bugs and the model
learnt the templates. It doesn't tell you how it does on Monday
morning's inbox.

## 7. The second exam: messages the way riders write them

```
  inbox sample: messages, per topic              -> (48, 12)
  inbox: rules, right first time                 -> 0.625
  inbox: model, right first time                 -> 0.979
  inbox: model, to person                        -> 0.479
  inbox: model, misrouted                        -> 0.021
  inbox: model, auto accuracy                    -> 0.96
  inbox: gain over rules, 95% interval           -> (0.352, 0.229, 0.479)

  for understanding only, never for choosing (this is still an exam):
  threshold  right first time  to a person  misrouted
       0.40             0.812        0.042      0.188
       0.50             0.896        0.208      0.104
       0.60             0.896        0.333      0.104
       0.70             0.979        0.479      0.021

  meets the brief on test (<= 15% to a person)   -> True
  meets the brief on the inbox sample            -> False
```

`inbox_sample.csv` holds 48 messages written the way real riders write:
plain ones, typos ("chian keeps slipin off"), Swahili and Sheng
("Nimelipa lakini sijapata risiti"), no keyword at all ("It makes a
horrible grinding noise uphill"), two problems in one message, and
sarcasm. Eight of each kind, twelve per team, each labelled with the
team that should get it. They're new: none of them are lesson 073's 24.

Treat a set like this as a **second sealed exam**. It wasn't used for
any choice above and isn't used for any choice now.

The result is the most important number in the lesson:

- The model is still much better than the rules (98% against 62.5%
  right first time, and the interval for the gain stays well above
  zero, even with only 48 messages).
- It misroutes almost nothing: one message in 48.
- But it gets there by sending **48% of messages to a person**. The
  brief's limit is 15%. On real-style messages, **it doesn't meet the
  brief.**

The second table shows the trade-off you'd face: lower the threshold
and fewer messages go to a person, but more get misrouted. At 0.5,
21% go to a person and 10% are misrouted. You might be tempted to pick
a threshold from this table. Don't: the moment you choose on it, it
stops being an exam. The right move is the one lesson 084 makes.

None of this means the work failed. It means the evaluation worked: it
found the gap between "tidy data" and "real data" **before** a single
rider's message went to the wrong team.

## 8. Error analysis: where it goes wrong, and why

```
  true \ top guess        app     bike  payment  station
  app                       11        0        1        0
  bike                       1       10        0        1
  payment                    1        3        8        0
  station                    0        3        0        9

  kind of message     right  to person  misrouted   (8 of each)
  plain                   6          2          0
  typos                   5          3          0
  Swahili/Sheng           2          5          1
  no keyword              3          5          0
  two topics              3          5          0
  sarcasm                 5          3          0

  words the model never saw: dev, inbox          -> (0.05, 0.53)
  average confidence: test, inbox                -> (0.96, 0.71)

  misroutes nobody would catch, most confident first:
    [station -> bike, 0.71] Kituo cha Market kimejaa kabisa
```

A score says *how much* it fails. Error analysis says *why*, and the
why tells you what to do next. Open `output/capstone/inbox_errors.csv`
and read every row. With 48 messages that takes five minutes, and it's
the best five minutes of the project. What it shows:

- **The confusion table** counts the model's *top guess*, even for the
  messages it passed to a person. Ten top guesses are wrong, and
  "bike" collects six of them. That's the model's fallback: when it
  recognises nothing, it leans towards bike. "Nowhere to return the bike
  near the mall" is a station problem that happens to say "bike".
- **The slices** (lesson 073) say where the trouble lives. Plain
  messages, typos and sarcasm mostly go fine; the character pieces earn
  their keep on "refnd" and "stashun", even if the model then hesitates.
  Swahili and Sheng, messages with no keyword, and two-topic messages
  are where it's lost: only two or three of eight right, the rest
  passed to a person.
- **Unknown words** explain it. On its own training data, 5% of words
  are new to the model. In the inbox sample it's **53%**. It can't route
  what it can't read. This is also a number you can measure on live
  traffic without any labels, which makes it a perfect monitor for
  lesson 084.
- **Confidence drops** from 0.96 on test to 0.71 on the inbox. That's
  good news: the model *knows* it's on unfamiliar ground. A model that
  was confidently wrong would be far more dangerous than one that says
  "I'm not sure".
- **The one silent misroute** is a Swahili message ("Market station is
  completely full") at confidence 0.71, a hair over the 0.70 line. A
  threshold is a line, and something always lands just on the wrong
  side of it.

Every one of these is a lead for the next round, not something to fix
right now and re-score on the same 48 messages. Write them down:

1. Collect and label real messages, especially Swahili and Sheng. (This
   was "not this time" in the scope. The data says it should move up.)
2. Teach the "no keyword" kinds: grinding, handlebars, deductions,
   password. These words simply never appeared in training.
3. Report scores on the real mix of message kinds, not only the tidy
   test set.

## 9. Save the model, its record and its mistakes for lesson 084

```
  model file, size in KB                         -> ('model.pkl', 140)
  record                                         -> {'candidate': 'lr_words_chars C=10', 'threshold': 0.7, 'data': '79a3efa5db0e'}
  project files                                  -> ['data/raw/feedback.csv', 'inbox_errors.csv', 'model.pkl', 'model_record.json', 'runs.jsonl']
```

The model goes into `model.pkl`, and `model_record.json` says
everything needed to trust it later: the data snapshot, seed, chosen
candidate, threshold, and the dev, test and inbox scores, next to the
rules' scores. `inbox_errors.csv` keeps every inbox prediction for the
write-up. Lesson 084 ships it.

## Your turn: build and evaluate your own capstone

About two or three hours:

1. Start every run by checking your snapshot fingerprint and rebuilding
   your sealed split.
2. Put your cleaning, features and model in one pipeline.
3. Compare three or four candidates (including the baseline) with
   cross-validation on dev. Log every run.
4. Choose any thresholds or settings on dev, using your brief's metric.
5. Bootstrap the gain over the baseline.
6. Open the test set once. Write the number down, whatever it is.
7. If you can, gather 30 to 50 examples that look like the real inputs,
   label them, and score them too.
8. Read every mistake. Slice by whatever kinds of input you have. Write
   down three leads.

If the result disappoints you, good: you've learnt something real, and
lesson 084 is about shipping what the evidence supports.

## What you can do now

- Check you're on the same data and split before any modelling.
- Keep cleaning, features and model in one pipeline.
- Compare candidates on dev only, with cross-validation and a run log.
- Choose a threshold from the brief's metric and limit.
- Tell a real win from the wobble with a paired bootstrap.
- Open a sealed test set once, and know what it can and can't tell you.
- Test on inputs that look like the real world, and read every mistake.

## What to do now

1. Run `lesson.py`, then open `output/capstone/inbox_errors.csv` and
   read all 48 rows.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   adds a candidate and sees what it does to the inbox exam.
3. Build and evaluate your own capstone as above.
4. Next, lesson 084, the last one: ship a minimal version, write it up,
   and decide what you'd do differently.

A perfect test score and a failed brief, in the same afternoon. That's
not a contradiction. That's what evaluation is for.
