# Lesson 082 - Capstone, part 1: choose and scope

**Phase 5 - Projects and production intuition** | Week 10, Day 3 | Wednesday 2026-11-11

> **Goal:** Pick a capstone project, write a plan, and gather your data.

Time: about 45 minutes to read and run, then an hour or so on your own
project. Same venv as lesson 013; nothing new to install.

---

The last three lessons of the course are one project: **your capstone**.
Today you choose it and set it up. Lesson 083 builds and evaluates it.
Lesson 084 ships a small version and asks what you'd do differently.

Most projects that fail don't fail at the modelling. They fail because
the question was fuzzy, the data wasn't really there, nobody could tell
whether it worked, or the work grew until it couldn't be finished. All
four of those are decided *today*, before any model exists. So today
trains no model, on purpose.

The lesson walks through the whole set-up with a worked example: **route
each support message to the right team**, using `feedback.csv`, the same
800 rider messages you met in lessons 061 to 073. The data is familiar
so you can watch the *process* without learning a new dataset. Your own
capstone can be about anything, as long as it passes section 1.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-5-projects/lesson-082-capstone-part-1-choose-and-scope/lesson.py
```

It takes a second or two. The one data file, `feedback.csv`, sits next
to the script. The script starts a real project folder in
`output/capstone/` (wiped every run) with everything it writes: the
brief, the plan, a data snapshot, an audit, the sealed test split and
the baselines. Every number below should match yours.

## 1. Score the ideas before you fall in love with one

```
  idea                                                total  passes gates
  Route support messages to the right team               12  True
  Forecast tomorrow's rides for the rebalancing van      10  True
  Predict which bikes will be stolen                      8  False
  Flag fraudulent refund requests                         5  False
  An AI assistant that answers any rider question         3  False

  chosen                                         -> 'Route support messages to the right team'
```

Write down three to five ideas, then score each from 0 (no) to 2 (yes)
on six questions:

- **Someone acts:** will a person or system do something differently
  because of the prediction? (Lesson 075's first question.)
- **Data today:** do you have the data *now*, not "once we get access"?
- **Can check:** can you tell whether a prediction was right? You need
  the real outcomes, and enough of them.
- **Baseline:** is there a simple rule to beat, so "good" has a meaning?
- **Fits the time:** about seven hours over three lessons.
- **Safe if wrong:** a wrong answer is annoying, not harmful.

Two of the six are **gates**: a 0 on *data today* or *can check* rules
an idea out, however exciting it is. The stolen-bike idea scores 8 but
fails a gate: with 20 thefts, you can't tell a good model from luck
(lesson 075, section 6). The city-wide AI assistant fails both: no
question-and-answer data, and no way to mark answers right or wrong
without reading every one. Fraud detection is a great problem with no
labelled fraud in hand.

The ride forecast passes, but only one month of rides exists, so
*data today* and *fits the time* get a 1. It would be a perfectly good
capstone; it's simply riskier. Message routing scores full marks:
real outcomes (each message has its topic), a rule to beat, and a
mistake costs a day's delay, not harm.

Ambitious ideas aren't bad. Keep them for after the course, once you've
finished one small project from end to end. **A finished small project
teaches more than an abandoned big one.**

## 2. Write the brief

```
  brief fields filled                            -> '8 of 8'
  missing                                        -> []
```

The same eight questions as lesson 075, written for this project. Open
`output/capstone/brief.json`. Two lines deserve a closer look:

- **Decision** includes "or a person if the system is unsure". That
  gives the model a third option besides right and wrong: *don't know*.
  It changes what we build in lesson 083 (a confidence threshold), and
  it came from writing the brief, not from the data.
- **Metric** has a limit built in: right first time, *with at most 15%
  sent to a person*. Without the limit, "send everything to a person"
  would be a perfect score. A metric with no constraint can usually be
  gamed by something silly.

## 3. Scope it: must, nice, not this time

```
  must-haves, nice-to-haves, cut                 -> (6, 3, 3)
  time-box, hours over three lessons             -> 7.0
```

Scope is a list of what you **won't** do, as much as what you will.
Three buckets:

- **Must:** without these, it isn't finished. A sealed test set, a
  baseline, a model that beats it, an "ask a person" threshold, a script
  that trains from scratch, and a short write-up.
- **Nice:** only if time is left. A tiny API (lesson 079), a drift check
  (lesson 080), tests for the cleaning code (lesson 076).
- **Not this time:** written down so they stop nagging you. Messages in
  Swahili or Sheng, messages about several topics at once, and a large
  language model.

Then a **time-box** per milestone: 1.5 hours today, 3 for building, 2.5
for shipping and the write-up. When a milestone runs over, you cut from
"nice", never from "must". It all goes into `output/capstone/PLAN.md`,
which is the first file anyone (including you, next month) should read.

## 4. Gather the data and snapshot it

```
  rows, columns                                  -> (800, 3)
  columns                                        -> ['feedback_id', 'text', 'topic']
  snapshot fingerprint                           -> '79a3efa5db0e'
```

Copy the data into the project as it arrived, in `data/raw/`, and never
edit that copy. All cleaning happens in code, so it can be re-run.
Then fingerprint it, as in lesson 078. Every score you report in the
next two lessons belongs to snapshot `79a3efa5db0e`. If the file ever
changes, the fingerprint will say so.

For your own project, this is where "data today" gets tested for real:
download it, export it, scrape it (politely, and only where that's
allowed), or simulate it if that's the honest option. If you can't get
it into `data/raw/` today, go back to section 1 and pick again.

## 5. Audit it before you trust it

```
  missing values                                 -> 0
  duplicate ids                                  -> 0
  exact duplicate texts                          -> 6
  same text after normalising                    -> 10
  messages per topic                             -> {'app': 200, 'bike': 200, 'payment': 200, 'station': 200}
  length chars median max                        -> (62, 139)
  no keyword at all                              -> 152
  keywords from two or more topics               -> 104
  example with two topics                        -> ("Quick note: the dock at Uni Square is badly lit when I needed it, and the login screen won't load. :(", 'station')
```

Ten minutes of auditing changes the plan more often than you'd think.
What this one found:

- **Duplicates.** Six exact repeats, and ten once case and punctuation
  are ignored. If a copy lands in training and its twin in the test set,
  the test score flatters the model (lesson 073 showed how much).
  Section 6 deals with it.
- **Perfect balance.** Exactly 200 messages per topic. Real inboxes are
  never that tidy. It means the data was *collected* balanced, so
  "right first time" on it won't equal "right first time" in real life,
  where perhaps half the messages are about payments. Note it in the
  write-up; it changes how you'd read every score.
- **Mixed messages.** 104 messages mention keywords from two or more
  topics, like the example: a dark dock *and* a broken login screen,
  labelled `station`. A person might have routed it either way. That's
  why "several topics per message" sits under "not this time", and why
  the "ask a person" option matters.
- **No keywords.** 152 messages hit no keyword at all, so keyword rules
  can only guess on those. Room for a model to help.

`output/capstone/audit.json` keeps the findings next to the data.

## 6. Seal the test set now

```
  dev rows, test rows                            -> (640, 160)
  test messages per topic                        -> {'app': 40, 'bike': 40, 'payment': 40, 'station': 40}
  same text on both sides                        -> 0
```

The test set is chosen **today**, before any model exists, and then
left alone until the very end of lesson 083. Everything else (features,
models, thresholds) is decided on the 640 **dev** messages, using
cross-validation inside them. That way the one test score you report is
honest, because nothing was tuned to it.

`StratifiedGroupKFold` takes 20% as test, with two rules: each topic
keeps its share (40 of each), and messages that are the same after
normalising form a **group** that stays together on one side. That's
why "same text on both sides" is 0. The test ids go into
`data/split.json`, next to the snapshot fingerprint and the seed, so
lesson 083 uses exactly this split.

## 7. Measure the bar to beat (on dev only)

```
  always the biggest topic: right first time     -> 0.25
  keyword rules: right first time                -> 0.756
  everything to a person: right, hand-routed     -> (1.0, 1.0)
  keyword rules, one-topic vs mixed messages     -> (0.797, 0.468)
```

Three baselines, and each one says something:

- **Always the biggest topic** gets 25%, because all four are the same
  size. Anything below that is broken.
- **Keyword rules** (lesson 073's) get 75.6% right first time with nobody
  hand-routing. This is the real bar: the model in lesson 083 has to
  beat 0.756 on dev, *and* do it while sending no more than 15% to a
  person, or it isn't worth the extra machinery.
- **Everything to a person** is always right and completely useless at
  scale. It's in the list because it reminds you why the 15% limit
  exists.

The last line is a gift for lesson 083: the rules get about 80% of
single-topic messages right, and fewer than half of mixed ones. That
says where to look first, and which messages a sensible model should
pass to a person.

Notice the baselines are measured on dev, not test. Rules don't learn,
so it would be *possible* to score them on the test set, but the habit
is the point: the test set is opened once, at the end, for everything.

## Your turn: set up your own capstone

Do this before lesson 083, in about an hour:

1. Write three to five ideas and score them with section 1's table. You
   can do it on paper. Pick the best one that passes both gates.
2. Fill in the eight lines of the brief. Leave a line blank rather than
   guess; a blank is a question to answer.
3. Write must, nice and not-this-time lists, and a time-box.
4. Get the data into `data/raw/`, fingerprint it, and spend ten minutes
   auditing it.
5. Seal a test set and record how you did it.
6. Measure at least two baselines on dev.

No project of your own in mind? Use the worked example. It's a proper
capstone, and lessons 083 and 084 will build it all the way through.

## What you can do now

- Score project ideas, and rule out the ones that fail on data or
  checkability, however exciting.
- Write a brief whose metric can't be gamed.
- Scope with must, nice and not-this-time lists and a time-box.
- Snapshot, fingerprint and audit data before trusting it.
- Seal a test set before any modelling, keeping duplicates together.
- Set the baselines a model has to beat.

## What to do now

1. Run `lesson.py`, then read `output/capstone/PLAN.md` and
   `audit.json`.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   adds a rubric row for your own idea and changes the time-box.
3. Set up your own capstone as above.
4. Next weekday, lesson 083: build the full pipeline and evaluate it
   against the baseline.

The decisions that matter most were all made today. Lesson 083 is just
building what you've already described.
