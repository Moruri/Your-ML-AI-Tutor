# Lesson 084 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle, the
last of the course. Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` and both
data files somewhere of your own and edit the copy (it writes an
`output/` folder next to wherever it sits). From the repo root:

```bash
cd lessons/phase-5-projects/lesson-084-capstone-part-3-ship-and-reflect
python lesson.py
mkdir -p ../../../my_capstone_084
cp lesson.py feedback.csv inbox_sample.csv ../../../my_capstone_084/
cd ../../../my_capstone_084
python lesson.py
```

---

## 1. Which shipping mode?

For each project, pick full auto, auto with a person for unsure cases,
or shadow mode, and say why. (a) Suggesting tags for photos in a family
album app, tested on 2,000 real photos; (b) flagging possibly
fraudulent M-Pesa transactions, tested only on simulated fraud; (c)
forecasting tomorrow's rides per station for the rebalancing van, with
a year of real history and a good backtest.

<details>
<summary>Check yourself</summary>

(a) Full auto is reasonable: tested on real inputs, and a wrong tag is
mildly annoying and easy to fix. (b) Shadow mode: simulated fraud says
little about real fraud, and a wrong flag can freeze someone's money.
Let analysts see the flags next to their usual work and measure
agreement first. (c) Auto with a person: the forecast drives the van,
but the driver or planner can override it on unusual days (a big
match, a strike), and those overrides become useful data. The mode
follows two things: how real your evaluation data was, and how much a
mistake costs.

</details>

## 2. Smoke test or evaluation?

Sort these into "smoke test" (run on every deploy, seconds) or
"evaluation" (run when the model changes, careful): (a) the service
answers `/health`; (b) accuracy on the inbox sample is at least 90%;
(c) "Refund my M-Pesa charge" goes to payment; (d) the bootstrap
interval of the gain over the rules stays above zero; (e) the model
version in `/health` matches the record.

<details>
<summary>Check yourself</summary>

Smoke tests: (a), (c) and (e). They're quick, they have one obvious
right answer, and they catch wiring mistakes: a dead service, the old
model, a broken pipeline. Evaluation: (b) and (d). They measure how good
the model is, need a careful dataset, and belong to lesson 083's
process. A smoke test with a model-quality threshold, like (b), tends
to fail at random after a retrain and gets switched off; keep the two
jobs separate.

</details>

## 3. The monitor that needs no labels

Of the three monitoring signals, which could you compute on the very
first day after launch, and which needs time? Why does that matter?

<details>
<summary>Check yourself</summary>

The unsure share and the unknown-word share can be computed from the
first request: they only need the model's own output and the message.
Agreement needs the person's decision, which in shadow mode arrives the
same day, but in full auto you'd only learn it when a misrouted message
gets forwarded, days later and only for some messages. Label-free
signals are your early warning; labelled ones are your proof. A good
plan has both.

</details>

## 4. Hands-on: tune the plan and add a smoke test

Edit your copy of `lesson.py`.

1. In section 5, change the unknown-words alert in `ALERTS` from `0.25`
   to `0.12`. In which week does the first alert fire now, and which
   monitor fires it?
2. In section 4, add a smoke test to `CHECKS`:
   `"Swahili message is marked unsure": call("/route", {"text": "Baiskeli haina breki kabisa"})[1]["unsure"],`
   Does it pass? Why is it a good test for *this* project?
3. Change `PER_DAY` in section 2 from 100 to 60. Does the arithmetic
   still say shadow mode? Would you change `MODE`?

<details>
<summary>Check yourself</summary>

1. Week 4, from `unknown_words` (0.15 is above 0.12), a week before
   the unsure share crosses 15%. A tighter line warns earlier, at the
   price of more false alarms on normal weeks. Set alert lines from a
   few weeks of normal traffic, not from a single guess.
2. It passes: confidence 0.668, below the 0.70 threshold. It's a good
   test because the model card promises that Swahili and Sheng go to a
   person. A smoke test that checks a promise from the card turns the
   card into something the deploy actually enforces.
3. At 60 a day, 48% is about 29 messages, under the 40 a person can
   handle. Tempting, but no: the 48% comes from 48 messages, so the
   true share could easily be 35% to 60% (lesson 083's puzzle), and we
   still don't know the real misroute rate. The written rule (four
   weeks of shadow logs) is there exactly for moments like this one.

</details>

## 5. Optional puzzle: your own look back

No code. Take your own capstone, or the worked example if you used it,
and answer the four look-back questions in two sentences each. Then
answer a fifth: *if you had to do the whole project again in half the
time, what would you cut, and what would you never cut?*

<details>
<summary>Check yourself</summary>

There's no single answer, but good ones share a pattern. What people
cut: extra candidates, hyperparameter searches, nicer plots, a fancy
service. What they never cut: the brief, the sealed test set, the
baseline, real-looking evaluation data, reading the mistakes, and the
write-up. If your "never cut" list looks like that, you've taken in the
most important thing this course had to teach.

Congratulations on finishing all 84 lessons.

</details>
