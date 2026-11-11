# Lesson 082 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere of
your own and edit the copy (it writes an `output/` folder next to
wherever it sits, and reads `feedback.csv` from the same folder, so copy
that too). From the repo root:

```bash
cd lessons/phase-5-projects/lesson-082-capstone-part-1-choose-and-scope
python lesson.py
mkdir -p ../../../my_capstone
cp lesson.py feedback.csv ../../../my_capstone/
cd ../../../my_capstone
python lesson.py
```

---

## 1. Gate or no gate?

Score these on *data today* and *can check* only. Which pass both
gates? (a) Predict next month's sales for your family shop from two
years of receipts in a notebook; (b) predict which job applicants will
be great employees; (c) classify photos of maize leaves as healthy or
diseased, from a public labelled dataset; (d) predict whether a tweet
will go viral, from tweets you'd collect from now on.

<details>
<summary>Check yourself</summary>

(a) Passes, if you'll type the receipts in: the data exists and next
month will tell you if you were right. Typing is a real cost, so be
honest on *fits the time*. (b) Fails *can check*: you only learn how
hired people did, never the rejected ones, and "great" isn't defined.
It also scores 0 on *safe if wrong*. (c) Passes both: data you can
download today, with labels. A good capstone. (d) Fails *data today*:
"from now on" means you'd wait weeks before having anything to train or
test on. Choosing a capstone is mostly about saying no quickly.

</details>

## 2. Why seal the test set before the audit is even finished?

Someone says: "I'll look at the whole dataset first, try a few models,
and then pick a test set." What goes wrong?

<details>
<summary>Check yourself</summary>

Every decision made while looking at the test rows (which words to
keep, which model family, which threshold) is a little bit tuned to
them, so the final test score is optimistic. Nobody does it on purpose;
it leaks in through choices. Sealing it first, and doing all the
choosing with cross-validation on dev, keeps one number honest. Light
auditing of the whole file (missing values, duplicates, counts) is
fine, because it doesn't pick anything that's scored later. Looking
for *patterns that predict the label* is the part to keep to dev.

</details>

## 3. Fix the metric

For each metric, find the silly system that games it: (a) "share of
messages routed right", with an "ask a person" option that's always
counted as right; (b) "share of messages sent to a person", lower is
better; (c) "accuracy on single-topic messages only".

<details>
<summary>Check yourself</summary>

(a) Send everything to a person: 100%. (b) Never send anything to a
person, however unsure: 0%, the "best" possible, with lots of misroutes.
(c) Ignore mixed messages entirely, which are exactly the hard ones.
Each is fixed by pairing a goal with a limit, as the brief does: right
first time, *with at most 15% to a person*, measured on all messages.

</details>

## 4. Hands-on: add your own idea and change the plan

Edit your copy of `lesson.py`.

1. Add your own capstone idea to `IDEAS` with honest scores. Where does
   it rank? If it fails a gate, which change to the idea would get it
   through?
2. You find you only have 5 hours, not 7. Change `MILESTONES` so it
   adds up to 5. What moves from "must" or "nice" to "not this time"?
3. Change `SEED` from 82 to 1. Does the
   keyword-rule score on dev change? By how much?

<details>
<summary>Check yourself</summary>

1. There's no single right score. The useful habit is the last
   question: an idea that fails *data today* can often be rescued by
   narrowing it ("all of Kenya" to "one county with published data"),
   and one that fails *can check* by choosing a target you'll actually
   observe.
2. Take time from 083 and 084, then move all three "nice" items to "not
   this time". Keep every "must": the sealed test set and the baseline
   cost minutes and make the rest believable. If it still doesn't fit,
   the honest fix is a smaller question, not skipping the evaluation.
3. The keyword rules move by a point or two (the dev set is a different
   640 messages), and the counts per topic stay at 40 each in the test
   set. A difference that size is noise. It's a preview of lesson 083:
   any model's "win" over the baseline should be bigger than the wobble
   you get just from re-splitting.

</details>

## 5. Optional puzzle: the balanced inbox

The audit found exactly 200 messages per topic. Suppose the real inbox
is 50% payment, 20% app, 20% station and 10% bike, and a model is right
90% of the time on payment and 70% on each of the others. What's its
accuracy on the balanced data, and in the real inbox?

<details>
<summary>Check yourself</summary>

Balanced: (0.9 + 0.7 + 0.7 + 0.7) / 4 = 0.75. Real inbox: 0.5 x 0.9 +
0.5 x 0.7 = 0.80. The same model scores differently because the mix
differs. It can go the other way too, if the model is weakest on the
biggest topic. That's why the write-up should report accuracy *per
topic*, and, if you know the real mix, re-weight the dev score to it.
A single number on balanced data only tells you about balanced data.

</details>
