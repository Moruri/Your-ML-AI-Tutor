# Lesson 083 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` and both
data files somewhere of your own and edit the copy (it writes an
`output/` folder next to wherever it sits). From the repo root:

```bash
cd lessons/phase-5-projects/lesson-083-capstone-part-2-build-and-evaluate
python lesson.py
mkdir -p ../../../my_capstone_083
cp lesson.py feedback.csv inbox_sample.csv ../../../my_capstone_083/
cd ../../../my_capstone_083
python lesson.py
```

---

## 1. Which number goes in the write-up?

You have three scores for the same model: dev cross-validation 0.992,
sealed test set 1.0, inbox sample 0.979 right first time with 48% to a
person. A manager asks "how good is it?" in one sentence. What do you
say?

<details>
<summary>Check yourself</summary>

Something like: "On tidy messages like our training data it's
essentially perfect; on messages written the way riders really write,
it routes almost nothing wrongly but is unsure about half of them, which
is three times the hand-routing we can afford." The dev score is for
choosing, not reporting. The test score is honest but only about tidy
data. The inbox numbers are closest to the real question, and they come
with a limit the brief cares about. One sentence can carry two numbers
if you say what each is about.

</details>

## 2. Spot the leak

Which of these quietly turns the test set into training data? (a)
Choosing C by its score on the test set; (b) fitting the TF-IDF
vocabulary on all 800 messages "because it's unsupervised"; (c)
fitting the final model on all 640 dev messages after cross-validation;
(d) reading the test-set mistakes, adding the missing words to the
keyword rules, and scoring the rules on the test set again.

<details>
<summary>Check yourself</summary>

(a), (b) and (d) leak. (a) is the classic. (b) is sneakier: the
vocabulary, and the IDF weights, now know which words appear in the
test set, so test messages look less unfamiliar than real new messages
would. That's why the vectoriser lives *inside* the pipeline, fitted
only on what the model trains on. (d) is tuning by hand: the rules
"learn" the test set through you. (c) is fine and normal: once the
choices are made, train on all the data you're allowed to use.

</details>

## 3. Unsure is a feature

Model A sends 48% of inbox messages to a person and misroutes 2%.
Model B sends 13% to a person and misroutes 15%. Which would you rather
ship, and what would you want to know first?

<details>
<summary>Check yourself</summary>

It depends on the costs in the brief: a misroute costs a rider about a
day; hand-routing costs a person's time, about 40 messages a day. At
100 messages a day, A means 48 hand-routed (over capacity) and 2 riders
waiting; B means 13 hand-routed and 15 riders waiting a day. Neither
fits the brief. A is the safer starting point, because its mistakes are
*visible*: a person sees every message it isn't sure about. B's are
silent. What you'd want to know first is how many messages a day really
arrive, and how much a day's wait really costs. The hands-on exercise
shows you've already met model B.

</details>

## 4. Hands-on: try other candidates on the second exam

Edit your copy of `lesson.py`.

1. Just after `best = max(...)` in section 3, force a different choice:
   `best = "lr_words C=10"` (words only, no character pieces). Run it.
   What happens to the inbox numbers in section 7?
2. Now try `best = "nb_words"`. What happens this time?
3. Put it back. Then, in section 4, change `MAX_TO_PERSON` to `0.05`.
   Which threshold is chosen, and what does it do on the inbox?

<details>
<summary>Check yourself</summary>

1. Words only: dev picks a threshold of 0.75 this time, and on the
   inbox it's right first time 1.0, but with 56% to a person (up from
   48%) and no misroutes. Without character pieces, typos like
   "refnd" and "stashun" are just unknown words, so it's unsure more
   often. Same dev score, different real-world behaviour: candidates
   that tie on dev can still differ where it matters.
2. Naive Bayes: 29% to a person, but 6% misrouted, three riders in 48
   sent to the wrong team. Naive Bayes is famously overconfident, so it
   passes fewer messages on, and some of the ones it keeps are wrong.
   This is model B from question 3. Its dev score was *lower* (0.967),
   and it still looks "better" on the to-person number. That's why one
   metric is never enough.
3. With at most 5% to a person, every threshold in the dev table is
   still allowed (the most is 3.9%), so 0.70 is chosen again and the
   inbox still sends 48% to a person: on tidy dev data the limit doesn't
   bite. The limit only matters on data the model finds hard, which is
   exactly why choosing it on tidy dev data is optimistic. You'd need
   real messages in dev to set it properly.

</details>

## 5. Optional puzzle: how big should the second exam be?

The inbox sample has 48 messages and the model sends 48% of them to a
person. Roughly how wide is the 95% interval on that 48%? How many
messages would you need for an interval of about plus or minus 5
points?

<details>
<summary>Check yourself</summary>

For a share p measured on n messages, the standard error is about
sqrt(p(1-p)/n). With p = 0.48 and n = 48 that's about 0.072, so the
interval is roughly plus or minus 14 points: anywhere from about 34% to
62%. Either way it's far above 15%, so the verdict holds. For plus or
minus 5 points you need 1.96 x sqrt(0.25/n) = 0.05, so n is about 384.
A few hundred real, labelled messages: that's the first thing to
collect, and lesson 084's shadow mode is how you collect them.

</details>
