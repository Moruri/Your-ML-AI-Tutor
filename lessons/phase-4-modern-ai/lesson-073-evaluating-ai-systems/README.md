# Lesson 073 - Evaluating AI systems

**Phase 4 - Modern AI: NLP, vision, generative and LLM basics** | Week 9, Day 4 | Thursday 2026-11-05

> **Goal:** Build small evaluation sets, measure quality honestly, and
> avoid fooling yourself.

Time: about 45 minutes. Same venv as lesson 013; nothing new to install.

---

Every lesson so far has ended with a number: an accuracy, a loss, a
precision@5. Today is about whether you should believe those numbers.

Evaluation is the least glamorous part of AI work and the part that
decides whether anything you ship actually helps. The trouble is rarely
the maths. It's that it's very easy to fool yourself: a test set that
quietly overlaps the training data, a score from 40 examples treated as
gospel, a grader that rewards the wrong thing, or a model "improved"
until it fits the test set instead of the problem.

We'll take a small classifier that routes rider feedback to the right
team (bike, app, payment or station, the same 800 messages as lesson
061) and put it through the questions a careful evaluator asks.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-4-modern-ai/lesson-073-evaluating-ai-systems/lesson.py
```

It reads `feedback.csv` from this folder, needs only scikit-learn and
pandas, and finishes in a few seconds. Every number below should match
yours exactly.

## 1. Two systems, one number each

```
messages: train, test                          -> (600, 200)
always guess 'app': accuracy                   -> 0.25
keyword rules: accuracy                        -> 0.79
TF-IDF + logistic regression: accuracy         -> 0.995
```

Always have a baseline. "Always guess app" gets 25%, which is just the
share of one topic. A dozen hand-written keyword rules (`rule_system`)
get 79%. The TF-IDF model from lesson 061 with logistic regression gets
99.5%: one mistake in 200.

That last number should make you suspicious rather than happy. Almost
nothing real is 99.5% easy. When a score looks too good, the first job
is to find out why.

## 2. Is the test set really unseen?

```
test messages also in train (exact)            -> 3
test messages with a near-twin (cos >= 0.9)    -> 17
model accuracy on near-twins                   -> 1.0
model accuracy on the rest                     -> 0.995
example test message                           -> "FYI my card was billed for a ride I didn't take please refund me."
  its twin in train                            -> "FYI my card was billed for a ride I didn't take please refund me."
hand-written real-world messages               -> 24
keyword rules on them: accuracy                -> 0.625
model on them: accuracy                        -> 0.667
```

Two checks, and both find problems.

**Overlap.** For each test message we find its most similar training
message (cosine similarity from lesson 067). Three test messages appear
word for word in training, and 17 have a near-twin. Those are free
marks: the model has effectively seen the answer. Real datasets are
full of this: repeated complaints, copied templates, the same email
forwarded twice. Deduplicate *across* your split before you trust it.
From here on we drop the 17, leaving a clean test set of 183.

**Who wrote the test set?** Our 800 messages were generated from
templates, so the test set is written in the same tidy style as the
training set. Real riders don't write like that. So the script includes
24 messages written by hand to look like what actually arrives: typos,
Swahili and Sheng, no obvious keyword, two problems in one message,
sarcasm. On those, the model drops from 99.5% to 67%, and the keyword
rules to 62.5%.

This is the most important lesson of the day: **your evaluation set
should look like the real inputs, not like your training data.** A
small set collected from real use beats a big set that's a mirror of
what you trained on.

## 3. How sure is the number?

```
clean test set: size                           -> 183
model on it: accuracy, 95% interval            -> (0.995, (0.984, 1.0))
model on real-world set: accuracy, interval    -> (0.667, (0.458, 0.833))
rules on real-world set: accuracy, interval    -> (0.625, (0.417, 0.792))
real-world: model minus rules, average gap     -> 0.042
share of resamples where model wins            -> 0.551
messages only the model gets right             -> 7
messages only the rules get right              -> 6
```

A score from a test set is an estimate, and small test sets give shaky
estimates. The **bootstrap** from lesson 026 shows how shaky: resample
the test set with replacement 2,000 times and see how much the score
moves.

On 183 clean messages, the model's score is pinned down tightly. On the
24 real-world messages, 67% could easily be anything from 46% to 83%.

The comparison is the useful part. The model beats the rules by 4
points on the real-world set, but in only 55% of resamples: close to a
coin toss. Look at the last two lines: seven messages only the model
gets right and six only the rules get right. With 24 examples you
cannot say which system is better, only that both struggle. If someone
shows you "ours: 67%, theirs: 62%" on a couple of dozen examples, ask
for the interval.

## 4. Where does it fail? Slices

```
typos         : right out of 4                 -> 2
Swahili/Sheng : right out of 4                 -> 4
no keyword    : right out of 4                 -> 4
two topics    : right out of 4                 -> 2
sarcasm       : right out of 4                 -> 2
plain         : right out of 4                 -> 2
most common mistakes                           -> {'station read as bike': 2, 'bike read as app': 2, 'bike read as station': 1}
  missed                                       -> 'brkes r gone on the green one at uni sq'
  missed                                       -> 'the dok wont release the bike'
  missed                                       -> 'Love the app, hate the squeaky brakes'
```

An average hides where the failures are. Splitting the real-world set
by kind of message (a **slice**) shows something surprising: the model
gets all four Swahili and Sheng messages right. Don't celebrate. Look
at why: three of them contain one familiar word ("pesa", "App", "bike"
and "Park Gate"), and "Baiskeli haina breki kabisa" contains no word
the model knows at all, so it fell back on its default guess, which
happened to be bike. Right for the wrong reason. Meanwhile it misses
half of the typos, the two-topic messages, the sarcasm, and even plain
messages like "Seat won't go down" and "Can't reset my password",
because "seat" and "password" never appeared in training.

The mistakes tell you what to fix: "the dok wont release the bike" goes
to the bike team because "dok" means nothing to a word-based model and
"bike" does. "Love the app, hate the squeaky brakes" goes to the app
team because the app is named and the brakes are only described.

Slices with four messages each are tiny, so treat them as leads, not
verdicts. The habit is what matters: before you ship, check the score
for every group you care about, not just the overall one.

## 5. Grading answers that aren't a label

```
exact match: agreement, kappa                  -> (0.5, 0.122)
normalised match: agreement, kappa             -> (0.583, 0.25)
word overlap >= 50%: agreement, kappa          -> (0.667, 0.226)
key-facts rubric: agreement, kappa             -> (1.0, 1.0)
```

Classifiers are easy to grade: the label is right or it isn't. A help
bot writes sentences, and many different sentences are correct. The
script has 12 answers to one question, each marked right or wrong by a
person, and tries four automatic graders. **Agreement** is the share of
answers where grader and person agree. **Kappa** (Cohen's kappa)
corrects for agreeing by chance: 0 means no better than luck, 1 means
perfect.

- **Exact match** only accepts a copy of the reference answer. Half the
  verdicts are wrong, because good paraphrases fail.
- **Normalised match** (lowercase, no punctuation) barely helps.
- **Word overlap** sounds clever, but "within 5 working days" shares
  almost every word with the reference and is wrong. Kappa 0.23.
- **A key-facts rubric** checks for what must be there (M-Pesa, 3
  working days) and what must not (5, 30, card, bank). Perfect
  agreement.

Don't trust that perfect score too much, though: I wrote the rubric
while looking at these same 12 answers. That's tuning on the test set
(next section). A rubric, or an LLM used as a judge with a written
rubric, is itself a model, and it needs its own check against human
labels on answers it hasn't seen.

## 6. Don't tune on the test set

```
ideas tried                                    -> 60
tuning set (30): average idea, best idea       -> (0.519, 0.667)
best idea on a fresh set (270)                 -> 0.537
average idea on the fresh set                  -> 0.536
```

Here's the trap in its purest form. We give each message a column with
no signal at all, a coin flip for "did the rider leave a tip?". No
model can honestly beat 50%.

Then we do what everyone does: try ideas. Sixty variations, different
settings, different slices of the training data, each scored on a
30-message tuning set. The average idea scores 52%, as expected. The
*best* idea scores 67%, which looks like a real result. On a fresh set
of 270 messages it's back to 54%, no better than the rest.

Nothing was learned; we just picked the luckiest coin. Every time you
look at a test score and change something, a little of that luck leaks
into your choice. The fix is the one from lesson 032, now with real
weight behind it: tune on a validation set, and look at the final test
set **once**, at the end. If you've peeked too often, collect fresh
examples.

## What you can do now

- Start any evaluation with a simple baseline.
- Check a test set for exact and near-duplicate overlap with training.
- Build a small evaluation set that looks like real inputs.
- Put a bootstrap interval on a score and on a difference between two
  systems.
- Slice results to find where a system fails.
- Grade open-ended answers with a key-facts rubric, and check the
  grader against human labels.
- Keep a test set clean of tuning decisions.

## What to do now

1. Run `lesson.py`. Then add five messages of your own to
   `REAL_WORLD`, the kind you'd really send, and see whether the
   interval in section 3 gets any narrower.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   tries a model that reads characters instead of words.
3. Next weekday, lesson 074: safety, bias and responsible use, where
   slicing results by group becomes a question of fairness.

Check that the test set is unseen, that it looks like real life, how
sure the number is and where it breaks, and then look at it only once.
