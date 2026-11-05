# Lesson 073 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy that sits
next to `feedback.csv`, so it can find the data. From the repo root:

```bash
cd lessons/phase-4-modern-ai/lesson-073-evaluating-ai-systems
cp lesson.py my_lesson_073.py
python my_lesson_073.py
```

`rule_system`, `model_system`, `bootstrap`, `keyword_hits` and the
`REAL_WORLD` list are there to reuse.

---

## 1. The suspicious 99.5%

A teammate reports 99.5% accuracy on a new spam filter and wants to
ship tomorrow. Name three questions you'd ask before agreeing.

<details>
<summary>Check yourself</summary>

Any three of these, all from today:

- **Does the test set overlap training?** Exact and near duplicates
  (section 2) hand out free marks.
- **Does the test set look like real traffic?** If it was sampled from
  the same tidy source as training, the real-world score can be far
  lower (99.5% became 67% for us).
- **What's the baseline?** If 97% of mail isn't spam, "never spam"
  already scores 97%.
- **How big is the test set, and what's the interval?** 99.5% on 200
  messages and on 20,000 are very different claims.
- **How often was it tuned against this test set?** (Section 6.)
- **How does it do on the slices that matter**, such as spam written
  in Swahili, or real mail from new senders?

</details>

## 2. Reading an interval

System A scores 81% on 50 test questions; system B scores 78% on the
same 50. A paired bootstrap says A wins in 64% of resamples. Your
manager asks "so A is better?" What do you say?

<details>
<summary>Check yourself</summary>

"We can't tell yet." If A were truly no better, it would still win
about half the resamples, and 64% is not far from that. Compare
section 3, where the model "won" by 4 points but in only 55% of
resamples. To decide, you'd want more test questions (the interval
shrinks roughly with the square root of the count), and a look at the
questions where the two disagree, since those are the only ones that
separate them.

</details>

## 3. Which grader?

A bot answers "What time does the Park Gate station open?" The
reference answer is "6am". Which grader from section 5 would you
start with, and what would its rubric check?

<details>
<summary>Check yourself</summary>

A key-facts rubric. It must find the time (6am, 6 a.m., 06:00, six in
the morning; normalise them all to one form first) and must *not*
contain a different time, or a different station. Exact match would
fail "It opens at 6am every day", and word overlap would happily pass
"It opens at 7am". Then check the rubric against a person's verdicts
on fresh answers before you rely on it.

</details>

## 4. Hands-on: read characters, not words

Section 4 showed typos are a weak spot: "dok" and "brkes" mean nothing
to a model that reads whole words. A model that reads **character
pieces** (`TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))`)
sees "dok" as "do", "ok", "dok" and so on, which overlaps with "dock".

Train one with the same logistic regression, then report its accuracy
on the clean test set and the real-world set, the share of bootstrap
resamples where it beats the word model on the real-world set, and its
score on each kind of message.

Predict first: will it fix the typos?

<details>
<summary>Expected results</summary>

```
clean test: words 0.995, characters 1.000
real world: words 0.667, characters 0.708
share of resamples where characters win: 0.629
typos          words 2/4, characters 2/4
Swahili/Sheng  words 4/4, characters 4/4
no keyword     words 4/4, characters 4/4
two topics     words 2/4, characters 2/4
sarcasm        words 2/4, characters 3/4
plain          words 2/4, characters 2/4
```

A small gain overall, but look where it came from: one sarcastic
message, not the typos. And 63% of resamples is nowhere near a clear
win. Honest conclusion: "maybe slightly better, can't tell on 24
messages." The real fix for typos and plain words like "seat" is more
real-world training examples, and a bigger real-world test set to
prove it.

</details>

<details>
<summary>One way to write it</summary>

Add this at the end of your copy:

```python
heading("Exercise 4")
char_model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2),
                           LogisticRegression(C=3.0, max_iter=2000)).fit(train["text"], train["topic"])
char_test_ok = (char_model.predict(test["text"]) == test["topic"].to_numpy())[clean]
char_real_ok = char_model.predict(real["text"]) == real["topic"].to_numpy()
print(f"clean test: words {model_ok.mean():.3f}, characters {char_test_ok.mean():.3f}")
print(f"real world: words {real_ok.mean():.3f}, characters {char_real_ok.mean():.3f}")
idx = np.random.default_rng(SEED).integers(0, len(real), size=(2000, len(real)))
gap = char_real_ok[idx].mean(axis=1) - real_ok[idx].mean(axis=1)
print(f"share of resamples where characters win: {(gap > 0).mean():.3f}")
for kind in real["kind"].unique():
    k = (real["kind"] == kind).to_numpy()
    print(f"{kind:<14} words {real_ok[k].sum()}/4, characters {char_real_ok[k].sum()}/4")
```

Note that `train` and `test` here are the ones from section 1; section
6 makes its own variables, so it doesn't disturb them.

</details>

## 5. Optional puzzle: the leak you can't see

Your test messages have no duplicates in training, not even near ones.
But every message from rider 4417 (who writes 60 messages a month,
always in the same style) is spread across both sets. Is that a
problem? How would you split instead?

<details>
<summary>Hint and answer</summary>

Yes. The model can learn rider 4417's style and recognise it in the
test set, so the score overstates how it does on *new* riders. Split by
rider instead of by message: every message from one rider goes to the
same side (scikit-learn's `GroupShuffleSplit` or `GroupKFold` do this).
The same idea applies to patients in medical data, users in app logs,
and days in time series: split by whatever the model will meet fresh
in real life.

</details>
