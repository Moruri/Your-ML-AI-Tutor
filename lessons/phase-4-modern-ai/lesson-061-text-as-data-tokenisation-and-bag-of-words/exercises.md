# Lesson 061 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-4-modern-ai/lesson-061-text-as-data-tokenisation-and-bag-of-words/lesson.py my_lesson_061.py
python my_lesson_061.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-4-modern-ai/lesson-061-text-as-data-tokenisation-and-bag-of-words").resolve()
```

`load_feedback`, `tokenise`, `bpe_merges`, `apply_merges` and `show` are
there to reuse.

---

## 1. Fit on what?

A teammate fits a `TfidfVectorizer` on all 800 messages, *then* splits
into train and test. What's wrong, and does it matter much here?

<details>
<summary>Check yourself</summary>

It's leakage (lesson 031): the vocabulary and the IDF values have seen
the test messages, so the test score is measured on features shaped by
the test set. A word that only appears in test messages gets a column
it shouldn't have. With 800 tidy messages the effect is small, but on
real data with rare words and new slang it can flatter the score. Fit
inside a pipeline on the training split and the problem disappears.

</details>

## 2. Counts or TF-IDF?

Swap `TfidfVectorizer()` for `CountVectorizer()` in section 5. Predict
first: better, worse, or the same?

<details>
<summary>Check yourself</summary>

```
CountVectorizer + logistic regression: accuracy  -> 1.0    (TF-IDF gave 0.99)
```

Slightly better here, on 200 test messages, which is one message's
difference. On clean, short, similar-length texts plain counts often do
as well as TF-IDF. TF-IDF tends to earn its keep on longer, messier
documents, where common words pile up and lengths vary a lot. As
always, compare on validation data rather than trusting a rule.

</details>

## 3. Throw away the small words?

Many tutorials remove **stop words** ("the", "is", "on") before
counting: `TfidfVectorizer(stop_words="english")`. Try it in section 5.
What happens, and why?

<details>
<summary>Check yourself</summary>

```
columns -> 129   (was 178)
accuracy -> 0.975   (was 0.99)
```

Fewer columns and a *worse* score. The English stop-word list throws
out words like "on", "off" and "nobody", and here "came off" and
"nobody replied" are real clues. TF-IDF already turns down words that
appear everywhere, so removing them by hand mostly risks losing signal.
Stop-word lists were essential when memory was tight; now, try with and
without.

</details>

## 4. Hands-on: how few labels does it need?

Someone has to tag messages by hand to make training data, and that
costs time. Train the section 5 pipeline on only the **first 20**
training messages, then on the **first 40**, and test both on the same
200 test messages. Use the same split (`test_size=0.25`,
`stratify=df.topic`, `random_state=61`) and slice the training set with
`X_tr[:20]`, `y_tr[:20]`. Predict first: how close to 0.99 will 40
messages get?

<details>
<summary>Expected results</summary>

```
first 20 training messages  -> accuracy 0.34
first 40 training messages  -> accuracy 0.86
all 600                     -> accuracy 0.99
```

Twenty messages is barely better than guessing: about five per topic,
so most useful words have never been seen next to a label. Forty
already gets 86%. The curve climbs fast and then flattens, which is
typical, and it's worth drawing for any real labelling project before
you pay for thousands of labels. It's also exactly the problem that
pretrained models (lesson 064) attack: they arrive already knowing what
words mean, so they need far fewer labels.

</details>

<details>
<summary>One way to write it</summary>

```python
df = load_feedback()
X_tr, X_te, y_tr, y_te = train_test_split(df.text, df.topic, test_size=0.25,
                                          stratify=df.topic, random_state=61)
for n in (20, 40, len(X_tr)):
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    model.fit(X_tr[:n], y_tr[:n])
    show(f"first {n} training messages", round(accuracy_score(y_te, model.predict(X_te)), 3))
```

</details>

## 5. (Optional) Tokens and cost

Large language model services charge per token, not per word. Using
section 2's BPE pieces, explain why a message full of names and typos
("Mwangi's bke at Kangemi brokee") costs more tokens than a plain one
of the same length.

<details>
<summary>Check yourself</summary>

BPE gives common words a single token because they were merged early.
Rare names and misspellings were never merged, so they break into many
small pieces: "bke" might be "b" + "ke", "brokee" might be "brok" + "ee".
Same number of words, more tokens, so more cost and less room in the
model's context. Languages that were rare in the tokeniser's training
text pay this tax on every word, which is a real fairness issue we'll
come back to in lesson 074.

</details>

---

That's text as data. If one thing sticks: *a model never sees text,
only tokens turned into numbers; how you cut and count them decides
what the model can possibly learn.*
