# Lesson 061 - Text as data: tokenisation and bag-of-words

**Phase 4 - Modern AI: NLP, vision, generative and LLM basics** | Week 8, Day 3 | Wednesday 2026-10-28

> **Goal:** Turn text into numbers with tokenisers, TF-IDF, and a baseline
> classifier.

Time: about 40 minutes. Same venv as before; everything today is in
`requirements.txt` already (scikit-learn and pandas).

---

Welcome to Phase 4. For twelve lessons we built networks from the ground
up. Now we turn to the systems people actually mean when they say "AI"
today: models that read, write, see and generate. Most of them start
with text, so we start there too.

Here's the problem. Every model we've built eats rows of numbers, all
the same length. Text is a string of characters, and every message is a
different length. Before any learning can happen, we have to answer two
questions: **what are the pieces of a text** (that's *tokenisation*),
and **how do we turn a pile of pieces into a row of numbers** (today
that's *bag-of-words* and *TF-IDF*).

Then we'll train the simplest text classifier that works. It's old,
it's fast, and it's surprisingly hard to beat. Every fancy model in the
rest of this phase should be compared against it.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-4-modern-ai/lesson-061-text-as-data-tokenisation-and-bag-of-words/lesson.py
```

It reads `feedback.csv` from the same folder and writes no files. It
takes a couple of seconds.

## 1. Meet the feedback

```
messages, columns                              -> (800, ['feedback_id', 'text', 'topic'])
messages per topic                             -> {'app': 200, 'bike': 200, 'payment': 200, 'station': 200}
example (payment)                              -> 'honestly, i was charged twice via m-pesa. so annoying.'
example (app)                                  -> 'Quick note: the app update drains my battery every time I open it.'
example (payment)                              -> 'Quick note: I was charged twice on my statement.'
characters per message: min, median, max       -> (24, 62, 139)
```

Our bike-share scheme from earlier phases has a feedback form, and
`feedback.csv` holds 800 messages from it. Each one has been tagged with
the team that should read it: **bike** (repairs), **app**, **station**
(docks) or **payment**. The messages are made up for this course, but
they look like real feedback: mixed case, punctuation, the odd "Ugh,",
and sometimes a second complaint tacked on the end.

The job for today: given a new message, route it to the right team. The
messages run from 24 to 139 characters, so before any model can touch
them we need a way to give every message the same shape.

## 2. Tokenisation: deciding what the pieces are

```
split on spaces                                -> ['Ugh,', 'the', 'app', "won't", 'load', 'since', 'the', 'update.', 'Please', 'fix!']
lowercase + letters/digits/apostrophes         -> ['ugh', 'the', 'app', "won't", 'load', 'since', 'the', 'update', 'please', 'fix']
distinct words in all 800 messages             -> 182
five most common                               -> [('the', 1059), ('i', 242), ('was', 214), ('my', 185), ('on', 169)]
is 'unlocking' in the vocabulary?              -> False
```

A **token** is one piece of text, the unit a model sees. The obvious
choice is "a word", but even that needs decisions. Splitting on spaces
gives `'update.'` and `'fix!'`, so "update" at the end of a sentence
would count as a different word from "update" in the middle. Lowercasing
and keeping only letters, digits and apostrophes fixes that:

```python
re.findall(r"[a-z0-9']+", text.lower())
```

The set of all distinct tokens is the **vocabulary**: 182 words here.
The most common words are the boring ones ("the", "I", "was"), which
will matter in section 4.

Word tokens have one big weakness. A word that never appeared in
training, like "unlocking", has no place in the vocabulary at all. The
model can't even see it.

```
first 8 merges a tiny BPE learned              -> ['th', 'the', 'on', 'in', 'at', 'no', 'as', 'an']
'station' as subword pieces                    -> ['station']
'unlocking' as subword pieces                  -> ['u', 'n', 'lo', 'ck', 'ing']
'stations' as subword pieces                   -> ['station', 's']
```

Modern language models solve this with **subword tokens**. The usual
method is called **byte-pair encoding (BPE)**: start with single
characters, find the pair of neighbours that appears most often, glue
it into one new token, and repeat. The script's `bpe_merges` does 60
rounds of that on our feedback, in about 25 lines. It learns "th", then
"the", then common endings.

The payoff: common words become a single token ("station"), and any
other word still breaks into known pieces ("station" + "s", or "lo",
"ck", "ing"). Nothing is ever unknown. Real tokenisers do tens of
thousands of merges on huge amounts of text, which is why a large
language model's vocabulary is about 50,000 to 200,000 pieces, and why
it "counts" text in tokens rather than words. For today's classifier,
word tokens are plenty.

## 3. Bag-of-words: counting the pieces

```
vocabulary, in column order                    -> ['and', 'app', 'crashes', 'dock', 'full', 'is', 'slow', 'the']
the app crashes                                -> [0, 1, 1, 0, 0, 0, 0, 1]
the app is slow                                -> [0, 1, 0, 0, 0, 1, 1, 1]
the dock is full and the app is slow           -> [1, 1, 0, 1, 1, 2, 1, 2]
all feedback: matrix shape                     -> (800, 178)
share of the matrix that is non-zero           -> 0.063
stored as                                      -> 'csr_matrix'
```

Now every message needs the same shape. The **bag-of-words** idea: give
every word in the vocabulary its own column, and fill in how many times
the message uses it. Imagine tipping the words into a bag and counting
them. Short or long, every message becomes a row of the same length.

```python
cv = CountVectorizer()
X = cv.fit_transform(df.text)      # one row per message, one column per word
```

`CountVectorizer` does the tokenising and counting in one go. (Its
default tokeniser drops one-letter words like "i", which is why it finds
178 columns rather than 182.) Like the scalers in lesson 035, it's
**fitted**: `fit` learns the vocabulary from the training text, and
`transform` counts new text using that fixed vocabulary.

Only 6% of the matrix is non-zero, since each message uses a handful of
the 178 words. So scikit-learn stores it as a **sparse matrix** that
remembers only the non-zero entries. With a real vocabulary of 50,000
words, that's the difference between fitting in memory and not.

## 4. TF-IDF: down-weighting the words everyone uses

```
idf of 'the', 'app', 'tyre', 'pesa'            -> (1.22, 3.22, 3.7, 4.43)
message                                        -> 'FYI the pedal came off this morning. Love the service though.'
its four heaviest words                        -> [('pedal', 0.38), ('off', 0.38), ('came', 0.38), ('this', 0.34)]
row length (L2 norm)                           -> 1.0
```

Raw counts treat every word as equally informative, but "the" appears
in nearly every message and tells us nothing about the topic. **TF-IDF**
fixes that. It multiplies two things:

- **TF**, term frequency: how often the word appears in *this* message.
- **IDF**, inverse document frequency: how *rare* the word is across all
  messages. scikit-learn uses `1 + log((1 + n) / (1 + df))`, where `n`
  is the number of messages and `df` is how many contain the word.

So "the" gets an IDF of 1.22, the lowest possible is 1, while "pesa"
(only in some payment messages) gets 4.43. In the example, "the" appears
twice but barely registers; "pedal", "came" and "off" carry the weight.
Finally each row is scaled to length 1, so a long rant doesn't outweigh
a short note just by having more words.

`TfidfVectorizer` is a drop-in replacement for `CountVectorizer`. It's
the standard first choice for text features.

## 5. A baseline classifier: which team should read this?

```
train / test messages                          -> (600, 200)
always guess the most common topic             -> 0.25
TF-IDF + logistic regression: accuracy         -> 0.99
macro F1                                       -> 0.99
```

Now it's just Phase 2 again. The text goes into a pipeline (lesson 036)
with TF-IDF first and logistic regression (lesson 033) second:

```python
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
model.fit(X_tr, y_tr)
```

The pipeline matters here: the vocabulary and IDF values are learned
from the training messages only, so nothing leaks from the test set.
Guessing one topic every time gets 25%; the baseline gets 99%. Our
made-up messages are cleaner than real ones, so expect lower numbers on
real feedback, but this simple recipe genuinely works well for routing
and tagging jobs.

```
strongest words for 'app'                      -> ['app', 'the', 'phone', 'update', 'notifications']
strongest words for 'bike'                     -> ['tyre', 'front', 'bike', '214', 'two']
strongest words for 'payment'                  -> ['nobody', 'paid', 'billed', 'statement', 'last']
strongest words for 'station'                  -> ['dock', 'station', 'at', 'monday', 'riverside']
```

The best thing about this model: you can **read it**. Each topic has one
weight per word, and the biggest weights show what it learned. Most are
sensible: "tyre" means bike, "billed" means payment, "dock" means
station. But look closer: "two", "last", "monday", "at", even "the" for
app. Those words aren't about the topic at all. They happen to turn up
more often in one team's messages ("after two minutes", "last week",
"on Monday"). The model will happily lean on them. That's the same
lesson as the planted bug in lesson 060: check what a model relies on,
not just its score.

```
mistakes on the test set                       -> 2
one of them                                    -> 'Quick note: the app keeps freezing on my phone, and my card was billed a fee nobody explained. :('
true topic, predicted                          -> ('app', 'payment')
brand-new messages                             -> ['app', 'payment']
```

The mistakes are fair ones: a message with two complaints, where the
tag went with the first. And it routes brand-new messages it never saw,
like "Paid twice for one trip, refund please", correctly.

## 6. What a bag forgets

```
same bag of words?                             -> True
same bag with word pairs (bigrams) added?      -> False
columns: words only, words + pairs             -> (6, 13)
```

"The bike is broken, not the app" and "the app is broken, not the bike"
say opposite things, and they have exactly the same bag of words. A bag
keeps *which* words appear and how often; it throws away the **order**.

A common patch is to add **n-grams**, runs of words treated as one
token. With `CountVectorizer(ngram_range=(1, 2))` you get every word
plus every pair of neighbours ("bike is", "app is"), and the two
sentences now differ. The cost is columns: 6 became 13 here, and on a
real vocabulary it's many times more.

Pairs help, but they still can't connect words that sit far apart, and
they don't know that "broken" in one place changes what "bike" means in
another. For that, every word needs to look at the others and decide
which ones matter. That mechanism is called **attention**, and
it's the very next lesson.

## What you can do now

- Explain what a token and a vocabulary are, and why tokenising needs
  decisions about case and punctuation.
- Describe how byte-pair encoding builds subword tokens, and why no word
  is ever unknown to it.
- Turn text into a bag-of-words matrix with `CountVectorizer`, and
  explain why it's stored sparse.
- Explain what TF-IDF rewards and penalises, and use `TfidfVectorizer`.
- Build, evaluate and read a TF-IDF + logistic regression text
  classifier inside a pipeline.
- Explain what bag-of-words throws away, and how n-grams win some of it
  back.

## What to do now

1. Run `lesson.py`. In section 5, swap `TfidfVectorizer()` for
   `CountVectorizer()` and compare.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   asks how few labelled messages the classifier can get by with.
3. Next up, lesson 062: attention, explained slowly. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Text becomes data in two steps: cut it into tokens, then count them.
Everything in this phase, up to large language models, still starts
with the first step.
