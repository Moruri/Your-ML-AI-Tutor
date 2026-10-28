"""
Lesson 061 - Text as data: tokenisation and bag-of-words

Models eat numbers, and text is a string of characters. Before any model can
learn from riders' feedback, we have to decide what the pieces of text are
(tokenisation) and how to turn a pile of pieces into a row of numbers
(bag-of-words and TF-IDF). Then we train a plain, honest baseline classifier,
the one every fancier text model has to beat.

Run it with (inside the venv from lesson 013, with requirements.txt installed):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files. Every random step has a fixed seed, so your
numbers will match the README. It takes a couple of seconds.
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
FEEDBACK_CSV = HERE / "feedback.csv"


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def load_feedback():
    return pd.read_csv(FEEDBACK_CSV)


# ---------------------------------------------------------------------------
# 1. Meet the feedback
# ---------------------------------------------------------------------------

def section_data():
    heading("1. Meet the feedback")
    df = load_feedback()
    show("messages, columns", (len(df), list(df.columns)))
    show("messages per topic", df["topic"].value_counts().sort_index().to_dict())
    for i in [2, 4, 6]:
        show(f"example ({df.topic[i]})", df.text[i])
    lengths = df["text"].str.len()
    show("characters per message: min, median, max", (int(lengths.min()), int(lengths.median()),
                                                     int(lengths.max())))
    print("  Every message is a different length. A model needs every row to be")
    print("  the same shape, so 'just use the characters' won't do.")


# ---------------------------------------------------------------------------
# 2. Tokenisation: deciding what the pieces are
# ---------------------------------------------------------------------------

def tokenise(text):
    """Lowercase, then keep runs of letters, digits and apostrophes."""
    return re.findall(r"[a-z0-9']+", text.lower())


def bpe_merges(words, n_merges):
    """A tiny byte-pair encoding: repeatedly glue the most common adjacent pair."""
    corpus = Counter(tuple(w) for w in words)
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for word, count in corpus.items():
            for a, b in zip(word, word[1:]):
                pairs[a, b] += count
        (a, b), _ = pairs.most_common(1)[0]
        merges.append(a + b)
        new = Counter()
        for word, count in corpus.items():
            out, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == (a, b):
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            new[tuple(out)] += count
        corpus = new
    return merges


def apply_merges(word, merges):
    pieces = list(word)
    for m in merges:
        i = 0
        while i < len(pieces) - 1:
            if pieces[i] + pieces[i + 1] == m:
                pieces[i:i + 2] = [m]
            else:
                i += 1
    return pieces


def section_tokenise():
    heading("2. Tokenisation: deciding what the pieces are")
    text = "Ugh, the app won't load since the update. Please fix!"
    show("split on spaces", text.split())
    show("lowercase + letters/digits/apostrophes", tokenise(text))
    df = load_feedback()
    vocab = Counter(t for s in df.text for t in tokenise(s))
    show("distinct words in all 800 messages", len(vocab))
    show("five most common", vocab.most_common(5))
    show("is 'unlocking' in the vocabulary?", "unlocking" in vocab)

    words = [t for s in df.text for t in tokenise(s)]
    merges = bpe_merges(words, 60)
    show("first 8 merges a tiny BPE learned", merges[:8])
    for w in ["station", "unlocking", "stations"]:
        show(f"'{w}' as subword pieces", apply_merges(w, merges))
    print("  Word tokens are simple but have no answer for a word they never saw.")
    print("  Subword tokens break any word into known pieces: that's how modern")
    print("  language models tokenise.")


# ---------------------------------------------------------------------------
# 3. Bag-of-words: counting the pieces
# ---------------------------------------------------------------------------

def section_bag():
    heading("3. Bag-of-words: counting the pieces")
    docs = ["the app crashes", "the app is slow", "the dock is full and the app is slow"]
    cv = CountVectorizer()
    X = cv.fit_transform(docs)
    show("vocabulary, in column order", list(cv.get_feature_names_out()))
    for d, row in zip(docs, X.toarray()):
        show(d, row.tolist())

    df = load_feedback()
    cv = CountVectorizer()
    X = cv.fit_transform(df.text)
    show("all feedback: matrix shape", X.shape)
    show("share of the matrix that is non-zero", round(X.nnz / (X.shape[0] * X.shape[1]), 3))
    show("stored as", type(X).__name__)
    print("  Each message becomes one row of word counts, all rows the same")
    print("  length. Most counts are zero, so scikit-learn stores it sparse.")


# ---------------------------------------------------------------------------
# 4. TF-IDF: down-weighting the words everyone uses
# ---------------------------------------------------------------------------

def section_tfidf():
    heading("4. TF-IDF: down-weighting the words everyone uses")
    df = load_feedback()
    tf = TfidfVectorizer()
    X = tf.fit_transform(df.text)
    names = tf.get_feature_names_out()
    idf = dict(zip(names, tf.idf_.round(2)))
    show("idf of 'the', 'app', 'tyre', 'pesa'", (float(idf["the"]), float(idf["app"]),
                                               float(idf["tyre"]), float(idf["pesa"])))
    i = 3
    row = X[i].toarray()[0]
    top = [(names[j], round(float(row[j]), 2)) for j in row.argsort()[::-1][:4]]
    show("message", df.text[i])
    show("its four heaviest words", top)
    show("row length (L2 norm)", round(float(np.linalg.norm(row)), 3))
    print("  'the' appears almost everywhere, so it says almost nothing. Rare,")
    print("  specific words get the weight. Each row is scaled to length 1.")


# ---------------------------------------------------------------------------
# 5. A baseline classifier: which team should read this?
# ---------------------------------------------------------------------------

def section_classifier():
    heading("5. A baseline classifier: which team should read this?")
    df = load_feedback()
    X_tr, X_te, y_tr, y_te = train_test_split(df.text, df.topic, test_size=0.25,
                                              stratify=df.topic, random_state=61)
    show("train / test messages", (len(X_tr), len(X_te)))
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    show("always guess the most common topic", round(accuracy_score(y_te, dummy.predict(X_te)), 3))

    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    show("TF-IDF + logistic regression: accuracy", round(accuracy_score(y_te, pred), 3))
    show("macro F1", round(f1_score(y_te, pred, average="macro"), 3))

    tf, lr = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
    names = tf.get_feature_names_out()
    for k, topic in enumerate(lr.classes_):
        top = names[np.argsort(lr.coef_[k])[::-1][:5]].tolist()
        show(f"strongest words for '{topic}'", top)

    wrong = [(t, a, p) for t, a, p in zip(X_te, y_te, pred) if a != p]
    show("mistakes on the test set", len(wrong))
    if wrong:
        t, a, p = wrong[0]
        show("one of them", t)
        show("true topic, predicted", (a, p))
    new = ["The scooter won't unlock and the screen is frozen",
           "Paid twice for one trip, refund please"]
    show("brand-new messages", list(model.predict(new)))
    print("  A few lines, a second to train, and you can read why it decides.")
    print("  That's the baseline every fancier text model has to beat.")


# ---------------------------------------------------------------------------
# 6. What a bag forgets
# ---------------------------------------------------------------------------

def section_limits():
    heading("6. What a bag forgets")
    a = "the bike is broken not the app"
    b = "the app is broken not the bike"
    cv = CountVectorizer().fit([a, b])
    show("same bag of words?", bool((cv.transform([a]) != cv.transform([b])).nnz == 0))
    cv2 = CountVectorizer(ngram_range=(1, 2)).fit([a, b])
    show("same bag with word pairs (bigrams) added?", bool((cv2.transform([a]) != cv2.transform([b])).nnz == 0))
    show("columns: words only, words + pairs", (len(cv.vocabulary_), len(cv2.vocabulary_)))
    print("  A bag keeps which words appear and how often, and throws away the")
    print("  order. Word pairs win some order back, at the cost of many more")
    print("  columns. Lesson 062's attention is the way to read words in context.")


def main():
    print("=" * 66)
    print("  Lesson 061: Text as data, tokenisation and bag-of-words")
    print("=" * 66)
    section_data()
    section_tokenise()
    section_bag()
    section_tfidf()
    section_classifier()
    section_limits()
    print()
    print("Tokenise, count, weight, classify. Simple, fast, and a baseline you")
    print("can explain to anyone.")
    print()


if __name__ == "__main__":
    main()
