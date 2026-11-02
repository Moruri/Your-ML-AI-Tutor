"""
Lesson 067 - Embeddings and semantic search

Lesson 064 turned a sentence into 384 numbers so a classifier could use them.
Today we use those numbers directly: if two sentences mean similar things,
their vectors point in similar directions. That one fact is enough to build a
search tool that finds rider messages by meaning, not by matching words.

Run it with (inside the venv from lesson 013, with torch and transformers
installed as in lessons 051 and 064):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

It reads feedback.csv (the same made-up rider messages as lessons 061 to 066)
from this folder and writes one file, output/index.npy, the saved search
index. The embedding model is the one from lesson 064 (about 90 MB, cached
after the first download). It takes under a minute on a laptop CPU.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

torch.set_num_threads(1)              # same numbers on every machine
hf_logging.set_verbosity_error()      # keep the library's chatter out of the lesson output

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # the model from lesson 064


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def r3(x):
    return round(float(x), 3) + 0.0      # + 0.0 turns -0.0 into 0.0


class Embedder:
    """Turn a list of texts into unit-length vectors, one row per text."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL).eval()

    def __call__(self, texts, batch_size=64):
        rows = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True,
                                   return_tensors="pt")
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            mean = (hidden * mask).sum(1) / mask.sum(1)        # average over real tokens (lesson 064)
            rows.append(F.normalize(mean, dim=1))               # length 1, so dot product = cosine
        return torch.cat(rows).numpy()


# ---------------------------------------------------------------------------
# 1. A sentence becomes a point
# ---------------------------------------------------------------------------

def section_point(embed):
    heading("1. A sentence becomes a point")
    v = embed(["no free docks at Uni Square again"])[0]
    show("numbers per sentence", v.shape[0])
    show("first five numbers", [r3(x) for x in v[:5]])
    show("length of the vector", r3(np.linalg.norm(v)))
    print("  384 numbers, scaled to length 1. On their own they mean nothing;")
    print("  what matters is how close two of these points are.")


# ---------------------------------------------------------------------------
# 2. Closeness is meaning: cosine similarity
# ---------------------------------------------------------------------------

PAIRS = [
    ("no free docks at Uni Square again", "the station was full, nowhere to leave the bike"),
    ("no free docks at Uni Square again", "the app logged me out twice"),
    ("I was charged twice for one ride", "money taken from my account two times"),
    ("the bike is great", "the bike is not great"),
]


def section_cosine(embed):
    heading("2. Closeness is meaning: cosine similarity")
    tfidf = TfidfVectorizer().fit([t for p in PAIRS for t in p])
    for a, b in PAIRS:
        ea, eb = embed([a, b])
        ta, tb = tfidf.transform([a, b]).toarray()
        print(f"    '{a}'")
        print(f"    '{b}'")
        print(f"      embedding cosine {r3(ea @ eb):>6}    word-count cosine {r3(ta @ tb):>6}")
    print("  The first and third pairs share almost no words but mean the same")
    print("  thing, and only the embedding sees it. The last pair is a warning:")
    print("  'great' and 'not great' look close to both. Similar is not equal.")


# ---------------------------------------------------------------------------
# 3. A search engine in a dozen lines
# ---------------------------------------------------------------------------

def search(query_vec, index, k=5):
    """Return (row numbers, scores) of the k rows most similar to the query."""
    scores = index @ query_vec
    top = np.argsort(-scores)[:k]
    return top, scores[top]


def section_search(embed, df):
    heading("3. A search engine in a dozen lines")
    t0 = time.time()
    index = embed(df["text"].tolist())
    show("messages embedded", index.shape[0])
    show("index shape", index.shape)
    show("seconds to build", round(time.time() - t0))
    query = "the wheel wobbles when I ride"
    show("query", query)
    rows, scores = search(embed([query])[0], index)
    for r, s in zip(rows, scores):
        print(f"    {r3(s):.3f}  [{df['topic'].iloc[r]:<7}] {df['text'].iloc[r][:58]}")
    print("  Embed everything once, embed the query, take the dot product with")
    print("  every row, sort. The top hit is about a 'wobbly' front tyre: other")
    print("  words, same problem. Keyword matching would never have linked them.")
    return index


# ---------------------------------------------------------------------------
# 4. Keywords versus meaning
# ---------------------------------------------------------------------------

QUERIES = [
    ("nowhere to return my bike", "station"),
    ("money taken from my account two times", "payment"),
    ("the wheel wobbles when I ride", "bike"),
    ("cannot sign in on my phone", "app"),
    ("refund still not received", "payment"),
    ("the seat is stuck too low", "bike"),
    ("map shows the wrong location", "app"),
    ("all the parking spots were taken", "station"),
]


def precision_at_k(rows, topics, want):
    return sum(topics[r] == want for r in rows) / len(rows)


def section_keywords(embed, df, index):
    heading("4. Keywords versus meaning")
    tfidf = TfidfVectorizer().fit(df["text"])
    tf_index = tfidf.transform(df["text"])
    topics = df["topic"].to_numpy()
    print(f"    {'query':<40} {'keywords':>8} {'meaning':>8}")
    kw, mn = [], []
    for q, want in QUERIES:
        kw_scores = (tf_index @ tfidf.transform([q]).T).toarray().ravel()
        kw_rows = np.argsort(-kw_scores)[:5]
        mn_rows, _ = search(embed([q])[0], index)
        kw.append(precision_at_k(kw_rows, topics, want))
        mn.append(precision_at_k(mn_rows, topics, want))
        print(f"    {q:<40} {kw[-1]:>8.1f} {mn[-1]:>8.1f}")
    show("keyword search: mean precision@5", r3(np.mean(kw)))
    show("semantic search: mean precision@5", r3(np.mean(mn)))
    print("  Precision@5: of the five results, the share on the right topic.")
    print("  Keyword search is lost when the rider uses different words.")


# ---------------------------------------------------------------------------
# 5. Measuring search honestly
# ---------------------------------------------------------------------------

def section_measure(embed, df):
    heading("5. Measuring search honestly")
    docs, queries = train_test_split(df, test_size=200, random_state=67, stratify=df["topic"])
    d_topics = docs["topic"].to_numpy()
    d_index = embed(docs["text"].tolist())
    q_index = embed(queries["text"].tolist())
    tfidf = TfidfVectorizer().fit(docs["text"])
    kw_sims = (tfidf.transform(queries["text"]) @ tfidf.transform(docs["text"]).T).toarray()
    mn_sims = q_index @ d_index.T
    hybrid = 0.5 * kw_sims + 0.5 * mn_sims
    for name, sims in [("keyword search", kw_sims), ("semantic search", mn_sims), ("hybrid (half each)", hybrid)]:
        top = np.argsort(-sims, axis=1)[:, :5]
        p5 = np.mean([precision_at_k(row, d_topics, want) for row, want in zip(top, queries["topic"])])
        show(f"{name}: precision@5 on 200 queries", r3(p5))
    print("  Every held-out message is a query, searched against the other 600,")
    print("  and a result counts as relevant when it shares the query's topic.")
    print("  Here riders reuse the same few phrasings, so keywords do just as")
    print("  well. The honest answer is to measure on your own data, and often")
    print("  to combine both scores.")


# ---------------------------------------------------------------------------
# 6. Practical search: save the index, know when nothing matches
# ---------------------------------------------------------------------------

def section_practical(embed, df, index):
    heading("6. Practical search: save the index, know when nothing matches")
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "index.npy", index.astype(np.float32))
    loaded = np.load(OUT / "index.npy")
    show("saved index (KB)", round((OUT / "index.npy").stat().st_size / 1024))
    show("loaded index matches", bool(np.allclose(loaded, index, atol=1e-6)))
    for q in ["charged twice for a single trip", "what is the capital of France"]:
        _, scores = search(embed([q])[0], loaded, k=1)
        show(f"best score for '{q}'", r3(scores[0]))
    print("  Embedding is the slow part, so do it once and save it. And search")
    print("  always returns something: an off-topic query still gets a 'best'")
    print("  match. A score threshold lets you say 'nothing relevant' instead.")


def main():
    print("=" * 66)
    print("  Lesson 067: Embeddings and semantic search")
    print("=" * 66)
    df = pd.read_csv(HERE / "feedback.csv")
    embed = Embedder()
    section_point(embed)
    section_cosine(embed)
    index = section_search(embed, df)
    section_keywords(embed, df, index)
    section_measure(embed, df)
    section_practical(embed, df, index)
    print()
    print("Embed once, compare by cosine, sort, and measure the results: that")
    print("is semantic search, and it is the first half of tomorrow's RAG.")
    print()


if __name__ == "__main__":
    main()
