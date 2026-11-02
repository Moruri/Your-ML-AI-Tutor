# Lesson 067 - Embeddings and semantic search

**Phase 4 - Modern AI: NLP, vision, generative and LLM basics** | Week 9, Day 1 | Monday 2026-11-02

> **Goal:** Embed documents, measure similarity, and build a small search
> tool.

Time: about 40 minutes. Same venv as lesson 064, with `torch` and
`transformers` installed.

---

In lesson 064 a pretrained model turned each rider message into 384
numbers, and a classifier learned from them. Today we skip the
classifier and use the numbers directly.

The idea is simple: the model was trained so that sentences with
similar meanings end up as vectors pointing in similar directions. So
"find messages like this one" becomes "find the nearest points", which
is a few lines of NumPy. That's **semantic search**, and it powers
"related articles", duplicate detection, recommendation, and the first
half of tomorrow's lesson on RAG.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-4-modern-ai/lesson-067-embeddings-and-semantic-search/lesson.py
```

It reads `feedback.csv` (the same 800 rider messages as lessons 061 to
066, copied into this folder) and writes one file,
`output/index.npy`. It uses lesson 064's model,
`sentence-transformers/all-MiniLM-L6-v2`, already in your cache if you
ran that lesson. It takes under a minute.

## 1. A sentence becomes a point

```
numbers per sentence                           -> 384
first five numbers                             -> [0.0, -0.017, 0.043, 0.005, 0.035]
length of the vector                           -> 1.0
```

`Embedder` is lesson 064's recipe wrapped in a class: tokenize, run the
model, and average the vectors of the real tokens (ignoring padding).
One new step at the end: `F.normalize` scales every vector to length 1.

The single numbers mean nothing to us. Nobody can say what number 3
"is". What matters is the *geometry*: which points are near which.

## 2. Closeness is meaning: cosine similarity

```
'no free docks at Uni Square again'
'the station was full, nowhere to leave the bike'
  embedding cosine  0.246    word-count cosine    0.0
'no free docks at Uni Square again'
'the app logged me out twice'
  embedding cosine  0.145    word-count cosine    0.0
'I was charged twice for one ride'
'money taken from my account two times'
  embedding cosine  0.473    word-count cosine    0.0
'the bike is great'
'the bike is not great'
  embedding cosine  0.819    word-count cosine  0.836
```

**Cosine similarity** measures the angle between two vectors: 1 means
the same direction, 0 means unrelated. Because our vectors have length
1, it's just the dot product, `a @ b`, which is why we normalised.

The word-count column uses lesson 061's TF-IDF vectors. When two
messages share no words, it scores them 0 no matter what they mean.
The embedding sees that "charged twice for one ride" and "money taken
two times" are the same complaint.

Read the numbers relatively, not as grades. 0.246 isn't "25% similar";
it's just clearly higher than the 0.145 of an unrelated pair. And the
last pair is a warning: "great" and "not great" look almost identical
to *both* methods. Small embedding models notice topic far more than
they notice negation. Similar is not the same as equal.

## 3. A search engine in a dozen lines

```
messages embedded                              -> 800
index shape                                    -> (800, 384)
seconds to build                               -> 3
query                                          -> 'the wheel wobbles when I ride'
  0.539  [bike   ] Hi, the front tyre felt wobbly this morning. So annoying.
  0.530  [bike   ] Not happy. the front tyre felt wobbly on my ride home. Ple
  0.495  [bike   ] Quick note: the pedal felt wobbly again. So annoying.
  0.479  [bike   ] The saddle felt wobbly halfway up the hill. Please fix!
  0.474  [bike   ] Ugh, the back tyre felt wobbly on bike 214. So annoying.
```

The whole search engine:

1. Embed every message once. That's the **index**: an 800 by 384 table.
2. Embed the query with the same model.
3. `index @ query` gives one score per message, all at once.
4. Sort, and take the top `k`.

The query said "wheel wobbles"; the top hit says "tyre felt wobbly".
Different words, same problem. Your seconds will differ from ours; on
any laptop it's a few.

## 4. Keywords versus meaning

```
query                                    keywords  meaning
nowhere to return my bike                     1.0      0.6
money taken from my account two times         1.0      1.0
the wheel wobbles when I ride                 0.0      1.0
cannot sign in on my phone                    1.0      1.0
refund still not received                     1.0      1.0
the seat is stuck too low                     1.0      1.0
map shows the wrong location                  1.0      1.0
all the parking spots were taken              0.0      1.0
keyword search: mean precision@5               -> 0.75
semantic search: mean precision@5              -> 0.95
```

To compare search methods we need a score. **Precision@5** is the share
of the top five results that are relevant. We don't have hand-marked
relevance, so we use a stand-in: a result counts if it has the topic
we'd expect for the query.

On eight queries written by hand, keyword search (TF-IDF, ranked by
the same dot product) scores 0 twice: the riders never wrote "wobbles"
or "parking spots", so it ranked on unhelpful words like "the". Semantic
search wins, 0.95 to 0.75. But look at the first row: there semantic
search did *worse*: two of its five results were payment complaints,
and all three "right" ones were about stations with *no bikes*, the
opposite problem. Our topic-based score even counts those as hits.
Close in meaning is not the same as right.

## 5. Measuring search honestly

```
keyword search: precision@5 on 200 queries     -> 0.932
semantic search: precision@5 on 200 queries    -> 0.931
hybrid (half each): precision@5 on 200 queries -> 0.942
```

Eight queries we wrote ourselves, knowing how the trick works, is not a
fair test. So here every one of 200 held-out messages is used as a
query against the other 600, and we average precision@5.

Now it's a tie. Real riders in this dataset reuse the same few
phrasings ("felt wobbly", "charged twice"), so a real message almost
always shares words with its neighbours, and keywords do fine. Adding
the two scores together, **hybrid search**, beats both a little, and
that's what a lot of production search does.

The lesson isn't "embeddings are better". It's that the answer depends
on how your users write, and you only find out by measuring on queries
that look like theirs.

## 6. Practical search: save the index, know when nothing matches

```
saved index (KB)                               -> 1200
loaded index matches                           -> True
best score for 'charged twice for a single trip' -> 0.641
best score for 'what is the capital of France' -> 0.184
```

Two habits:

- **Embed once, save, reuse.** Embedding is the slow part. `np.save`
  writes the index (800 x 384 x 4 bytes = 1200 KB); next time you load
  it and only embed the query. When documents change, embed just the
  new ones. Past a few hundred thousand rows, a **vector database**
  does this storing and searching for you, but the idea is exactly
  this.
- **Search always returns something.** Ask about the capital of France
  and you still get five "results". The best score gives it away: 0.18
  against 0.64 for a real query. A threshold, chosen by looking at
  scores on real queries, lets you say "nothing relevant found".
  Tomorrow this matters a lot.

## What you can do now

- Turn texts into unit-length embeddings and compare them with a dot
  product.
- Explain why cosine scores are relative, and where embeddings stumble
  (negation, look-alike problems).
- Build a search index and query it in a few lines of NumPy.
- Measure search with precision@k, on queries that look like real ones.
- Combine keyword and semantic scores, save an index, and use a score
  threshold for "no match".

## What to do now

1. Run `lesson.py`. Add your own queries to `QUERIES` in section 4,
   including ones written in Swahili or Sheng, and predict which method
   wins before you run it.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   tunes the hybrid weight.
3. Next lesson, 068: retrieval-augmented generation. We point this
   search at a set of help articles and hand what it finds to lesson
   066's chat model, so it answers from our documents instead of
   making things up.

An embedding puts meaning on a map; search is just asking what's
nearby, and measuring tells you whether "nearby" is what you wanted.
