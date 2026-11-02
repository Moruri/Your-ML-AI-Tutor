# Lesson 067 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv (with `transformers` installed, as in lesson 064).
For the hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-4-modern-ai/lesson-067-embeddings-and-semantic-search/lesson.py my_lesson_067.py
cp lessons/phase-4-modern-ai/lesson-067-embeddings-and-semantic-search/feedback.csv .
python my_lesson_067.py
```

The script reads `feedback.csv` from its own folder, which is why the
second line copies it next to your copy. `Embedder`, `search` and
`precision_at_k` are there to reuse.

---

## 1. Why normalise?

Section 1 scales every vector to length 1. What would go wrong in
`index @ query` if we skipped that step?

<details>
<summary>Check yourself</summary>

The dot product would mix up two things: the angle between the vectors
(meaning) and their lengths. A message whose vector happened to be long
would score high against every query and float to the top of every
search. With every length set to 1, the dot product *is* the cosine of
the angle, so only direction counts. It also means one matrix multiply
scores all 800 messages at once, with no division needed.

</details>

## 2. "Great" versus "not great"

Section 2 scored "the bike is great" and "the bike is not great" at
0.82. You're building a search for unhappy riders. Why is this a
problem, and what could you do?

<details>
<summary>Check yourself</summary>

A search for complaints would happily return praise, because the model
mostly encodes *what a sentence is about* (the bike) and only weakly
*what it says about it* (good or bad). Options: use the embedding to
find the topic and a separate sentiment classifier (lessons 061 and
064 built classifiers) to filter; use a larger or task-specific
embedding model; or check results by hand before trusting them for
this job. Always test the cases that matter to you, not just the easy
ones.

</details>

## 3. Is the topic label a fair judge?

Sections 4 and 5 count a result as relevant if it has the right topic.
Give one way this stand-in could mislead you.

<details>
<summary>Check yourself</summary>

Topic is much coarser than relevance. For "nowhere to return my bike",
a message about a station with no bikes is the right topic (`station`)
but the wrong problem, and it would count as a hit. It can also go the
other way: some messages mention two problems (look at
`feedback.csv`), so a relevant result can carry the "wrong" label. For
a real search tool you'd mark relevance by hand for a few dozen real
queries. It's tedious, and it's the only score you can fully trust.

</details>

## 4. Hands-on: tune the hybrid weight

Section 5 mixed the two scores half and half. Try weights of 0, 0.3,
0.5, 0.7 and 1 on the keyword side, on the same 200 queries. Predict
the shape first: where's the best weight, and how much does it matter?

<details>
<summary>Expected results</summary>

```
keyword weight 0.0  precision@5 0.931
keyword weight 0.3  precision@5 0.94
keyword weight 0.5  precision@5 0.942
keyword weight 0.7  precision@5 0.943
keyword weight 1.0  precision@5 0.932
```

Any mix beats either method alone, and the exact weight barely
matters once both are in. Be careful, though: you just picked a weight
by looking at the queries you're scoring, the Phase 2 mistake of tuning
on the test set. To choose it properly, tune on one set of queries and
report on another.

</details>

<details>
<summary>One way to write it</summary>

In your copy, replace the `main()` call at the very bottom with:

```python
if __name__ == "__main__":
    df = pd.read_csv(HERE / "feedback.csv")
    embed = Embedder()
    docs, queries = train_test_split(df, test_size=200, random_state=67, stratify=df["topic"])
    mn = embed(queries["text"].tolist()) @ embed(docs["text"].tolist()).T
    tfidf = TfidfVectorizer().fit(docs["text"])
    kw = (tfidf.transform(queries["text"]) @ tfidf.transform(docs["text"]).T).toarray()
    for w in [0, 0.3, 0.5, 0.7, 1]:
        top = np.argsort(-(w * kw + (1 - w) * mn), axis=1)[:, :5]
        p5 = np.mean([precision_at_k(r, docs["topic"].to_numpy(), t) for r, t in zip(top, queries["topic"])])
        print(f"keyword weight {w:.1f}  precision@5 {round(p5, 3)}")
```

</details>

## 5. (Optional) Duplicate messages

Support wants to know how many messages are near-copies of another
message. Using only the index from section 3, how would you find them,
and what's the catch?

<details>
<summary>Check yourself</summary>

Compute `index @ index.T`, an 800 by 800 table of similarities, set the
diagonal to -1 so nothing matches itself, and flag every pair above a
high threshold such as 0.95. The catch is choosing that threshold:
print pairs just above and just below it and read them. In this
dataset many messages share a template ("Quick note: ... So
annoying."), so "near-duplicate" can mean "same complaint" or just
"same opening words". The full table also grows with the square of the
number of messages, which is fine at 800 and not at a million.

</details>

---

That's embeddings and semantic search. If one thing sticks: *search
by meaning is a dot product, and whether it beats keywords is
something you measure, not something you assume.*
