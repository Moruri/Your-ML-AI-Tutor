# Lesson 062 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-4-modern-ai/lesson-062-attention-explained-slowly/lesson.py my_lesson_062.py
python my_lesson_062.py
```

This lesson needs no data file, so the copy runs as it is.
`attention`, `make_batch`, `Lookup`, `train_lookup`, `r3` and `show`
are there to reuse.

---

## 1. Shapes

A sentence has 10 tokens, and each query and key has 64 numbers. What
shape is the grid of scores? How many scores is that for a document of
1,000 tokens?

<details>
<summary>Check yourself</summary>

10 by 10: one score for every (query, key) pair. For 1,000 tokens it's
1,000 x 1,000 = 1,000,000 scores, per attention layer. The cost grows
with the *square* of the length, which is why language models have a
limit on how much text they can read at once (their "context window"),
and why so much research goes into making attention cheaper.

</details>

## 2. Order?

Shuffle the four words of "the river bank fish" in section 1. Does
"bank"'s new vector change?

<details>
<summary>Check yourself</summary>

No. Attention is a weighted average over a *set*: the scores depend on
what the vectors are, not where they sit, so shuffling the words gives
exactly the same result for "bank". On its own, attention is as blind
to order as a bag of words. Transformers fix this by adding a
**positional encoding** to each token's vector, so position becomes part
of what's matched. That's part of lesson 063.

</details>

## 3. Minus infinity

Why do masks set scores to `-inf` rather than setting weights to 0
after the softmax?

<details>
<summary>Check yourself</summary>

Because softmax rescales so that each row adds to 1. If you zeroed some
weights afterwards, the rest would add to less than 1, and you'd have
to renormalise by hand. Setting the score to minus infinity makes
`exp(-inf) = 0` inside the softmax, so masked positions get exactly zero
weight and the allowed ones still add to 1. One gotcha: if *every*
score in a row is masked, softmax divides 0 by 0 and returns `nan`.
Make sure each query can see at least one key (with a causal mask, it
can always see itself).

</details>

## 4. Hands-on: a bigger dictionary

The lookup task in section 5 used 6 pairs. Make it **10 pairs** (keys
are still drawn from 12, so there are only 2 keys left over). Set
`SLOTS = 10` before training, use the same test batch
(`make_batch(2000, torch.Generator().manual_seed(620))`), and train both
models with `train_lookup`. Predict first: which model suffers more?

<details>
<summary>Expected results</summary>

```
average the pairs, no attention  -> test accuracy 0.255   (was 0.338 with 6 pairs)
learned attention                -> test accuracy 1.0     (was 1.0)
```

The average gets worse: ten values blur together, so knowing "which
values are in there" narrows things down less. Attention doesn't care.
It scores each pair separately and picks out the match, however many
pairs there are. That's the big strength of attention over averaging or
squeezing a sequence into one hidden state: what it can retrieve
doesn't fade as the input grows. The price is the score grid from
exercise 1.

</details>

<details>
<summary>One way to write it</summary>

```python
SLOTS = 10        # at the top of your copy, replacing 6
...
test = make_batch(2000, torch.Generator().manual_seed(620))
keys, values, query, answer, pick = test
for use_attention in (False, True):
    model = train_lookup(use_attention)
    with torch.no_grad():
        logits, _ = model(keys, values, query)
    show(f"attention={use_attention}", r3((logits.argmax(1) == answer).float().mean()))
```

</details>

## 5. (Optional) Many heads

Real models run several attention computations side by side, each with
its own `Wq`, `Wk` and `Wv`, and glue the outputs together. This is
called **multi-head attention**. In words, why might eight small heads
beat one big one?

<details>
<summary>Check yourself</summary>

One softmax gives one set of weights, so one head can only follow one
idea of "relevant" per position. In a sentence, a word might need to
look at its subject, at the previous word, and at a name mentioned
earlier, all at once. Eight heads can each specialise in one kind of
relationship, at about the same total cost as one wide head, since each
works on a smaller slice of the numbers. PyTorch's
`nn.MultiheadAttention` does exactly this, and it's part of the
Transformer block in lesson 063.

</details>

---

That's attention. If one thing sticks: *each position scores every key
against its query, turns the scores into weights with a softmax, and
takes a weighted average of the values; training decides what to look
for.*
