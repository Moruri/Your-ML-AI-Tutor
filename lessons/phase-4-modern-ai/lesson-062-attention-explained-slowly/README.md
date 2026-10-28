# Lesson 062 - Attention, explained slowly

**Phase 4 - Modern AI: NLP, vision, generative and LLM basics** | Week 8, Day 3 | Wednesday 2026-10-28

> **Goal:** Implement scaled dot-product attention in a few lines and see
> what it computes.

Time: about 45 minutes. Same venv as lesson 051, with `torch` installed.

---

Lesson 061 ended on a problem. A bag of words forgets order, and even
word pairs can't tell that "broken" changes what "bike" means three
words away. Lesson 058's recurrent networks read in order, but one step
at a time, squeezing everything seen so far into one hidden state.

**Attention** takes a different route. Every position in a sequence
gets to look at *every* other position at once, decide which ones
matter to it, and pull in what it finds. That one idea is the engine of
the Transformer (lesson 063), and so of every large language model.

The good news: it's not complicated. It's dot products (the same sums
inside lesson 049's neuron), a softmax and a weighted average. We'll go
slowly, one piece at a time, then build it in four lines and train a
model that needs it.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-4-modern-ai/lesson-062-attention-explained-slowly/lesson.py
```

It needs no data file and writes no files. Every random step has a
fixed seed, so your numbers will match. It takes a few seconds.

## 1. Context changes meaning

```
'bank' on its own                              -> [0.5, 0.5]
the river bank fish: 'bank' becomes            -> [0.765, 0.191]
  how much 'bank' looked at each word          -> {'the': 0.043, 'river': 0.319, 'bank': 0.319, 'fish': 0.319}
the bank loan cash: 'bank' becomes             -> [0.191, 0.765]
  how much 'bank' looked at each word          -> {'the': 0.043, 'bank': 0.319, 'loan': 0.319, 'cash': 0.319}
```

Start with the intuition. The script gives six words hand-made vectors
of two numbers: the first means "watery", the second "money". "River"
is `[1, 0]`, "loan" is `[0, 1]`, and "bank", which could be either, sits
in the middle at `[0.5, 0.5]`.

Now let every word look at every word in its sentence, and replace
itself with a weighted average of what it sees. The weights come from
how well the vectors match (their dot product), turned into proportions
that add to 1. "The" is `[0, 0]`, matches nothing, and gets almost no
weight.

Same word, same starting vector, two different results. Next to
"river" and "fish", "bank" becomes `[0.77, 0.19]`: mostly watery. Next
to "loan" and "cash", it becomes `[0.19, 0.77]`: mostly money. The
word's new vector now carries its *context*. That's what attention is
for.

## 2. Scores, softmax, weighted average

```
scores = keys @ query                          -> [1.0, 0.0, 0.7]
weights = softmax(scores)                      -> [0.474, 0.174, 0.351]
weights add up to                              -> 1.0
output = weights @ values                      -> [6.499, 3.501]
```

Section 1 used each word's vector for three jobs at once. Real attention
separates them, and gives each a name. Think of looking something up:

- The **query** is what this position is looking for.
- Each **key** is a label saying what a position offers.
- Each **value** is the content that position hands over if chosen.

Three steps, always the same:

1. **Score** every key against the query with a dot product. Matching
   directions score high.
2. **Softmax** the scores into weights: all positive, adding to 1.
3. **Average** the values with those weights.

Here the query `[1, 0]` matches the first key best, so the output leans
towards the first value, `[10, 0]`. But softmax never gives *all* the
weight to one key, so the others still contribute a little. That's
deliberate: it keeps everything smooth, so gradients can flow and the
model can learn what to look for.

## 3. Why we divide by the square root of d

```
d=4: score spread, top weight raw/scaled       -> (1.99, 0.49, 0.303)
d=64: score spread, top weight raw/scaled      -> (7.908, 0.864, 0.316)
d=512: score spread, top weight raw/scaled     -> (22.239, 0.949, 0.317)
```

Real queries and keys have 64 or more numbers, not 2. And the longer
the vectors, the bigger dot products get: a dot product adds up `d`
terms, so for random vectors its spread grows like $\sqrt{d}$. With
`d=512` the scores spread over about ±22.

Softmax hates big gaps. Feed it scores that far apart and it puts
nearly all the weight (0.95 on average here) on one key, out of ten.
That's almost a hard choice, and a hard choice has almost no gradient
(the saturation problem from lesson 053). Training stalls.

The fix is simple: divide the scores by $\sqrt{d}$. Now the spread is
about 1 whatever the size, and the top weight stays around 0.31 in every
case. This is the "scaled" in **scaled dot-product attention**.

## 4. Scaled dot-product attention in four lines

```
Q, K, V shape (batch, positions, d)            -> (2, 5, 8)
output shape                                   -> (2, 5, 8)
weights shape (batch, queries, keys)           -> (2, 5, 5)
every row of weights sums to 1                 -> True
matches F.scaled_dot_product_attention         -> True
```

Put it together and you get the most famous formula in modern AI:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

In code, with every query handled at once as a matrix:

```python
def attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V, weights
```

Read the shapes. With 5 positions, `Q @ K.T` is a 5 by 5 grid of
scores: row `i` says how much position `i` cares about each position.
The softmax runs along each row, so every row of weights adds to 1.
Then `weights @ V` gives each position its own blend. The extra batch
dimension at the front just comes along for the ride.

PyTorch ships the same thing, heavily optimised, as
`F.scaled_dot_product_attention`. Ours matches it exactly. Use theirs
in real code; now you know what's inside.

## 5. Learning what to look at: a lookup task

```
one example: keys                              -> [7, 8, 4, 3, 11, 9]
             values                            -> [9, 0, 1, 4, 9, 8]
             query key, answer                 -> (9, 8)
guessing a value at random                     -> 0.1
average the pairs, no attention: test accuracy -> 0.338
learned attention: test accuracy               -> 1.0
```

So far we chose the vectors by hand. In a real model, the queries, keys
and values are **learned**. Each comes from its own linear layer
(lesson 052) applied to the same input:

```python
mixed, w = attention(self.Wq(q), self.Wk(slots), self.Wv(slots))
```

Here's a task that needs attention. Each example is six (key, value)
pairs, like a tiny dictionary: key 7 stores 9, key 8 stores 0, and so
on. Then comes a query key, and the model must answer with the value
stored next to it. In the example, the query is 9, and key 9 stores 8.
Keys and values are just numbers, embedded as in lesson 059, and each
pair's two embeddings are added together into one slot.

Two models, trained the same way for 1,500 steps:

- **No attention:** average the six slots, then predict. 34% accuracy.
  Better than 10% guessing, because it at least knows which values are
  in the dictionary, but it can't tell which one belongs to the query.
- **With attention:** 100%.

```
attention weights for the example above        -> [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
weight on the matching pair, on average        -> 0.995
top weight lands on the matching pair          -> 1.0
```

And we can see *why*. For the example, all the weight is on the sixth
slot, which is exactly the pair with key 9. Across 2,000 test examples,
the matching pair gets 99.5% of the weight on average. Nobody told the
model where to look. Training shaped `Wq` and `Wk` so that a query
scores high against its own key, and `Wv` so that the value comes
through cleanly. The model learned to look things up.

That's what happens inside language models too, on a grand scale:
attention lets a word fetch information from the right earlier word,
like a pronoun finding the name it refers to.

## 6. Masks: what a position is allowed to see

```
causal mask (True = may look)                  -> [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
weights with the causal mask                   -> [[1.0, 0.0, 0.0, 0.0], [0.07, 0.93, 0.0, 0.0], [0.008, 0.003, 0.988, 0.0], [0.002, 0.091, 0.023, 0.884]]
weights with the padding position hidden       -> [[0.944, 0.021, 0.036, 0.0], [0.067, 0.886, 0.047, 0.0], [0.008, 0.003, 0.988, 0.0], [0.017, 0.784, 0.199, 0.0]]
```

Sometimes a position *must not* look at certain others. That's what the
`mask` argument is for: masked scores are set to minus infinity, and
softmax turns those into a weight of exactly zero.

Two masks you'll see everywhere:

- **Causal mask.** A language model learns by predicting the next word.
  If position 2 could see position 3, it would just copy the answer. So
  position `i` may only look at positions up to `i`, and the weights
  form a lower triangle. Note the first row: with only itself to look
  at, the first position puts all its weight there.
- **Padding mask.** To batch sentences of different lengths, short ones
  get padded with filler tokens (like lesson 055's batching). No one
  should attend to filler, so its column is all zeros.

Here queries, keys and values are all the same random vectors (this is
called **self-attention**), so each position mostly matches itself.
In a trained model, `Wq` and `Wk` would make the pattern far more
interesting.

## What you can do now

- Explain in plain words what attention does: each position blends in
  information from the positions that match what it's looking for.
- Name the roles of queries, keys and values, and the three steps:
  score, softmax, average.
- Explain why scores are divided by $\sqrt{d}$.
- Write scaled dot-product attention yourself, and read its shapes.
- Train a model whose attention learns where to look, and inspect the
  weights.
- Use causal and padding masks, and explain what each prevents.

## What to do now

1. Run `lesson.py`. In section 1, change the `* 4` in `blend` to `* 1`
   and to `* 20`, and watch how sharp "bank"'s attention becomes.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   makes the dictionary bigger.
3. Next weekday, lesson 063: the Transformer architecture, where
   attention gets combined with feed-forward layers and positions into
   a working block. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

Attention is a soft lookup: score the keys, softmax the scores, average
the values. Four lines of code, and the basis of every modern language
model.
