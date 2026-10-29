# Lesson 063 - The Transformer architecture

**Phase 4 - Modern AI: NLP, vision, generative and LLM basics** | Week 8, Day 4 | Thursday 2026-10-29

> **Goal:** Put attention, feed-forward layers and positional encodings
> together into a working block.

Time: about 45 minutes. Same venv as lesson 051, with `torch` installed.

---

Lesson 062 gave us attention: every position scores every other
position, softmaxes the scores and blends in what it finds. It's
powerful, but on its own it's not a model. It can't tell what order
things came in, it only ever averages, and one round of it is shallow.

The **Transformer** fixes all three with a handful of parts you already
know: an embedding (lesson 059), linear layers (lesson 052), a little
normalisation and one very simple trick called a residual connection.
Put together, they make a **block**. Stack blocks and you have the
architecture behind BERT, GPT and every large language model since.

Today we build that block from scratch, check it against PyTorch's own,
train a tiny Transformer, and finish by counting the parameters of
GPT-2.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-4-modern-ai/lesson-063-the-transformer-architecture/lesson.py
```

It needs no data file and writes no files. Every random step has a
fixed seed, so your numbers will match. It takes about ten seconds.

## 1. Attention can't tell order

```
shuffle in, shuffled out (same numbers)        -> True
average over positions: identical?             -> True
```

Exercise 2 of lesson 062 hinted at this. Feed a block five token
vectors, then the same five shuffled. Every token gets *exactly* the
same output as before, just in its new place. Average over the
positions and the two results are identical.

That's because attention scores depend on what the vectors are, never
on where they sit. So, as far as attention is concerned, "dog bites
man" and "man bites dog" are the same sentence. For language that's
fatal. We need to tell the model where each token is.

## 2. Positional encodings

```
shape (positions, d)                           -> (50, 16)
position 0, first 4 numbers                    -> [0.0, 1.0, 0.0, 1.0]
position 1, first 4 numbers                    -> [0.841, 0.54, 0.311, 0.95]
similarity of position 10 to 11, 13, 20, 40    -> [0.936, 0.693, 0.456, 0.337]
similarity of 30 to 31 (same gap as 10 to 11)  -> 0.936
```

The fix is almost cheeky: make a vector for each *position* and **add**
it to the token's vector. Now "bike at position 2" and "bike at
position 7" are different vectors, and attention can use the difference.

The original Transformer paper built these vectors out of sine and
cosine waves. Each pair of numbers is one wave: the first pair wiggles
fast (it changes a lot from one position to the next), later pairs
wiggle more and more slowly. Read across all of them and you get a
unique fingerprint per position, a bit like reading a clock by its
second, minute and hour hands together.

Two properties make it useful:

- **Nearby positions look alike.** Position 10 is 0.94 similar to 11,
  0.69 to 13, and only 0.34 to 40.
- **Only the gap matters.** 30 vs 31 scores exactly the same as 10 vs
  11. So the model can learn "look one word back" once, and use it
  everywhere.

Many models skip the waves and simply *learn* a position embedding, an
`nn.Embedding` with one row per position, exactly like lesson 059's
category embeddings. Both work; exercise 4 tries it.

## 3. Multi-head attention

```
input shape (batch, positions, d)              -> (2, 5, 16)
weights shape (batch, heads, n, n)             -> (2, 4, 5, 5)
output shape                                   -> (2, 5, 16)
numbers each head works with                   -> 4
matches nn.MultiheadAttention                  -> True
```

One attention gives one set of weights per position, so it can follow
one idea of "relevant" at a time. Real models run several **heads**
side by side (exercise 5 of lesson 062 asked why).

The trick is not to make the model bigger, but to slice it. With 16
numbers per token and 4 heads, each head gets its own 4 numbers:

1. One linear layer makes queries, keys and values for every token.
2. Reshape so each head has its own slice: `(batch, heads, n, d/heads)`.
3. Run lesson 062's `attention` on all heads at once. The weights now
   have an extra `heads` dimension: four separate 5 by 5 grids.
4. Glue the heads back together and mix them with one more linear
   layer.

That's `MultiHeadAttention` in `lesson.py`, about a dozen lines. To
check it, the script copies our weights into PyTorch's
`nn.MultiheadAttention` and gets the same numbers.

## 4. The block: attention, feed-forward, residuals, layer norm

```
input shape, output shape                      -> ((2, 6, 32), (2, 6, 32))
one position before layer norm: mean, std      -> (2.014, 2.394)
after layer norm: mean, std                    -> (0.0, 1.0)
parameters: attention, feed-forward, norms     -> (4224, 8352, 128)
our block, PyTorch's block: parameters         -> (12704, 12704)
```

Here's the whole block:

```python
def forward(self, x, mask=None):
    a, _ = self.attn(self.norm1(x), mask)
    x = x + a                          # residual around attention
    x = x + self.ffn(self.norm2(x))    # residual around the feed-forward
    return x
```

Four ideas, each small:

- **Attention** lets the tokens talk to each other. It's the only place
  in the block where information moves between positions.
- **Feed-forward network** (`ffn`): two linear layers with a GELU in
  between, applied to each position on its own. Attention gathers; this
  part thinks about what was gathered. It's 4 times wider in the middle,
  and it holds two thirds of the block's parameters.
- **Residual connections** (the `x = x + ...`): each part *adds* to the
  token's vector instead of replacing it. Nothing is lost by default,
  and gradients get a clean path straight back through every block.
  It's what lets these models stack dozens of blocks deep without the
  training failures from lesson 053.
- **Layer norm** rescales each token's vector to mean 0 and spread 1
  (with two learned numbers per feature to adjust that) before each
  part. Like lesson 053's careful initialisation, it keeps the numbers
  in a range where training behaves.

The most useful property is the first line of output: **shape in equals
shape out**. A block takes `(batch, positions, d)` and gives back the
same shape, so you can stack as many as you like. PyTorch's
`nn.TransformerEncoderLayer` with the same settings has exactly the
same number of parameters as ours, because it's the same design.

## 5. A tiny Transformer that needs to know the order

```
one example (10 = A, 11 = B)                   -> [8, 0, 2, 10, 4, 5, 11, 9]
A comes before B?                              -> True
share of test examples where A is first        -> 0.495
no positional encoding: test accuracy          -> 0.504
with positional encoding: test accuracy        -> 1.0
parameters in the tiny Transformer             -> 25922
```

Time to train one. Each example is 8 tokens: random filler digits plus
one `A` (token 10) and one `B` (token 11) somewhere. The question: does
`A` come before `B`? It's a pure order question; the *set* of tokens
says nothing about the answer.

The model is everything from today in a row: embed the tokens (lesson
059), add positional encodings, two blocks, a final layer norm, average
the positions, and a linear layer to two classes. About 26,000
parameters. Two versions, trained identically for 600 steps:

- **Without positional encodings:** 50%, a coin flip. Section 1 already
  told us why: shuffling the tokens can't change its answer, so it has
  no way to know.
- **With positional encodings:** 100% on 2,000 test examples.

One line of code, `x = x + self.pe`, is the whole difference.

## 6. Stack it up: from toy to GPT-2

```
parameters in one GPT-2-sized block            -> 7087872
token and position embeddings                  -> 39383808
12 blocks + embeddings + final norm            -> 124439808
in millions                                    -> 124.4
```

Now scale our `Block` up to GPT-2 small's settings: 768 numbers per
token, 12 heads, 12 blocks, a vocabulary of 50,257 tokens and room for
1,024 positions (learned, not waves). Count the parameters and you get
124,439,808, the exact published size of GPT-2 small. Same block you
just wrote, only wider and stacked twelve deep.

Transformers come in three families, all built from this block:

- **Encoders** (like BERT): every token sees every other token. Good for
  understanding text: classifying, tagging, searching. Lesson 064 uses
  one.
- **Decoders** (like GPT): the same block with lesson 062's **causal
  mask**, so each token sees only the ones before it. Trained to
  predict the next token, which makes them good at writing. Lesson 065
  covers how.
- **Encoder-decoders** (like the original paper's translator): an
  encoder reads the input, and a decoder writes the output while also
  attending to the encoder's result.

## What you can do now

- Explain why attention alone ignores order, and how positional
  encodings fix it.
- Build multi-head attention by slicing the numbers across heads, and
  read its shapes.
- Write a Transformer block: layer norm, attention, residual, layer
  norm, feed-forward, residual.
- Say what each part is for, and why blocks can be stacked.
- Train a small Transformer and show that positions matter.
- Count the parameters of a real model from its settings.

## What to do now

1. Run `lesson.py`. In section 5, change `depth=2` to `depth=1` and to
   `depth=4`, and watch the parameter count and the accuracy.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   swaps the sine waves for learned positions.
3. Next weekday, lesson 064: pretrained models and transfer learning,
   where we download a Transformer someone else already trained, and
   teach it our task with only 20 examples. It needs one new install;
   see [PROGRESS.md](../../../curriculum/PROGRESS.md).

A Transformer block is attention plus a small feed-forward network,
each wrapped in a layer norm and a residual, with positions added at
the start. Everything bigger is the same block, stacked.
