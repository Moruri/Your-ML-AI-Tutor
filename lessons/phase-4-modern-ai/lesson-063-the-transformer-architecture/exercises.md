# Lesson 063 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-4-modern-ai/lesson-063-the-transformer-architecture/lesson.py my_lesson_063.py
python my_lesson_063.py
```

This lesson needs no data file, so the copy runs as it is.
`Block`, `MultiHeadAttention`, `positional_encoding`, `TinyTransformer`,
`make_batch`, `train_order`, `r3` and `show` are there to reuse.

---

## 1. Splitting the heads

A model has 512 numbers per token and 8 heads. How many numbers does
each head work with? What happens if you ask for 7 heads?

<details>
<summary>Check yourself</summary>

512 / 8 = 64 per head. With 7 heads, 512 doesn't split evenly, so
`MultiHeadAttention` stops at its `assert` (PyTorch's version raises an
error too). Heads are slices of the same numbers, so the slices have to
be equal sizes. That's why model sizes are always round multiples of
the head count.

</details>

## 2. Take away the residuals

Change the block to `x = a` and `x = self.ffn(self.norm2(x))`, with no
`x +`. What do you lose, especially in a deep stack?

<details>
<summary>Check yourself</summary>

Two things. First, each part now *replaces* the token's vector instead
of adding to it, so anything a block doesn't actively pass on is gone,
including the token's own identity and its position. Second, gradients
have to squeeze back through every attention and feed-forward layer in
turn, which brings back lesson 053's vanishing gradients. With two
blocks you might get away with it; with twenty-four, training would
struggle badly. Residuals are what make deep Transformers trainable.

</details>

## 3. Where the parameters live

Section 4 printed 4,224 parameters for attention and 8,352 for the
feed-forward network, with `d = 32`. Explain both numbers.

<details>
<summary>Check yourself</summary>

Attention: the `qkv` layer is 32 x 96 weights plus 96 biases = 3,168,
and the `out` layer is 32 x 32 + 32 = 1,056. Total 4,224, or about
`4 x d x d`.

Feed-forward: 32 x 128 + 128 = 4,224 into the wide middle, then
128 x 32 + 32 = 4,128 back out. Total 8,352, about `8 x d x d`.

So the feed-forward layers hold roughly twice the parameters of
attention, in every block of every size. The number of heads doesn't
change either count: heads slice the numbers, they don't add any.

</details>

## 4. Hands-on: learned positions

Replace the sine and cosine waves with a learned position embedding:
one trainable row per position, like lesson 059. Train it on the same
task with the same seeds, and test it on the same 2,000 examples.
Predict first: will it be as good, given it has to learn positions
from scratch?

<details>
<summary>Expected results</summary>

```
learned positions: test accuracy -> 1.0
```

Just as good on this task. Eight positions is easy to learn. The waves
still have a quiet advantage: they exist for any position, even ones
never seen in training, while a learned table has exactly as many rows
as you gave it. That's why GPT-2 can't read past 1,024 tokens: its
table stops there.

</details>

<details>
<summary>One way to write it</summary>

```python
class Learned(TinyTransformer):
    def __init__(self, d=32):
        super().__init__(True, d)
        self.pos = nn.Embedding(LENGTH, d)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos.weight   # one learned row per position
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x).mean(dim=1))


x, y = make_batch(2000, torch.Generator().manual_seed(632))
torch.manual_seed(630)
gen = torch.Generator().manual_seed(631)
model = Learned()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for _ in range(600):
    xb, yb = make_batch(64, gen)
    loss = F.cross_entropy(model(xb), yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
model.eval()
with torch.no_grad():
    show("learned positions: test accuracy", r3((model(x).argmax(1) == y).float().mean()))
```

</details>

## 5. (Optional) Turn it into a decoder

What would you change in `TinyTransformer` to make it a GPT-style
decoder that predicts the next token at every position?

<details>
<summary>Check yourself</summary>

Three changes. Pass a causal mask to every block,
`torch.tril(torch.ones(n, n, dtype=torch.bool))`, so no position can
see the future. Drop the `.mean(dim=1)`, and keep one output vector per
position. And make the head predict a whole vocabulary,
`nn.Linear(d, VOCAB)`, trained with cross-entropy against the input
shifted by one (position `i` predicts token `i + 1`). That's the whole
recipe for a language model's architecture. Lesson 065 is about what
happens when you train one on a lot of text.

</details>

---

That's the Transformer. If one thing sticks: *add positions to the
tokens, then repeat a block of attention and a feed-forward network,
each wrapped in a layer norm and a residual.*
