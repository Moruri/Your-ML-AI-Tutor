"""
Lesson 063 - The Transformer architecture

Lesson 062 built attention: every position scores every other position,
softmaxes the scores and blends in what it finds. Today we put attention
inside the block that runs every modern language model: positional
encodings so order matters, several attention heads side by side, a small
feed-forward network, residual connections and layer norm. Then we train a
tiny Transformer on a task it can only solve if it knows the order.

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files and needs no data file. Every random step has a
fixed seed, so your numbers will match the README. It takes about ten
seconds.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

torch.set_num_threads(1)   # same numbers on every machine


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def r3(x):
    if torch.is_tensor(x):
        x = x.detach()
    return round(float(x), 3)


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention, exactly as in lesson 062."""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V, weights


# ---------------------------------------------------------------------------
# The pieces of a Transformer block
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Several attention heads side by side, each on its own slice of the numbers."""

    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0, "d must split evenly across the heads"
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)      # makes queries, keys and values in one go
        self.out = nn.Linear(d, d)          # mixes the heads back together

    def forward(self, x, mask=None):
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (b, n, d) -> (b, heads, n, d / heads): each head gets its own slice
        split = lambda t: t.view(b, n, self.heads, d // self.heads).transpose(1, 2)
        mixed, weights = attention(split(q), split(k), split(v), mask)
        mixed = mixed.transpose(1, 2).reshape(b, n, d)   # glue the heads back together
        return self.out(mixed), weights


class Block(nn.Module):
    """One Transformer block (the 'pre-norm' layout used by GPT-style models)."""

    def __init__(self, d, heads, ff=None):
        super().__init__()
        ff = ff or 4 * d
        self.norm1 = nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, heads)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, mask=None):
        a, _ = self.attn(self.norm1(x), mask)
        x = x + a                          # residual: keep what you had, add what attention found
        x = x + self.ffn(self.norm2(x))    # then let every position think on its own
        return x


def positional_encoding(n, d):
    """The sine/cosine position signal from the original Transformer paper."""
    pos = torch.arange(n).float()[:, None]
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2).float() / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


def n_params(module):
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# 1. Attention can't tell order
# ---------------------------------------------------------------------------

def section_order():
    heading("1. Attention can't tell order")
    torch.manual_seed(63)
    block = Block(d=16, heads=2).eval()
    x = torch.randn(1, 5, 16)                 # five token vectors
    perm = torch.tensor([3, 0, 4, 1, 2])      # the same tokens, shuffled
    with torch.no_grad():
        out, out_shuffled = block(x), block(x[:, perm])
    show("shuffle in, shuffled out (same numbers)", torch.allclose(out[:, perm], out_shuffled, atol=1e-6))
    show("average over positions: identical?", torch.allclose(out.mean(1), out_shuffled.mean(1), atol=1e-6))
    print("  Shuffle the tokens and each one gets exactly the same output, just")
    print("  in a new place. 'dog bites man' and 'man bites dog' look identical.")


# ---------------------------------------------------------------------------
# 2. Positional encodings
# ---------------------------------------------------------------------------

def section_positions():
    heading("2. Positional encodings")
    pe = positional_encoding(50, 16)
    show("shape (positions, d)", tuple(pe.shape))
    show("position 0, first 4 numbers", [r3(v) for v in pe[0, :4]])
    show("position 1, first 4 numbers", [r3(v) for v in pe[1, :4]])
    cos = lambda i, j: r3(F.cosine_similarity(pe[i], pe[j], dim=0))
    show("similarity of position 10 to 11, 13, 20, 40", [cos(10, 11), cos(10, 13), cos(10, 20), cos(10, 40)])
    show("similarity of 30 to 31 (same gap as 10 to 11)", cos(30, 31))
    print("  Each position gets its own fingerprint of waves, fast and slow.")
    print("  Neighbours look alike, far-apart positions don't, and the pattern")
    print("  depends on the gap, not where you are. Add it to each token vector.")


# ---------------------------------------------------------------------------
# 3. Multi-head attention
# ---------------------------------------------------------------------------

def section_heads():
    heading("3. Multi-head attention")
    torch.manual_seed(63)
    d, heads = 16, 4
    mha = MultiHeadAttention(d, heads)
    x = torch.randn(2, 5, d)
    out, w = mha(x)
    show("input shape (batch, positions, d)", tuple(x.shape))
    show("weights shape (batch, heads, n, n)", tuple(w.shape))
    show("output shape", tuple(out.shape))
    show("numbers each head works with", d // heads)

    ref = nn.MultiheadAttention(d, heads, batch_first=True)
    with torch.no_grad():                     # give PyTorch's version our exact weights
        ref.in_proj_weight.copy_(mha.qkv.weight)
        ref.in_proj_bias.copy_(mha.qkv.bias)
        ref.out_proj.weight.copy_(mha.out.weight)
        ref.out_proj.bias.copy_(mha.out.bias)
        ref_out, _ = ref(x, x, x)
    show("matches nn.MultiheadAttention", torch.allclose(out, ref_out, atol=1e-6))
    print("  Four heads, each attending with its own 4 numbers, so each can")
    print("  follow a different kind of relationship. Same cost as one wide head.")


# ---------------------------------------------------------------------------
# 4. The block: attention, feed-forward, residuals, layer norm
# ---------------------------------------------------------------------------

def section_block():
    heading("4. The block: attention, feed-forward, residuals, layer norm")
    torch.manual_seed(63)
    d = 32
    block = Block(d, heads=4)
    x = torch.randn(2, 6, d) * 3 + 1
    y = block(x)
    show("input shape, output shape", (tuple(x.shape), tuple(y.shape)))
    normed = block.norm1(x)
    show("one position before layer norm: mean, std", (r3(x[0, 0].mean()), r3(x[0, 0].std(unbiased=False))))
    show("after layer norm: mean, std", (r3(normed[0, 0].mean()), r3(normed[0, 0].std(unbiased=False))))
    show("parameters: attention, feed-forward, norms",
         (n_params(block.attn), n_params(block.ffn), n_params(block.norm1) + n_params(block.norm2)))
    ref = nn.TransformerEncoderLayer(d, 4, dim_feedforward=4 * d, norm_first=True, batch_first=True)
    show("our block, PyTorch's block: parameters", (n_params(block), n_params(ref)))
    print("  Shape in equals shape out, so blocks stack like bricks. Two thirds")
    print("  of the parameters sit in the plain feed-forward layers.")


# ---------------------------------------------------------------------------
# 5. A tiny Transformer that needs to know the order
# ---------------------------------------------------------------------------

LENGTH, A, B, VOCAB = 8, 10, 11, 12    # tokens 0-9 are filler digits, 10 is 'A', 11 is 'B'


def make_batch(size, gen):
    """Sequences of 8 tokens with one A and one B somewhere. Label: is A before B?"""
    x = torch.randint(0, 10, (size, LENGTH), generator=gen)
    places = torch.rand(size, LENGTH, generator=gen).argsort(dim=1)[:, :2]
    rows = torch.arange(size)
    x[rows, places[:, 0]] = A
    x[rows, places[:, 1]] = B
    y = (places[:, 0] < places[:, 1]).long()
    return x, y


class TinyTransformer(nn.Module):
    def __init__(self, use_positions, d=32, heads=4, depth=2):
        super().__init__()
        self.use_positions = use_positions
        self.embed = nn.Embedding(VOCAB, d)
        self.register_buffer("pe", positional_encoding(LENGTH, d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, tokens):
        x = self.embed(tokens)
        if self.use_positions:
            x = x + self.pe
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x).mean(dim=1))   # average the positions, then classify


def train_order(use_positions, steps=600):
    torch.manual_seed(630)
    gen = torch.Generator().manual_seed(631)
    model = TinyTransformer(use_positions)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x, y = make_batch(64, gen)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def section_train():
    heading("5. A tiny Transformer that needs to know the order")
    x, y = make_batch(2000, torch.Generator().manual_seed(632))
    show("one example (10 = A, 11 = B)", x[0].tolist())
    show("A comes before B?", bool(y[0]))
    show("share of test examples where A is first", r3(y.float().mean()))
    for use_positions, label in ((False, "no positional encoding"), (True, "with positional encoding")):
        model = train_order(use_positions)
        with torch.no_grad():
            acc = (model(x).argmax(1) == y).float().mean()
        show(f"{label}: test accuracy", r3(acc))
    show("parameters in the tiny Transformer", n_params(model))
    print("  Same architecture, same training. Without positions it's a coin")
    print("  flip, because it literally can't see order. Add positions and it")
    print("  gets every test example right.")


# ---------------------------------------------------------------------------
# 6. Stack it up: from toy to GPT-2
# ---------------------------------------------------------------------------

def section_scale():
    heading("6. Stack it up: from toy to GPT-2")
    vocab, context, d, heads, depth = 50257, 1024, 768, 12, 12   # GPT-2 small's settings
    per_block = n_params(Block(d, heads))
    embeddings = vocab * d + context * d
    total = embeddings + depth * per_block + 2 * d                # + the final layer norm
    show("parameters in one GPT-2-sized block", per_block)
    show("token and position embeddings", embeddings)
    show("12 blocks + embeddings + final norm", total)
    show("in millions", round(total / 1e6, 1))
    print("  The exact count for GPT-2 small. Same block as ours, just wider,")
    print("  stacked twelve deep, with a causal mask so it can't peek ahead.")


def main():
    print("=" * 66)
    print("  Lesson 063: The Transformer architecture")
    print("=" * 66)
    section_order()
    section_positions()
    section_heads()
    section_block()
    section_train()
    section_scale()
    print()
    print("A Transformer block: add positions, let attention mix the tokens,")
    print("let a feed-forward layer think about each one, keep residuals and")
    print("norms around both. Stack and repeat.")
    print()


if __name__ == "__main__":
    main()
