"""
Lesson 062 - Attention, explained slowly

A word's meaning depends on the words around it. Attention is the mechanism
that lets each position in a sequence look at every other position, decide
which ones matter, and blend in what it finds. It is a few lines of maths:
dot products, a softmax and a weighted average. Today we build it by hand,
check it against PyTorch's version, and train a tiny model that needs it.

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files and needs no data file. Every random step has a
fixed seed, so your numbers will match the README. It takes a few seconds.
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


def rows(t):
    return [[r3(v) for v in row] for row in t]


# ---------------------------------------------------------------------------
# 1. Context changes meaning
# ---------------------------------------------------------------------------

# Hand-made 2-number word vectors: first number "water-ness", second "money-ness".
WORDS = {"river": [1.0, 0.0], "fish": [0.9, 0.1], "bank": [0.5, 0.5],
         "loan": [0.0, 1.0], "cash": [0.1, 0.9], "the": [0.0, 0.0]}


def blend(sentence):
    """Every word looks at every word (itself included) and averages them, weighted."""
    X = torch.tensor([WORDS[w] for w in sentence])
    weights = torch.softmax(X @ X.T * 4, dim=-1)   # x4 just sharpens this toy example
    return weights @ X, weights


def section_context():
    heading("1. Context changes meaning")
    show("'bank' on its own", WORDS["bank"])
    for sentence in (["the", "river", "bank", "fish"], ["the", "bank", "loan", "cash"]):
        out, w = blend(sentence)
        i = sentence.index("bank")
        show(" ".join(sentence) + ": 'bank' becomes", [r3(v) for v in out[i]])
        show("  how much 'bank' looked at each word", dict(zip(sentence, [r3(v) for v in w[i]])))
    print("  Same word, same starting vector, but after looking around it leans")
    print("  towards water in one sentence and money in the other.")


# ---------------------------------------------------------------------------
# 2. Scores, softmax, weighted average
# ---------------------------------------------------------------------------

def section_steps():
    heading("2. Scores, softmax, weighted average")
    query = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    values = torch.tensor([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
    scores = keys @ query
    weights = torch.softmax(scores, dim=0)
    show("scores = keys @ query", [r3(s) for s in scores])
    show("weights = softmax(scores)", [r3(w) for w in weights])
    show("weights add up to", r3(weights.sum()))
    show("output = weights @ values", [r3(v) for v in weights @ values])
    print("  The query asks a question, each key says how well it matches, and")
    print("  the output is the matching values, mixed in proportion.")


# ---------------------------------------------------------------------------
# 3. Why we divide by the square root of d
# ---------------------------------------------------------------------------

def section_scale():
    heading("3. Why we divide by the square root of d")
    torch.manual_seed(62)
    for d in (4, 64, 512):
        q, k = torch.randn(1000, d), torch.randn(1000, 10, d)
        raw = torch.einsum("nd,ntd->nt", q, k)
        top_raw = torch.softmax(raw, dim=-1).max(-1).values.mean()
        top_scaled = torch.softmax(raw / math.sqrt(d), dim=-1).max(-1).values.mean()
        show(f"d={d}: score spread, top weight raw/scaled",
             (r3(raw.std()), r3(top_raw), r3(top_scaled)))
    print("  Dot products of longer vectors spread wider, the softmax piles all")
    print("  the weight on one key, and gradients vanish. Dividing by sqrt(d)")
    print("  keeps the spread near 1 whatever the size.")


# ---------------------------------------------------------------------------
# 4. Scaled dot-product attention in four lines
# ---------------------------------------------------------------------------

def attention(Q, K, V, mask=None):
    """Q: (..., n_queries, d), K: (..., n_keys, d), V: (..., n_keys, d_v)."""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V, weights


def section_function():
    heading("4. Scaled dot-product attention in four lines")
    torch.manual_seed(62)
    Q, K, V = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    out, w = attention(Q, K, V)
    show("Q, K, V shape (batch, positions, d)", tuple(Q.shape))
    show("output shape", tuple(out.shape))
    show("weights shape (batch, queries, keys)", tuple(w.shape))
    show("every row of weights sums to 1", bool(torch.allclose(w.sum(-1), torch.ones(2, 5))))
    show("matches F.scaled_dot_product_attention", bool(torch.allclose(out, F.scaled_dot_product_attention(Q, K, V), atol=1e-6)))


# ---------------------------------------------------------------------------
# 5. Learning what to look at: a lookup task
# ---------------------------------------------------------------------------

N_KEYS, N_VALUES, SLOTS = 12, 10, 6


def make_batch(n, gen):
    """Each example: 6 (key, value) pairs with different keys, and a query key.
    The answer is the value stored next to the query key."""
    keys = torch.stack([torch.randperm(N_KEYS, generator=gen)[:SLOTS] for _ in range(n)])
    values = torch.randint(0, N_VALUES, (n, SLOTS), generator=gen)
    pick = torch.randint(0, SLOTS, (n,), generator=gen)
    query = keys[torch.arange(n), pick]
    answer = values[torch.arange(n), pick]
    return keys, values, query, answer, pick


class Lookup(nn.Module):
    def __init__(self, d=32, use_attention=True):
        super().__init__()
        self.key_emb = nn.Embedding(N_KEYS, d)
        self.value_emb = nn.Embedding(N_VALUES, d)
        self.use_attention = use_attention
        self.Wq, self.Wk, self.Wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out = nn.Linear(2 * d, N_VALUES)

    def forward(self, keys, values, query):
        slots = self.key_emb(keys) + self.value_emb(values)   # (batch, 6, d)
        q = self.key_emb(query).unsqueeze(1)                   # (batch, 1, d)
        if self.use_attention:
            mixed, w = attention(self.Wq(q), self.Wk(slots), self.Wv(slots))
        else:                                                  # no attention: plain average
            mixed, w = slots.mean(1, keepdim=True), None
        return self.out(torch.cat([mixed, q], -1).squeeze(1)), w


def train_lookup(use_attention, steps=1500):
    torch.manual_seed(62)
    gen = torch.Generator().manual_seed(62)
    model = Lookup(use_attention=use_attention)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        keys, values, query, answer, _ = make_batch(128, gen)
        loss = F.cross_entropy(model(keys, values, query)[0], answer)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def section_lookup():
    heading("5. Learning what to look at: a lookup task")
    test = make_batch(2000, torch.Generator().manual_seed(620))
    keys, values, query, answer, pick = test
    show("one example: keys", keys[0].tolist())
    show("             values", values[0].tolist())
    show("             query key, answer", (int(query[0]), int(answer[0])))
    show("guessing a value at random", r3(1 / N_VALUES))
    for use_attention, name in ((False, "average the pairs, no attention"), (True, "learned attention")):
        model = train_lookup(use_attention)
        with torch.no_grad():
            logits, w = model(keys, values, query)
        show(f"{name}: test accuracy", r3((logits.argmax(1) == answer).float().mean()))
    w = w.squeeze(1)
    show("attention weights for the example above", [r3(v) for v in w[0]])
    show("weight on the matching pair, on average", r3(w[torch.arange(len(pick)), pick].mean()))
    show("top weight lands on the matching pair", r3((w.argmax(1) == pick).float().mean()))
    print("  Nobody told the model where to look. The query and key projections")
    print("  learned to make the matching key score highest, so attention fetches")
    print("  that pair's value. The average can't single anything out.")


# ---------------------------------------------------------------------------
# 6. Masks: what a position is allowed to see
# ---------------------------------------------------------------------------

def section_masks():
    heading("6. Masks: what a position is allowed to see")
    torch.manual_seed(62)
    X = torch.randn(1, 4, 8)
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    _, w = attention(X, X, X, mask=causal)
    show("causal mask (True = may look)", causal.int().tolist())
    show("weights with the causal mask", rows(w[0]))
    pad = torch.tensor([True, True, True, False])          # the last position is padding
    _, w = attention(X, X, X, mask=pad.expand(4, 4))
    show("weights with the padding position hidden", rows(w[0]))
    print("  A language model predicting the next word must not peek ahead, so")
    print("  position i only sees positions up to i. Padding added to make a")
    print("  batch rectangular should never be looked at by anyone.")


def main():
    print("=" * 66)
    print("  Lesson 062: Attention, explained slowly")
    print("=" * 66)
    section_context()
    section_steps()
    section_scale()
    section_function()
    section_lookup()
    section_masks()
    print()
    print("Attention: score every key against the query, softmax the scores,")
    print("and average the values with those weights.")
    print()


if __name__ == "__main__":
    main()
