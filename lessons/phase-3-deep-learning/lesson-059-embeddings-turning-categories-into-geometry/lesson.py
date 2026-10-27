"""
Lesson 059 - Embeddings: turning categories into geometry

A category is a name, not a number: Monday, Park Gate, "coffee". One-hot
columns keep categories apart but treat every pair as equally different,
and they get enormous when there are thousands of names. An embedding gives
every category a short list of learned numbers instead: a point in space.
Training moves similar categories close together, so the geometry ends up
meaning something.

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files. Every random step has a fixed seed, so your
numbers will match the README. It takes a few seconds.
"""

import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)   # same numbers on every machine

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


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


# ---------------------------------------------------------------------------
# 1. Why one-hot runs out of road
# ---------------------------------------------------------------------------

def section_one_hot():
    heading("1. Why one-hot runs out of road")
    days = F.one_hot(torch.arange(7), num_classes=7).float()
    show("Monday, one-hot", days[0].int().tolist())
    show("distance Mon-Tue, Mon-Sat, Sat-Sun", (r3(torch.dist(days[0], days[1])),
                                                r3(torch.dist(days[0], days[5])),
                                                r3(torch.dist(days[5], days[6]))))
    show("one-hot columns for a 50,000-word vocabulary", 50_000)
    show("embedding columns for the same vocabulary", 64)
    print("  One-hot says every pair of days is exactly as different as every")
    print("  other pair. We know Saturday and Sunday are alike; it can't say so.")


# ---------------------------------------------------------------------------
# 2. nn.Embedding is a lookup table
# ---------------------------------------------------------------------------

def section_lookup():
    heading("2. nn.Embedding is a lookup table")
    torch.manual_seed(59)
    emb = nn.Embedding(num_embeddings=7, embedding_dim=2)
    show("weight shape (categories, dimensions)", tuple(emb.weight.shape))
    idx = torch.tensor([0, 5, 5])
    show("emb([Mon, Sat, Sat])", [[r3(v) for v in row] for row in emb(idx)])
    one_hot = F.one_hot(idx, num_classes=7).float()
    show("one_hot @ weight gives the same rows", torch.allclose(one_hot @ emb.weight, emb(idx)))
    show("weights in Embedding(50_000, 64)", 50_000 * 64)
    print("  Looking up row i is the same as multiplying a one-hot vector by the")
    print("  weight matrix, just without building the huge mostly-zero vector.")


# ---------------------------------------------------------------------------
# 3. Embeddings for IDs: when do people ride?
# ---------------------------------------------------------------------------

def hourly_table():
    """One row per hour of September: its day of week, hour of day and ride count."""
    rides = pd.read_csv(RIDES_CSV, parse_dates=["started_at"])
    start = rides["started_at"].min().floor("D")
    end = rides["started_at"].max().ceil("D")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    counts = rides.set_index("started_at").resample("h").size().reindex(hours, fill_value=0)
    day = torch.tensor(hours.dayofweek.to_numpy())
    hour = torch.tensor(hours.hour.to_numpy())
    y = torch.tensor(counts.to_numpy(), dtype=torch.float32)
    return day, hour, y


class RideRate(nn.Module):
    """A day vector and an hour vector, side by side, into a small network.

    The output is the log of the expected number of rides in that hour.
    """

    def __init__(self, dim=2):
        super().__init__()
        self.day = nn.Embedding(7, dim)
        self.hour = nn.Embedding(24, dim)
        for emb in (self.day, self.hour):
            nn.init.normal_(emb.weight, std=0.1)   # start close together
        self.head = nn.Sequential(nn.Linear(2 * dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, day, hour):
        x = torch.cat([self.day(day), self.hour(hour)], dim=1)
        return self.head(x).squeeze(-1)


def nearest(weight, i, names, k=2):
    """The k names whose vectors sit closest to name i (straight-line distance)."""
    d = torch.cdist(weight[i:i + 1], weight)[0]
    d[i] = math.inf
    return [names[j] for j in d.argsort()[:k].tolist()]


def section_ids():
    heading("3. Embeddings for IDs: when do people ride?")
    day, hour, y = hourly_table()
    tr, te = slice(0, 21 * 24), slice(21 * 24, None)   # three weeks to learn, the rest to test
    show("hours: train / test", (len(y[tr]), len(y[te])))
    by_hour = torch.zeros(24).index_add_(0, hour[tr], y[tr]) / 21
    show("baseline: average for that hour, test MAE", r3((by_hour[hour[te]] - y[te]).abs().mean()))
    torch.manual_seed(59)
    model = RideRate()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.PoissonNLLLoss(log_input=True)   # the usual loss for counts
    for epoch in range(400):
        opt.zero_grad()
        loss_fn(model(day[tr], hour[tr]), y[tr]).backward()
        opt.step()
    with torch.no_grad():
        pred = model(day[te], hour[te]).exp()
    show("embedding model, test MAE", r3((pred - y[te]).abs().mean()))
    W = model.day.weight.detach()
    print("  learned day vectors (2 numbers each):")
    for i, name in enumerate(DAYS):
        show(f"  {name}", [r3(v) for v in W[i]])
    show("closest to Sat", nearest(W, 5, DAYS))
    show("closest to Tue", nearest(W, 1, DAYS))
    hours = [f"{h:02d}h" for h in range(24)]
    H = model.hour.weight.detach()
    for h in (8, 12, 3):
        show(f"closest to {hours[h]}", nearest(H, h, hours, k=3))
    print("  Nobody told the model which days are the weekend, or which hours")
    print("  are rush hours. It put them together because they behave alike.")


# ---------------------------------------------------------------------------
# 4. Embeddings for words
# ---------------------------------------------------------------------------

def make_corpus(n=3000, seed=59):
    """Tiny made-up sentences. Words used in the same places mean similar things."""
    g = torch.Generator().manual_seed(seed)
    groups = {
        "animal": ["cat", "dog", "horse", "goat"],
        "food": ["bread", "rice", "beans", "mango"],
        "place": ["nairobi", "mombasa", "kisumu", "nakuru"],
        "ride": ["bike", "bus", "train", "matatu"],
    }
    templates = [
        ["the", "animal", "eats", "food"],
        ["the", "animal", "sleeps", "in", "place"],
        ["we", "take", "the", "ride", "to", "place"],
        ["we", "buy", "food", "in", "place"],
        ["the", "ride", "leaves", "place", "early"],
    ]
    sentences = []
    for _ in range(n):
        t = templates[torch.randint(len(templates), (1,), generator=g).item()]
        words = []
        for slot in t:
            if slot in groups:
                options = groups[slot]
                slot = options[torch.randint(len(options), (1,), generator=g).item()]
            words.append(slot)
        sentences.append(words)
    return sentences, groups


def skipgram_pairs(sentences, vocab, window=2):
    """(word, neighbour) for every word and each neighbour within the window."""
    pairs = []
    for s in sentences:
        ids = [vocab[w] for w in s]
        for i, w in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((w, ids[j]))
    return torch.tensor(pairs)


class SkipGram(nn.Module):
    """Predict a neighbouring word from a word's embedding."""

    def __init__(self, n_words, dim=8):
        super().__init__()
        self.emb = nn.Embedding(n_words, dim)
        self.out = nn.Linear(dim, n_words)

    def forward(self, w):
        return self.out(self.emb(w))


def most_similar(weight, word, words, k=3):
    v = F.normalize(weight, dim=1)
    sims = v @ v[words.index(word)]
    sims[words.index(word)] = -math.inf
    top = sims.topk(k)
    return [(words[i], r3(s)) for s, i in zip(top.values, top.indices.tolist())]


def section_words():
    heading("4. Embeddings for words")
    sentences, groups = make_corpus()
    words = sorted({w for s in sentences for w in s})
    vocab = {w: i for i, w in enumerate(words)}
    show("sentences, vocabulary size", (len(sentences), len(words)))
    show("two sentences", [" ".join(s) for s in sentences[:2]])
    pairs = skipgram_pairs(sentences, vocab)
    show("(word, neighbour) training pairs", len(pairs))
    torch.manual_seed(59)
    model = SkipGram(len(words))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loader = DataLoader(TensorDataset(pairs[:, 0], pairs[:, 1]), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(59))
    for epoch in range(5):
        for w, ctx in loader:
            opt.zero_grad()
            F.cross_entropy(model(w), ctx).backward()
            opt.step()
    W = model.emb.weight.detach()
    print("  most similar words (cosine similarity):")
    for w in ("cat", "mango", "kisumu", "matatu"):
        show(f"  {w}", most_similar(W, w, words))
    v = F.normalize(W, dim=1)
    same, diff = [], []
    names = list(groups)
    for a in names:
        for b in names:
            for x in groups[a]:
                for z in groups[b]:
                    if x < z:
                        s = (v[vocab[x]] @ v[vocab[z]]).item()
                        (same if a == b else diff).append(s)
    show("average similarity: same group, different", (r3(sum(same) / len(same)),
                                                       r3(sum(diff) / len(diff))))
    print("  The model only ever tried to guess neighbouring words. Words that")
    print("  turn up in the same places ended up pointing the same way.")


# ---------------------------------------------------------------------------
# 5. What an embedding captures, and what it doesn't
# ---------------------------------------------------------------------------

def section_limits():
    heading("5. What an embedding captures, and what it doesn't")
    torch.manual_seed(59)
    emb = nn.Embedding(4, 3)
    unseen = emb.weight[3].detach()
    show("a row that training never touched", [r3(v) for v in unseen])
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([3.0, 0.1])
    show("cosine similarity, straight-line distance", (r3(F.cosine_similarity(a, b, dim=0)),
                                                       r3(torch.dist(a, b))))
    print("  A category the model never saw keeps its random starting vector.")
    print("  And 'close' depends on how you measure: cosine looks at direction,")
    print("  distance at position. Pick the one that matches how it was trained.")


def main():
    print("=" * 66)
    print("  Lesson 059: Embeddings, turning categories into geometry")
    print("=" * 66)
    section_one_hot()
    section_lookup()
    section_ids()
    section_words()
    section_limits()
    print()
    print("An embedding is a learned table of vectors, one per category. Train")
    print("it on a task and similar categories end up close together.")
    print()


if __name__ == "__main__":
    main()
