"""
Lesson 055 - Datasets, DataLoaders and batching

Lesson 054's schedule section trained on 16 random points per step, picked
by hand with torch.randint. That's how real training works, but doing it by
hand gets messy fast. torch has two small tools for it:

    Dataset     - "how many examples, and give me number i"
    DataLoader  - "hand them to me in shuffled batches, epoch after epoch"

Today we build both, see why the shuffle matters more than it looks, and
split data cleanly so the test set stays honest.

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files. Every random step has a fixed seed, so your
numbers will match the README. It takes about ten seconds.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

RIDE_FEATURES = ["duration_min", "distance_km", "weekend", "hour", "temp_c", "rain_mm"]


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


def make_ring(n=400, seed=50):
    """Lesson 050's ring: label 1 inside a circle of radius 1, 3% noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.6, 1.6, size=(n, 2))
    y = (np.hypot(X[:, 0], X[:, 1]) < 1.0).astype(float)
    flip = rng.random(n) < 0.03
    y[flip] = 1 - y[flip]
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


class RingNet(nn.Module):
    """Lesson 052's network: inputs -> hidden layer (tanh) -> one score."""

    def __init__(self, n_inputs=2, n_hidden=8):
        super().__init__()
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x):
        return self.out(torch.tanh(self.hidden(x))).squeeze(1)


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.sigmoid(model(X))


def accuracy(p, y):
    return r3(((p >= 0.5).float() == y).float().mean())


def auc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    pos, neg = p[y == 1], p[y == 0]
    return r3((pos[:, None] > neg[None, :]).mean()
              + 0.5 * (pos[:, None] == neg[None, :]).mean())


def train_epochs(model, loader, optimiser, epochs):
    """The standard loop: for each epoch, for each batch, one step."""
    loss_fn = nn.BCEWithLogitsLoss()
    steps = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimiser.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimiser.step()
            steps += 1
    return steps


# ---------------------------------------------------------------------------
# 1. Why batches at all
# ---------------------------------------------------------------------------

def section_why_batches():
    heading("1. Why batches at all")
    X, y = make_ring()
    torch.manual_seed(54)
    model = RingNet()
    loss_fn = nn.BCEWithLogitsLoss()

    def grad_of(idx):
        model.zero_grad()
        loss_fn(model(X[idx]), y[idx]).backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()]).clone()

    full = grad_of(torch.arange(len(y)))
    show("weights in RingNet", full.numel())
    show("examples", len(y))
    picker = torch.Generator().manual_seed(55)
    show("batch size: average error vs full gradient", "")
    for b in (1, 16, 64, 256):
        errs = []
        for _ in range(200):
            idx = torch.randperm(len(y), generator=picker)[:b]
            errs.append((grad_of(idx) - full).norm().item())
        show(f"  batch of {b}", r3(np.mean(errs)))
    show("full gradient's own size", r3(full.norm()))
    print("  A batch gives a noisy estimate of the full gradient. Bigger batches")
    print("  are less noisy, but each step costs more. 16 to 256 is the usual range.")


# ---------------------------------------------------------------------------
# 2. A Dataset is two methods
# ---------------------------------------------------------------------------

class RidesDataset(Dataset):
    """One ride per item: six features in, 'is it a casual rider?' out."""

    def __init__(self, csv_path):
        rides = pd.read_csv(csv_path, parse_dates=["started_at"])
        rides["weekend"] = (rides["started_at"].dt.dayofweek >= 5).astype(int)
        rides["hour"] = rides["started_at"].dt.hour
        self.X = torch.tensor(rides[RIDE_FEATURES].to_numpy(dtype=float), dtype=torch.float32)
        self.y = torch.tensor((rides["rider_type"] == "casual").to_numpy(dtype=float),
                              dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def section_dataset():
    heading("2. A Dataset is two methods")
    ds = RidesDataset(RIDES_CSV)
    x0, y0 = ds[0]
    show("len(ds)", len(ds))
    show("ds[0] features", [round(v, 2) for v in x0.tolist()])
    show("ds[0] label", r3(y0))
    show("ds[-1] label", r3(ds[-1][1]))
    ring = TensorDataset(*make_ring())
    show("TensorDataset(ring): len, ring[0]", (len(ring), [r3(v) for v in ring[0][0]], r3(ring[0][1])))
    print("  __len__ says how many; __getitem__ hands back example i. That's the")
    print("  whole contract, whether the data is a CSV, a folder of images or a database.")


# ---------------------------------------------------------------------------
# 3. DataLoader: batches, epochs and the last batch
# ---------------------------------------------------------------------------

def section_loader():
    heading("3. DataLoader: batches, epochs and the last batch")
    ds = RidesDataset(RIDES_CSV)
    loader = DataLoader(ds, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(55))
    xb, yb = next(iter(loader))
    show("one batch: features shape", tuple(xb.shape))
    show("one batch: labels shape", tuple(yb.shape))
    show("batches per epoch (len(loader))", len(loader))
    sizes = [len(b[1]) for b in loader]
    show("first and last batch sizes", (sizes[0], sizes[-1]))
    show("examples seen in one epoch", sum(sizes))
    dropped = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    show("with drop_last=True: batches", len(dropped))
    print("  1176 = 36 x 32 + 24, so the last batch has 24 rides. drop_last=True")
    print("  throws it away, which matters when tiny batches would upset training.")


# ---------------------------------------------------------------------------
# 4. Shuffle, or learn the order
# ---------------------------------------------------------------------------

def section_shuffle():
    heading("4. Shuffle, or learn the order")
    X, y = make_ring()
    X_new, y_new = make_ring(seed=51)
    order = torch.argsort(y, stable=True)          # all the 0s, then all the 1s
    sorted_ds = TensorDataset(X[order], y[order])
    peek = DataLoader(sorted_ds, batch_size=50, shuffle=False)
    show("share of 1s per batch, unshuffled", [r3(yb.mean()) for _, yb in peek])
    for shuffle in (False, True):
        torch.manual_seed(54)
        model = RingNet()
        loader = DataLoader(sorted_ds, batch_size=16, shuffle=shuffle,
                            generator=torch.Generator().manual_seed(55))
        train_epochs(model, loader, torch.optim.Adam(model.parameters(), lr=0.01), 60)
        p = predict_proba(model, X_new)
        with torch.no_grad():
            full_loss = nn.BCEWithLogitsLoss()(model(X), y)
        show(f"shuffle={shuffle}, 60 epochs: training loss", r3(full_loss))
        show("  accuracy on new points", accuracy(p, y_new))
        show("  share predicted as 1", r3((p >= 0.5).float().mean()))
    show("true share of 1s on new points", r3(y_new.mean()))
    print("  Unshuffled, every epoch is 'all 0s' then 'all 1s'. The model gets dragged")
    print("  one way then the other and ends each epoch leaning towards 1. Shuffle it.")


# ---------------------------------------------------------------------------
# 5. Splitting cleanly
# ---------------------------------------------------------------------------

def section_split():
    heading("5. Splitting cleanly")
    ds = RidesDataset(RIDES_CSV)
    train_ds, val_ds, test_ds = random_split(ds, [0.7, 0.15, 0.15],
                                             generator=torch.Generator().manual_seed(55))
    show("train / val / test sizes", (len(train_ds), len(val_ds), len(test_ds)))
    again = random_split(ds, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(55))
    show("same seed, same split", train_ds.indices[:5] == again[0].indices[:5])
    show("overlap between train and test", len(set(train_ds.indices) & set(test_ds.indices)))
    tr = torch.tensor(train_ds.indices)
    mean, std = ds.X[tr].mean(dim=0), ds.X[tr].std(dim=0)
    show("scaling stats from train only: mean", [round(v, 2) for v in mean.tolist()])
    show("casual share: train / val / test",
         tuple(r3(ds.y[torch.tensor(s.indices)].mean()) for s in (train_ds, val_ds, test_ds)))
    print("  Split once, with a seeded generator, before anything learns from the")
    print("  data. Scaling numbers come from the training part only (lesson 036).")


# ---------------------------------------------------------------------------
# 6. The full loop on the rides, and batch size
# ---------------------------------------------------------------------------

class Standardised(Dataset):
    """Wrap a split and scale its features with numbers from the training split."""

    def __init__(self, subset, mean, std):
        self.subset, self.mean, self.std = subset, mean, std

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y = self.subset[i]
        return (x - self.mean) / self.std, y


def section_rides():
    heading("6. The full loop on the rides, and batch size")
    ds = RidesDataset(RIDES_CSV)
    train_ds, val_ds, test_ds = random_split(ds, [0.7, 0.15, 0.15],
                                             generator=torch.Generator().manual_seed(55))
    tr = torch.tensor(train_ds.indices)
    mean, std = ds.X[tr].mean(dim=0), ds.X[tr].std(dim=0)
    train_s = Standardised(train_ds, mean, std)
    test_s = Standardised(test_ds, mean, std)
    X_te, y_te = next(iter(DataLoader(test_s, batch_size=len(test_s))))
    show("epochs for every run", 15)
    for batch in (8, 32, 128, len(train_s)):
        torch.manual_seed(52)
        model = RingNet(n_inputs=6, n_hidden=8)
        loader = DataLoader(train_s, batch_size=batch, shuffle=True,
                            generator=torch.Generator().manual_seed(55))
        steps = train_epochs(model, loader, torch.optim.Adam(model.parameters(), lr=0.01), 15)
        label = "full batch" if batch == len(train_s) else f"batch {batch}"
        show(f"{label}: steps, test AUC", (steps, auc(predict_proba(model, X_te), y_te)))
    print("  Same 15 passes over the data. Small batches take many cheap, noisy")
    print("  steps; one giant batch takes 15 careful ones and barely moves.")


def main():
    print("=" * 66)
    print("  Lesson 055: Datasets, DataLoaders and batching")
    print("=" * 66)
    section_why_batches()
    section_dataset()
    section_loader()
    section_shuffle()
    section_split()
    section_rides()
    print()
    print("Dataset says what one example is. DataLoader shuffles and batches.")
    print("Split once, with a seed, before anything learns.")
    print()


if __name__ == "__main__":
    main()
