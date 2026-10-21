"""
Lesson 052 - Building networks with nn.Module

Lesson 051 kept the weights in a plain dict and updated them by hand
inside torch.no_grad(). That works, and it's exactly what PyTorch does
under the hood, but nobody writes real code that way. Real PyTorch code
has three pieces:

    a model      - a class built from nn.Module, holding layers like nn.Linear
    a loss       - from torch.nn, e.g. nn.BCEWithLogitsLoss
    an optimiser - from torch.optim, which does the "w -= lr * w.grad" for you

Today we rebuild the ring network those three ways, write the five-line
training loop you'll see in every PyTorch project, save and reload the
trained model, then point the same pattern at the bike-share rides.

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes one file, output/ring_net.pt, next to this file. Every
random step has a fixed seed, so your numbers will match the README. It
takes a few seconds.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"
OUTPUT = HERE / "output"

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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# The model, written the way real PyTorch code writes it
# ---------------------------------------------------------------------------

class RingNet(nn.Module):
    """Inputs -> hidden layer (tanh) -> one output score."""

    def __init__(self, n_inputs=2, n_hidden=8):
        super().__init__()                       # always call this first
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        return self.out(h).squeeze(1)            # a raw score (a "logit"), not a probability


def train(model, X, y, lr, steps, report=()):
    """The five-line loop you'll see in every PyTorch project."""
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        optimiser.zero_grad()                    # 1. clear last step's gradients
        logits = model(X)                        # 2. forward pass
        loss = loss_fn(logits, y)                # 3. how wrong are we?
        loss.backward()                          # 4. autograd fills every .grad
        optimiser.step()                         # 5. nudge every weight
        if step in report:
            show(f"loss at step {step}", r3(loss))
    return loss


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.sigmoid(model(X))


def accuracy(p, y):
    return r3(((p >= 0.5).float() == y).float().mean())


# ---------------------------------------------------------------------------
# 1. A layer is an object: nn.Linear
# ---------------------------------------------------------------------------

def section_linear():
    heading("1. A layer is an object: nn.Linear")
    torch.manual_seed(52)
    layer = nn.Linear(2, 8)
    show("layer", str(layer))
    show("layer.weight.shape (out, in)", tuple(layer.weight.shape))
    show("layer.bias.shape", tuple(layer.bias.shape))
    show("weight requires_grad, out of the box", layer.weight.requires_grad)
    show("numbers to learn in this layer", count_params(layer))
    x = torch.tensor([[0.5, -1.0], [1.2, 0.3]])
    by_hand = x @ layer.weight.T + layer.bias
    show("layer(x).shape for 2 points", tuple(layer(x).shape))
    show("layer(x) equals x @ W.T + b", bool(torch.allclose(layer(x), by_hand)))
    show("starting weights, first row", [r3(v) for v in layer.weight[0]])
    print("  nn.Linear is lesson 051's W1 and b1, bundled up, already marked")
    print("  requires_grad, and started at small random values for you.")


# ---------------------------------------------------------------------------
# 2. Your first nn.Module
# ---------------------------------------------------------------------------

def section_module():
    heading("2. Your first nn.Module")
    torch.manual_seed(52)
    model = RingNet()
    print("  print(model):")
    for line in str(model).splitlines():
        print("    " + line)
    show("parameters to learn", count_params(model))
    show("named parameters", [name for name, _ in model.named_parameters()])
    X, _ = make_ring()
    show("model(X).shape (one score per point)", tuple(model(X).shape))
    seq = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    show("nn.Sequential version: parameters", count_params(seq))
    print("  Layers you assign to self.something are registered automatically,")
    print("  so model.parameters() finds all 33 numbers without you listing them.")


# ---------------------------------------------------------------------------
# 3. Losses and optimisers come in the box
# ---------------------------------------------------------------------------

def section_loss_and_optimiser():
    heading("3. Losses and optimisers come in the box")
    logits = torch.tensor([2.0, -1.0, 40.0])
    y = torch.tensor([1.0, 0.0, 0.0])
    with_logits = nn.BCEWithLogitsLoss(reduction="none")(logits, y)
    sig_first = nn.BCELoss(reduction="none")(torch.sigmoid(logits), y)
    show("per-point loss, BCEWithLogitsLoss", [r3(v) for v in with_logits])
    show("per-point loss, sigmoid then BCELoss", [r3(v) for v in sig_first])
    show("sigmoid(40) in float32", torch.sigmoid(torch.tensor(40.0)).item())
    w = nn.Parameter(torch.tensor(3.0))
    opt = torch.optim.SGD([w], lr=0.1)
    (w * w).backward()
    show("before optimiser.step(): w, w.grad", (r3(w), r3(w.grad)))
    opt.step()
    show("after step: w = 3 - 0.1 * 6", r3(w))
    opt.zero_grad()
    show("after optimiser.zero_grad(): w.grad", w.grad)
    print("  The third point is confidently wrong. sigmoid(40) rounds to exactly")
    print("  1.0 in float32, so BCELoss can't see how wrong it is and caps the")
    print("  damage at 100. Working from raw scores keeps the true loss of 40.")


# ---------------------------------------------------------------------------
# 4. The standard training loop
# ---------------------------------------------------------------------------

def section_train():
    heading("4. The standard training loop")
    X, y = make_ring()
    X_new, y_new = make_ring(seed=51)
    torch.manual_seed(52)
    model = RingNet()
    train(model, X, y, lr=1.0, steps=3000, report=(0, 100, 500, 1000, 2999))
    show("training accuracy", accuracy(predict_proba(model, X), y))
    show("on 400 fresh points from the same ring", accuracy(predict_proba(model, X_new), y_new))
    show("model.training after predict_proba", model.training)
    print("  Different starting weights from lesson 050 (nn.Linear picks its own),")
    print("  so the path is different, but it lands in the same place.")
    return model, X


# ---------------------------------------------------------------------------
# 5. Saving and loading a trained model
# ---------------------------------------------------------------------------

def section_save_load(model, X):
    heading("5. Saving and loading a trained model")
    state = model.state_dict()
    show("state_dict keys", list(state.keys()))
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "ring_net.pt"
    torch.save(state, path)
    show("saved to", "output/" + path.name)
    show("file size in bytes, roughly", round(path.stat().st_size, -2))
    fresh = RingNet()
    show("fresh model agrees before loading", bool(torch.equal(
        predict_proba(fresh, X), predict_proba(model, X))))
    fresh.load_state_dict(torch.load(path, weights_only=True))
    show("fresh model agrees after loading", bool(torch.equal(
        predict_proba(fresh, X), predict_proba(model, X))))
    print("  Save the state_dict (the numbers), not the whole object. To load,")
    print("  build the same class, then pour the numbers back in.")


# ---------------------------------------------------------------------------
# 6. Same pattern, the rides
# ---------------------------------------------------------------------------

def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    return r3((pos[:, None] > neg[None, :]).mean()
              + 0.5 * (pos[:, None] == neg[None, :]).mean())


def section_rides():
    heading("6. Same pattern, the rides")
    rides = pd.read_csv(RIDES_CSV, parse_dates=["started_at"])
    rides["weekend"] = (rides["started_at"].dt.dayofweek >= 5).astype(int)
    rides["hour"] = rides["started_at"].dt.hour
    X = rides[RIDE_FEATURES].to_numpy(dtype=float)
    y = (rides["rider_type"] == "casual").to_numpy(dtype=float)
    order = np.random.default_rng(0).permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = order[:cut], order[cut:]
    mean, std = X[tr].mean(axis=0), X[tr].std(axis=0)
    X_tr = torch.tensor((X[tr] - mean) / std, dtype=torch.float32)
    X_te = torch.tensor((X[te] - mean) / std, dtype=torch.float32)
    y_tr = torch.tensor(y[tr], dtype=torch.float32)

    for n_hidden in (8, 32):
        torch.manual_seed(52)
        model = RingNet(n_inputs=6, n_hidden=n_hidden)
        loss = train(model, X_tr, y_tr, lr=0.5, steps=2000)
        p_tr = predict_proba(model, X_tr).numpy()
        p_te = predict_proba(model, X_te).numpy()
        show(f"{n_hidden} hidden ({count_params(model)} params): train / test AUC",
             (auc(p_tr, y[tr]), auc(p_te, y[te])))
    print("  Changing the network's size was one argument. Nothing else moved:")
    print("  same loss, same optimiser, same loop. That's the point of nn.Module.")
    print("  The bigger net fits the training rides better and the test rides")
    print("  worse: overfitting, which lesson 056 is all about.")


def main():
    print("=" * 66)
    print("  Lesson 052: Building networks with nn.Module")
    print("=" * 66)
    section_linear()
    section_module()
    section_loss_and_optimiser()
    model, X = section_train()
    section_save_load(model, X)
    section_rides()
    print()
    print("Model, loss, optimiser. Then: zero_grad, forward, loss, backward,")
    print("step. Every PyTorch project you'll ever read is built on those.")
    print()


if __name__ == "__main__":
    main()
