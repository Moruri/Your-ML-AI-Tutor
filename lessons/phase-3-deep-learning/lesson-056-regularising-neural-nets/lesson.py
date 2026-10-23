"""
Lesson 056 - Regularising neural nets

A network with enough weights can memorise its training data: every
point, every bit of noise. Its training loss keeps falling while its
score on new data stalls or gets worse. That's overfitting, the same
enemy as lesson 041, now in a model with thousands of weights.

Four standard defences, each a line or two of torch:

    weight decay       - keep the weights small (lesson 041's Ridge, again)
    dropout            - randomly switch off units while training
    early stopping     - stop when the validation score stops improving
    data augmentation  - show the model new, label-preserving versions of the data

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files. Every random step has a fixed seed, so your
numbers will match the README. It takes about twenty seconds.
"""

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

RIDE_FEATURES = ["duration_min", "distance_km", "weekend", "hour", "temp_c", "rain_mm"]

EPOCHS = 300


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


def big_net(n_inputs=2, width=64, dropout=0.0):
    """Far more weights than 60 points need: room to memorise."""
    return nn.Sequential(
        nn.Linear(n_inputs, width), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(width, 1), nn.Flatten(0),
    )


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.sigmoid(model(X))


@torch.no_grad()
def loss_on(model, X, y):
    model.eval()
    return r3(nn.BCEWithLogitsLoss()(model(X), y))


def accuracy(p, y):
    return r3(((p >= 0.5).float() == y).float().mean())


def auc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    pos, neg = p[y == 1], p[y == 0]
    return r3((pos[:, None] > neg[None, :]).mean()
              + 0.5 * (pos[:, None] == neg[None, :]).mean())


def train(model, train_ds, val, optimiser, epochs=EPOCHS, batch=16,
          augment=None, patience=None, report=()):
    """Lesson 055's loop, plus: validation loss each epoch, optional
    augmentation, and optional early stopping that restores the best weights."""
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(56))
    best, best_epoch, best_state, seen = math.inf, 0, None, {}
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            if augment is not None:
                xb = augment(xb)
            optimiser.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimiser.step()
        val_loss = loss_on(model, *val)
        if epoch in report:
            seen[epoch] = (loss_on(model, *train_ds.tensors), val_loss)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif patience is not None and epoch - best_epoch >= patience:
            model.load_state_dict(best_state)
            return {"stopped": epoch, "best": best_epoch, "seen": seen}
    if patience is not None:
        model.load_state_dict(best_state)
    return {"stopped": epochs, "best": best_epoch, "seen": seen}


def ring_data():
    X_tr, y_tr = make_ring(n=60, seed=56)     # a small training set
    val = make_ring(n=200, seed=57)           # for choices
    test = make_ring(n=400, seed=51)          # touched once per model
    return TensorDataset(X_tr, y_tr), val, test


def report_ring(model, train_ds, test):
    X_te, y_te = test
    show("  train / test loss", (loss_on(model, *train_ds.tensors), loss_on(model, X_te, y_te)))
    show("  test accuracy", accuracy(predict_proba(model, X_te), y_te))


# ---------------------------------------------------------------------------
# 1. Watching a network overfit
# ---------------------------------------------------------------------------

def section_overfit():
    heading("1. Watching a network overfit")
    train_ds, val, test = ring_data()
    torch.manual_seed(56)
    model = big_net()
    show("training points", len(train_ds))
    show("weights in the network", sum(p.numel() for p in model.parameters()))
    out = train(model, train_ds, val, torch.optim.Adam(model.parameters(), lr=0.01),
                report=(10, 30, 100, 300))
    show("(train loss, val loss) at epoch", "")
    for epoch, pair in out["seen"].items():
        show(f"  epoch {epoch}", pair)
    show("best val loss was at epoch", out["best"])
    print("  No defences, 300 epochs:")
    report_ring(model, train_ds, test)
    print("  Training loss heads for zero while validation loss turns round and")
    print("  climbs: the network is learning the 60 points, not the circle.")


# ---------------------------------------------------------------------------
# 2. Weight decay
# ---------------------------------------------------------------------------

def section_weight_decay():
    heading("2. Weight decay")
    train_ds, val, test = ring_data()
    for wd in (0.0, 0.1, 1.0):
        torch.manual_seed(56)
        model = big_net()
        train(model, train_ds, val, torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=wd))
        size = sum(p.pow(2).sum() for n, p in model.named_parameters() if "weight" in n).sqrt()
        print(f"  AdamW, weight_decay={wd}")
        show("  size of all weights", r3(size))
        report_ring(model, train_ds, test)
    print("  Every step shrinks each weight a little towards zero. Big weights")
    print("  are what sharp, wiggly boundaries are made of, so decay smooths them.")


# ---------------------------------------------------------------------------
# 3. Dropout
# ---------------------------------------------------------------------------

def section_dropout():
    heading("3. Dropout")
    drop = nn.Dropout(p=0.5)
    torch.manual_seed(56)
    ones = torch.ones(8)
    drop.train()
    show("Dropout(0.5) in train mode, on ones", [round(v, 1) for v in drop(ones).tolist()])
    drop.eval()
    show("same layer in eval mode", [round(v, 1) for v in drop(ones).tolist()])
    train_ds, val, test = ring_data()
    for p in (0.0, 0.3):
        torch.manual_seed(56)
        model = big_net(dropout=p)
        train(model, train_ds, val, torch.optim.Adam(model.parameters(), lr=0.01))
        print(f"  dropout p={p}")
        report_ring(model, train_ds, test)
    print("  In training, each unit is zeroed half the time (the rest are doubled")
    print("  to keep the average). No unit can rely on another, so none can memorise")
    print("  alone. model.eval() switches it off for predictions.")


# ---------------------------------------------------------------------------
# 4. Early stopping
# ---------------------------------------------------------------------------

def section_early_stopping():
    heading("4. Early stopping")
    train_ds, val, test = ring_data()
    torch.manual_seed(56)
    model = big_net()
    out = train(model, train_ds, val, torch.optim.Adam(model.parameters(), lr=0.01), patience=20)
    show("patience", 20)
    show("best epoch, stopped at epoch", (out["best"], out["stopped"]))
    print("  weights restored from the best epoch:")
    report_ring(model, train_ds, test)
    print("  Watch validation loss; when it hasn't improved for 20 epochs, stop")
    print("  and roll back to the best weights. Cheap, and it saves time too.")


# ---------------------------------------------------------------------------
# 5. Data augmentation
# ---------------------------------------------------------------------------

def rotate(xb, gen=torch.Generator().manual_seed(56)):
    """Spin every point by its own random angle. Distance from the centre,
    and so the label, doesn't change: free, honest new training points."""
    angle = torch.rand(len(xb), generator=gen) * 2 * math.pi
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack([c * xb[:, 0] - s * xb[:, 1], s * xb[:, 0] + c * xb[:, 1]], dim=1)


def section_augmentation():
    heading("5. Data augmentation")
    p = torch.tensor([[0.6, 0.8]])
    q = rotate(p)
    show("a point, its distance from centre", ([r3(v) for v in p[0]], r3(p.norm())))
    show("rotated, its distance from centre", ([r3(v) for v in q[0]], r3(q.norm())))
    train_ds, val, test = ring_data()
    torch.manual_seed(56)
    model = big_net()
    train(model, train_ds, val, torch.optim.Adam(model.parameters(), lr=0.01), augment=rotate)
    print("  random rotation of every batch, 300 epochs:")
    report_ring(model, train_ds, test)
    print("  Every epoch the network sees 60 points it has never seen before, so")
    print("  there's nothing fixed to memorise. Only the circle stays the same.")


# ---------------------------------------------------------------------------
# 6. Putting it together on the rides
# ---------------------------------------------------------------------------

def rides_splits():
    rides = pd.read_csv(RIDES_CSV, parse_dates=["started_at"])
    rides["weekend"] = (rides["started_at"].dt.dayofweek >= 5).astype(int)
    rides["hour"] = rides["started_at"].dt.hour
    X = torch.tensor(rides[RIDE_FEATURES].to_numpy(dtype=float), dtype=torch.float32)
    y = torch.tensor((rides["rider_type"] == "casual").to_numpy(dtype=float), dtype=torch.float32)
    tr, va, te = random_split(range(len(y)), [0.7, 0.15, 0.15],
                              generator=torch.Generator().manual_seed(55))
    tr, va, te = (torch.tensor(s.indices) for s in (tr, va, te))
    mean, std = X[tr].mean(dim=0), X[tr].std(dim=0)
    X = (X - mean) / std
    return TensorDataset(X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def section_rides():
    heading("6. Putting it together on the rides")
    train_ds, val, (X_te, y_te) = rides_splits()
    show("train / val / test rides", (len(train_ds), len(val[1]), len(y_te)))
    runs = [
        ("big net, no defences", 0.0, 0.0, None),
        ("+ dropout 0.3, AdamW decay 0.1, patience 20", 0.3, 0.1, 20),
    ]
    for name, p, wd, patience in runs:
        torch.manual_seed(56)
        model = big_net(n_inputs=6, dropout=p)
        out = train(model, train_ds, val,
                    torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=wd),
                    epochs=150, batch=32, patience=patience)
        X_tr, y_tr = train_ds.tensors
        print(f"  {name}")
        show("  epochs run, best epoch", (out["stopped"], out["best"]))
        show("  train / test AUC", (auc(predict_proba(model, X_tr), y_tr),
                                    auc(predict_proba(model, X_te), y_te)))
    print("  Undefended, the big net memorises: train AUC 0.97, test 0.84. Defended,")
    print("  it stops early, and train and test agree.")


def main():
    print("=" * 66)
    print("  Lesson 056: Regularising neural nets")
    print("=" * 66)
    section_overfit()
    section_weight_decay()
    section_dropout()
    section_early_stopping()
    section_augmentation()
    section_rides()
    print()
    print("Start with AdamW weight decay and early stopping on a validation set.")
    print("Add dropout or augmentation when the gap between train and val stays wide.")
    print()


if __name__ == "__main__":
    main()
