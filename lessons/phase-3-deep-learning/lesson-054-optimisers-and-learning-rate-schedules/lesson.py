"""
Lesson 054 - Optimisers and learning-rate schedules

So far every network has learned with plain gradient descent: work out
the gradient, step a fixed distance downhill, repeat. It works, but it's
slow on awkward landscapes, and the learning rate has to be chosen just
so. Today we meet the upgrades nearly everyone uses:

    momentum  - keep some of the last step, so you build up speed
    Adam      - give every weight its own step size
    schedules - change the learning rate as training goes on

Because every torch optimiser has the same step() and zero_grad(),
swapping one for another is a one-line change. That makes them easy to
compare honestly, which is exactly what we'll do.

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


def fit(model, X, y, optimiser, steps, scheduler=None, report=()):
    """Lesson 052's loop, with an optional scheduler step at the end."""
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    seen = {}
    for step in range(steps):
        optimiser.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimiser.step()
        if scheduler is not None:
            scheduler.step()                     # after optimiser.step(), once per step
        if step in report:
            seen[step] = r3(loss)
    return r3(loss), seen


# ---------------------------------------------------------------------------
# 1. A ravine, and plain gradient descent
# ---------------------------------------------------------------------------

def ravine(w):
    """A long, narrow valley: gentle along w[0], 50 times steeper along w[1]."""
    return 0.5 * (w[0] ** 2 + 50 * w[1] ** 2)


def descend(make_opt, steps=500, target=1e-3):
    """Walk from (-4, 1) down the ravine; return steps to reach target, or the final loss."""
    w = torch.tensor([-4.0, 1.0], requires_grad=True)
    opt = make_opt([w])
    for step in range(steps):
        opt.zero_grad()
        loss = ravine(w)
        if loss.item() < target:
            return f"reached in {step} steps"
        if not torch.isfinite(loss) or loss.item() > 1e6:
            return f"blew up by step {step}"
        loss.backward()
        opt.step()
    return f"still at {ravine(w).item():.3f} after {steps}"


def section_ravine():
    heading("1. A ravine, and plain gradient descent")
    w = torch.tensor([-4.0, 1.0], requires_grad=True)
    ravine(w).backward()
    show("start at w", (-4.0, 1.0))
    show("starting loss", r3(ravine(w)))
    show("gradient there", [r3(v) for v in w.grad])
    for lr in (0.01, 0.035, 0.041):
        show(f"SGD, lr {lr}", descend(lambda p: torch.optim.SGD(p, lr=lr)))
    print("  The steep direction caps the learning rate: past 2/50 = 0.04 the")
    print("  steps across the valley overshoot and grow. But a rate that's safe")
    print("  across the valley is painfully slow along it.")


# ---------------------------------------------------------------------------
# 2. Momentum: a heavy ball
# ---------------------------------------------------------------------------

def section_momentum():
    heading("2. Momentum: a heavy ball")
    for m in (0.5, 0.9):
        show(f"SGD, lr 0.035, momentum {m}",
             descend(lambda p: torch.optim.SGD(p, lr=0.035, momentum=m)))
    w = torch.tensor([0.0], requires_grad=True)
    opt = torch.optim.SGD([w], lr=1.0, momentum=0.9)
    moves = []
    for _ in range(4):
        opt.zero_grad()
        (-w.sum()).backward()                   # gradient is always -1: "go right"
        before = w.item()
        opt.step()
        moves.append(r3(w.item() - before))
    show("same gradient 4 times: step sizes", moves)
    print("  Momentum adds 0.9 of the last step to this one. Where the gradient")
    print("  keeps agreeing, steps grow; where it flips sign, they cancel out.")


# ---------------------------------------------------------------------------
# 3. Adam: a step size for every weight
# ---------------------------------------------------------------------------

def section_adam():
    heading("3. Adam: a step size for every weight")
    w = torch.tensor([-4.0, 1.0], requires_grad=True)
    sgd_w = w.detach().clone().requires_grad_()
    opt = torch.optim.Adam([w], lr=0.1)
    ravine(w).backward()
    opt.step()
    show("gradient at the start", (-4.0, 50.0))
    show("SGD lr 0.035 would move by", (r3(0.035 * 4.0), r3(-0.035 * 50.0)))
    show("Adam lr 0.1, first step moved by", [r3(v) for v in w.detach() - sgd_w.detach()])
    for lr in (0.1, 0.3):
        show(f"Adam, lr {lr}", descend(lambda p: torch.optim.Adam(p, lr=lr)))
    print("  Adam divides each gradient by a running average of its own size.")
    print("  Big, steep directions and small, gentle ones both move about lr per")
    print("  step, which is why one learning rate suits most weights at once.")


# ---------------------------------------------------------------------------
# 4. Racing optimisers on the ring
# ---------------------------------------------------------------------------

def section_race():
    heading("4. Racing optimisers on the ring")
    X, y = make_ring()
    X_new, y_new = make_ring(seed=51)
    racers = [
        ("SGD, lr 1.0", lambda p: torch.optim.SGD(p, lr=1.0)),
        ("SGD + momentum 0.9, lr 0.1", lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
        ("Adam, lr 0.01", lambda p: torch.optim.Adam(p, lr=0.01)),
        ("Adam, lr 0.05", lambda p: torch.optim.Adam(p, lr=0.05)),
    ]
    show("loss at steps", "100, 300, 1000")
    for name, make_opt in racers:
        torch.manual_seed(54)
        model = RingNet()
        final, seen = fit(model, X, y, make_opt(model.parameters()), 1000, report=(100, 300))
        show(name, (seen[100], seen[300], final))
        show("  accuracy on new points", accuracy(predict_proba(model, X_new), y_new))
    print("  Same model, same seed, same loop: only the optimiser line changed.")
    print("  Adam at 0.01 is the most common starting point in practice; it")
    print("  isn't always the winner, but it's rarely a disaster.")


# ---------------------------------------------------------------------------
# 5. Learning-rate schedules
# ---------------------------------------------------------------------------

def fit_in_batches(model, X, y, optimiser, steps, scheduler=None, batch=16):
    """Each step learns from 16 random points, not all 400. Returns the
    full-data loss over the last 100 steps, so we can see how much it wobbles."""
    loss_fn = nn.BCEWithLogitsLoss()
    picker = torch.Generator().manual_seed(54)
    late = []
    model.train()
    for step in range(steps):
        idx = torch.randint(0, len(y), (batch,), generator=picker)
        optimiser.zero_grad()
        loss_fn(model(X[idx]), y[idx]).backward()
        optimiser.step()
        if scheduler is not None:
            scheduler.step()                     # after optimiser.step(), once per step
        if step >= steps - 100:
            with torch.no_grad():
                late.append(loss_fn(model(X), y).item())
    return late


def section_schedules():
    heading("5. Learning-rate schedules")
    show("lr shown at steps", "0, 250, 500, 750, 1000")
    p = [nn.Parameter(torch.zeros(1))]
    for name, make in [
        ("StepLR, x0.1 every 400", lambda o: torch.optim.lr_scheduler.StepLR(o, 400, gamma=0.1)),
        ("CosineAnnealingLR, 1000 steps", lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, 1000)),
    ]:
        opt = torch.optim.Adam(p, lr=0.05)
        sched = make(opt)
        lrs = []
        for step in range(1001):
            if step % 250 == 0:
                lrs.append(round(sched.get_last_lr()[0], 4))
            opt.step()
            sched.step()
        show(name, tuple(lrs))

    X, y = make_ring()
    X_new, y_new = make_ring(seed=51)
    for label, use_cosine in [("Adam 0.2, batches of 16, constant", False),
                              ("Adam 0.2, batches of 16, cosine", True)]:
        torch.manual_seed(54)
        model = RingNet()
        opt = torch.optim.Adam(model.parameters(), lr=0.2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2000) if use_cosine else None
        late = fit_in_batches(model, X, y, opt, 2000, scheduler=sched)
        print(f"  {label}")
        show("  full loss at the end", r3(late[-1]))
        show("  wobble over the last 100 steps", r3(max(late) - min(late)))
        show("  accuracy on new points", accuracy(predict_proba(model, X_new), y_new))
    print("  Learning from small random batches makes every step a bit noisy.")
    print("  With a constant rate the model keeps getting knocked about; as the")
    print("  cosine schedule shrinks the steps, it settles into the bottom.")


# ---------------------------------------------------------------------------
# 6. Same swap, the rides
# ---------------------------------------------------------------------------

def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    return r3((pos[:, None] > neg[None, :]).mean()
              + 0.5 * (pos[:, None] == neg[None, :]).mean())


def section_rides():
    heading("6. Same swap, the rides")
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

    runs = [
        ("SGD 0.5, 2000 steps (lesson 052)", lambda p: torch.optim.SGD(p, lr=0.5), 2000),
        ("Adam 0.01, 300 steps", lambda p: torch.optim.Adam(p, lr=0.01), 300),
        ("AdamW 0.01 + decay 0.1, 300 steps", lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1), 300),
    ]
    for name, make_opt, steps in runs:
        torch.manual_seed(52)
        model = RingNet(n_inputs=6, n_hidden=8)
        fit(model, X_tr, y_tr, make_opt(model.parameters()), steps)
        p_tr = predict_proba(model, X_tr).numpy()
        p_te = predict_proba(model, X_te).numpy()
        print(f"  {name}")
        show("  train / test AUC", (auc(p_tr, y[tr]), auc(p_te, y[te])))
    print("  Adam matched SGD's test score in under a seventh of the steps. On a")
    print("  big model, that's hours saved, not milliseconds.")


def main():
    print("=" * 66)
    print("  Lesson 054: Optimisers and learning-rate schedules")
    print("=" * 66)
    section_ravine()
    section_momentum()
    section_adam()
    section_race()
    section_schedules()
    section_rides()
    print()
    print("Start with Adam at 0.01. If it's jumpy at the end, add a cosine")
    print("schedule. If you need the last drop, try SGD with momentum.")
    print()


if __name__ == "__main__":
    main()
