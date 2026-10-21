"""
Lesson 051 - Hello, PyTorch: tensors and autograd

Yesterday we wrote the backward pass by hand: five lines of chain rule, a
gradient check, and a lot of care with shapes. Today we hand that job to
PyTorch. Its tensors are NumPy arrays with two superpowers: they can
remember how they were made, and they can run that history backwards to
get gradients. That second one is called autograd.

We rebuild lesson 050's two-layer network on the same ring of points,
check that autograd's gradients match our hand-written ones, and train
it with a loop that has no backward() of our own at all.

Run it with (inside the venv from lesson 013, after `pip install -r
requirements.txt`, which now includes torch):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

No files are written. Every random step has a fixed seed, so your numbers
will match the README. It takes a few seconds.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

RIDE_FEATURES = ["duration_min", "distance_km", "weekend", "hour", "temp_c", "rain_mm"]

torch.manual_seed(51)


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


def r4(x):
    return round(float(x), 4)


def make_ring(n=400, seed=50):
    """Lesson 050's ring: label 1 inside a circle of radius 1, 3% noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.6, 1.6, size=(n, 2))
    y = (np.hypot(X[:, 0], X[:, 1]) < 1.0).astype(float)
    flip = rng.random(n) < 0.03
    y[flip] = 1 - y[flip]
    return X, y


def init_numpy(n_inputs, n_hidden, seed=0):
    """Exactly lesson 050's starting weights, so we can compare like for like."""
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 1 / np.sqrt(n_inputs), size=(n_inputs, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, 1 / np.sqrt(n_hidden), size=(n_hidden, 1)),
        "b2": np.zeros(1),
    }


def to_params(P_np):
    """NumPy weights -> tensors that autograd will track."""
    return {k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for k, v in P_np.items()}


def forward(X, P):
    """Lesson 050's forward pass, word for word, in torch."""
    h = torch.tanh(X @ P["W1"] + P["b1"])
    return torch.sigmoid(h @ P["W2"] + P["b2"]).squeeze(1)


def log_loss(p, y):
    p = p.clamp(1e-12, 1 - 1e-12)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()


def accuracy(p, y):
    return r3(((p >= 0.5).double() == y).double().mean())


def numpy_backward(X, y, P):
    """Lesson 050's hand-written backward pass, kept here to check autograd."""
    n = len(y)
    h = np.tanh(X @ P["W1"] + P["b1"])
    p = (1 / (1 + np.exp(-(h @ P["W2"] + P["b2"])))).ravel()
    dz2 = (p - y).reshape(-1, 1) / n
    dh = dz2 @ P["W2"].T
    dz1 = dh * (1 - h ** 2)
    return {"W1": X.T @ dz1, "b1": dz1.sum(axis=0),
            "W2": h.T @ dz2, "b2": dz2.sum(axis=0)}


# ---------------------------------------------------------------------------
# 1. Tensors: NumPy's close cousin
# ---------------------------------------------------------------------------

def section_tensors():
    heading("1. Tensors: NumPy's close cousin")
    t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    show("t.shape", tuple(t.shape))
    show("t.dtype (torch's default float)", str(t.dtype))
    show("t.sum(dim=0)  (NumPy says axis=0)", t.sum(dim=0).tolist())
    show("t @ t.T", (t @ t.T).tolist())
    a = np.array([1.0, 2.0, 3.0])
    shared = torch.from_numpy(a)
    copied = torch.tensor(a)
    a[0] = 100.0
    show("dtype after from_numpy(float64 array)", str(shared.dtype))
    show("from_numpy after changing the array", shared.tolist())
    show("torch.tensor after changing the array", copied.tolist())
    show("back to NumPy with .numpy()", copied.numpy().tolist())
    show("a single number out with .item()", t.sum().item())
    print("  Same shapes, same broadcasting, same @. The differences that bite:")
    print("  dim= instead of axis=, float32 by default, and from_numpy shares memory.")


# ---------------------------------------------------------------------------
# 2. Autograd on one number at a time
# ---------------------------------------------------------------------------

def section_autograd_scalar():
    heading("2. Autograd on one number at a time")
    x = torch.tensor(2.0)
    w1 = torch.tensor(0.5, requires_grad=True)
    w2 = torch.tensor(-1.0, requires_grad=True)
    h = torch.tanh(w1 * x)
    p = torch.sigmoid(w2 * h)
    loss = -torch.log(p)                 # the true label is 1
    show("loss (same tiny network as lesson 050)", r3(loss))
    show("loss.grad_fn (how it was made)", type(loss.grad_fn).__name__)
    show("w1.grad before backward()", w1.grad)
    loss.backward()
    show("w2.grad after backward()", r3(w2.grad))
    show("w1.grad after backward()", r3(w1.grad))
    show("lesson 050's hand-derived answers", (-0.519, 0.573))
    print("  requires_grad=True tells torch to record every operation. backward()")
    print("  walks that record from the loss outwards: our chain rule, automated.")


# ---------------------------------------------------------------------------
# 3. Autograd on the whole network, checked against lesson 050
# ---------------------------------------------------------------------------

def section_autograd_network(X_np, y_np):
    heading("3. Autograd on the whole network, checked against lesson 050")
    P_np = init_numpy(2, 8)
    P = to_params(P_np)
    X, y = torch.from_numpy(X_np), torch.from_numpy(y_np)
    loss = log_loss(forward(X, P), y)
    loss.backward()
    hand = numpy_backward(X_np, y_np, P_np)
    show("loss at the start", r3(loss))
    show("W2 gradient, first three (autograd)", [r4(v) for v in P["W2"].grad.ravel()[:3]])
    show("W2 gradient, first three (by hand)", [r4(v) for v in hand["W2"].ravel()[:3]])
    gap = max(float(np.abs(P[k].grad.numpy() - hand[k]).max()) for k in P)
    show("biggest gap over all 33 gradients", float(f"{gap:.1e}"))
    print("  Agreement to about 15 decimal places: floating-point noise. Autograd")
    print("  did yesterday's whole backward() for us, and it can't forget tanh.")


# ---------------------------------------------------------------------------
# 4. A training loop with autograd
# ---------------------------------------------------------------------------

def section_train(X_np, y_np):
    heading("4. A training loop with autograd")
    P = to_params(init_numpy(2, 8))
    X, y = torch.from_numpy(X_np), torch.from_numpy(y_np)
    lr = 1.0
    for step in range(3000):
        loss = log_loss(forward(X, P), y)
        loss.backward()                  # fills P[k].grad
        with torch.no_grad():            # the update itself isn't part of the model
            for w in P.values():
                w -= lr * w.grad
                w.grad.zero_()           # clear it, or next step adds to it
        if step in (0, 100, 500, 1000, 2999):
            show(f"loss at step {step}", r3(loss))
    with torch.no_grad():
        show("training accuracy", accuracy(forward(X, P), y))
        X_new, y_new = (torch.from_numpy(a) for a in make_ring(seed=51))
        show("on 400 fresh points from the same ring", accuracy(forward(X_new, P), y_new))
    print("  Same losses and accuracy as lesson 050, because it's the same maths")
    print("  from the same starting weights. Only the backward pass changed hands.")


# ---------------------------------------------------------------------------
# 5. The two traps: piling-up gradients and no_grad
# ---------------------------------------------------------------------------

def section_traps():
    heading("5. The two traps: piling-up gradients and no_grad")
    w = torch.tensor(3.0, requires_grad=True)
    for i in range(3):
        (w * w).backward()               # d/dw of w^2 is 2w = 6, every time
        show(f"w.grad after backward() call {i + 1}, no zeroing", r3(w.grad))
    w.grad.zero_()
    (w * w).backward()
    show("after zero_() and one more backward()", r3(w.grad))
    v = w * 2
    show("w * 2 normally: requires_grad", v.requires_grad)
    with torch.no_grad():
        v = w * 2
    show("w * 2 inside no_grad: requires_grad", v.requires_grad)
    show("w.detach(): requires_grad", w.detach().requires_grad)
    print("  Gradients add up until you clear them. That's on purpose (it lets you")
    print("  sum over several batches), but forgetting to zero is a classic bug.")


# ---------------------------------------------------------------------------
# 6. Back to the rides, in float32
# ---------------------------------------------------------------------------

def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    return r3((pos[:, None] > neg[None, :]).mean()
              + 0.5 * (pos[:, None] == neg[None, :]).mean())


def section_rides():
    heading("6. Back to the rides, in float32")
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

    P = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True)
         for k, v in init_numpy(6, 8).items()}
    show("weights dtype", str(P["W1"].dtype))
    for _ in range(2000):
        loss = log_loss(forward(X_tr, P), y_tr)
        loss.backward()
        with torch.no_grad():
            for w in P.values():
                w -= 0.5 * w.grad
                w.grad.zero_()
    show("final training loss", r3(loss))
    with torch.no_grad():
        p_te = forward(X_te, P).numpy()
    show("test AUC (lesson 050 by hand: 0.841)", auc(p_te, y[te]))
    print("  float32 is half the memory of float64 and what real networks use.")
    print("  The answer matches yesterday's NumPy net to three decimals.")


def main():
    print("=" * 66)
    print("  Lesson 051: Hello, PyTorch: tensors and autograd")
    print("=" * 66)
    show("torch version", torch.__version__.split("+")[0])
    section_tensors()
    section_autograd_scalar()
    X, y = make_ring()
    section_autograd_network(X, y)
    section_train(X, y)
    section_traps()
    section_rides()
    print()
    print("A tensor is a NumPy array that can remember how it was made.")
    print("Mark the weights with requires_grad, compute the loss, call")
    print("backward(), update under no_grad, zero the grads. Repeat.")
    print()


if __name__ == "__main__":
    main()
