"""
Lesson 050 - Backpropagation, step by step

Yesterday's neuron draws one straight line through the data. Plenty of
problems don't split along a straight line. The fix is to stack neurons:
a hidden layer that bends the space, then an output neuron on top. The
catch is that we now need the gradient for every weight in every layer.
Backpropagation is how we get it: the chain rule, applied from the loss
backwards, one layer at a time. Today we derive it, code it, check it,
and train a two-layer network with no framework at all.

We start on a small made-up dataset of points in a ring, where one neuron
is hopeless, then come back to the bike-share rides in rides_clean.csv.

Run it with (inside the venv from lesson 013, after `pip install -r
requirements.txt`):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes one picture, output/decision_boundary.png, next to this
file. Every random step has a fixed seed, so your numbers will match the
README. It takes about five seconds.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    return round(float(x), 3)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(p, y):
    return r3(np.mean((p >= 0.5) == y))


def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    return r3((pos[:, None] > neg[None, :]).mean()
              + 0.5 * (pos[:, None] == neg[None, :]).mean())


def make_ring(n=400, seed=50):
    """Points in a square; label 1 if they fall inside a circle of radius 1."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.6, 1.6, size=(n, 2))
    y = (np.hypot(X[:, 0], X[:, 1]) < 1.0).astype(float)
    flip = rng.random(n) < 0.03          # a little label noise, like real data
    y[flip] = 1 - y[flip]
    return X, y


# ---------------------------------------------------------------------------
# The single neuron from lesson 049, for comparison
# ---------------------------------------------------------------------------

def train_neuron(X, y, lr=0.5, steps=2000):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        error = sigmoid(X @ w + b) - y
        w -= lr * X.T @ error / len(y)
        b -= lr * error.mean()
    return w, b


# ---------------------------------------------------------------------------
# The two-layer network
# ---------------------------------------------------------------------------

def init_params(n_inputs, n_hidden, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 1 / np.sqrt(n_inputs), size=(n_inputs, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, 1 / np.sqrt(n_hidden), size=(n_hidden, 1)),
        "b2": np.zeros(1),
    }


def forward(X, P):
    """Inputs -> hidden layer (tanh) -> output neuron (sigmoid)."""
    z1 = X @ P["W1"] + P["b1"]        # (n, hidden)
    h = np.tanh(z1)                   # (n, hidden)
    z2 = h @ P["W2"] + P["b2"]        # (n, 1)
    p = sigmoid(z2).ravel()           # (n,)
    return p, {"X": X, "h": h}


def backward(p, y, cache, P, forget_tanh=False):
    """The chain rule, from the loss back to every weight."""
    X, h = cache["X"], cache["h"]
    n = len(y)
    dz2 = (p - y).reshape(-1, 1) / n             # loss -> output pre-activation
    dW2 = h.T @ dz2                              # output weights
    db2 = dz2.sum(axis=0)
    dh = dz2 @ P["W2"].T                         # back through the output weights
    dz1 = dh if forget_tanh else dh * (1 - h ** 2)   # back through tanh
    dW1 = X.T @ dz1                              # hidden weights
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def loss_of(X, y, P):
    return log_loss(forward(X, P)[0], y)


def numeric_grads(X, y, P, eps=1e-6):
    grads = {}
    for name, arr in P.items():
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + eps
            up = loss_of(X, y, P)
            arr[idx] = old - eps
            down = loss_of(X, y, P)
            arr[idx] = old
            g[idx] = (up - down) / (2 * eps)
        grads[name] = g
    return grads


def rel_error(a, b):
    """Biggest relative gap across all parameters: ~1e-7 or less means correct."""
    worst = 0.0
    for k in a:
        num = np.abs(a[k] - b[k])
        den = np.maximum(np.abs(a[k]) + np.abs(b[k]), 1e-12)
        worst = max(worst, float((num / den).max()))
    return float(f"{worst:.1e}")


def train_net(X, y, n_hidden, lr, steps, seed=0):
    P = init_params(X.shape[1], n_hidden, seed)
    history = []
    for _ in range(steps):
        p, cache = forward(X, P)
        history.append(log_loss(p, y))
        grads = backward(p, y, cache, P)
        for k in P:
            P[k] -= lr * grads[k]
    return P, history


# ---------------------------------------------------------------------------
# 1. A problem one neuron can't solve
# ---------------------------------------------------------------------------

def section_ring():
    heading("1. A problem one neuron can't solve")
    X, y = make_ring()
    show("points / share inside the ring", (len(y), r3(y.mean())))
    w, b = train_neuron(X, y)
    p = sigmoid(X @ w + b)
    show("one neuron: training accuracy", accuracy(p, y))
    show("always saying 'outside': accuracy", accuracy(np.zeros_like(y), y))
    show("one neuron: weights", [r3(v) for v in w])
    print("  A single neuron can only draw a straight line. No straight line")
    print("  separates the inside of a circle from the outside, so it gives up")
    print("  and does no better than guessing 'outside' every time.")
    return X, y


# ---------------------------------------------------------------------------
# 2. Two layers: the forward pass
# ---------------------------------------------------------------------------

def section_forward(X):
    heading("2. Two layers: the forward pass")
    P = init_params(2, 8)
    p, cache = forward(X, P)
    show("X shape (points, inputs)", X.shape)
    show("W1 shape (inputs, hidden units)", P["W1"].shape)
    show("h shape, after tanh", cache["h"].shape)
    show("W2 shape (hidden units, 1)", P["W2"].shape)
    show("p shape, one probability per point", p.shape)
    show("parameters to learn", sum(a.size for a in P.values()))
    show("first three predictions, untrained", [r3(v) for v in p[:3]])
    print("  Eight hidden neurons each draw their own line and squash it with")
    print("  tanh. The output neuron combines those eight bent signals.")


# ---------------------------------------------------------------------------
# 3. The chain rule, with one number at a time
# ---------------------------------------------------------------------------

def section_chain_rule():
    heading("3. The chain rule, with one number at a time")
    x, y, w1, w2 = 2.0, 1.0, 0.5, -1.0
    h = np.tanh(w1 * x)
    p = sigmoid(w2 * h)
    L = -np.log(p)
    show("x, w1, w2, true label y", (x, w1, w2, y))
    show("h = tanh(w1 * x)", r3(h))
    show("p = sigmoid(w2 * h)", r3(p))
    show("loss = -log(p)", r3(L))
    dz2 = p - y                       # dL/dz2, same as lesson 049
    dw2 = dz2 * h                     # z2 = w2 * h
    dh = dz2 * w2                     # pass the blame back to h
    dz1 = dh * (1 - h ** 2)           # tanh'(z) = 1 - tanh(z)^2
    dw1 = dz1 * x                     # z1 = w1 * x
    show("dL/dw2 = (p - y) * h", r3(dw2))
    show("dL/dw1 = (p - y) * w2 * (1 - h^2) * x", r3(dw1))
    eps = 1e-6

    def loss(a, b):
        return -np.log(sigmoid(b * np.tanh(a * x)))

    show("nudge-and-measure dL/dw2", r3((loss(w1, w2 + eps) - loss(w1, w2 - eps)) / (2 * eps)))
    show("nudge-and-measure dL/dw1", r3((loss(w1 + eps, w2) - loss(w1 - eps, w2)) / (2 * eps)))
    print("  Each step backwards multiplies by one local slope. That's all")
    print("  backpropagation is: the chain rule, read from the loss outward.")


# ---------------------------------------------------------------------------
# 4. The backward pass in code, and the gradient check
# ---------------------------------------------------------------------------

def section_backward(X, y):
    heading("4. The backward pass in code, and the gradient check")
    P = init_params(2, 8)
    Xs, ys = X[:20], y[:20]           # a small batch keeps the check quick
    p, cache = forward(Xs, P)
    grads = backward(p, ys, cache, P)
    numeric = numeric_grads(Xs, ys, P)
    show("gradient shapes match parameter shapes",
         all(grads[k].shape == P[k].shape for k in P))
    show("W2 gradient, first three (backprop)", [r3(v) for v in grads["W2"].ravel()[:3]])
    show("W2 gradient, first three (nudges)", [r3(v) for v in numeric["W2"].ravel()[:3]])
    show("worst relative error, backprop vs nudges", rel_error(grads, numeric))
    buggy = backward(p, ys, cache, P, forget_tanh=True)
    show("same check, but forgetting tanh's slope", rel_error(buggy, numeric))
    print("  Backprop gets all 33 gradients from one backward sweep. The nudge")
    print("  method needed 66 extra forward passes. On a real network with")
    print("  millions of weights, that's the difference between possible and not.")


# ---------------------------------------------------------------------------
# 5. Training the network
# ---------------------------------------------------------------------------

def section_train(X, y):
    heading("5. Training the network")
    P, history = train_net(X, y, n_hidden=8, lr=1.0, steps=3000)
    for step in (0, 100, 500, 1000, 2999):
        show(f"loss at step {step}", r3(history[step]))
    p, _ = forward(X, P)
    show("two-layer net: training accuracy", accuracy(p, y))
    X_new, y_new = make_ring(seed=51)
    show("on 400 fresh points from the same ring", accuracy(forward(X_new, P)[0], y_new))

    OUTPUT.mkdir(exist_ok=True)
    grid = np.linspace(-1.6, 1.6, 200)
    gx, gy = np.meshgrid(grid, grid)
    gp = forward(np.c_[gx.ravel(), gy.ravel()], P)[0].reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.contourf(gx, gy, gp, levels=20, cmap="RdBu_r", alpha=0.6)
    ax.contour(gx, gy, gp, levels=[0.5], colors="black")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=14)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, linestyle="--", color="grey"))
    ax.set_aspect("equal")
    ax.set_title("Two-layer network: where it says 'inside'")
    fig.tight_layout()
    path = OUTPUT / "decision_boundary.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    show("picture written to", "output/" + path.name)


# ---------------------------------------------------------------------------
# 6. Back to the rides
# ---------------------------------------------------------------------------

def section_rides():
    heading("6. Back to the rides")
    rides = pd.read_csv(RIDES_CSV, parse_dates=["started_at"])
    rides["weekend"] = (rides["started_at"].dt.dayofweek >= 5).astype(int)
    rides["hour"] = rides["started_at"].dt.hour
    X = rides[RIDE_FEATURES].to_numpy(dtype=float)
    y = (rides["rider_type"] == "casual").to_numpy(dtype=float)
    order = np.random.default_rng(0).permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = order[:cut], order[cut:]
    mean, std = X[tr].mean(axis=0), X[tr].std(axis=0)
    X_tr, X_te = (X[tr] - mean) / std, (X[te] - mean) / std
    y_tr, y_te = y[tr], y[te]

    w, b = train_neuron(X_tr, y_tr)
    show("one neuron, six inputs: test AUC", auc(sigmoid(X_te @ w + b), y_te))
    P, _ = train_net(X_tr, y_tr, n_hidden=8, lr=0.5, steps=2000)
    show("two-layer net, 8 hidden: test AUC", auc(forward(X_te, P)[0], y_te))
    show("two-layer net: train AUC", auc(forward(X_tr, P)[0], y_tr))
    print("  On the rides, the extra layer buys nothing: test AUC is the same")
    print("  to two decimals. With six tidy columns and 882 rows, a straight")
    print("  line already captures most of the pattern. Depth pays off when")
    print("  there's a lot more data and a lot more shape to find, like the")
    print("  images and text coming later in Phases 3 and 4.")


def main():
    print("=" * 66)
    print("  Lesson 050: Backpropagation, step by step")
    print("=" * 66)
    X, y = section_ring()
    section_forward(X)
    section_chain_rule()
    section_backward(X, y)
    section_train(X, y)
    section_rides()
    print()
    print("Backpropagation is the chain rule, applied from the loss backwards:")
    print("each layer passes the blame to the one before it, multiplied by its")
    print("own local slope. Check it with nudges once, then trust it. Next")
    print("lesson, PyTorch does this sweep for us.")
    print()


if __name__ == "__main__":
    main()
