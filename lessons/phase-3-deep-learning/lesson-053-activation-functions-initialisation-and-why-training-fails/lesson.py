"""
Lesson 053 - Activation functions, initialisation and why training fails

Lesson 052 gave us the five-line loop, and on the ring it just worked.
It won't always. A network can sit at the same loss for a thousand
steps, jump to nan, or "train" while half its neurons do nothing at all.
Almost every one of those failures comes from three places:

    the activation function  - what each neuron does after its weighted sum
    the starting weights     - how big the random numbers are on day one
    the learning rate        - how far each step moves

Today we look at each one, break a network on purpose in each way, and
learn the three numbers worth printing when training goes wrong: the
loss, the size of the gradients, and how many neurons are still alive.

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


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.sigmoid(model(X))


def accuracy(p, y):
    return r3(((p >= 0.5).float() == y).float().mean())


def deep_net(act, depth=10, width=32, n_inputs=2):
    """depth hidden layers of the same width, each followed by `act`."""
    layers, n_in = [], n_inputs
    for _ in range(depth):
        layers += [nn.Linear(n_in, width), act()]
        n_in = width
    layers += [nn.Linear(n_in, 1), nn.Flatten(0)]
    return nn.Sequential(*layers)


def linears(model):
    return [m for m in model if isinstance(m, nn.Linear)]


def train(model, X, y, lr, steps, report=()):
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        optimiser.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimiser.step()
        if step in report:
            show(f"loss at step {step}", r3(loss))
    return loss


# ---------------------------------------------------------------------------
# 1. Activation functions, side by side
# ---------------------------------------------------------------------------

def section_activations():
    heading("1. Activation functions, side by side")
    z = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0], requires_grad=True)
    show("inputs z", [r3(v) for v in z])
    for name, f in [("sigmoid", torch.sigmoid), ("tanh", torch.tanh),
                    ("relu", torch.relu),
                    ("leaky_relu", lambda t: nn.functional.leaky_relu(t, 0.01)),
                    ("gelu", nn.functional.gelu)]:
        out = f(z)
        (grad,) = torch.autograd.grad(out.sum(), z)
        show(f"{name}(z)", [r3(v) for v in out])
        show(f"  its slope at z", [r3(v) for v in grad])
    grid = torch.linspace(-6, 6, 1201, requires_grad=True)
    (g,) = torch.autograd.grad(torch.sigmoid(grid).sum(), grid)
    show("steepest sigmoid slope anywhere", r3(g.max()))
    print("  The slope is what backprop multiplies by at every layer. Sigmoid's")
    print("  is never more than 0.25, and almost 0 once |z| passes 4.")


# ---------------------------------------------------------------------------
# 2. Vanishing gradients: ten layers deep
# ---------------------------------------------------------------------------

def grad_norms(model, X, y):
    """Size of the weight gradient in each Linear layer, first to last."""
    model.zero_grad()
    nn.BCEWithLogitsLoss()(model(X), y).backward()
    return [float(layer.weight.grad.norm()) for layer in linears(model)]


def sci(values):
    return [f"{v:.1e}" for v in values]


def section_vanishing():
    heading("2. Vanishing gradients: ten layers deep")
    X, y = make_ring()
    X_new, y_new = make_ring(seed=51)
    show("layers shown", "1st, 4th, 7th, 10th hidden, then output")
    for name, act in [("sigmoid", nn.Sigmoid), ("tanh", nn.Tanh), ("relu", nn.ReLU)]:
        torch.manual_seed(53)
        norms = grad_norms(deep_net(act), X, y)
        show(f"{name}: gradient size by layer", sci([norms[i] for i in (0, 3, 6, 9, 10)]))
    torch.manual_seed(53)
    model = deep_net(nn.Sigmoid)
    loss = train(model, X, y, lr=0.5, steps=1000)
    show("sigmoid, 10 deep, 1000 steps: loss", r3(loss))
    show("accuracy on new points", accuracy(predict_proba(model, X_new), y_new))
    show("share of new points that are outside", r3(1 - y_new.mean()))
    print("  Gradients reach the first layer as a product of slopes. Ten")
    print("  sigmoids is ten numbers under 0.25 multiplied together: nothing")
    print("  arrives, so the early layers never learn. And ReLU didn't save us.")


# ---------------------------------------------------------------------------
# 3. Initialisation: how big should the starting weights be?
# ---------------------------------------------------------------------------

def signal_sizes(std, act, depth=10, width=256):
    """Push 1000 random inputs through `depth` random layers; track the spread."""
    torch.manual_seed(53)
    x = torch.randn(1000, width)
    sizes = []
    for _ in range(depth):
        W = torch.randn(width, width) * std
        x = act(x @ W)
        sizes.append(r3(x.std()))
    return sizes, x


def init_deep(model, how):
    for layer in linears(model)[:-1]:
        if how == "kaiming":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        else:
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.zeros_(layer.bias)
    return model


def section_init():
    heading("3. Initialisation: how big should the starting weights be?")
    fan_in = 256
    show("signal size after layers", "1, 5 and 10 of 10, each 256 wide")
    for label, std in [("relu, std 0.01 (too small)", 0.01),
                       ("relu, std 1.0 (too big)", 1.0),
                       ("relu, Kaiming: sqrt(2 / 256)", (2 / fan_in) ** 0.5)]:
        s, _ = signal_sizes(std, torch.relu)
        show(label, (s[0], s[4], s[9]))
    _, x = signal_sizes(1.0, torch.tanh)
    show("tanh, std 1.0: outputs stuck near +-1", r3((x.abs() > 0.99).float().mean()))
    s, _ = signal_sizes((5 / 3) / fan_in ** 0.5, torch.tanh)
    show("tanh, Xavier: (5/3) / sqrt(256)", (s[0], s[4], s[9]))
    layer = nn.Linear(256, 256)
    show("nn.Linear's default weight spread", r3(layer.weight.std()))
    show("Kaiming would want", r3((2 / fan_in) ** 0.5))

    X, y = make_ring()
    X_new, y_new = make_ring(seed=51)
    for label, act, how in [("relu, default init", nn.ReLU, None),
                            ("relu, Kaiming init", nn.ReLU, "kaiming"),
                            ("tanh, Xavier init", nn.Tanh, "xavier")]:
        torch.manual_seed(53)
        model = deep_net(act)
        if how:
            init_deep(model, how)
        first = grad_norms(model, X, y)[0]
        loss = train(model, X, y, lr=0.5, steps=1000)
        show(f"{label}: first-layer grad, loss",
             (f"{first:.1e}", r3(loss)))
        show("  accuracy on new points", accuracy(predict_proba(model, X_new), y_new))
    print("  Each layer scales the signal by roughly std * sqrt(fan_in). Pick")
    print("  the std that makes that about 1 for your activation, and ten")
    print("  layers train fine. Too small fades, too big explodes.")


# ---------------------------------------------------------------------------
# 4. Dead ReLUs
# ---------------------------------------------------------------------------

def dead_share(model, X):
    """Share of hidden ReLU units that output 0 for every input in X."""
    with torch.no_grad():
        h = torch.relu(model[0](X))
    return r3((h.max(dim=0).values == 0).float().mean())


def relu_net(width=32):
    return nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, 1), nn.Flatten(0))


def section_dead_relus():
    heading("4. Dead ReLUs")
    X, y = make_ring()
    X_new, y_new = make_ring(seed=51)
    torch.manual_seed(53)
    model = relu_net()
    show("healthy start: dead units", dead_share(model, X))
    train(model, X, y, lr=0.5, steps=2000)
    show("after training, lr 0.5: dead units", dead_share(model, X))
    show("accuracy on new points", accuracy(predict_proba(model, X_new), y_new))

    torch.manual_seed(53)
    model = relu_net()
    with torch.no_grad():
        model[0].bias.fill_(-3.0)          # every unit starts below zero everywhere
    show("bias started at -3: dead units", dead_share(model, X))
    loss = train(model, X, y, lr=0.5, steps=2000)
    show("after 2000 steps: dead units, loss", (dead_share(model, X), r3(loss)))
    show("accuracy on new points", accuracy(predict_proba(model, X_new), y_new))

    torch.manual_seed(53)
    model = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1), nn.Flatten(0))
    with torch.no_grad():
        model[0].bias.fill_(-3.0)
    loss = train(model, X, y, lr=0.5, steps=2000)
    show("same start, LeakyReLU(0.01): loss", r3(loss))
    show("accuracy on new points", accuracy(predict_proba(model, X_new), y_new))
    print("  A ReLU that outputs 0 for every input also has slope 0 for every")
    print("  input, so no gradient ever reaches it. It's dead, and it stays dead.")
    print("  LeakyReLU keeps a trickle of slope, but from a start this bad the")
    print("  trickle is too slow. The real fix is not to start there.")


# ---------------------------------------------------------------------------
# 5. Learning rates: too small, about right, too big
# ---------------------------------------------------------------------------

def section_learning_rate():
    heading("5. Learning rates: too small, about right, too big")
    X, y = make_ring()
    for lr in (0.01, 1.0, 30.0):
        torch.manual_seed(53)
        model = relu_net()
        losses = []
        loss_fn = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.SGD(model.parameters(), lr=lr)
        for step in range(1000):
            optimiser.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimiser.step()
            losses.append(r3(loss))
        show(f"lr {lr}: loss at steps 0, 10, 100, 999",
             (losses[0], losses[10], losses[100], losses[999]))
        show(f"lr {lr}: dead units at the end", dead_share(model, X))
    print("  Too small: steady but slow. Too big: the loss jumps about and")
    print("  big updates knock ReLUs dead. Try a few powers of 10 first.")


# ---------------------------------------------------------------------------
# 6. A health check, on the rides
# ---------------------------------------------------------------------------

def health(model, X, y):
    """The three numbers worth printing: loss, first-layer gradient, dead share."""
    first = grad_norms(model, X, y)[0]
    loss = nn.BCEWithLogitsLoss()(model(X), y)
    return r3(loss), round(first, 3), dead_share(model, X)


def section_rides():
    heading("6. A health check, on the rides")
    rides = pd.read_csv(RIDES_CSV, parse_dates=["started_at"])
    rides["weekend"] = (rides["started_at"].dt.dayofweek >= 5).astype(int)
    rides["hour"] = rides["started_at"].dt.hour
    X = rides[RIDE_FEATURES].to_numpy(dtype=float)
    y = torch.tensor((rides["rider_type"] == "casual").to_numpy(dtype=float),
                     dtype=torch.float32)
    raw = torch.tensor(X, dtype=torch.float32)
    scaled = torch.tensor((X - X.mean(axis=0)) / X.std(axis=0), dtype=torch.float32)
    show("biggest raw feature value", r3(raw.abs().max()))

    for label, data in [("raw features", raw), ("scaled features", scaled)]:
        torch.manual_seed(53)
        model = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1), nn.Flatten(0))
        show(f"{label}, before: loss, grad, dead", health(model, data, y))
        train(model, data, y, lr=0.5, steps=500)
        show(f"{label}, after 500: loss, grad, dead", health(model, data, y))
    print("  Raw distances and durations are tens of times bigger than the")
    print("  weekend flag, so the first steps are huge, units die, and the")
    print("  loss stalls. Scale your inputs (lesson 035) before blaming the model.")


def main():
    print("=" * 66)
    print("  Lesson 053: Activation functions, initialisation, why training fails")
    print("=" * 66)
    section_activations()
    section_vanishing()
    section_init()
    section_dead_relus()
    section_learning_rate()
    section_rides()
    print()
    print("When training fails, print three things: the loss, the gradient")
    print("size in the first layer, and the share of dead units.")
    print()


if __name__ == "__main__":
    main()
