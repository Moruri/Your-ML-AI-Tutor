"""
Lesson 058 - Recurrent networks and sequences

Some data only makes sense in order: words in a sentence, prices day by
day, rides hour by hour. Shuffle it and the meaning goes. A recurrent
network (RNN) reads a sequence one step at a time and carries a small
memory, the hidden state, from each step to the next.

    h_t = tanh(W_x x_t + W_h h_(t-1) + b)

Same weights at every step, like a convolution's kernel sliding across
time instead of space.

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files. Every random step has a fixed seed, so your
numbers will match the README. It takes about half a minute.
"""

import copy
import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)   # same numbers on every machine

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"

WINDOW = 24     # hours of history the model sees
SCALE = 10.0    # rides per hour are divided by this, to keep inputs small


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
# 1. Rides as a sequence
# ---------------------------------------------------------------------------

def hourly_counts():
    rides = pd.read_csv(RIDES_CSV, parse_dates=["started_at"])
    start = rides["started_at"].min().floor("D")
    end = rides["started_at"].max().ceil("D")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    counts = rides.set_index("started_at").resample("h").size()
    return counts.reindex(hours, fill_value=0)


def windows(series):
    """Every run of WINDOW hours as an input, the hour after it as the target.

    Each hour carries three features: its ride count (scaled), and the hour
    of day as a point on a circle (sin, cos), so 23:00 sits next to 00:00.
    """
    v = torch.tensor(series.to_numpy(dtype=float) / SCALE, dtype=torch.float32)
    angle = torch.tensor(series.index.hour.to_numpy(), dtype=torch.float32) * 2 * math.pi / 24
    steps = torch.stack([v, torch.sin(angle), torch.cos(angle)], dim=1)
    X = torch.stack([steps[i:i + WINDOW] for i in range(len(v) - WINDOW)])
    y = v[WINDOW:]
    return X, y


def section_sequence(counts):
    heading("1. Rides as a sequence")
    show("hours, from, to", (len(counts), str(counts.index[0]), str(counts.index[-1])))
    show("rides in each hour of 1 September", counts.iloc[:24].tolist())
    X, y = windows(counts)
    show("windows, shape (samples, time steps, features)", tuple(X.shape))
    show("targets, shape", tuple(y.shape))
    a = torch.tensor([0., 0., 8., 1., 0.])
    b = torch.tensor([8., 1., 0., 0., 0.])
    show("two sequences, their means", (r3(a.mean()), r3(b.mean())))
    print("  Same values, same mean, different stories. A busy hour that just")
    print("  happened means something different from one five hours ago.")


# ---------------------------------------------------------------------------
# 2. A recurrent cell, by hand
# ---------------------------------------------------------------------------

def section_cell():
    heading("2. A recurrent cell, by hand")
    torch.manual_seed(58)
    rnn = nn.RNN(input_size=1, hidden_size=3, batch_first=True)
    W_x, W_h = rnn.weight_ih_l0.detach(), rnn.weight_hh_l0.detach()
    b = (rnn.bias_ih_l0 + rnn.bias_hh_l0).detach()
    xs = torch.tensor([[0.0], [0.8], [0.1], [0.0]])
    h = torch.zeros(3)
    print("  hidden state after each step (by hand):")
    for t, x in enumerate(xs):
        h = torch.tanh(W_x @ x + W_h @ h + b)
        show(f"  step {t}, input {x.item():.1f}", [r3(v) for v in h])
    out, h_last = rnn(xs.unsqueeze(0))
    show("nn.RNN's final hidden state", [r3(v) for v in h_last[0, 0]])
    show("nn.RNN weights: W_x, W_h, biases", (tuple(W_x.shape), tuple(W_h.shape), 2 * len(b)))
    print("  Each step mixes the new input with the old hidden state. The same")
    print("  weights are reused at every step, however long the sequence.")


# ---------------------------------------------------------------------------
# 3. Forecasting the next hour
# ---------------------------------------------------------------------------

class Forecaster(nn.Module):
    """Read the window with a recurrent layer, predict from its last hidden state."""

    def __init__(self, cell="rnn", n_inputs=1, hidden=16):
        super().__init__()
        layer = {"rnn": nn.RNN, "gru": nn.GRU}[cell]
        self.recurrent = layer(input_size=n_inputs, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, h = self.recurrent(x)        # out: every step, h: the last one
        return self.head(h[-1]).squeeze(-1)


def fit(model, train_set, val, loss_fn, epochs=60, patience=10, lr=0.01, batch=32):
    """Lesson 056's loop: AdamW, validation every epoch, early stopping."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loader = DataLoader(TensorDataset(*train_set), batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(58))
    best, best_epoch, best_state = math.inf, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val[0]), val[1]).item()
        if val_loss < best:
            best, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return best_epoch


def mae(pred, y):
    """Mean absolute error, back in rides per hour."""
    return r3((pred - y).abs().mean() * SCALE)


def rmse(pred, y):
    """Root mean squared error, in rides per hour: big misses count extra."""
    return r3((pred - y).pow(2).mean().sqrt() * SCALE)


def section_forecast(counts):
    heading("3. Forecasting the next hour")
    X, y = windows(counts)
    # Time order, never shuffled across the split: train on the past, test on the future.
    n_train, n_val = 16 * 24, 3 * 24
    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, None)
    show("train / val / test windows", (n_train, n_val, len(y) - n_train - n_val))
    X_te, y_te = X[te], y[te]
    show("(MAE, RMSE) in rides per hour", "")
    show("  baseline: same as the hour before", (mae(X_te[:, -1, 0], y_te), rmse(X_te[:, -1, 0], y_te)))
    show("  baseline: same hour yesterday", (mae(X_te[:, 0, 0], y_te), rmse(X_te[:, 0, 0], y_te)))
    models = {}
    for cell in ("rnn", "gru"):
        torch.manual_seed(58)
        model = Forecaster(cell, n_inputs=3)
        best_epoch = fit(model, (X[tr], y[tr]), (X[va], y[va]), nn.MSELoss(),
                         epochs=100, patience=15)
        model.eval()
        with torch.no_grad():
            pred = model(X_te)
        models[cell] = model
        show(f"  {cell.upper()} (best epoch {best_epoch})", (mae(pred, y_te), rmse(pred, y_te)))
    with torch.no_grad():
        day = models["gru"](X_te[:24]) * SCALE
    hours = counts.index[WINDOW + n_train + n_val:][:24]
    show("first test day", str(hours[0].date()))
    show("  hours", [h.hour for h in hours[7:20]])
    show("  GRU predicted", [round(v) for v in day[7:20].tolist()])
    show("  actual", [round(v) for v in (y_te[7:20] * SCALE).tolist()])
    print("  'Same hour yesterday' is a tough baseline. The GRU just beats it, by")
    print("  reading the whole day so far; the plain RNN doesn't.")


# ---------------------------------------------------------------------------
# 4. Where plain RNNs struggle: long memory
# ---------------------------------------------------------------------------

def memory_task(n, length, seed):
    """The label is the sign of the FIRST value; everything after is noise."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, length, 1, generator=g)
    y = (X[:, 0, 0] > 0).float()
    return X, y


def remember_by_default(model, bias=2.0):
    """Push a GRU's update gate towards 'keep the old memory' at the start.

    torch stores a GRU's biases as [reset | update | new], hidden_size each.
    """
    h = model.recurrent.hidden_size
    with torch.no_grad():
        model.recurrent.bias_ih_l0[h:2 * h].fill_(bias)
    return model


def memory_models():
    return [
        ("RNN", lambda: Forecaster("rnn")),
        ("GRU", lambda: Forecaster("gru")),
        ("GRU, remembering by default", lambda: remember_by_default(Forecaster("gru"))),
    ]


def section_memory():
    heading("4. Where plain RNNs struggle: long memory")
    print("  Task: say whether the FIRST number was positive. The rest is noise.")
    for length in (10, 100):
        train_set = memory_task(2000, length, seed=1)
        val = memory_task(300, length, seed=2)
        X_te, y_te = memory_task(500, length, seed=3)
        print(f"  sequences of length {length}, test accuracy:")
        for name, make in memory_models():
            torch.manual_seed(58)
            model = make()
            fit(model, train_set, val, nn.BCEWithLogitsLoss(), epochs=20, patience=20, batch=64)
            model.eval()
            with torch.no_grad():
                acc = ((model(X_te) > 0).float() == y_te).float().mean()
            show(f"  {name}", r3(acc))
    print("  Ten steps back is easy for everyone. A hundred steps back, the plain")
    print("  RNN and the default GRU are guessing; one bias setting fixes the GRU.")


# ---------------------------------------------------------------------------
# 5. Why: the vanishing gradient
# ---------------------------------------------------------------------------

def section_gradients():
    heading("5. Why: the vanishing gradient")
    X, _ = memory_task(1, 100, seed=4)
    X.requires_grad_(True)
    print("  size of the gradient reaching the input at step 99, 90, 50, 0:")
    for name, make in memory_models():
        torch.manual_seed(58)
        model = make()
        X.grad = None
        model(X).sum().backward()
        g = X.grad[0, :, 0].abs()
        show(f"  {name}", tuple(float(f"{g[t].item():.1e}") for t in (99, 90, 50, 0)))
    rnn, gru = nn.RNN(1, 16), nn.GRU(1, 16)
    show("weights in RNN(1, 16), GRU(1, 16)", (sum(p.numel() for p in rnn.parameters()),
                                               sum(p.numel() for p in gru.parameters())))
    print("  Backprop multiplies by W_h and tanh's slope once per step. A hundred")
    print("  small factors make an astronomically small number, so step 0 never")
    print("  gets a say. A GRU whose gate says 'keep' lets the signal through.")


def main():
    print("=" * 66)
    print("  Lesson 058: Recurrent networks and sequences")
    print("=" * 66)
    counts = hourly_counts()
    section_sequence(counts)
    section_cell()
    section_forecast(counts)
    section_memory()
    section_gradients()
    print()
    print("RNNs read in order and carry a memory. Plain ones forget fast; gated")
    print("ones (GRU, LSTM) remember longer. Lessons 062-063 meet what replaced them.")
    print()


if __name__ == "__main__":
    main()
