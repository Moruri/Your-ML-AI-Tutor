"""
Lesson 060 - Mini-project: train and debug a real network

Phase 3 ends with a real job: a network that tells members from casual
riders, built the way you'd build one at work. Baselines first, sanity
checks before the long run, a log of every epoch, checkpoints you can
reload, and the test set opened exactly once at the end.

Along the way we break a training run on purpose and use the log to find
out why, because most of deep learning in practice is debugging.

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

It writes output/train_log.csv and output/best.pt next to this script.
Every random step has a fixed seed, so your numbers will match the README.
It takes a few seconds.
"""

import copy
import csv
import math
from pathlib import Path

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)   # same numbers on every machine

HERE = Path(__file__).resolve().parent
RIDES_CSV = HERE / "rides_clean.csv"
OUTPUT = HERE / "output"
NUMERIC = ["distance_km", "duration_min", "temp_c", "rain_mm"]


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
# 1. The task, the split and the bar to beat
# ---------------------------------------------------------------------------

def load_data():
    """Numeric columns, category codes and labels as tensors, plus the split."""
    rides = pd.read_csv(RIDES_CSV, parse_dates=["started_at"])
    num = torch.tensor(rides[NUMERIC].to_numpy(), dtype=torch.float32)
    num = torch.cat([num, torch.tensor((rides["bike_type"] == "electric").to_numpy(),
                                       dtype=torch.float32).unsqueeze(1)], dim=1)
    hour = torch.tensor(rides["started_at"].dt.hour.to_numpy()).long()
    day = torch.tensor(rides["started_at"].dt.dayofweek.to_numpy()).long()
    y = torch.tensor((rides["rider_type"] == "member").to_numpy(), dtype=torch.float32)
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(60))
    split = {"train": order[:824], "val": order[824:1000], "test": order[1000:]}
    return (num, hour, day, y), split


def subset(data, idx):
    return tuple(t[idx] for t in data)


def accuracy(logits, y):
    return r3(((logits > 0).float() == y).float().mean())


def section_task(data, split):
    heading("1. The task, the split and the bar to beat")
    num, hour, day, y = data
    show("rides, input columns", (len(y), len(NUMERIC) + 1 + 2))
    show("train / val / test", tuple(len(v) for v in split.values()))
    show("share of members in train", r3(y[split["train"]].mean()))
    show("baseline: always 'member', val accuracy", r3(y[split["val"]].mean()))
    # A plain logistic regression on scaled numbers, hour and day as one-hot.
    def table(idx):
        mean, std = num[split["train"]].mean(0), num[split["train"]].std(0)
        onehot = torch.cat([nn.functional.one_hot(hour[idx], 24),
                            nn.functional.one_hot(day[idx], 7)], dim=1)
        return torch.cat([(num[idx] - mean) / std, onehot], dim=1).numpy()
    logreg = LogisticRegression(max_iter=1000).fit(table(split["train"]), y[split["train"]].numpy())
    show("baseline: logistic regression, val accuracy",
         round(logreg.score(table(split["val"]), y[split["val"]].numpy()), 3))
    print("  Two bars to beat before any network earns its place: guessing the")
    print("  common class, and the simplest sensible model.")
    return logreg, table


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class RiderNet(nn.Module):
    """Scaled numbers plus small hour and day embeddings (lesson 059), into an MLP."""

    def __init__(self, mean, std, hidden=32, dropout=0.1):
        super().__init__()
        # Keep the scaling inside the model, so a checkpoint carries it too.
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.hour = nn.Embedding(24, 3)
        self.day = nn.Embedding(7, 2)
        self.body = nn.Sequential(
            nn.Linear(len(mean) + 3 + 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, num, hour, day):
        x = torch.cat([(num - self.mean) / self.std, self.hour(hour), self.day(day)], dim=1)
        return self.body(x).squeeze(-1)


def make_model(data, split, scale=True, **kw):
    num = data[0][split["train"]]
    if scale:
        mean, std = num.mean(0), num.std(0)
    else:
        mean, std = torch.zeros(num.shape[1]), torch.ones(num.shape[1])
    torch.manual_seed(60)
    return RiderNet(mean, std, **kw)


# ---------------------------------------------------------------------------
# 2. Sanity checks before the long run
# ---------------------------------------------------------------------------

def section_checks(data, split):
    heading("2. Sanity checks before the long run")
    model = make_model(data, split)
    loss_fn = nn.BCEWithLogitsLoss()
    num, hour, day, y = subset(data, split["train"])
    with torch.no_grad():
        start = loss_fn(model(num, hour, day), y).item()
    show("output shape for 824 rides", tuple(model(num, hour, day).shape))
    show("loss before training, and ln(2)", (r3(start), r3(math.log(2))))
    # Can it memorise 16 rides? If not, something is broken.
    tiny = subset(data, split["train"][:16])
    model.eval()                   # no dropout for this check
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in range(300):
        opt.zero_grad()
        loss = loss_fn(model(*tiny[:3]), tiny[3])
        loss.backward()
        opt.step()
    show("loss after 300 steps on 16 rides", r3(loss))
    show("trainable weights", sum(p.numel() for p in model.parameters()))
    print("  A fresh binary classifier should start near ln(2), unsure about")
    print("  everything, and a working model can memorise a tiny batch. Both")
    print("  pass, so the plumbing works. Now the real run.")


# ---------------------------------------------------------------------------
# Training, with a log
# ---------------------------------------------------------------------------

def grad_norm(model):
    return math.sqrt(sum(p.grad.pow(2).sum().item() for p in model.parameters()
                         if p.grad is not None))


def train(model, data, split, lr=0.003, epochs=200, patience=20,
          log_path=None, ckpt_path=None, misalign_labels=False):
    """AdamW, early stopping on validation loss, one log row per epoch."""
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_set = list(subset(data, split["train"]))
    if misalign_labels:   # the bug: labels shuffled separately from the inputs
        g = torch.Generator().manual_seed(1)
        train_set[3] = train_set[3][torch.randperm(len(train_set[3]), generator=g)]
    loader = DataLoader(TensorDataset(*train_set), batch_size=64,
                        shuffle=True, generator=torch.Generator().manual_seed(60))
    val = subset(data, split["val"])
    log, best, best_epoch, best_state = [], math.inf, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total, norms = 0.0, []
        for num, hour, day, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(num, hour, day), y)
            loss.backward()
            norms.append(grad_norm(model))
            opt.step()
            total += loss.item() * len(y)
        model.eval()
        with torch.no_grad():
            logits = model(*val[:3])
            val_loss = loss_fn(logits, val[3]).item()
        row = {"epoch": epoch, "train_loss": r3(total / len(split["train"])),
               "val_loss": r3(val_loss), "val_acc": accuracy(logits, val[3]),
               "grad_norm": r3(max(norms))}
        log.append(row)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            if ckpt_path:
                torch.save({"epoch": epoch, "model": best_state,
                            "optimizer": opt.state_dict(), "val_loss": val_loss}, ckpt_path)
        elif epoch - best_epoch >= patience:
            break
    if log_path:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log[0]))
            writer.writeheader()
            writer.writerows(log)
    model.load_state_dict(best_state)
    return log, best_epoch


def print_log(log, epochs):
    print("  epoch  train_loss  val_loss  val_acc  grad_norm")
    for row in log:
        if row["epoch"] in epochs:
            print(f"  {row['epoch']:>5}  {row['train_loss']:>10}  {row['val_loss']:>8}"
                  f"  {row['val_acc']:>7}  {row['grad_norm']:>9}")


# ---------------------------------------------------------------------------
# 3. A run that goes wrong, and reading the log
# ---------------------------------------------------------------------------

def section_broken(data, split):
    heading("3. A run that goes wrong, and reading the log")
    model = make_model(data, split)
    log, best_epoch = train(model, data, split, epochs=30, patience=30, misalign_labels=True)
    print("  a run with a hidden bug:")
    print_log(log, {1, 2, 5, 10, 20, 30})
    p = data[3][split["train"]].mean()
    show("loss of just guessing the 0.697 share", r3(-(p * p.log() + (1 - p) * (1 - p).log())))
    show("val accuracy of always saying 'member'", r3(data[3][split["val"]].mean()))
    print("  The code runs, nothing crashes, and the model learns nothing. Train")
    print("  loss sits at the loss of guessing the class share, and validation")
    print("  accuracy at the 'always member' baseline. The bug: the labels were")
    print("  shuffled separately from the inputs, so they no longer match.")


# ---------------------------------------------------------------------------
# 4. The real run: logging and checkpoints
# ---------------------------------------------------------------------------

def section_real(data, split):
    heading("4. The real run: logging and checkpoints")
    OUTPUT.mkdir(exist_ok=True)
    log_path, ckpt_path = OUTPUT / "train_log.csv", OUTPUT / "best.pt"
    model = make_model(data, split)
    log, best_epoch = train(model, data, split, log_path=log_path, ckpt_path=ckpt_path)
    print("  the same run with the labels lined up:")
    print_log(log, {1, 2, 5, 10, 20, best_epoch, len(log)})
    show("epochs run, best epoch", (len(log), best_epoch))
    show("files written", sorted(p.name for p in OUTPUT.iterdir()))
    print("  Train loss keeps falling; validation loss bottoms out and turns")
    print("  up. Early stopping kept the best epoch, and so did the checkpoint.")
    return model


# ---------------------------------------------------------------------------
# 5. Reload, then open the test set once
# ---------------------------------------------------------------------------

def section_test(data, split, trained, logreg, table):
    heading("5. Reload, then open the test set once")
    ckpt = torch.load(OUTPUT / "best.pt", weights_only=True)
    fresh = make_model(data, split)
    fresh.load_state_dict(ckpt["model"])
    fresh.eval()
    trained.eval()
    val = subset(data, split["val"])
    with torch.no_grad():
        same = torch.equal(fresh(*val[:3]), trained(*val[:3]))
    show("checkpoint: epoch, val loss", (ckpt["epoch"], r3(ckpt["val_loss"])))
    show("reloaded model gives identical outputs", same)
    num, hour, day, y = subset(data, split["test"])
    with torch.no_grad():
        logits = fresh(num, hour, day)
    pred = (logits > 0).float()
    show("test accuracy: always 'member'", r3(y.mean()))
    show("test accuracy: logistic regression",
         round(logreg.score(table(split["test"]), y.numpy()), 3))
    show("test accuracy: the network", accuracy(logits, y))
    show("casuals caught, members caught", (f"{int(((pred == 0) & (y == 0)).sum())} of {int((y == 0).sum())}",
                                            f"{int(((pred == 1) & (y == 1)).sum())} of {int((y == 1).sum())}"))
    wrong = pred != y
    show("median distance (km): wrong, right", (r3(num[wrong, 0].median()), r3(num[~wrong, 0].median())))
    print("  The test set was used once, after every choice was made. That")
    print("  number is the honest one to report.")


def main():
    print("=" * 66)
    print("  Lesson 060: Mini-project, train and debug a real network")
    print("=" * 66)
    data, split = load_data()
    logreg, table = section_task(data, split)
    section_checks(data, split)
    section_broken(data, split)
    model = section_real(data, split)
    section_test(data, split, model, logreg, table)
    print()
    print("Baselines, sanity checks, a log, checkpoints, one look at the test")
    print("set. That's the routine for every network from here on.")
    print()


if __name__ == "__main__":
    main()
