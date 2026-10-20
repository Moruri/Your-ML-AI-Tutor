"""
Lesson 049 - A neuron in pure NumPy

Phase 3 starts here. Before we touch PyTorch, we build the smallest piece
of a neural network by hand: one neuron. It takes a few numbers, weighs
them, adds a bias, and squashes the result into a probability. Then we
give it a way to measure its mistakes (a loss) and a way to improve
(gradient descent). No frameworks, no magic, about thirty lines of maths.

Same cleaned September bike-share rides as Phase 2, in rides_clean.csv
next to this file. The neuron's job is lesson 033's: tell casual riders
from members.

Run it with (inside the venv from lesson 013, after `pip install -r
requirements.txt`):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes one picture, output/loss_curve.png, next to this file.
Every random step has a fixed seed, so your numbers will match the README.
It takes a couple of seconds.
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

FEATURES = ["duration_min", "distance_km", "weekend"]

np.set_printoptions(precision=3, suppress=True)


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        print(f"  {label}:")
        for line in value.to_string().splitlines():
            print(f"      {line}")
    else:
        print(f"  {label:<46} -> {value!r}")


def r3(x):
    return round(float(x), 3)


def vec(a):
    """A NumPy array as a short list of rounded numbers, for printing."""
    return [r3(v) + 0.0 for v in np.ravel(a)]  # + 0.0 turns -0.0 into 0.0


def load_rides(path):
    rides = pd.read_csv(path, parse_dates=["started_at"])
    rides["weekend"] = (rides["started_at"].dt.dayofweek >= 5).astype(int)
    rides["casual"] = (rides["rider_type"] == "casual").astype(int)
    return rides


def split_and_scale(rides, seed=0):
    """A 75/25 split by hand, then standardise with the *training* stats."""
    X = rides[FEATURES].to_numpy(dtype=float)
    y = rides["casual"].to_numpy(dtype=float)
    order = np.random.default_rng(seed).permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = order[:cut], order[cut:]
    mean, std = X[tr].mean(axis=0), X[tr].std(axis=0)
    return (X[tr] - mean) / std, (X[te] - mean) / std, y[tr], y[te]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict(X, w, b):
    """The whole neuron: weigh the inputs, add the bias, squash."""
    return sigmoid(X @ w + b)


def log_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)  # never take log(0)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gradients(X, y, w, b):
    """How the loss changes as each weight (and the bias) changes."""
    error = predict(X, w, b) - y          # one number per ride
    return X.T @ error / len(y), float(error.mean())


def train(X, y, lr, steps):
    w, b = np.zeros(X.shape[1]), 0.0
    history = []
    for _ in range(steps):
        history.append(log_loss(predict(X, w, b), y))
        dw, db = gradients(X, y, w, b)
        w, b = w - lr * dw, b - lr * db
    return w, b, history


def accuracy(p, y):
    return r3(np.mean((p >= 0.5) == y))


def auc(p, y):
    """Chance a random casual ride scores above a random member ride."""
    pos, neg = p[y == 1], p[y == 0]
    wins = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return r3(wins + 0.5 * ties)


# ---------------------------------------------------------------------------
# 1. The inputs, on one ruler
# ---------------------------------------------------------------------------

def section_data(rides):
    heading("1. The inputs, on one ruler")
    X_train, X_test, y_train, y_test = split_and_scale(rides)
    show("train rows / test rows", (len(y_train), len(y_test)))
    show("share of casual rides (train)", r3(y_train.mean()))
    show("first three training rows, scaled", [vec(row) for row in X_train[:3]])
    show("column means after scaling", vec(X_train.mean(axis=0)))
    show("column std devs after scaling", vec(X_train.std(axis=0)))
    print("  Each column now has mean 0 and spread 1, measured on the training")
    print("  rows only (lesson 035). Gradient descent is much happier this way.")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. One neuron: weigh, add, squash
# ---------------------------------------------------------------------------

def section_neuron(X_train):
    heading("2. One neuron: weigh, add, squash")
    w = np.array([1.0, 0.0, 0.5])
    b = -1.0
    x = X_train[0]
    z = float(x @ w + b)
    show("one ride's inputs x", vec(x))
    show("weights w, bias b", (w.tolist(), b))
    show("z = x . w + b", r3(z))
    show("sigmoid(z), a probability", r3(sigmoid(z)))
    show("sigmoid at -5, 0, 5", [r3(sigmoid(v)) for v in (-5, 0, 5)])
    show("first five predictions with these weights",
         vec(predict(X_train[:5], w, b)))
    print("  A neuron is a weighted sum plus a bias, pushed through a squashing")
    print("  function. These weights are made up. Next: how wrong are they?")


# ---------------------------------------------------------------------------
# 3. A loss: one number for how wrong we are
# ---------------------------------------------------------------------------

def section_loss(X_train, y_train):
    heading("3. A loss: one number for how wrong we are")
    zeros = np.zeros(X_train.shape[1])
    show("loss with all weights 0 (always says 0.5)",
         r3(log_loss(predict(X_train, zeros, 0.0), y_train)))
    show("ln 2, for comparison", r3(np.log(2)))
    show("loss of the made-up weights from section 2",
         r3(log_loss(predict(X_train, np.array([1.0, 0.0, 0.5]), -1.0), y_train)))
    show("cost of saying 0.9 when the truth is 1", r3(-np.log(0.9)))
    show("cost of saying 0.1 when the truth is 1", r3(-np.log(0.1)))
    show("cost of saying 0.01 when the truth is 1", r3(-np.log(0.01)))
    print("  Log loss punishes confident mistakes hard. Lower is better, and")
    print("  'always say 0.5' scores ln 2: the number to beat.")


# ---------------------------------------------------------------------------
# 4. The gradient, and a check that it's right
# ---------------------------------------------------------------------------

def section_gradient(X_train, y_train):
    heading("4. The gradient, and a check that it's right")
    w, b = np.array([1.0, 0.0, 0.5]), -1.0
    dw, db = gradients(X_train, y_train, w, b)
    show("formula gradient for w", vec(dw))
    show("formula gradient for b", r3(db))
    # Nudge each weight a tiny bit both ways and watch the loss.
    eps = 1e-6
    numeric = []
    for i in range(len(w)):
        up, down = w.copy(), w.copy()
        up[i] += eps
        down[i] -= eps
        numeric.append((log_loss(predict(X_train, up, b), y_train)
                        - log_loss(predict(X_train, down, b), y_train)) / (2 * eps))
    numeric = np.array(numeric)
    show("nudge-and-measure gradient for w", vec(numeric))
    show("biggest difference between the two", float(f"{np.abs(dw - numeric).max():.1e}"))
    print("  The formula is just (prediction - truth) times the input, averaged.")
    print("  Checking it against tiny nudges is the habit that catches bugs.")


# ---------------------------------------------------------------------------
# 5. Gradient descent: take many small steps downhill
# ---------------------------------------------------------------------------

def section_descent(X_train, X_test, y_train, y_test):
    heading("5. Gradient descent: take many small steps downhill")
    w, b, history = train(X_train, y_train, lr=0.5, steps=300)
    for step in (0, 10, 50, 100, 299):
        show(f"loss at step {step}", r3(history[step]))
    show("learned weights (duration, distance, weekend)", vec(w))
    show("learned bias", r3(b))
    p_test = predict(X_test, w, b)
    show("test accuracy", accuracy(p_test, y_test))
    show("test AUC", auc(p_test, y_test))
    show("test accuracy of always saying 'member'", accuracy(np.zeros_like(y_test), y_test))

    OUTPUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, color="tab:blue")
    ax.axhline(np.log(2), color="grey", linestyle="--", label="always say 0.5 (ln 2)")
    ax.set_xlabel("step")
    ax.set_ylabel("training log loss")
    ax.set_title("One neuron learning, step by step")
    ax.legend()
    fig.tight_layout()
    path = OUTPUT / "loss_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    show("picture written to", "output/" + path.name)
    return w, b


# ---------------------------------------------------------------------------
# 6. The learning rate, and a familiar face
# ---------------------------------------------------------------------------

def section_learning_rate(X_train, y_train, w):
    heading("6. The learning rate, and a familiar face")
    for lr in (0.01, 0.5, 5.0, 50.0):
        _, _, hist = train(X_train, y_train, lr=lr, steps=300)
        show(f"lr={lr}: loss after 300 steps", r3(hist[-1]))
    _, _, hist = train(X_train, y_train, lr=50.0, steps=8)
    show("lr=50.0: first eight losses", [r3(v) for v in hist])

    from sklearn.linear_model import LogisticRegression
    sk = LogisticRegression(C=1e6, max_iter=1000).fit(X_train, y_train)
    w_long, _, _ = train(X_train, y_train, lr=0.5, steps=5000)
    show("our weights after 300 steps", vec(w))
    show("our weights after 5000 steps", vec(w_long))
    show("scikit-learn LogisticRegression weights", vec(sk.coef_[0]))
    print("  Too small a step crawls, too big a step overshoots and bounces.")
    print("  Duration and distance are twins (lesson 047), so the last bit of")
    print("  the walk is a long flat valley: the loss stopped moving at 0.45")
    print("  long before the weights settled.")
    print("  And one sigmoid neuron trained on log loss *is* logistic")
    print("  regression: same model as lesson 033, now built with our own hands.")


def main():
    print("=" * 66)
    print("  Lesson 049: A neuron in pure NumPy")
    print("=" * 66)
    rides = load_rides(RIDES_CSV)
    X_train, X_test, y_train, y_test = section_data(rides)
    section_neuron(X_train)
    section_loss(X_train, y_train)
    section_gradient(X_train, y_train)
    w, _ = section_descent(X_train, X_test, y_train, y_test)
    section_learning_rate(X_train, y_train, w)
    print()
    print("A neuron weighs its inputs, adds a bias and squashes. A loss says")
    print("how wrong it is, the gradient says which way is downhill, and")
    print("gradient descent walks there in small steps. Stack these and you")
    print("have a neural network: that's lesson 050.")
    print()


if __name__ == "__main__":
    main()
