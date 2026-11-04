"""
Lesson 072 - Diffusion models, intuitively

The image generators everyone talks about (Stable Diffusion and friends) rest
on one surprisingly modest trick. Take real data, add a little noise, again
and again, until nothing is left but static. Then train a network to undo a
single small step of that. Start from fresh static, undo step after step,
and data appears that was never in the training set.

We build the whole thing from scratch, small enough to run on a laptop:
first on 2-D points, where you can watch every step, then on 8x8 handwritten
digits, where we also learn to ask for a particular digit.

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

No downloads: the points come from scikit-learn's make_moons and the digits
from its built-in set. It writes four pictures to output/ and takes about 40
seconds on a laptop CPU.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from torch import nn

torch.set_num_threads(1)              # same numbers on every machine
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
SEED = 72

# The noise schedule: 200 small steps. betas[t] is how much noise step t adds.
T = 200
BETAS = torch.linspace(1e-4, 0.05, T)
ALPHAS = 1 - BETAS
ALPHA_BAR = torch.cumprod(ALPHAS, 0)  # how much of the original survives after t+1 steps


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def r3(x):
    return round(float(x), 3) + 0.0


def add_noise(x0, t, noise):
    """Jump straight to step t: keep sqrt(alpha_bar) of the data, fill the rest with noise."""
    a = ALPHA_BAR[t].view(-1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


class Denoiser(nn.Module):
    """Looks at a noisy point and the step number, and guesses the noise in it."""

    def __init__(self, dim, width, n_labels=0):
        super().__init__()
        torch.manual_seed(SEED)
        self.time = nn.Embedding(T, 32)                    # lesson 059: a learned vector per step
        self.label = nn.Embedding(n_labels + 1, 32) if n_labels else None   # +1 for "no label"
        self.net = nn.Sequential(nn.Linear(dim + 32 + (32 if n_labels else 0), width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, dim))

    def forward(self, x, t, y=None):
        parts = [x, self.time(t)] + ([self.label(y)] if self.label is not None else [])
        return self.net(torch.cat(parts, 1))


def train(model, X, iters, y=None, n_labels=0, batch=256):
    """The whole training loop: noise a real example, ask the model to name the noise."""
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    losses = []
    for _ in range(iters):
        idx = torch.randint(len(X), (batch,))
        x0 = X[idx]
        t = torch.randint(T, (batch,))
        noise = torch.randn_like(x0)
        labels = None
        if y is not None:
            labels = y[idx].clone()
            labels[torch.rand(batch) < 0.15] = n_labels         # sometimes hide the label
        loss = ((model(add_noise(x0, t, noise), t, labels) - noise) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return model.eval(), losses


@torch.no_grad()
def predict_noise(model, x, t, label=None, n_labels=0, guidance=0.0):
    tt = torch.full((len(x),), t)
    if label is None:
        return model(x, tt)
    eps = model(x, tt, label)
    if guidance:
        # Guidance: compare "with the label" to "without it" and lean further
        # in the direction the label pushes.
        free = model(x, tt, torch.full_like(label, n_labels))
        eps = eps + guidance * (eps - free)
    return eps


@torch.no_grad()
def sample(model, n, dim, steps=T, label=None, n_labels=0, guidance=0.0, snapshots=()):
    """Start from pure static and remove noise, one step at a time.

    With steps=T this is the original recipe (DDPM): every step also adds a
    little fresh noise back. With fewer steps it skips along the schedule
    without the fresh noise (DDIM), which is how fast samplers work.
    """
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(n, dim, generator=g)
    taken = {}
    ts = np.linspace(T - 1, 0, steps).round().astype(int)
    for i, t in enumerate(ts):
        eps = predict_noise(model, x, int(t), label, n_labels, guidance)
        a_bar = ALPHA_BAR[t]
        if steps == T:
            mean = (x - BETAS[t] / (1 - a_bar).sqrt() * eps) / ALPHAS[t].sqrt()
            x = mean + (BETAS[t].sqrt() * torch.randn(x.shape, generator=g) if t > 0 else 0)
        else:
            x0_guess = (x - (1 - a_bar).sqrt() * eps) / a_bar.sqrt()
            a_next = ALPHA_BAR[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
            x = a_next.sqrt() * x0_guess + (1 - a_next).sqrt() * eps
        if int(t) in snapshots:
            taken[int(t)] = x.clone()
    return x, taken


# ---------------------------------------------------------------------------
# 1. Adding noise, step by step
# ---------------------------------------------------------------------------

def section_noising(digits):
    heading("1. Adding noise, step by step")
    torch.manual_seed(SEED)
    x0 = digits[:300]
    noise = torch.randn_like(x0)
    show("steps in the schedule", T)
    show("noise added at the first and last step", (round(float(BETAS[0]), 4), r3(BETAS[-1])))
    fig, axes = plt.subplots(1, 6, figsize=(7, 1.6))
    for ax, t in zip(axes, (0, 10, 30, 60, 100, 199)):
        xt = add_noise(x0, torch.full((len(x0),), t), noise)
        corr = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(x0.numpy(), xt.numpy())])
        show(f"after step {t:>3}: original kept, likeness", (r3(ALPHA_BAR[t].sqrt()), r3(corr)))
        ax.imshow(xt[0].reshape(8, 8), cmap="gray_r", vmin=-1.5, vmax=1.5)
        ax.set_title(f"step {t}", fontsize=8)
        ax.axis("off")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "noising.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    show("saved", "output/noising.png")
    print("  Each step keeps most of the picture and adds a little static. By the")
    print("  last step nothing of the digit survives: it's pure noise.")


# ---------------------------------------------------------------------------
# 2. The learning task: guess the noise
# 3. From static to data, one step at a time
# 4. Fewer, bigger steps
# ---------------------------------------------------------------------------

def moons(n, seed):
    X, _ = make_moons(n_samples=n, noise=0.05, random_state=seed)
    return torch.tensor((X - [0.5, 0.25]) / [0.9, 0.5], dtype=torch.float32)   # roughly centred, spread about 1


def section_task(train_pts):
    heading("2. The learning task: guess the noise")
    model, losses = train(Denoiser(2, 128), train_pts, iters=8000, batch=512)
    show("training points (2-D moons)", tuple(train_pts.shape))
    show("loss if you always guess 'no noise'", 1.0)
    show("loss, first 100 steps (average)", r3(np.mean(losses[:100])))
    show("loss, last 100 steps (average)", r3(np.mean(losses[-100:])))
    torch.manual_seed(SEED)
    x0 = train_pts[:1000]
    noise = torch.randn_like(x0)
    for t in (5, 50, 150):
        tt = torch.full((len(x0),), t)
        with torch.no_grad():
            err = ((model(add_noise(x0, tt, noise), tt) - noise) ** 2).mean()
        show(f"loss at step {t}", r3(err))
    print("  A plain regression task, nothing more. Late steps are easy (the point")
    print("  is nearly all noise); early ones are hard (the noise is tiny).")
    return model


def near_share(points, reference, within=0.05):
    """Share of points within 0.05 of some real point: how many landed on a moon."""
    nn_ = NearestNeighbors(n_neighbors=1).fit(reference.numpy())
    return r3((nn_.kneighbors(points.numpy())[0] < within).mean())


def section_reverse(model, train_pts, test_pts):
    heading("3. From static to data, one step at a time")
    snaps = (199, 150, 100, 50, 0)
    pts, taken = sample(model, 1000, 2, snapshots=snaps)
    static = torch.randn(1000, 2, generator=torch.Generator().manual_seed(1))
    show("pure static: share on a moon", near_share(static, train_pts))
    show("after all 200 steps: share on a moon", near_share(pts, train_pts))
    show("real held-out points, for comparison", near_share(test_pts, train_pts))
    show("share of new points on the left moon's side", r3((pts[:, 0] < 0).float().mean()))
    fig, axes = plt.subplots(1, len(snaps), figsize=(12, 2.6), sharex=True, sharey=True)
    for ax, t in zip(axes, snaps):
        p = taken[t].numpy()
        ax.scatter(p[:, 0], p[:, 1], s=2)
        ax.set_title(f"after step {t}" if t < 199 else "start: static", fontsize=9)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    fig.savefig(OUT / "moons.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    show("saved", "output/moons.png")
    print("  Each step the model guesses the noise and we remove a bit of it.")
    print("  Two hundred small cleanups turn a blob of static into two moons.")


def section_fewer_steps(model, train_pts):
    heading("4. Fewer, bigger steps")
    for steps in (200, 50, 10, 3, 1):
        start = time.perf_counter()
        pts, _ = sample(model, 1000, 2, steps=steps)
        ms = (time.perf_counter() - start) * 1000
        show(f"{steps:>3} steps: share on a moon, time in ms", (near_share(pts, train_pts), round(ms)))
    print("  Skip along the schedule and you need far fewer model calls. Fifty")
    print("  is as good as 200; below ten the shortcut starts to fall apart.")


# ---------------------------------------------------------------------------
# 5. Diffusion on real digits
# 6. Asking for a particular digit
# ---------------------------------------------------------------------------

def save_grid(images, path, rows, title):
    cols = len(images) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 0.9 + 0.4))
    for ax, img in zip(np.ravel(axes), images):
        ax.imshow(np.asarray(img).reshape(8, 8), cmap="gray_r", vmin=-1, vmax=1)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def section_digits(train_x, train_y, test_x):
    heading("5. Diffusion on real digits")
    model, losses = train(Denoiser(64, 256, n_labels=10), train_x, iters=8000, y=train_y, n_labels=10)
    show("loss, last 100 steps (average)", r3(np.mean(losses[-100:])))
    nn_ = NearestNeighbors(n_neighbors=1).fit(train_x.numpy())
    gap = lambda imgs: nn_.kneighbors(imgs.numpy())[0].ravel()
    static = torch.randn(len(test_x), 64, generator=torch.Generator().manual_seed(SEED)).clamp(-1, 1)
    imgs, _ = sample(model, 500, 64, steps=50, label=torch.full((500,), 10), n_labels=10)
    imgs = imgs.clamp(-1, 1)
    show("distance to nearest training digit: static", r3(gap(static).mean()))
    show("  new digits (no label asked)", r3(gap(imgs).mean()))
    show("  real held-out digits", r3(gap(test_x).mean()))
    show("closest any new digit gets to a training one", r3(gap(imgs).min()))
    show("closest any held-out digit gets", r3(gap(test_x).min()))
    save_grid(imgs[:32], OUT / "digits.png", rows=4, title="32 digits drawn out of static")
    show("saved", "output/digits.png")
    print("  Same recipe, 64 numbers instead of 2. The new digits sit much nearer")
    print("  real ones than static does, and none is a copy of a training digit.")
    return model, gap


def section_guidance(model, gap, judge, judge_accuracy):
    heading("6. Asking for a particular digit")
    show("judge's accuracy on real held-out digits", judge_accuracy)
    wanted = torch.arange(10).repeat(50)
    for g in (0.0, 1.0, 3.0):
        imgs, _ = sample(model, len(wanted), 64, steps=50, label=wanted, n_labels=10, guidance=g)
        imgs = imgs.clamp(-1, 1)
        got = judge.predict(((imgs.numpy() + 1) / 2))
        show(f"guidance {g}: right digit, distance to real", (r3((got == wanted.numpy()).mean()), r3(gap(imgs).mean())))
    imgs, _ = sample(model, 20, 64, steps=50, label=torch.arange(10).repeat(2), n_labels=10, guidance=1.0)
    save_grid(imgs.clamp(-1, 1), OUT / "asked_for.png", rows=2, title="asked for 0 to 9, twice (guidance 1)")
    show("saved", "output/asked_for.png")
    print("  The label is a hint fed in at every step. A little guidance makes the")
    print("  model listen harder; too much overcooks the picture. Swap the label")
    print("  for a sentence and you have the core of a text-to-image model.")


def main():
    print("=" * 66)
    print("  Lesson 072: Diffusion models, intuitively")
    print("=" * 66)
    d = load_digits()
    X = (d.data / 8.0 - 1.0).astype(np.float32)                        # pixels from -1 to 1
    X_tr, X_te, y_tr, y_te = train_test_split(X, d.target, test_size=0.25, random_state=SEED, stratify=d.target)
    # A separate digit classifier, used only to check which digit we got.
    judge = SVC().fit((X_tr + 1) / 2, y_tr)
    train_x, test_x = torch.tensor(X_tr), torch.tensor(X_te)
    section_noising(train_x)
    train_pts, test_pts = moons(4000, SEED), moons(1000, SEED + 1)
    model = section_task(train_pts)
    section_reverse(model, train_pts, test_pts)
    section_fewer_steps(model, train_pts)
    digit_model, gap = section_digits(train_x, torch.tensor(y_tr), test_x)
    section_guidance(digit_model, gap, judge, r3(judge.score((X_te + 1) / 2, y_te)))
    print()
    print("Add noise until nothing is left, teach a network to take one small")
    print("step back, then walk all the way back from fresh static.")
    print()


if __name__ == "__main__":
    main()
