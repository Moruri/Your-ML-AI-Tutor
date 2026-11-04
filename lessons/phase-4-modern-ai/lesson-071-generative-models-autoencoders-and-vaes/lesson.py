"""
Lesson 071 - Generative models: autoencoders and VAEs

Until now every model has looked at data and said something about it: a label,
a number, a box. Today the model makes data. We squeeze 8x8 handwritten digits
(64 numbers each) through a narrow middle of 8 numbers, and ask a network to rebuild them.
That's an autoencoder. Then we change one idea and get a variational
autoencoder (VAE), which can invent new digits nobody ever wrote.

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

It uses scikit-learn's built-in digits (no download), writes four pictures to
output/, and takes about 20 seconds on a laptop CPU.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from torch import nn

torch.set_num_threads(1)              # same numbers on every machine
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
SEED = 71
LATENT = 8                            # the narrow middle: 8 numbers per digit, down from 64


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def r3(x):
    return round(float(x), 3) + 0.0


def load_data():
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)          # pixels from 0 (white) to 1 (black)
    return train_test_split(X, digits.target, test_size=0.25, random_state=SEED, stratify=digits.target)


def mse(a, b):
    return r3(((np.asarray(a) - np.asarray(b)) ** 2).mean())


def save_grid(images, path, rows, title):
    cols = len(images) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 0.9 + 0.4))
    for ax, img in zip(np.ravel(axes), images):
        ax.imshow(np.asarray(img).reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# The two networks: an encoder squeezes, a decoder rebuilds
# ---------------------------------------------------------------------------

class AutoEncoder(nn.Module):
    def __init__(self, variational=False, latent=LATENT):
        super().__init__()
        torch.manual_seed(SEED)                           # same starting weights every run
        self.variational = variational
        self.encoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.to_code = nn.Linear(64, latent)
        self.to_logvar = nn.Linear(64, latent)            # only the VAE uses this
        self.decoder = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        return self.to_code(h), self.to_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.variational and self.training:
            # The reparameterisation trick: a random point near mu, written so
            # gradients can still flow back through mu and the spread.
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return self.decoder(z), mu, logvar


def train(model, X, epochs=150):
    torch.manual_seed(SEED)
    X = torch.tensor(X)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(X)).split(128):
            x = X[idx]
            rebuilt, mu, logvar = model(x)
            # How far the rebuild is from the real digit, pixel by pixel. Binary
            # cross-entropy (lesson 033's log loss) suits pixels between 0 and 1.
            loss = nn.functional.binary_cross_entropy(rebuilt, x, reduction="none").sum(1).mean()
            if model.variational:
                # KL term: pull every code towards a plain bell curve around 0.
                kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
                loss = loss + kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def codes_of(model, X):
    with torch.no_grad():
        return model.encode(torch.tensor(X))[0].numpy()


def decode(model, Z):
    with torch.no_grad():
        return model.decoder(torch.tensor(np.asarray(Z, dtype=np.float32))).numpy()


def rebuild(model, X):
    with torch.no_grad():
        return model(torch.tensor(X))[0].numpy()


# ---------------------------------------------------------------------------
# 1. Squeeze and rebuild: what PCA already does
# ---------------------------------------------------------------------------

def section_pca(X_tr, X_te):
    heading("1. Squeeze and rebuild: what PCA already does")
    show("numbers per digit image", X_tr.shape[1])
    show("rebuild error if we always guess the average", mse(X_te, np.broadcast_to(X_tr.mean(0), X_te.shape)))
    for k in (2, 10, 30):
        pca = PCA(n_components=k, random_state=SEED).fit(X_tr)
        show(f"PCA, squeezed to {k} numbers, rebuild error", mse(X_te, pca.inverse_transform(pca.transform(X_te))))
    print("  Squeeze 64 numbers into a few, rebuild, and measure what was lost.")
    print("  PCA (lesson 045) can only squeeze along straight lines.")


# ---------------------------------------------------------------------------
# 2. An autoencoder: the same idea, learned and bendy
# ---------------------------------------------------------------------------

def section_autoencoder(X_tr, X_te):
    heading("2. An autoencoder: the same idea, learned and bendy")
    ae = train(AutoEncoder(), X_tr)
    pca = PCA(n_components=LATENT, random_state=SEED).fit(X_tr)
    show("parameters in the autoencoder", sum(p.numel() for p in ae.parameters()))
    show(f"PCA, {LATENT} numbers, rebuild error (test)", mse(X_te, pca.inverse_transform(pca.transform(X_te))))
    show(f"autoencoder, {LATENT} numbers, rebuild error (test)", mse(X_te, rebuild(ae, X_te)))
    show("autoencoder, rebuild error (train)", mse(X_tr, rebuild(ae, X_tr)))
    OUT.mkdir(exist_ok=True)
    save_grid(np.concatenate([X_te[:8], rebuild(ae, X_te[:8])]), OUT / "rebuilds.png", rows=2,
              title=f"top: real digits, bottom: rebuilt from {LATENT} numbers")
    show("saved", "output/rebuilds.png")
    print("  Encoder squeezes, decoder rebuilds, and the only teacher is the")
    print("  input itself. No labels anywhere.")
    return ae


# ---------------------------------------------------------------------------
# 3. The latent space: where the codes live
# ---------------------------------------------------------------------------

def section_latent(X_tr, y_tr, X_te, y_te):
    heading("3. The latent space: where the codes live")
    # Two numbers per digit is too few to rebuild well, but we can draw it.
    flat = {name: train(AutoEncoder(variational=v, latent=2), X_tr) for name, v in (("autoencoder", False), ("VAE", True))}
    z = codes_of(flat["autoencoder"], X_te)
    show("one digit's 2-number code", tuple(r3(v) for v in z[0]))
    show("its real label", int(y_te[0]))
    for name, model in flat.items():
        clf = LogisticRegression(max_iter=2000).fit(codes_of(model, X_tr), y_tr)
        show(f"label guessed from 2-number {name} codes", r3(clf.score(codes_of(model, X_te), y_te)))
    za, zv = codes_of(flat["autoencoder"], X_tr), codes_of(flat["VAE"], X_tr)
    show("autoencoder codes: centre and spread", (tuple(r3(v) for v in za.mean(0)), tuple(r3(v) for v in za.std(0))))
    show("VAE codes: centre and spread", (tuple(r3(v) for v in zv.mean(0)), tuple(r3(v) for v in zv.std(0))))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (name, model) in zip(axes, flat.items()):
        codes = codes_of(model, X_te)
        pts = ax.scatter(codes[:, 0], codes[:, 1], c=y_te, cmap="tab10", s=10)
        ax.set_title(f"{name}: test digits, coloured by label")
    fig.colorbar(pts, ax=axes, ticks=range(10))
    fig.savefig(OUT / "latent.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    show("saved", "output/latent.png")
    print("  Nobody told either model which digit is which, yet similar digits")
    print("  land near each other. The VAE keeps its codes in a tidy ball round 0.")


# ---------------------------------------------------------------------------
# 4. Generating with an autoencoder: the empty gaps
# 5. The VAE: making the space smooth
# ---------------------------------------------------------------------------

def judge(judge_clf, images):
    """How digit-like are these images? Share a strict judge calls with 90%+ confidence."""
    return r3((judge_clf.predict_proba(images).max(1) >= 0.9).mean())


def section_sample_ae(ae, judge_clf, X_tr, X_te):
    heading("4. Generating with an autoencoder: the empty gaps")
    show("real test digits the judge is 90%+ sure of", judge(judge_clf, X_te))
    show("rebuilt test digits, 90%+ sure", judge(judge_clf, rebuild(ae, X_te)))
    rng = np.random.default_rng(SEED)
    z = codes_of(ae, X_tr)
    near = rng.standard_normal((500, LATENT)) * z.std(0) + z.mean(0)
    show("500 new codes, centred and spread like its own", judge(judge_clf, decode(ae, near)))
    print("  Decode a code it has seen and you get a digit. Make up a new code")
    print("  and you often land in a gap it never learned, and get a smudge.")


def section_vae(ae, vae, judge_clf, X_te):
    heading("5. The VAE: making the space smooth")
    rng = np.random.default_rng(SEED)
    show("rebuild error, VAE (test)", mse(X_te, rebuild(vae, X_te)))
    show("same, plain autoencoder (section 2)", mse(X_te, rebuild(ae, X_te)))
    samples = decode(vae, rng.standard_normal((500, LATENT)))
    show("500 new codes from a bell curve, 90%+ sure", judge(judge_clf, samples))
    show("digits the judge thinks it drew (0-9 counts)", np.bincount(judge_clf.predict(samples), minlength=10).tolist())
    save_grid(samples[:32], OUT / "new_digits.png", rows=4, title="32 digits nobody wrote: VAE samples")
    show("saved", "output/new_digits.png")
    print("  Sample a point from a plain bell curve, decode it, and get a digit")
    print("  nobody wrote. The price: rebuilds come out a bit blurrier.")


# ---------------------------------------------------------------------------
# 6. Walking between two digits
# ---------------------------------------------------------------------------

def section_walk(vae, judge_clf, X_te, y_te):
    heading("6. Walking between two digits")
    a, b = X_te[np.where(y_te == 0)[0][0]], X_te[np.where(y_te == 1)[0][0]]
    steps = np.linspace(0, 1, 8)[:, None]
    za, zb = codes_of(vae, np.stack([a, b]))
    walk = decode(vae, (1 - steps) * za + steps * zb)
    pixels = (1 - steps) * a + steps * b
    show("plain pixel blend: judge's reading", judge_clf.predict(pixels).tolist())
    show("VAE walk: judge's reading", judge_clf.predict(walk).tolist())
    save_grid(np.concatenate([pixels, walk]), OUT / "walk.png", rows=2,
              title="top: fading pixels from 0 to 1; bottom: walking the VAE's latent space")
    show("saved", "output/walk.png")
    print("  Blending pixels gives a double exposure: a 0 and a 1 at once. The")
    print("  VAE walk changes shape instead, one believable digit at a time.")


def main():
    print("=" * 66)
    print("  Lesson 071: Generative models: autoencoders and VAEs")
    print("=" * 66)
    X_tr, X_te, y_tr, y_te = load_data()
    # A separate digit classifier plays judge: how digit-like is a made-up image?
    judge_clf = CalibratedClassifierCV(SVC(), ensemble=False).fit(X_tr, y_tr)
    section_pca(X_tr, X_te)
    ae = section_autoencoder(X_tr, X_te)
    section_latent(X_tr, y_tr, X_te, y_te)
    section_sample_ae(ae, judge_clf, X_tr, X_te)
    vae = train(AutoEncoder(variational=True), X_tr)
    section_vae(ae, vae, judge_clf, X_te)
    section_walk(vae, judge_clf, X_te, y_te)
    print()
    print("An autoencoder learns to squeeze and rebuild. A VAE also keeps its")
    print("squeezed space smooth, so any point you pick decodes into something real.")
    print()


if __name__ == "__main__":
    main()
