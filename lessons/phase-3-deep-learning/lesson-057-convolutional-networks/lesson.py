"""
Lesson 057 - Convolutional networks

So far every network has treated its input as a flat list of numbers.
For an image that throws away the most useful fact about it: which pixels
are next to which. A convolutional network (CNN) keeps the grid, and
looks at it through small windows that slide across the picture.

Two building blocks do almost all the work:

    convolution  - slide a small grid of weights over the image; the same
                   weights look for the same pattern everywhere
    pooling      - shrink the result, keeping the strongest response in
                   each little patch

Today's images are scikit-learn's handwritten digits: 1,797 tiny 8x8
greyscale pictures of 0 to 9. They ship with scikit-learn, so there's
nothing to download.

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files. Every random step has a fixed seed, so your
numbers will match the README. It takes about ten seconds.
"""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.set_num_threads(1)   # same numbers on every machine


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


def ascii_digit(img):
    """Draw an 8x8 image (values 0..1) with characters, darkest = '#'."""
    shades = " .:+#"
    rows = []
    for row in img:
        rows.append("".join(shades[min(int(v * 5), 4)] * 2 for v in row))
    return rows


def load():
    """Digits as a (n, 1, 8, 8) tensor scaled to 0..1, plus labels 0..9."""
    digits = load_digits()
    X = torch.tensor(digits.images / 16.0, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(digits.target, dtype=torch.long)
    return X, y


def splits(X, y):
    tr, va, te = random_split(range(len(y)), [0.6, 0.2, 0.2],
                              generator=torch.Generator().manual_seed(57))
    tr, va, te = (torch.tensor(s.indices) for s in (tr, va, te))
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


# ---------------------------------------------------------------------------
# 1. An image is a grid of numbers
# ---------------------------------------------------------------------------

def section_grid(X, y):
    heading("1. An image is a grid of numbers")
    show("images, shape", tuple(X.shape))
    show("pixel values run from, to", (r3(X.min()), r3(X.max())))
    show("label of image 0", int(y[0]))
    print("  image 0, drawn with characters:")
    for line in ascii_digit(X[0, 0]):
        print("      " + line)
    show("its top row, as numbers", [r3(v) for v in X[0, 0, 0]])
    print("  The shape is (images, channels, height, width). One channel for")
    print("  greyscale; a colour photo would have three.")


# ---------------------------------------------------------------------------
# 2. A convolution, by hand
# ---------------------------------------------------------------------------

def convolve(img, kernel):
    """Slide a k x k kernel over img (no padding): multiply and add at each spot."""
    k = kernel.shape[0]
    h, w = img.shape
    out = np.zeros((h - k + 1, w - k + 1))
    for i in range(h - k + 1):
        for j in range(w - k + 1):
            out[i, j] = (img[i:i + k, j:j + k] * kernel).sum()
    return out


def section_convolution(X):
    heading("2. A convolution, by hand")
    img = X[0, 0].numpy()
    vertical = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)
    mine = convolve(img, vertical)
    show("kernel (finds dark-to-light, left to right)", vertical.astype(int).tolist())
    show("input 8x8 -> output shape", mine.shape)
    show("row 3 of the output", [round(float(v), 2) for v in mine[3]])
    torch_out = F.conv2d(X[:1], torch.tensor(vertical).view(1, 1, 3, 3))
    show("torch's conv2d gives the same numbers", bool(np.allclose(mine, torch_out[0, 0].numpy(), atol=1e-5)))
    padded = F.conv2d(X[:1], torch.tensor(vertical).view(1, 1, 3, 3), padding=1)
    show("with padding=1, output shape", tuple(padded.shape[-2:]))
    print("  Positive where the stroke's left edge is, negative at its right edge,")
    print("  near zero where nothing changes. The same nine weights are used at")
    print("  every position: that sharing is the whole trick.")


# ---------------------------------------------------------------------------
# 3. Pooling, and why it helps
# ---------------------------------------------------------------------------

def section_pooling():
    heading("3. Pooling")
    fmap = torch.tensor([[1., 0., 0., 0.],
                         [0., 0., 3., 0.],
                         [2., 0., 0., 0.],
                         [0., 0., 0., 0.]]).view(1, 1, 4, 4)
    pooled = F.max_pool2d(fmap, 2)
    show("4x4 feature map, row by row", fmap[0, 0].int().tolist())
    show("after max_pool2d(2)", pooled[0, 0].int().tolist())
    moved = torch.roll(fmap, shifts=1, dims=3)
    show("same map, shifted one pixel right", moved[0, 0].int().tolist())
    show("pooled again", F.max_pool2d(moved, 2)[0, 0].int().tolist())
    print("  Each 2x2 patch keeps only its strongest response, so the map is a")
    print("  quarter of the size, and a shift that stays inside a patch changes")
    print("  nothing at all.")


# ---------------------------------------------------------------------------
# 4. Two networks, and their weights
# ---------------------------------------------------------------------------

def mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


def cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),   # 16 x 8 x 8
        nn.MaxPool2d(2),                                         # 16 x 4 x 4
        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),  # 32 x 4 x 4
        nn.MaxPool2d(2),                                         # 32 x 2 x 2
        nn.Flatten(),                                            # 128
        nn.Linear(128, 10),
    )


def count(model):
    return sum(p.numel() for p in model.parameters())


def section_shapes(X):
    heading("4. Two networks, and their weights")
    torch.manual_seed(57)
    m, c = mlp(), cnn()
    show("plain network (flatten, 64 hidden), weights", count(m))
    show("CNN (two conv + pool blocks), weights", count(c))
    show("first conv layer, weight shape", tuple(c[0].weight.shape))
    x = X[:1]
    print("  one image's shape, layer by layer through the CNN:")
    show("  input", tuple(x.shape))
    for layer in c:
        x = layer(x)
        if not isinstance(layer, nn.ReLU):
            show(f"  after {type(layer).__name__}", tuple(x.shape))
    print("  The conv layer's 16 filters are each 3x3, whatever the image size.")


# ---------------------------------------------------------------------------
# 5. Training both on the digits
# ---------------------------------------------------------------------------

def train(model, train_set, val, epochs=40, patience=8):
    """Lesson 056's loop, for 10 classes: cross-entropy, AdamW, early stopping."""
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    loader = DataLoader(TensorDataset(*train_set), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(57))
    best, best_epoch, best_state = math.inf, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
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


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(X).argmax(dim=1)


def acc(model, X, y):
    return r3((predict(model, X) == y).float().mean())


def shift(X, dx):
    """Move every image dx pixels sideways, filling the gap with blank paper."""
    out = torch.roll(X, shifts=dx, dims=3)
    if dx > 0:
        out[..., :dx] = 0
    else:
        out[..., dx:] = 0
    return out


def section_train(X, y):
    heading("5. Training both on the digits")
    train_set, val, (X_te, y_te) = splits(X, y)
    show("train / val / test images", (len(train_set[1]), len(val[1]), len(y_te)))
    models = {}
    for name, make in [("plain network", mlp), ("CNN", cnn)]:
        torch.manual_seed(57)
        model = make()
        best_epoch = train(model, train_set, val)
        models[name] = model
        print(f"  {name}")
        show("  best epoch", best_epoch)
        show("  test accuracy", acc(model, X_te, y_te))
        show("  test accuracy, images shifted 1 pixel right", acc(model, shift(X_te, 1), y_te))
    print("  On the original images the two are close. Nudge the digits one pixel")
    print("  and the plain network suffers more: it learned 'ink at pixel 27', the")
    print("  CNN learned 'a curve here-ish'.")
    return models, (X_te, y_te)


# ---------------------------------------------------------------------------
# 6. What the CNN still gets wrong
# ---------------------------------------------------------------------------

def section_mistakes(models, test):
    heading("6. What the CNN still gets wrong")
    X_te, y_te = test
    pred = predict(models["CNN"], X_te)
    wrong = (pred != y_te).nonzero().flatten()
    show("test images it got wrong", (len(wrong), len(y_te)))
    pairs = sorted((int(y_te[i]), int(pred[i])) for i in wrong.tolist())
    show("each mistake as (true, predicted)", pairs)
    i = wrong[0].item()
    print(f"  first mistake: true label {int(y_te[i])}, predicted {int(pred[i])}")
    for line in ascii_digit(X_te[i, 0]):
        print("      " + line)
    print("  Look at the mistakes before you trust the accuracy. Some of these")
    print("  would fool a person at 8x8 pixels too.")


def main():
    print("=" * 66)
    print("  Lesson 057: Convolutional networks")
    print("=" * 66)
    X, y = load()
    section_grid(X, y)
    section_convolution(X)
    section_pooling()
    section_shapes(X)
    models, test = section_train(X, y)
    section_mistakes(models, test)
    print()
    print("Convolutions share a small set of weights across the whole image, and")
    print("pooling shrinks it. Together they find patterns wherever they appear.")
    print()


if __name__ == "__main__":
    main()
