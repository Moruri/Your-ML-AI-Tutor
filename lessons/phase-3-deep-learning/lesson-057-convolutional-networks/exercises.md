# Lesson 057 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-3-deep-learning/lesson-057-convolutional-networks/lesson.py my_lesson_057.py
python my_lesson_057.py
```

The digits ship with scikit-learn, so the copy needs no data file.
`load`, `splits`, `mlp`, `cnn`, `count`, `train`, `predict`, `acc`,
`shift`, `convolve`, `r3` and `show` are there to reuse:

```python
X, y = load()
train_set, val, (X_te, y_te) = splits(X, y)
```

---

## 1. Output sizes

A 28x28 greyscale image goes through `nn.Conv2d(1, 8, kernel_size=5)`
(no padding), then `nn.MaxPool2d(2)`. What's the shape after each
layer, for one image?

<details>
<summary>Check yourself</summary>

After the conv: `(1, 8, 24, 24)`. A 5-wide kernel fits in
$28 - 5 + 1 = 24$ positions each way, and there are 8 kernels. After
pooling: `(1, 8, 12, 12)`. The rule without padding is
$\text{out} = \text{in} - k + 1$; `padding=2` would keep it at 28.

</details>

## 2. Counting weights

How many weights (including biases) does `nn.Conv2d(16, 32,
kernel_size=3)` have? Check with `count(...)`.

<details>
<summary>Check yourself</summary>

4,640. Each of the 32 kernels looks at all 16 input channels through a
3x3 window: $16 \times 3 \times 3 = 144$ weights, plus one bias, so
145. Times 32 is 4,640. The image size doesn't appear anywhere.

</details>

## 3. A kernel that does nothing

What does this kernel do to an image (with `padding=1`)?

```
[[0, 0, 0],
 [0, 1, 0],
 [0, 0, 0]]
```

And what about the same thing with the 1 moved to the middle of the
left column?

<details>
<summary>Check yourself</summary>

The first copies the image exactly: each output is just the pixel under
the centre. The second copies the pixel to the *left* of centre, so the
whole image moves one pixel right. Convolutions can blur, sharpen, find
edges and shift, all with nine numbers. A CNN picks whichever numbers
help.

</details>

## 4. Hands-on: teach the CNN to ignore shifts

Lesson 056's data augmentation, for images. Write a `jitter` function
that shifts each batch by -1, 0 or +1 pixels at random, using the
script's `shift`:

```python
def jitter(xb, gen=torch.Generator().manual_seed(57)):
    d = torch.randint(-1, 2, (1,), generator=gen).item()
    return shift(xb, d) if d else xb
```

Copy `train` into `train_aug(model, train_set, val, augment)` and apply
`xb = augment(xb)` to every batch, as lesson 056's loop did. Train both
`mlp()` and `cnn()` with seed 57, and report test accuracy on the
original images and on images shifted one pixel right and one pixel
left. Predict first: which model gains more?

<details>
<summary>Expected results</summary>

```
                 best epoch   original   shifted right   shifted left
plain network        38         0.967        0.944           0.955
CNN                  26         0.986        0.981           0.986
```

Both improve enormously on shifted digits: the plain network from 43%
to 94%, the CNN from 69% to 98%. The CNN ends up almost perfectly
indifferent to a one-pixel nudge, and slightly better on the original
images too (98.6% against 98.1%).

The plain network gains the most, because it had the most to learn: it
had to memorise every digit at every position. The CNN only needed a
small push, because convolution already gives it most of that for
free. That's the usual story: the right architecture plus the right
augmentation beats either alone.

</details>

<details>
<summary>One way to write it</summary>

```python
def train_aug(model, train_set, val, augment, epochs=40, patience=8):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    loader = DataLoader(TensorDataset(*train_set), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(57))
    best, best_epoch, best_state = math.inf, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = augment(xb)
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


X, y = load()
train_set, val, (X_te, y_te) = splits(X, y)
for name, make in [("plain network", mlp), ("CNN", cnn)]:
    torch.manual_seed(57)
    model = make()
    best_epoch = train_aug(model, train_set, val, jitter)
    print(name, best_epoch, acc(model, X_te, y_te),
          acc(model, shift(X_te, 1), y_te), acc(model, shift(X_te, -1), y_te))
```

</details>

## 5. (Optional) Look inside the first layer

After training the CNN, its first layer's kernels are in
`model[0].weight`, shape `(16, 1, 3, 3)`. Pick a few and run them over
image 0 with the script's `convolve` (convert with `.detach().numpy()`).
Print the outputs rounded to one decimal. Can you find one that behaves
like section 2's edge finder, and one that responds to something else?

<details>
<summary>Check yourself</summary>

There's no single right answer here, because the kernels depend on the
seed. You'll usually find a few that look like edge or line finders
(positive on one side, negative on the other), some that respond to ink
in general, and a few that barely do anything. Learned kernels are
rarely as tidy as hand-made ones, but they're doing the same kind of
job.

</details>

---

That's convolution. If one thing sticks: *a CNN slides the same small
set of weights across the whole image, so it finds a pattern wherever
it appears; pooling then shrinks the map and blurs exact positions.*
