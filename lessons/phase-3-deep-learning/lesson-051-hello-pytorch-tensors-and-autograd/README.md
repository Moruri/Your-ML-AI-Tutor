# Lesson 051 - Hello, PyTorch: tensors and autograd

**Phase 3 - Deep learning foundations** | Week 7, Day 3 | Wednesday 2026-10-21

> **Goal:** Move the NumPy network to PyTorch and let autograd compute the
> gradients.

Time: about 40 minutes. This is the first lesson that needs `torch`.

---

Yesterday you wrote a backward pass by hand. Five lines of chain rule,
every shape tracked, then a gradient check to prove it was right. It was
worth doing once: you now know exactly what "training a network" means.
It is not worth doing again for every network you ever build.

**PyTorch** is the library most deep learning research and a lot of
production code is written in. Today we meet its two core ideas. A
**tensor** is a NumPy array with a few extras. **Autograd** is the extra
that matters: a tensor can remember every operation that produced it,
and run that history backwards to get gradients. In other words,
yesterday's `backward()`, written for you, for any network, and never
forgetting tanh's slope.

We'll rebuild lesson 050's network on the same ring of points, from the
same starting weights, and check that autograd and our hand-written
gradients agree to the last decimal place.

## Installing torch

`torch` is already in [`requirements.txt`](../../../requirements.txt). The
default download includes GPU support and is large. Everything in this
course runs fine on a CPU, so a smaller CPU-only build is enough. In your
venv:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"
```

On a Mac, plain `pip install torch` is already small; use that. If the
version prints, you're set. The script prints your version first; it
doesn't need to match mine (2.14.1).

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-3-deep-learning/lesson-051-hello-pytorch-tensors-and-autograd/lesson.py
```

It writes no files. Every random step has a fixed seed, so your numbers
will match. It takes about five seconds.

## 1. Tensors: NumPy's close cousin

```
t.shape                                        -> (2, 3)
t.dtype (torch's default float)                -> 'torch.float32'
t.sum(dim=0)  (NumPy says axis=0)              -> [5.0, 7.0, 9.0]
t @ t.T                                        -> [[14.0, 32.0], [32.0, 77.0]]
dtype after from_numpy(float64 array)          -> 'torch.float64'
from_numpy after changing the array            -> [100.0, 2.0, 3.0]
torch.tensor after changing the array          -> [1.0, 2.0, 3.0]
back to NumPy with .numpy()                    -> [1.0, 2.0, 3.0]
a single number out with .item()               -> 21.0
```

If you know NumPy, you mostly know tensors already. Shapes, indexing,
broadcasting, `@` and `.T` all work the way they did in lessons 014 and
015. Three differences catch everyone at least once:

- **`dim=` instead of `axis=`.** Same meaning, different word. (Recent
  versions accept `axis=` too, but you'll read `dim=` everywhere.)
- **float32 by default.** NumPy uses 64-bit floats; torch uses 32-bit.
  Half the memory, plenty of precision for training, and what GPUs are
  fast at. Mixing the two in one calculation gives you a dtype error,
  so convert with `dtype=torch.float32` when you bring data in.
- **`torch.from_numpy` shares memory.** The tensor and the array are two
  views of the same numbers, so changing one changes the other. That's
  fast, but surprising. `torch.tensor(a)` makes a copy.

`.numpy()` takes you back to NumPy, and `.item()` pulls a single number
out as a plain Python float.

## 2. Autograd on one number at a time

```
loss (same tiny network as lesson 050)         -> 1.145
loss.grad_fn (how it was made)                 -> 'NegBackward0'
w1.grad before backward()                      -> None
w2.grad after backward()                       -> -0.519
w1.grad after backward()                       -> 0.573
lesson 050's hand-derived answers              -> (-0.519, 0.573)
```

Here's lesson 050's one-number network again, input 2.0, weights 0.5 and
-1.0, in torch:

```python
w1 = torch.tensor(0.5, requires_grad=True)
w2 = torch.tensor(-1.0, requires_grad=True)
h = torch.tanh(w1 * x)
p = torch.sigmoid(w2 * h)
loss = -torch.log(p)
loss.backward()
```

`requires_grad=True` is the switch. It tells torch, "this is a number I
want to learn, so keep a record of everything you do with it." Every
result computed from `w1` or `w2` carries a `grad_fn`, a note saying
which operation made it (the last step was a minus sign, so
`NegBackward0`). Chained together, those notes are the full recipe from
weights to loss.

`loss.backward()` walks that recipe in reverse, applying each step's
local slope: yesterday's table, row by row, done for you. The answers
land in `w1.grad` and `w2.grad`, and they're the same -0.519 and 0.573
you derived by hand. Before `backward()`, `.grad` is `None`: nothing has
been computed yet.

## 3. Autograd on the whole network, checked against lesson 050

```
loss at the start                              -> 0.689
W2 gradient, first three (autograd)            -> [-0.0008, -0.0001, -0.0031]
W2 gradient, first three (by hand)             -> [-0.0008, -0.0001, -0.0031]
biggest gap over all 33 gradients              -> 5.6e-17
```

Now the full two-layer network on the 400 ring points. The script
builds the starting weights exactly as lesson 050 did (same seed, same
NumPy code), then turns them into tensors with `requires_grad=True`. The
forward pass is lesson 050's, word for word: `torch.tanh` for `np.tanh`,
`torch.sigmoid` for our hand-made sigmoid.

There's no `backward()` function of ours this time, just
`loss.backward()`. Then we compare with yesterday's hand-written backward
pass. The biggest gap across all 33 gradients is 0.00000000000000006.
That's not "close"; that's the same numbers, up to the rounding any
computer does. (The tensors here are float64, so the comparison with
NumPy is fair. Exercise 4 tries float32.)

Remember yesterday's gradient check, the one that caught a forgotten
tanh slope? With autograd you rarely need it for your own code. You do
still need it if you ever write a custom backward rule, and that's rare.

## 4. A training loop with autograd

```
loss at step 0                                 -> 0.689
loss at step 100                               -> 0.361
loss at step 500                               -> 0.191
loss at step 1000                              -> 0.161
loss at step 2999                              -> 0.148
training accuracy                              -> 0.955
on 400 fresh points from the same ring         -> 0.948
```

Here's the loop:

```python
for step in range(3000):
    loss = log_loss(forward(X, P), y)
    loss.backward()                  # fills P[k].grad
    with torch.no_grad():            # the update itself isn't part of the model
        for w in P.values():
            w -= lr * w.grad
            w.grad.zero_()           # clear it, or next step adds to it
```

Every number matches lesson 050: loss 0.689 down to 0.148, 95.5% on the
training points, 94.8% on fresh ones. Same maths, same starting point,
same answer. Only who does the calculus changed.

Two lines are new, and both matter:

- **`with torch.no_grad():`** Updating a weight is not part of the
  model; it's us adjusting the model. Inside this block torch stops
  recording, so the update doesn't get added to the recipe that the next
  `backward()` would walk.
- **`w.grad.zero_()`** Clears the gradient after using it. Section 5
  shows why. (The trailing underscore is torch's way of saying "changes
  the tensor in place".)

## 5. The two traps: piling-up gradients and no_grad

```
w.grad after backward() call 1, no zeroing     -> 6.0
w.grad after backward() call 2, no zeroing     -> 12.0
w.grad after backward() call 3, no zeroing     -> 18.0
after zero_() and one more backward()          -> 6.0
w * 2 normally: requires_grad                  -> True
w * 2 inside no_grad: requires_grad            -> False
w.detach(): requires_grad                      -> False
```

The slope of $w^2$ at $w = 3$ is 6. Call `backward()` three times without
clearing and `.grad` says 6, then 12, then 18. Torch **adds** new
gradients to whatever is already in `.grad`; it never replaces them.

That's deliberate. Sometimes you want to add up gradients from several
small batches before taking one step. But in an ordinary loop, forgetting
to zero means every step uses the sum of every gradient so far, and
training goes strange (exercise 4 shows how strange). It's one of the
most common PyTorch bugs, and it doesn't crash; it just trains badly.

The other half: inside `torch.no_grad()`, results stop tracking history
(`requires_grad` is `False`). Use it for weight updates and for making
predictions, where you don't need gradients and don't want to pay for
recording them. `.detach()` does the same for a single tensor: "give me
these numbers, without the history".

## 6. Back to the rides, in float32

```
weights dtype                                  -> 'torch.float32'
final training loss                            -> 0.413
test AUC (lesson 050 by hand: 0.841)           -> 0.841
```

Last, the real data. Same six scaled ride features, same split, same
eight hidden units and starting weights as lesson 050, but this time
everything is float32, the way real networks run. Test AUC: 0.841,
yesterday's number exactly. The lower precision made no difference you
can see.

## What you can do now

- Create tensors, move between NumPy and torch, and spot the three
  differences that bite.
- Mark weights with `requires_grad` and get gradients with `backward()`.
- Explain what autograd records and why its answers match hand-written
  backprop.
- Write a training loop with `torch.no_grad()` updates and zeroed grads.
- Recognise the piling-up gradients bug.

## What to do now

1. Install torch, run `lesson.py`, and check your numbers match.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   deliberately forgets `zero_()` and watches what training does.
3. Next up, lesson 052: the same network the way real PyTorch code is
   written, with `nn.Module` layers, a built-in loss and an optimiser
   doing the update for you. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

A tensor is a NumPy array that can remember how it was made. Mark your
weights with `requires_grad`, compute the loss, call `backward()`,
update inside `no_grad`, zero the grads. Repeat. That's yesterday's
lesson with the calculus done for you.
