# Lesson 052 - Building networks with `nn.Module`

**Phase 3 - Deep learning foundations** | Week 7, Day 3 | Wednesday 2026-10-21

> **Goal:** Define layers, forward passes and a training loop the way real
> PyTorch code does.

Time: about 40 minutes. Same venv as lesson 051, with `torch` installed.

---

Lesson 051's network kept its weights in a dictionary, and we updated
each one by hand inside `torch.no_grad()`. That's exactly what happens
under the hood, and it's worth having seen. But open any real PyTorch
project and you won't find it written that way. You'll find three
pieces:

| Piece | Where it comes from | What it replaces from lesson 051 |
|-------|--------------------|----------------------------------|
| A **model** | a class built on `nn.Module`, made of layers like `nn.Linear` | the `P` dict and the `forward()` function |
| A **loss** | `torch.nn`, e.g. `nn.BCEWithLogitsLoss` | our `log_loss()` |
| An **optimiser** | `torch.optim`, e.g. `torch.optim.SGD` | the `w -= lr * w.grad` and `zero_()` lines |

Learn these three and the five-line loop that ties them together, and
you can read most PyTorch code you'll come across, from a homework
notebook to a research repo.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-3-deep-learning/lesson-052-building-networks-with-nn-module/lesson.py
```

It writes one file, `output/ring_net.pt`, next to the script. Every
random step has a fixed seed, so your numbers will match. It takes
about ten seconds.

## 1. A layer is an object: `nn.Linear`

```
layer                                          -> 'Linear(in_features=2, out_features=8, bias=True)'
layer.weight.shape (out, in)                   -> (8, 2)
layer.bias.shape                               -> (8,)
weight requires_grad, out of the box           -> True
numbers to learn in this layer                 -> 24
layer(x).shape for 2 points                    -> (2, 8)
layer(x) equals x @ W.T + b                    -> True
starting weights, first row                    -> [0.306, 0.401]
```

`nn.Linear(2, 8)` is lesson 051's `W1` and `b1`, bundled into one
object: two inputs in, eight outputs out, 2×8 weights plus 8 biases is
24 numbers. You call it like a function, `layer(x)`, and it computes
the weighted sum.

It also does three jobs you used to do yourself. Its weights already
have `requires_grad=True`. They start at small random values, using a
sensible recipe based on the number of inputs (lesson 053 explains why
the recipe matters). And they're registered as **parameters**, so
anything that asks "what can be learned here?" will find them.

One surprise: `weight` is stored as (out, in), 8-by-2, the transpose of
our `W1`. So the layer computes $xW^\top + b$. The script checks that
it equals `x @ layer.weight.T + layer.bias`, and it does. You'll rarely
need to know that, but when you do (reading weights, loading them from
elsewhere), you'll be glad you did.

## 2. Your first `nn.Module`

```
print(model):
  RingNet(
    (hidden): Linear(in_features=2, out_features=8, bias=True)
    (out): Linear(in_features=8, out_features=1, bias=True)
  )
parameters to learn                            -> 33
named parameters                               -> ['hidden.weight', 'hidden.bias', 'out.weight', 'out.bias']
model(X).shape (one score per point)           -> (400,)
nn.Sequential version: parameters              -> 33
```

Here's lesson 050's network as a proper PyTorch model:

```python
class RingNet(nn.Module):
    def __init__(self, n_inputs=2, n_hidden=8):
        super().__init__()                       # always call this first
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        return self.out(h).squeeze(1)            # a raw score (a "logit")
```

It's a class, like the ones from lesson 010, with two methods:

- **`__init__`** creates the layers and stores them as attributes. The
  `super().__init__()` line sets up the machinery that notices them;
  leave it out and PyTorch complains as soon as you add a layer.
- **`forward`** says how data flows through those layers. You never
  call it directly. You call `model(X)`, and PyTorch calls `forward` for
  you (plus some bookkeeping).

Because the layers are attributes, `model.parameters()` finds all 33
numbers on its own, and `named_parameters()` tells you whose they are.
`print(model)` gives you a neat summary. Same 33 as lesson 050: 2×8 + 8
+ 8×1 + 1.

Notice `forward` stops *before* the sigmoid. It returns a raw score,
called a **logit**: any number, where positive means "leans inside" and
negative "leans outside". Section 3 explains why.

For a model that's just one layer after another, `nn.Sequential` saves
writing a class:

```python
nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
```

Same 33 parameters. Use `Sequential` for straight stacks, and a class
the moment you need anything else (two inputs, a skip, a print for
debugging).

## 3. Losses and optimisers come in the box

```
per-point loss, BCEWithLogitsLoss              -> [0.127, 0.313, 40.0]
per-point loss, sigmoid then BCELoss           -> [0.127, 0.313, 100.0]
sigmoid(40) in float32                         -> 1.0
before optimiser.step(): w, w.grad             -> (3.0, 6.0)
after step: w = 3 - 0.1 * 6                    -> 2.4
after optimiser.zero_grad(): w.grad            -> None
```

**The loss.** `nn.BCEWithLogitsLoss` is our log loss (BCE stands for
binary cross-entropy, the textbook name), but it takes raw scores and
does the sigmoid itself, inside. Why not sigmoid first, then
`nn.BCELoss`? Look at the third point: a score of 40, so the model is
supremely sure it's a 1, but the truth is 0. Its real loss is 40. In
float32, `sigmoid(40)` rounds to exactly 1.0, so `BCELoss` sees
$-\log(1 - 1) = -\log(0)$, infinity, and quietly caps it at 100. The
number is wrong, and worse, the gradient through a sigmoid stuck at 1.0
is zero, so the model can't learn from its worst mistakes.
`BCEWithLogitsLoss` combines the two steps in a form that never rounds
to 0, and gets 40.

So the rule is: **models return logits, and you use the `WithLogits`
loss.** You apply `torch.sigmoid` only when you want an actual
probability out, after training.

**The optimiser.** `torch.optim.SGD(model.parameters(), lr=1.0)` takes
the list of parameters and does the update for you. `optimiser.step()`
is our `w -= lr * w.grad` for every parameter (3 - 0.1 × 6 = 2.4), and
`optimiser.zero_grad()` clears every `.grad` (to `None`, which torch
treats as zero). SGD stands for stochastic gradient descent; it's plain
lesson 049 gradient descent. Lesson 054 swaps in smarter optimisers, and
because they all have the same `step()` and `zero_grad()`, it'll be a
one-line change.

## 4. The standard training loop

```
loss at step 0                                 -> 0.68
loss at step 100                               -> 0.381
loss at step 500                               -> 0.193
loss at step 1000                              -> 0.166
loss at step 2999                              -> 0.152
training accuracy                              -> 0.952
on 400 fresh points from the same ring         -> 0.942
model.training after predict_proba             -> False
```

Here's the loop. Learn it by heart; you'll write it a hundred times:

```python
loss_fn = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=lr)
model.train()
for step in range(steps):
    optimiser.zero_grad()                    # 1. clear last step's gradients
    logits = model(X)                        # 2. forward pass
    loss = loss_fn(logits, y)                # 3. how wrong are we?
    loss.backward()                          # 4. autograd fills every .grad
    optimiser.step()                         # 5. nudge every weight
```

Zero, forward, loss, backward, step. Putting `zero_grad()` first, not
last, is the common habit: whatever happened before, each step starts
clean.

The network learns the ring: loss 0.68 down to 0.152, 95.2% on the
training points, 94.2% on fresh ones. Not quite lesson 050's numbers,
because `nn.Linear` chose its own starting weights, so training took a
different path. It ends in the same place.

For predictions, the script uses this:

```python
@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.sigmoid(model(X))
```

`@torch.no_grad()` is lesson 051's `with torch.no_grad():`, wrapped
round a whole function. `model.eval()` switches the model into
**evaluation mode**, and `model.train()` switches it back. For this
network it changes nothing, but some layers behave differently while
training (dropout, in lesson 056, is the famous one). Make it a habit
now: `train()` before training, `eval()` before predicting.

## 5. Saving and loading a trained model

```
state_dict keys                                -> ['hidden.weight', 'hidden.bias', 'out.weight', 'out.bias']
saved to                                       -> 'output/ring_net.pt'
file size in bytes, roughly                    -> 2500
fresh model agrees before loading              -> False
fresh model agrees after loading               -> True
```

A trained model is just its numbers. `model.state_dict()` hands them
over as a dictionary, keyed by the same names as `named_parameters()`,
and `torch.save` writes it to a file: about 2.5 KB for 33 numbers plus
labels. (Your file size may differ slightly between torch versions.)

To load, build a fresh `RingNet()` (random weights, different
predictions) and pour the numbers in with `load_state_dict`. Now its
predictions are identical to the trained model's, to the last bit.

```python
torch.save(model.state_dict(), path)

fresh = RingNet()
fresh.load_state_dict(torch.load(path, weights_only=True))
```

Save the `state_dict`, not the whole model object. It's smaller, it
doesn't break when you tidy up your code, and `weights_only=True` means
loading it can't run code hidden in a file someone sent you. Treat `.pt`
files from strangers the way you'd treat any download.

## 6. Same pattern, the rides

```
8 hidden (65 params): train / test AUC         -> (0.864, 0.827)
32 hidden (257 params): train / test AUC       -> (0.879, 0.809)
```

Last, the payoff for doing it properly. The same `RingNet` class, with
`n_inputs=6`, fits the six scaled ride features from lesson 050. Trying a
bigger network is one argument, `n_hidden=32`. The loss, the optimiser
and the loop didn't change at all.

The results are honest rather than flattering. With eight hidden units,
test AUC is 0.827, a little below lesson 050's 0.841 (different
starting weights, small test set). With 32, the training AUC goes *up*
to 0.879 and the test AUC goes *down* to 0.809. The bigger network has
room to memorise quirks of the 661 training rides that don't carry over
to new ones. That's overfitting, the same story as lesson 031, and
lesson 056 is all about fighting it in neural networks. On this small,
tidy table, Phase 2's gradient boosting is still the better tool.

## What you can do now

- Use `nn.Linear` and explain its weight shape.
- Write an `nn.Module` with `__init__` and `forward`, or an
  `nn.Sequential` for a straight stack.
- Explain why models return logits and train with `BCEWithLogitsLoss`.
- Write the five-line loop: zero, forward, loss, backward, step.
- Switch between `train()` and `eval()`, and predict under `no_grad`.
- Save and reload a model with `state_dict`.

## What to do now

1. Run `lesson.py`. Then open `RingNet` and add a `print(h.shape)` to
   `forward`, run it, and remove it again. Printing inside `forward` is
   the most useful debugging trick in PyTorch.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   swaps tanh for ReLU in one line, and builds a deeper network.
3. Next weekday, lesson 053: why training fails, from dead ReLUs to
   vanishing gradients and bad learning rates, and how to spot each one.
   See [PROGRESS.md](../../../curriculum/PROGRESS.md).

Model, loss, optimiser. Then zero, forward, loss, backward, step. Every
PyTorch project, however big, is built on that skeleton; the rest is
detail you can now learn one piece at a time.
