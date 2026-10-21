# Lesson 052 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-3-deep-learning/lesson-052-building-networks-with-nn-module/lesson.py my_lesson_052.py
python my_lesson_052.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-3-deep-learning/lesson-052-building-networks-with-nn-module").resolve()
```

`make_ring`, `RingNet`, `train`, `predict_proba`, `accuracy`,
`count_params`, `r3` and `show` are there to reuse. The ring data is:

```python
X, y = make_ring()                 # training points (float32 tensors)
X_new, y_new = make_ring(seed=51)  # fresh points to check on
```

---

## 1. Count before you print

How many parameters does `nn.Linear(6, 32)` have? And the whole
`RingNet(n_inputs=6, n_hidden=32)` from section 6? Work it out, then
check with `count_params`.

<details>
<summary>Check yourself</summary>

`nn.Linear(6, 32)`: 6×32 weights + 32 biases = 224. The output layer
`nn.Linear(32, 1)`: 32 + 1 = 33. Total 257, as section 6 printed. A
layer from `a` inputs to `b` outputs always has $a \times b + b$
numbers.

</details>

## 2. What's wrong with this model?

```python
class Broken(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(2, 8), nn.Linear(8, 1)]

    def forward(self, x):
        return self.layers[1](torch.tanh(self.layers[0](x))).squeeze(1)
```

It runs. But `count_params(Broken())` says 0, and it never learns. Why?

<details>
<summary>Check yourself</summary>

The layers are inside a plain Python list, and `nn.Module` only
registers layers that are attributes themselves (or inside its own
containers). So `model.parameters()` is empty, the optimiser has nothing
to update, and the weights stay random forever. Fix it with
`nn.ModuleList([...])` instead of `[...]`, or use `nn.Sequential`.

</details>

## 3. Where does the sigmoid go?

A friend's model ends `return torch.sigmoid(self.out(h))`, and they train
it with `nn.BCEWithLogitsLoss()`. The loss treats whatever the model
returns as a raw score. What range of probabilities can the loss think
the model is predicting, and why will that hurt on the ring?

<details>
<summary>Check yourself</summary>

The sigmoid gets applied twice: once in `forward`, once inside the loss.
The model's outputs are between 0 and 1, so the loss sees
`sigmoid(0)` = 0.5 up to `sigmoid(1)` = 0.73, never below 0.5. To the
loss, the model can never say "outside", so every outside point costs
at least $-\log(1 - 0.5) = 0.69$ however well it trains, and the
gradients push in odd directions. Pick one place for the sigmoid:
return logits from `forward`, and use `BCEWithLogitsLoss`.

</details>

## 4. Hands-on: ReLU, a forgotten zero_grad, and going deeper

**a)** Build the ring network as an `nn.Sequential` with **ReLU** instead
of tanh:

```python
torch.manual_seed(52)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Flatten(0))
```

(`nn.Flatten(0)` does the `.squeeze(1)` job.) Train it with
`train(model, X, y, lr=1.0, steps=3000)` and print the final loss and
accuracy on `X` and `X_new`.

**b)** Write the loop yourself for a `RingNet()` (after
`torch.manual_seed(52)`), but leave out `optimiser.zero_grad()`. Print the
loss at steps 0, 10, 100 and 2999, and the accuracies. Compare with
lesson 051's exercise 4.

<details>
<summary>Expected results</summary>

```
a) ReLU: final loss 0.176, accuracy train 0.95, new 0.935
b) loss at step 0     -> 0.68
   loss at step 10    -> 0.606
   loss at step 100   -> 0.208
   loss at step 2999  -> 0.348
   accuracy: training 0.88, new points 0.86
```

(a) ReLU does about as well as tanh here, in one changed line. That's
the payoff of building from layers.

(b) Exactly lesson 051's bug, now inside an optimiser: fast early
progress (0.208 at step 100, better than the correct loop's 0.381), then
it overshoots and ends at 0.348 instead of 0.152. The optimiser does what
it's told; it doesn't know you forgot to clear the gradients.

</details>

<details>
<summary>One way to write it</summary>

```python
torch.manual_seed(52)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Flatten(0))
loss = train(model, X, y, lr=1.0, steps=3000)
show("ReLU: loss, train acc, new acc", (r3(loss),
     accuracy(predict_proba(model, X), y), accuracy(predict_proba(model, X_new), y_new)))

torch.manual_seed(52)
model = RingNet()
loss_fn = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=1.0)
for step in range(3000):
    loss = loss_fn(model(X), y)        # no optimiser.zero_grad() here
    loss.backward()
    optimiser.step()
    if step in (0, 10, 100, 2999):
        show(f"loss at step {step}", r3(loss))
show("accuracy: train / new", (accuracy(predict_proba(model, X), y),
                               accuracy(predict_proba(model, X_new), y_new)))
```

</details>

## 5. (Optional) Two hidden layers

Write a `DeepRingNet` class with **two** hidden layers of 8 units each
(2 → 8 → 8 → 1, tanh after each hidden layer). Before running: how many
parameters? Then train it with the same `train` call (`lr=1.0`, 3000
steps, `torch.manual_seed(52)` first) and compare with `RingNet`.

<details>
<summary>Check yourself</summary>

2×8 + 8 = 24, then 8×8 + 8 = 72, then 8 + 1 = 9: 105 parameters. It
reaches a final loss of 0.102, with 0.965 accuracy on the training points
and 0.957 on new ones, a little better than `RingNet`'s 0.952 and 0.942.
The second layer bends the already-bent signals again, so the boundary
gets rounder.

```python
class DeepRingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(2, 8)
        self.h2 = nn.Linear(8, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        h = torch.tanh(self.h1(x))
        h = torch.tanh(self.h2(h))
        return self.out(h).squeeze(1)
```

Going from one hidden layer to two was three lines. That's what
`nn.Module` buys you.

</details>

---

That's `nn.Module`. If one thing sticks: *a model is a class of layers
that returns logits; train it with a matching loss, an optimiser, and
the loop zero, forward, loss, backward, step.*
