# Lesson 054 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-3-deep-learning/lesson-054-optimisers-and-learning-rate-schedules/lesson.py my_lesson_054.py
python my_lesson_054.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-3-deep-learning/lesson-054-optimisers-and-learning-rate-schedules").resolve()
```

`make_ring`, `RingNet`, `fit`, `ravine`, `descend`, `predict_proba`,
`accuracy`, `r3` and `show` are there to reuse. The ring data is:

```python
X, y = make_ring()                 # training points (float32 tensors)
X_new, y_new = make_ring(seed=51)  # fresh points to check on
```

---

## 1. Steeper valley, smaller limit

Section 1's valley was 50 times steeper in one direction, and plain
SGD blew up above a learning rate of 2/50 = 0.04. If the valley were
200 times steeper, `0.5 * (w[0]**2 + 200 * w[1]**2)`, what would the
limit be? What does that tell you about plain SGD on badly shaped
problems?

<details>
<summary>Check yourself</summary>

$2/200 = 0.01$. The steepest direction sets the largest safe learning
rate, and the gentle directions then crawl at that rate. The worse the
shape, the slower SGD gets. That's exactly what momentum and Adam are
for, and also why lesson 035's feature scaling helps: unscaled features
make the valley lopsided.

</details>

## 2. How fast can momentum go?

With momentum 0.9 and the same gradient every step, section 2's step
sizes went 1, 1.9, 2.71, 3.44. What number are they heading for? And
for momentum 0.99?

<details>
<summary>Check yourself</summary>

Each step adds 0.9 times the previous one: $1 + 0.9 + 0.9^2 + \dots$,
which adds up to $1/(1 - 0.9) = 10$. With 0.99 it's
$1/(1 - 0.99) = 100$. That's why momentum is paired with a smaller
learning rate: at 0.9, the effective step can be ten times the plain
one.

</details>

## 3. A schedule that's too short

You train for 300 steps, but set
`CosineAnnealingLR(optimiser, T_max=100)` by mistake. Starting at 0.05,
what's the learning rate at steps 50, 100, 150, 200 and 300? Predict,
then check with section 5's code.

<details>
<summary>Check yourself</summary>

```
step 0, 50, 100, 150, 200, 300  -> (0.05, 0.025, 0.0, 0.025, 0.05, 0.0)
```

It doesn't stop at zero. After `T_max` steps the cosine keeps going,
climbing back up to 0.05 at step 200 and down again at 300. Your model
gets a sudden burst of big steps in the middle of training. Set `T_max`
to your total number of steps (or of `scheduler.step()` calls).

</details>

## 4. Hands-on: Adam to the rescue, and a learning-rate sweep

**a)** Lesson 053's ten-layer ReLU network, with the default
initialisation, couldn't learn with SGD. Paste in its builder:

```python
def deep_net(act, depth=10, width=32, n_inputs=2):
    layers, n_in = [], n_inputs
    for _ in range(depth):
        layers += [nn.Linear(n_in, width), act()]
        n_in = width
    layers += [nn.Linear(n_in, 1), nn.Flatten(0)]
    return nn.Sequential(*layers)
```

Train it twice for 1000 steps with `fit`, calling
`torch.manual_seed(53)` before building each one: once with
`torch.optim.SGD(..., lr=0.5)`, once with `torch.optim.Adam(..., lr=0.01)`.
Print the final loss and the accuracy on new points.

**b)** Sweep Adam's learning rate on `RingNet()` (seed 54, 1000 steps):
0.001, 0.01, 0.1 and 1.0.

<details>
<summary>Expected results</summary>

```
a) SGD lr 0.5:   loss 0.631, accuracy on new 0.69
   Adam lr 0.01: loss 0.053, accuracy on new 0.96
b) Adam lr 0.001: loss 0.434, accuracy on new 0.848
   Adam lr 0.01:  loss 0.158, accuracy on new 0.938
   Adam lr 0.1:   loss 0.137, accuracy on new 0.947
   Adam lr 1.0:   loss 0.236, accuracy on new 0.94
```

(a) Adam rescues it. The early layers' gradients were tiny (lesson 053
measured 0.000012), but Adam divides each gradient by its own typical
size, so even tiny ones produce steps of about `lr`. Kaiming
initialisation is still the cleaner fix, but this is a big part of why
Adam is so forgiving.

(b) A thousandfold range of learning rates, and everything from 0.01 to
1.0 lands between 94% and 95%. Only 0.001 is clearly too slow for 1,000
steps. Compare lesson 053's SGD sweep, where 30 killed every unit.

</details>

<details>
<summary>One way to write it</summary>

```python
for name, make_opt in [("SGD lr 0.5", lambda p: torch.optim.SGD(p, lr=0.5)),
                       ("Adam lr 0.01", lambda p: torch.optim.Adam(p, lr=0.01))]:
    torch.manual_seed(53)
    model = deep_net(nn.ReLU)
    final, _ = fit(model, X, y, make_opt(model.parameters()), 1000)
    show(name, (final, accuracy(predict_proba(model, X_new), y_new)))

for lr in (0.001, 0.01, 0.1, 1.0):
    torch.manual_seed(54)
    model = RingNet()
    final, _ = fit(model, X, y, torch.optim.Adam(model.parameters(), lr=lr), 1000)
    show(f"Adam lr {lr}", (final, accuracy(predict_proba(model, X_new), y_new)))
```

</details>

## 5. (Optional) Momentum by hand

Write momentum yourself on the ravine, no `torch.optim`, and check you
get section 2's step counts (58 for momentum 0.5 and 87 for 0.9, at
`lr=0.035`). The gradient of the ravine at `w` is `(w[0], 50 * w[1])`.

<details>
<summary>Check yourself</summary>

```python
for momentum in (0.5, 0.9):
    w = torch.tensor([-4.0, 1.0])
    velocity = torch.zeros(2)
    for step in range(500):
        if ravine(w).item() < 1e-3:
            break
        grad = torch.tensor([w[0].item(), 50 * w[1].item()])
        velocity = momentum * velocity + grad
        w = w - 0.035 * velocity
    show(f"momentum {momentum}: steps", step)
```

You get 58 and 87, identical to `torch.optim.SGD`. That's all momentum
is: one extra line and one extra variable per weight. Adam is two
extra variables and a division; torch's versions are just careful,
fast implementations of a few lines like these.

</details>

---

That's optimisers and schedules. If one thing sticks: *start with Adam
or AdamW at about 0.01, give it a cosine schedule over all your steps,
and swap optimisers in one line when you want to compare.*
