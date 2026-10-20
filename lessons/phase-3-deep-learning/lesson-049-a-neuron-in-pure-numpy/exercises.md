# Lesson 049 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-3-deep-learning/lesson-049-a-neuron-in-pure-numpy/lesson.py my_lesson_049.py
python my_lesson_049.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-3-deep-learning/lesson-049-a-neuron-in-pure-numpy").resolve()
```

`load_rides`, `split_and_scale`, `predict`, `log_loss`, `gradients`,
`train`, `accuracy`, `auc`, `r3`, `vec` and `show` are there to reuse.
This gives you the same split as the lesson:

```python
rides = load_rides(RIDES_CSV)
X_train, X_test, y_train, y_test = split_and_scale(rides)
```

---

## 1. What does the sigmoid buy us?

Why not just use $z = x \cdot w + b$ as the prediction and skip the
sigmoid?

<details>
<summary>Check yourself</summary>

$z$ can be any number, like -3 or 12, and that isn't a probability.
The sigmoid squashes it into the range 0 to 1, so we can read it as a
chance and score it with log loss. Without it, $\log p$ wouldn't
even be defined for negative values.

</details>

## 2. Reading a weight

The learned `weekend` weight is 0.531. In plain words, what does a
positive weight here mean?

<details>
<summary>Check yourself</summary>

Holding duration and distance fixed, a weekend ride gets a higher $z$
and so a higher chance of being casual. Because the column was scaled,
the 0.531 is per standard deviation of the weekend flag, not per
weekend, so compare signs and rough sizes rather than reading it as an
exact effect. And as in lesson 047, it describes the model, not cause
and effect.

</details>

## 3. Why start at zero?

We started every weight at 0. Is that a problem for a single neuron?

<details>
<summary>Check yourself</summary>

No. Log loss for one sigmoid neuron is a smooth bowl with one bottom, so
any starting point rolls to the same place, and zero (predict 0.5 for
everything) is a neat neutral start. In lesson 050 you'll see that
starting a whole *layer* at zero is a real problem.

</details>

## 4. Hands-on: skip the ruler

**a)** Train on `duration_min` alone (`X_train[:, :1]`), with `lr=0.5`
and 300 steps. What's the final training loss and the test AUC?

**b)** Now skip scaling. Build the raw arrays and the same split
yourself, then train on all three raw columns with `lr=0.5` for 300
steps. Print the first six losses, the final loss and the test AUC. Then
try `lr=0.01`.

<details>
<summary>Expected results</summary>

```
a) duration only: final loss 0.477, test AUC 0.83
b) raw, lr=0.5:  first six losses [0.693, 1.616, 17.259, 6.446, 16.531, 7.712]
                 final loss 8.884, test AUC 0.833
   raw, lr=0.01: final loss 0.628, test AUC 0.841
```

Duration alone gets most of the way (0.83 against 0.85 with all three).
Unscaled, the columns live on very different rulers: duration runs to
60 and more, the weekend flag is 0 or 1. A step size that suits one is
wild for the other, so at 0.5 the loss explodes and bounces around.
Shrink the step to 0.01 and it stops exploding, but now the small
columns barely learn (the weekend weight is still only 0.06 after 300
steps) and the loss is stuck at 0.63. Scaling
lets one learning rate suit every column.

The test AUC stays decent even in the wild run because AUC only cares
about the *order* of the predictions, and all three weights still
point the right way. Log loss sees the confident nonsense.

</details>

<details>
<summary>One way to write it</summary>

```python
w, b, hist = train(X_train[:, :1], y_train, lr=0.5, steps=300)
show("a) final loss", r3(hist[-1]))
show("a) test AUC", auc(predict(X_test[:, :1], w, b), y_test))

X = rides[FEATURES].to_numpy(dtype=float)
y = rides["casual"].to_numpy(dtype=float)
order = np.random.default_rng(0).permutation(len(y))
tr, te = order[:882], order[882:]
for lr in (0.5, 0.01):
    w, b, hist = train(X[tr], y[tr], lr=lr, steps=300)
    show(f"b) lr={lr} first six losses", [r3(v) for v in hist[:6]])
    show(f"b) lr={lr} final loss", r3(hist[-1]))
    show(f"b) lr={lr} test AUC", auc(predict(X[te], w, b), y[te]))
```

</details>

## 5. (Optional) Know when to stop

Three hundred steps was a guess. Write a loop (scaled data, `lr=0.5`,
start from zero) that stops as soon as one step improves the training
loss by less than 0.000001. Before you run: will it stop before or after
step 300?

<details>
<summary>Check yourself</summary>

Before: it stops at step 147 with loss 0.45 and weights
`[0.909, 0.701, 0.53]`. The loss has gone flat, even though section 6
showed the weights would keep drifting for thousands more steps. A
"stop when the loss stops improving" rule is how most real training
loops end, and it's the seed of early stopping in lesson 056.

```python
w, b, prev = np.zeros(3), 0.0, None
for step in range(100_000):
    loss = log_loss(predict(X_train, w, b), y_train)
    if prev is not None and prev - loss < 1e-6:
        break
    prev = loss
    dw, db = gradients(X_train, y_train, w, b)
    w, b = w - 0.5 * dw, b - 0.5 * db
show("stopped at step", step)
```

</details>

---

That's one neuron. If one thing sticks: *weigh, add, squash; score it
with a loss; step against the gradient; and check your gradient with
tiny nudges before you trust it.*
