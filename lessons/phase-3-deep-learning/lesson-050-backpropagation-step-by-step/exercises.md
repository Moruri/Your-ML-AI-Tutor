# Lesson 050 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-3-deep-learning/lesson-050-backpropagation-step-by-step/lesson.py my_lesson_050.py
python my_lesson_050.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-3-deep-learning/lesson-050-backpropagation-step-by-step").resolve()
```

`make_ring`, `init_params`, `forward`, `backward`, `train_net`,
`numeric_grads`, `rel_error`, `log_loss`, `accuracy`, `r3` and `show`
are there to reuse. The lesson's data is:

```python
X, y = make_ring()                 # training points
X_new, y_new = make_ring(seed=51)  # fresh points to check on
```

---

## 1. Why does the hidden layer need tanh?

Suppose you deleted the `np.tanh` and let the hidden layer pass $XW_1 + b_1$
straight through. Could the network still learn the ring?

<details>
<summary>Check yourself</summary>

No. A weighted sum of weighted sums is just one weighted sum:
$(XW_1 + b_1)W_2 + b_2 = X(W_1W_2) + (b_1W_2 + b_2)$. The whole network
collapses into a single neuron and can only draw a straight line again.
The squashing function between layers is what lets depth add anything.

</details>

## 2. Where does $(1 - h^2)$ come from?

The backward pass multiplies by `1 - h ** 2`. Why that, and why is it
written in terms of `h` instead of `z1`?

<details>
<summary>Check yourself</summary>

It's the slope of tanh: if $h = \tanh(z)$, then
$dh/dz = 1 - \tanh(z)^2 = 1 - h^2$. Writing it with `h` is handy because
we already saved `h` during the forward pass, so the backward pass needs
no extra work. Notice it's close to 0 when $h$ is near -1 or 1: a
saturated unit barely passes any blame back. Lesson 053 returns to that.

</details>

## 3. When to gradient-check

The check in section 4 runs on just 20 points, once. Why not on all 400
points, every training step?

<details>
<summary>Check yourself</summary>

It's slow: two extra forward passes per weight. Its job is to prove the
backward code is right, and if the formula is right on 20 points it's
right on 400. Check once when you write or change `backward()`, then
train with backprop alone.

</details>

## 4. Hands-on: how small can it be, and why not start at zero?

**a)** Train the network on the ring with 1, 2, 3 and 4 hidden units
(`lr=1.0`, 3000 steps). For each, print training accuracy and accuracy on
`X_new`. What's the smallest that works?

**b)** Now start every weight at **zero** instead of small random
numbers, with 8 hidden units, and train for 3000 steps. What's the
training accuracy? Look at `W1` and `W2` afterwards. What happened?

<details>
<summary>Expected results</summary>

```
a) hidden 1: train 0.675, new 0.69
   hidden 2: train 0.752, new 0.728
   hidden 3: train 0.945, new 0.938
   hidden 4: train 0.938, new 0.95
b) zero start: train accuracy 0.675, and W1 and W2 are still exactly 0
```

One hidden unit is still a single bent line. Two can carve out a band.
Three is the smallest that works: three lines make a triangle, and tanh
rounds off its corners into something close to a circle.

In (b) the weights never move; only the output bias `b2` drifts, until
the net says 0.325 for every point. With $W_1 = 0$, every $h$ is $\tanh(0) = 0$,
so the gradient for $W_2$ (which is $h^\top dz_2$) is zero. With
$W_2 = 0$, the blame passed back to the hidden layer
($dz_2 W_2^\top$) is zero too. Both layers wait for the other one
forever. That's why we start from small *random* numbers.

</details>

<details>
<summary>One way to write it</summary>

```python
for n_hidden in (1, 2, 3, 4):
    P, _ = train_net(X, y, n_hidden=n_hidden, lr=1.0, steps=3000)
    show(f"hidden {n_hidden}: train / new",
         (accuracy(forward(X, P)[0], y), accuracy(forward(X_new, P)[0], y_new)))

P = {"W1": np.zeros((2, 8)), "b1": np.zeros(8),
     "W2": np.zeros((8, 1)), "b2": np.zeros(1)}
for _ in range(3000):
    p, cache = forward(X, P)
    grads = backward(p, y, cache, P)
    for k in P:
        P[k] -= 1.0 * grads[k]
show("zero start: train accuracy", accuracy(forward(X, P)[0], y))
show("largest weight anywhere", float(max(np.abs(P["W1"]).max(), np.abs(P["W2"]).max())))
```

</details>

## 5. (Optional) Swap tanh for ReLU

**ReLU** is $\max(0, z)$, and it's the most common hidden activation in
modern networks. Change `forward` to use `np.maximum(z1, 0)` and save
`z1` in the cache. In `backward`, replace `(1 - h ** 2)` with ReLU's
slope. Run the gradient check, then train with 8 hidden units
(`lr=1.0`, 3000 steps). Before you run: what *is* ReLU's slope?

<details>
<summary>Check yourself</summary>

1 where $z_1 > 0$ and 0 elsewhere, so the line is
`dz1 = dh * (cache["z1"] > 0)`. The gradient check gives about
`2e-06`: a little bigger than tanh's because ReLU has a sharp corner at
0, and a nudge that crosses the corner sees a slightly different slope.
Still correct. Training reaches 0.915 on the training points and 0.91
on new ones, with a loss of 0.223. That's a bit worse than tanh here:
ReLU draws its boundary from straight pieces, so eight of them make a
blunter polygon. It wins on deep networks for reasons lesson 053 covers.

</details>

---

That's backpropagation. If one thing sticks: *the gradient of an early
weight is the product of every local slope between it and the loss, so
work backwards, reuse what you've computed, and gradient-check once
before you trust it.*
