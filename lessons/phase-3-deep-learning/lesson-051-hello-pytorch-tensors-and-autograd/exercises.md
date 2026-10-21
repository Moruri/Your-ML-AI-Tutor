# Lesson 051 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-3-deep-learning/lesson-051-hello-pytorch-tensors-and-autograd/lesson.py my_lesson_051.py
python my_lesson_051.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-3-deep-learning/lesson-051-hello-pytorch-tensors-and-autograd").resolve()
```

`make_ring`, `init_numpy`, `to_params`, `forward`, `log_loss`,
`accuracy`, `numpy_backward`, `r3` and `show` are there to reuse. The
ring data as tensors is:

```python
X_np, y_np = make_ring()
X, y = torch.from_numpy(X_np), torch.from_numpy(y_np)
X_new, y_new = (torch.from_numpy(a) for a in make_ring(seed=51))
```

---

## 1. Which tensors need `requires_grad`?

In section 4, the weights in `P` have `requires_grad=True`, but `X` and
`y` don't. Why not? Would anything break if `X` had it too?

<details>
<summary>Check yourself</summary>

We only want gradients for things we're going to change, and we never
change the data. Nothing would break if `X` had it; torch would just
compute and store a gradient for every input number that nobody uses,
which costs time and memory. (Gradients with respect to inputs do have
uses, like asking "which input pixels mattered most?", but that's a
different question from training.)

</details>

## 2. Why `torch.no_grad()` around the update?

What would go wrong if you wrote `w -= lr * w.grad` *without* the
`with torch.no_grad():` block?

<details>
<summary>Check yourself</summary>

Torch refuses: changing a weight that requires gradients in place, while
it's recording, raises a `RuntimeError`, because it would scramble the
recipe that `backward()` needs. Even if it were allowed, the update would
become part of the history, and the next `backward()` would try to
differentiate through your own optimiser. The update is us changing the
model, not the model computing something, so we switch recording off.

</details>

## 3. `from_numpy` or `torch.tensor`?

You load a CSV with pandas, convert it with `torch.from_numpy(df.to_numpy())`,
then later fill missing values in the DataFrame in place. Could that
change your tensor? What would you use to be safe?

<details>
<summary>Check yourself</summary>

It can. If `to_numpy()` handed back the DataFrame's own memory (common
for a single-dtype frame), the tensor shares it, and an in-place change
shows up in both. Use `torch.tensor(..., dtype=torch.float32)`, which
always copies and also gives you the float32 you want for training.

</details>

## 4. Hands-on: forget to zero, and check a slope

**a)** Copy section 4's loop and delete the `w.grad.zero_()` line. Train
for 3000 steps at `lr=1.0` from the same starting weights. Print the
loss at steps 0, 10, 100 and 2999, then accuracy on `X` and `X_new`.
Before you run: will it fail to learn, or something stranger?

**b)** Use autograd to get the slope of tanh at seven points:

```python
x = torch.linspace(-3, 3, 7, requires_grad=True)
torch.tanh(x).sum().backward()
```

Compare `x.grad` with lesson 050's formula, `1 - tanh(x) ** 2`. Why did
we need `.sum()` before `backward()`?

<details>
<summary>Expected results</summary>

```
a) loss at step 0     -> 0.689
   loss at step 10    -> 0.649
   loss at step 100   -> 0.247
   loss at step 2999  -> 0.521
   accuracy: training 0.877, new points 0.875
b) x.grad          -> [0.01, 0.071, 0.42, 1.0, 0.42, 0.071, 0.01]
   1 - tanh(x)**2  -> [0.01, 0.071, 0.42, 1.0, 0.42, 0.071, 0.01]
```

Stranger. At first it learns *faster* than the correct loop (0.247 at
step 100 versus 0.361), because each step pushes with the sum of every
gradient so far, like a ball rolling downhill that never slows down.
Then it can't stop: it overshoots, swings about, and finishes at a loss
of 0.521, much worse than 0.148. No error, no warning, just a worse
model. That's why this bug is so easy to miss.

In (b) they match. `backward()` starts from a single number, the loss.
`tanh(x)` is seven numbers, so we sum them into one; the slope of a sum
with respect to each `x` is just the slope of that one term, so each
`x.grad` is tanh's slope at that point.

</details>

<details>
<summary>One way to write it</summary>

```python
P = to_params(init_numpy(2, 8))
for step in range(3000):
    loss = log_loss(forward(X, P), y)
    loss.backward()
    with torch.no_grad():
        for w in P.values():
            w -= 1.0 * w.grad          # no zero_() this time
    if step in (0, 10, 100, 2999):
        show(f"loss at step {step}", r3(loss))
with torch.no_grad():
    show("accuracy: train / new", (accuracy(forward(X, P), y),
                                   accuracy(forward(X_new, P), y_new)))

x = torch.linspace(-3, 3, 7, requires_grad=True)
torch.tanh(x).sum().backward()
show("x.grad", [r3(v) for v in x.grad])
show("1 - tanh(x)**2", [r3(v) for v in 1 - torch.tanh(x.detach()) ** 2])
```

</details>

## 5. (Optional) How much does float32 cost?

Repeat section 3, but build the weights and data as float32
(`dtype=torch.float32`, and `X.float()`, `y.float()`). Compare the
gradients with `numpy_backward` again. How big is the biggest gap now?
Should it worry you?

<details>
<summary>Check yourself</summary>

About `1.3e-08` instead of `5.6e-17`. float32 keeps roughly 7 significant
digits instead of 16, so the gradients agree to about 7 places. That's
far smaller than the changes a learning rate makes each step, which is
why section 6 lands on the same AUC. It's fine for training. Where it
does matter is gradient checks: use float64 for those, or the rounding
noise can look like a bug.

</details>

---

That's autograd. If one thing sticks: *mark what you want to learn with
`requires_grad`, call `backward()` on one number, update inside
`no_grad`, and always zero the gradients before the next step.*
