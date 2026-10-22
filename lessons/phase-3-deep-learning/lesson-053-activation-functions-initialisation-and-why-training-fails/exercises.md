# Lesson 053 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-3-deep-learning/lesson-053-activation-functions-initialisation-and-why-training-fails/lesson.py my_lesson_053.py
python my_lesson_053.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-3-deep-learning/lesson-053-activation-functions-initialisation-and-why-training-fails").resolve()
```

`make_ring`, `deep_net`, `init_deep`, `relu_net`, `train`,
`grad_norms`, `dead_share`, `predict_proba`, `accuracy`, `r3` and
`show` are there to reuse. The ring data is:

```python
X, y = make_ring()                 # training points (float32 tensors)
X_new, y_new = make_ring(seed=51)  # fresh points to check on
```

---

## 1. A ceiling on the gradient

Sigmoid's slope is never above 0.25. Ignoring the weights, what's the
most that ten sigmoid layers can pass back, as a fraction of what
arrives at the top? How does that compare with what section 2 measured?

<details>
<summary>Check yourself</summary>

$0.25^{10} \approx 9.5 \times 10^{-7}$, under one in a million. Section 2
measured a first-layer gradient of `1.1e-10` against `3.9e-01` at the
output, a fraction of about $3 \times 10^{-10}$, smaller still, because
most units don't sit at the steepest point and the default weights
shrink the signal too.

</details>

## 2. Kaiming by hand

A ReLU layer has 64 inputs. What standard deviation does Kaiming
initialisation give its weights? And what's `nn.Linear(64, 64)`'s
default spread?

<details>
<summary>Check yourself</summary>

Kaiming: $\sqrt{2/64} = \sqrt{0.03125} \approx 0.177$. The default:
$1/\sqrt{3 \times 64} = 1/\sqrt{192} \approx 0.072$. Check the second
with `nn.Linear(64, 64).weight.std()`, which will be close but not
exact, since it's measured on random numbers.

</details>

## 3. Dead or just quiet?

A friend's ReLU network reports 40% dead units after the first step of
training, and 40% after the last. Another reports 5% after the first
and 60% after the last. Which one should worry more, and what would you
check first?

<details>
<summary>Check yourself</summary>

The second. A steady 40% from the start means the units began dead
(often a bad initialisation or a big negative bias) and the rest of
the network trained around them: wasteful but stable. Going from 5% to
60% *during* training means something is killing them: most often a
learning rate that's too big, or unscaled inputs, as in sections 5 and
6. Check those first.

</details>

## 4. Hands-on: break it, then fix it

**a)** Build the ten-layer ReLU network with Kaiming initialisation and
train it at three learning rates:

```python
for lr in (0.05, 0.5, 3.0):
    torch.manual_seed(53)
    model = init_deep(deep_net(nn.ReLU), "kaiming")
    loss = train(model, X, y, lr=lr, steps=1000)
    show(f"lr {lr}: loss, accuracy on new",
         (r3(loss), accuracy(predict_proba(model, X_new), y_new)))
```

**b)** Swap ReLU for GELU, `deep_net(nn.GELU)`. First with the default
initialisation, then with `init_deep(..., "kaiming")`. For each, print
the first-layer gradient size before training (`grad_norms(model, X,
y)[0]`), then the loss and accuracy after 1000 steps at `lr=0.5`.

<details>
<summary>Expected results</summary>

```
a) lr 0.05: loss, accuracy on new           -> (0.083, 0.938)
   lr 0.5: loss, accuracy on new            -> (0.075, 0.957)
   lr 3.0: loss, accuracy on new            -> (nan, 0.69)
b) GELU, default init: first grad 6.9e-07, loss 0.631, accuracy 0.69
   GELU, Kaiming init: first grad 1.2e-02, loss 0.073, accuracy 0.96
```

(a) There's your `nan`. At 3.0 the deep network's weights blow up,
the loss becomes "not a number", and the predictions collapse to the
"always outside" answer. Notice that 0.05, ten times smaller than the
best rate, still does well: too small costs time, too big costs
everything.

(b) GELU behaves like ReLU here: it fades with the default start and
trains beautifully with Kaiming's. Activation and initialisation go
together.

</details>

## 5. (Optional) Clip the gradients

Gradient clipping caps how big a step can get: if the gradients' total
size is above a limit, they're scaled down to it. Re-run section 5's
`lr=30.0` case, but add this line between `loss.backward()` and
`optimiser.step()`:

```python
nn.utils.clip_grad_norm_(model.parameters(), 0.1)
```

Print the loss at steps 0, 10, 100 and 999, the dead share, and the
accuracy on new points. Does clipping rescue a learning rate of 30?

<details>
<summary>Check yourself</summary>

```
loss at steps 0, 10, 100, 999  -> (0.702, 0.517, 0.288, 0.466)
dead units at the end          -> 0.156
accuracy on new points         -> 0.873
```

Partly. Each step is now at most 30 × 0.1 = 3 long, so far fewer units
die (15.6% instead of 100%) and the model learns something, 87.3% on
new points. But the loss went up again between step 100 and the end,
so it's still bouncing around. Clipping is a seatbelt, not a steering
wheel: it's common in big models to stop one bad batch from wrecking
training, but it's no substitute for a sensible learning rate.

</details>

---

That's why training fails. If one thing sticks: *when a network won't
learn, print the loss, the first layer's gradient and the dead share,
then check the learning rate, the initialisation and the input scale.*
