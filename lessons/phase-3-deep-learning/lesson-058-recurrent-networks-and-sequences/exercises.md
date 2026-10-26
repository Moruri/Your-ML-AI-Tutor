# Lesson 058 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-3-deep-learning/lesson-058-recurrent-networks-and-sequences/lesson.py my_lesson_058.py
python my_lesson_058.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-3-deep-learning/lesson-058-recurrent-networks-and-sequences").resolve()
```

`hourly_counts`, `windows`, `Forecaster`, `fit`, `mae`, `rmse`,
`memory_task`, `remember_by_default`, `r3` and `show` are there to
reuse.

---

## 1. Shapes in, shapes out

`gru = nn.GRU(input_size=3, hidden_size=16, batch_first=True)` gets a
batch of shape `(32, 24, 3)`. What are the shapes of `out` and `h` in
`out, h = gru(x)`?

<details>
<summary>Check yourself</summary>

`out` is `(32, 24, 16)`: the hidden state at every one of the 24 steps,
for all 32 windows. `h` is `(1, 32, 16)`: just the last step's hidden
state. The leading 1 is the number of stacked layers, which is why the
forecaster uses `h[-1]`, the top layer's state. For a forecast you
usually want `h`; to label every step (say, tag every word in a
sentence) you'd use `out`.

</details>

## 2. Counting weights

Explain the 304 weights in `nn.RNN(1, 16)`, and why `nn.GRU(1, 16)` has
exactly three times as many.

<details>
<summary>Check yourself</summary>

$W_x$ is 16x1 (16), $W_h$ is 16x16 (256), plus two bias vectors of 16
(32): 304. A GRU computes three things with that same shape of weights,
the reset gate, the update gate and the candidate memory, so it has
three of each: 912. An LSTM has four, so 1,216.

</details>

## 3. The shuffled split

A teammate splits the ride windows at random into train and test, and
the GRU's test MAE drops well below 1. Why shouldn't you believe it?

<details>
<summary>Check yourself</summary>

Neighbouring windows overlap by 23 hours. With a random split, almost
every test window has near-copies in training, from just before *and*
just after it, so the model has effectively seen the answer. It's
leakage, in time-series form. Train on the past and test
on the future, every time, because that's the only way the model will
ever be used.

</details>

## 4. Hands-on: what were the hour features worth?

Section 3 gave every hour three features: the count, and the hour of
day as sine and cosine. Train the same GRU forecaster with the **count
only**. Keep everything else the same: the same windows (take just the
first feature with `X[..., :1]`), the same split, seed 58,
`Forecaster("gru", n_inputs=1)`, `nn.MSELoss()`, `epochs=100,
patience=15`. Report the best epoch, MAE and RMSE. Predict first: does
it still beat "same hour yesterday" (1.071, 2.032)?

<details>
<summary>Expected results</summary>

```
GRU, count only:     best epoch 19, (MAE, RMSE) -> (1.254, 2.221)
GRU, count + hour:   best epoch 29, (MAE, RMSE) -> (1.039, 1.729)
```

Without the hour, the GRU falls behind the baseline on both scores. In
principle it could work out the time of day from the pattern of the
last 24 hours (the overnight zeros are a clue), but with only 16 days
to learn from, it doesn't. Two cheap, obvious features did more than
the network could find on its own. That's the usual lesson with small
data: a little domain knowledge is worth a lot of model.

</details>

<details>
<summary>One way to write it</summary>

```python
counts = hourly_counts()
X, y = windows(counts)
n_train, n_val = 16 * 24, 3 * 24
tr = slice(0, n_train)
va = slice(n_train, n_train + n_val)
te = slice(n_train + n_val, None)

torch.manual_seed(58)
model = Forecaster("gru", n_inputs=1)
best_epoch = fit(model, (X[tr][..., :1], y[tr]), (X[va][..., :1], y[va]),
                 nn.MSELoss(), epochs=100, patience=15)
model.eval()
with torch.no_grad():
    pred = model(X[te][..., :1])
print(best_epoch, mae(pred, y[te]), rmse(pred, y[te]))
```

</details>

## 5. (Optional) A hundred small factors

Suppose that, on average, each step backwards through an RNN multiplies
the gradient by 0.7. What's left after 100 steps? And if a gate lets
each step multiply by 0.99 instead?

<details>
<summary>Check yourself</summary>

$0.7^{100} \approx 3 \times 10^{-16}$: gone. $0.99^{100} \approx 0.37$:
most of it survives. That's the whole difference between section 5's
rows. Nothing about one step is dramatic; it's the repetition. (The
opposite problem exists too: factors a little above 1 make gradients
*explode*, which is why the script's `fit` clips them with
`clip_grad_norm_`.)

</details>

---

That's recurrence. If one thing sticks: *an RNN carries a memory from
step to step with the same weights each time; plain ones forget fast
because gradients shrink at every step, and gates are how a network
learns what to keep.*
