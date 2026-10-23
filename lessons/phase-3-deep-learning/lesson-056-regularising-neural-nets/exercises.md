# Lesson 056 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-3-deep-learning/lesson-056-regularising-neural-nets/lesson.py my_lesson_056.py
python my_lesson_056.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-3-deep-learning/lesson-056-regularising-neural-nets").resolve()
```

`ring_data`, `big_net`, `train`, `rotate`, `report_ring`, `loss_on`,
`predict_proba`, `accuracy`, `r3` and `show` are there to reuse. The
ring data is:

```python
train_ds, val, test = ring_data()   # 60 training points, 200 val, 400 test
```

---

## 1. Dropout's scaling

With `nn.Dropout(p=0.2)` in train mode, what values can a unit whose
output was 1.0 come out as, and how often? Why the scaling?

<details>
<summary>Check yourself</summary>

0 with probability 0.2, and $1/(1-0.2) = 1.25$ with probability 0.8.
The average is $0.8 \times 1.25 = 1.0$, the same as the original. In
eval mode nothing is dropped, so without the scaling the next layer
would see bigger numbers at prediction time than it ever saw in
training.

</details>

## 2. The forgotten `eval()`

A teammate's model has dropout. They call it twice on the same ride,
without `model.eval()`, and get 0.549 then 0.508. What happened, and
what's the fix?

<details>
<summary>Check yourself</summary>

The model is still in train mode, so each call drops a different random
set of units and gives a different answer. Call `model.eval()` before
predicting (and wrap it in `torch.no_grad()` while you're there), then
`model.train()` before training again. Lesson 052's `predict_proba`
does both.

</details>

## 3. Choosing on test

To pick the best epoch, why not just watch the *test* loss instead of
setting aside a validation set?

<details>
<summary>Check yourself</summary>

Because then the test set has helped make a choice, and its score is
no longer an honest guess about new data. It's the same leak as lesson
042's hyperparameter search: choose on validation, report on test.
With only one or two choices the leak is small, but it adds up fast
when you're tuning epochs, decay, dropout and learning rate all at once.

</details>

## 4. Hands-on: stack two defences, then lie

**a)** Train `big_net()` on the ring with both AdamW
`weight_decay=0.1` *and* `augment=rotate`, 300 epochs, seed 56. Report
train and test loss and test accuracy with `report_ring`. Does stacking
beat rotation alone (section 5)?

**b)** Now an augmentation that isn't true for this problem: shift
every point 0.5 to the right,
`lambda xb: xb + torch.tensor([0.5, 0.0])`. Plain Adam at 0.01, 300
epochs, seed 56. Predict first: better or worse than no augmentation at
all?

<details>
<summary>Expected results</summary>

```
a) decay 0.1 + rotation:  train / test loss (0.126, 0.218), test accuracy 0.96
b) shift right by 0.5:    train / test loss (3.142, 3.156), test accuracy 0.822
```

(a) A slightly better test loss than rotation alone (0.218 against
0.236), and a point lower on accuracy (96% against 97%). Defences
overlap: once rotation has taken away most of the memorising, there's
less left for decay to fix. Stack them, but check each one earns its
place on validation.

(b) Much worse than no defence at all (93%). The model learned the
circle in the wrong place, so even its *training* loss on the real,
unshifted points is over 3. An augmentation that changes the meaning of
the data teaches the model something false, very thoroughly.

</details>

<details>
<summary>One way to write it</summary>

```python
train_ds, val, test = ring_data()

torch.manual_seed(56)
model = big_net()
train(model, train_ds, val,
      torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1),
      augment=rotate)
print("decay 0.1 + rotation")
report_ring(model, train_ds, test)

torch.manual_seed(56)
model = big_net()
train(model, train_ds, val, torch.optim.Adam(model.parameters(), lr=0.01),
      augment=lambda xb: xb + torch.tensor([0.5, 0.0]))
print("shift right by 0.5")
report_ring(model, train_ds, test)
```

</details>

## 5. (Optional) Weight decay is a penalty

Show that, with plain SGD, `weight_decay` is exactly lesson 041's
squared-weight penalty. Make two identical networks
(`torch.manual_seed(56); a = big_net(); b = copy.deepcopy(a)`). Train
both for 100 full-batch steps at `lr=0.1` on the 60 ring points: `a`
with `torch.optim.SGD(..., weight_decay=0.01)`, and `b` with plain SGD
but `0.5 * 0.01 * sum(p.pow(2).sum() for p in b.parameters())` added to
its loss. How far apart do their weights end up?

<details>
<summary>Check yourself</summary>

```
largest difference between any two weights -> about 0.00000006
```

The same, down to float32 rounding. The gradient of
$\tfrac{\lambda}{2} w^2$ is $\lambda w$, which is exactly what
`weight_decay` adds to each gradient. (Note that torch's version decays
the biases too; lesson 041's Ridge didn't penalise the intercept.)
With Adam the two are *not* the same, because Adam rescales the penalty
along with the gradient. AdamW applies the decay separately, which is
the whole point of the "W", and why it's the version to use.

</details>

---

That's regularisation. If one thing sticks: *watch validation, not
training; keep the weights small, stop at the best epoch, and only
augment with changes you know are true.*
