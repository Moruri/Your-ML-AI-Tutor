# Lesson 060 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-3-deep-learning/lesson-060-mini-project-train-and-debug-a-real-network/lesson.py my_lesson_060.py
python my_lesson_060.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-3-deep-learning/lesson-060-mini-project-train-and-debug-a-real-network").resolve()
```

`load_data`, `subset`, `make_model`, `RiderNet`, `train`, `print_log`,
`accuracy`, `r3` and `show` are there to reuse.

---

## 1. The first loss

A colleague's fresh 10-class classifier starts with a loss of 2.30.
Another's starts at 41. Which one would you worry about, and what
number would you expect?

<details>
<summary>Check yourself</summary>

The second. An untrained classifier that's unsure about all 10 classes
gives each about 1/10, and cross-entropy of that is $\ln 10 \approx
2.30$, so the first one is exactly right. A starting loss of 41 means
the model is very confident and very wrong from the first step:
usually huge initial outputs (unscaled inputs, a bad initialisation),
or labels that aren't what the loss expects.

</details>

## 2. Why overfit on purpose?

Section 2 trained on 16 rides until the loss hit zero. Isn't
overfitting the thing we're meant to avoid?

<details>
<summary>Check yourself</summary>

In real training, yes. As a *test*, it's the quickest check that the
whole pipeline can learn at all: the model has enough capacity, the
gradients reach every layer, the optimiser updates the weights and the
labels line up with the inputs. If a model can't memorise 16 examples,
it certainly can't learn from 824, and you've found that out in one
second instead of after a long run.

</details>

## 3. Which number to report?

After section 4, a teammate wants to tune ten more settings, each time
checking the *test* accuracy and keeping the best. What goes wrong?

<details>
<summary>Check yourself</summary>

The test set turns into a second validation set. Pick the best of ten
tries on 176 rides and you'll partly be picking whichever setting got
lucky on those particular rides, so the final number is optimistic.
Tune on validation, as many times as you like, then run the test set
once. If you need to keep experimenting after that, the honest move is
a fresh test set, or cross-validation (lesson 032) on the training
data.

</details>

## 4. Hands-on: improve it, using validation only

Try two changes to `RiderNet` and compare them with the section 4 run
(best epoch 37, validation loss 0.473):

- smaller and more regularised: `make_model(data, split, hidden=8, dropout=0.3)`
- bigger with no dropout: `make_model(data, split, hidden=64, dropout=0.0)`

Train each with `train(model, data, split)` (the default settings), and
report the best epoch, its validation loss and the validation accuracy
of the reloaded best model. Predict first: which way should you go on
a dataset this small? Don't touch the test set.

<details>
<summary>Expected results</summary>

```
hidden=8,  dropout=0.3   best epoch 64, val loss 0.464, val accuracy 0.767
hidden=32, dropout=0.1   best epoch 37, val loss 0.473, val accuracy 0.756   (section 4)
hidden=64, dropout=0.0   best epoch 25, val loss 0.480, val accuracy 0.75
```

Smaller wins. With 824 rides, the big network starts memorising sooner
(its best epoch comes earliest) and ends worse. The small, heavily
regularised one learns more slowly but generalises best. That fits
section 5's finding that a straight line already does well here: the
data doesn't need much model. If you were going to report a test score
for the small network, you'd now run the test set, once.

</details>

<details>
<summary>One way to write it</summary>

```python
data, split = load_data()
for kw in [dict(hidden=8, dropout=0.3), dict(hidden=64, dropout=0.0)]:
    model = make_model(data, split, **kw)
    log, best_epoch = train(model, data, split)
    model.eval()
    with torch.no_grad():
        val = subset(data, split["val"])
        acc = accuracy(model(*val[:3]), val[3])
    print(kw, best_epoch, log[best_epoch - 1]["val_loss"], acc)
```

</details>

## 5. (Optional) A different bug

Suppose, instead of misaligned labels, someone forgets `opt.zero_grad()`
in the training loop. Nothing crashes. What would you expect to see in
the log, and which column gives it away first?

<details>
<summary>Check yourself</summary>

Without `zero_grad`, PyTorch *adds* each batch's gradients to the ones
before, so the gradient keeps growing for the whole epoch. The
`grad_norm` column climbs batch after batch and epoch after epoch, far
beyond the healthy 0.2 to 0.7 in section 4. Plain SGD would take bigger
and bigger steps and the loss would bounce or blow up; Adam rescales
its steps, which can hide the damage for a while, but it's stepping in
the direction of a stale running total rather than the current batch.
A gradient norm that only ever grows is the clue. Try it in your copy.

</details>

---

That's the mini-project, and Phase 3. If one thing sticks: *a network
that trains without errors can still be broken; baselines, sanity
checks and a log are how you know whether it's really learning.*
