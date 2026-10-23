# Lesson 056 - Regularising neural nets

**Phase 3 - Deep learning foundations** | Week 7, Day 5 | Friday 2026-10-23

> **Goal:** Use dropout, weight decay, early stopping and data
> augmentation to fight overfitting.

Time: about 45 minutes. Same venv as lesson 051, with `torch` installed.

---

Lesson 041 tamed overfitting in linear models with a penalty on big
weights. Neural networks need it more: a modest network has thousands
of weights, and given enough epochs it will happily memorise every
training example, noise and all. Today we watch that happen, then try
the four defences everyone uses:

| Defence | The idea in one line | In torch |
|---------|---------------------|----------|
| **Weight decay** | Keep the weights small | `torch.optim.AdamW(..., weight_decay=0.1)` |
| **Dropout** | Randomly switch off units while training | `nn.Dropout(p=0.3)` |
| **Early stopping** | Stop when validation stops improving, keep the best weights | A few lines in your loop |
| **Data augmentation** | Train on altered copies that keep the same label | A function applied to each batch |

All of them use lesson 055's split: train on one part, make choices on
**validation**, and touch **test** only to report.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-3-deep-learning/lesson-056-regularising-neural-nets/lesson.py
```

It writes no files. Every random step has a fixed seed, so your numbers
will match. It takes about twenty seconds.

## 1. Watching a network overfit

```
training points                                -> 60
weights in the network                         -> 4417
(train loss, val loss) at epoch                -> ''
  epoch 10                                     -> (0.104, 0.31)
  epoch 30                                     -> (0.039, 0.402)
  epoch 100                                    -> (0.031, 0.577)
  epoch 300                                    -> (0.001, 0.952)
best val loss was at epoch                     -> 8
No defences, 300 epochs:
  train / test loss                            -> (0.001, 0.823)
  test accuracy                                -> 0.933
```

To make overfitting easy to see, the script gives a big network a small
problem: just 60 ring points to learn from, and a network of two hidden
layers of 64 ReLU units, 4,417 weights. That's over 70 weights per
training point. Separate rings of 200 points (validation) and 400
points (test) check its work.

Read the pairs down the list. Training loss falls all the way to 0.001:
the network gets every one of the 60 points right, with total
confidence. Validation loss does the opposite. It's best at epoch 8,
then climbs all the way to 0.95. Every epoch after 8 is spent learning
things about these particular 60 points that aren't true of the circle.

One more thing worth noticing: test *accuracy* is still 93%. Accuracy
only asks "which side of 0.5?", while loss also asks "how sure were
you?". The overfitted network is wrong about the same points a sensible
one would be, but now it's wildly, confidently wrong about them, and
that's what a test loss of 0.823 is telling you. Overconfidence is
often the first sign of overfitting, before accuracy moves at all.

## 2. Weight decay

```
AdamW, weight_decay=0.0
  size of all weights                          -> 22.798
  train / test loss                            -> (0.001, 0.823)
  test accuracy                                -> 0.933
AdamW, weight_decay=0.1
  size of all weights                          -> 16.716
  train / test loss                            -> (0.023, 0.438)
  test accuracy                                -> 0.923
AdamW, weight_decay=1.0
  size of all weights                          -> 10.133
  train / test loss                            -> (0.13, 0.272)
  test accuracy                                -> 0.91
```

**Weight decay** shrinks every weight a little towards zero at each
step. It's lesson 041's Ridge penalty in neural-network clothing (with
plain SGD the two are exactly the same; exercise 5 proves it). You met
it at the end of lesson 054: it's the "W" in AdamW.

```python
optimiser = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
```

Why does it help? Very confident, very sharp boundaries need big
weights. Keep the weights small and the network's answers stay smooth
and modest. The "size of all weights" line (the square root of the sum
of squared weights, ignoring biases) drops from 22.8 to 10.1, and test
loss more than halves, from 0.823 to 0.438 at 0.1, and 0.272 at 1.0.

Be honest about the accuracy column, though: it dips a little, from
93.3% to 91%. Decay of 1.0 is pulling hard enough to round off the
circle's edge slightly. Like Ridge's `alpha`, `weight_decay` is a dial
to tune on validation, and 0.01 to 0.1 is the usual starting range.

## 3. Dropout

```
Dropout(0.5) in train mode, on ones            -> [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]
same layer in eval mode                        -> [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
dropout p=0.0
  train / test loss                            -> (0.001, 0.823)
  test accuracy                                -> 0.933
dropout p=0.3
  train / test loss                            -> (0.03, 0.638)
  test accuracy                                -> 0.93
```

**Dropout** is the strangest-sounding idea in this lesson, and one of
the most used. During training, each unit's output is set to zero with
probability `p`, a fresh random choice every batch. The survivors are
scaled up by $1/(1-p)$ so the average stays the same. With `p=0.5`,
that's zero or two, as the first line shows (six zeros out of eight is
just this draw's luck).

Why would knocking out units help? Because no unit can count on any
other being there. A network can't build a delicate chain of units that
together memorise one odd training point; everything it learns has to
work even with pieces missing. That pushes it towards simpler, more
robust patterns.

In the network it's just another layer, after each activation:

```python
nn.Linear(2, 64), nn.ReLU(), nn.Dropout(0.3),
```

The second line is the part people get wrong. Dropout only acts in
**train mode**. `model.eval()` switches it off, so predictions use the
whole network, every time. Forget `model.eval()` and your predictions
come out different every time you ask; that's why lesson 052's
`predict_proba` has always started with it.

Here, dropout 0.3 cuts test loss from 0.823 to 0.638: it helps, but on
this tiny problem it's the weakest of the four. Dropout shines on big
networks with lots of units to share the work.

## 4. Early stopping

```
patience                                       -> 20
best epoch, stopped at epoch                   -> (8, 28)
weights restored from the best epoch:
  train / test loss                            -> (0.135, 0.237)
  test accuracy                                -> 0.95
```

Section 1 already told us the answer: validation loss was best at epoch
8. **Early stopping** turns that into a rule. After every epoch, check
validation loss. If it's the best yet, save a copy of the weights. If
it hasn't improved for `patience` epochs, stop, and load the saved best
weights back in.

```python
if val_loss < best:
    best, best_epoch = val_loss, epoch
    best_state = copy.deepcopy(model.state_dict())
elif epoch - best_epoch >= patience:
    model.load_state_dict(best_state)
    break
```

`state_dict()` is every weight in the model, by name. The
`copy.deepcopy` matters: without it, you'd save a reference to weights
that keep changing.

With a patience of 20, training stops at epoch 28 and rolls back to
epoch 8. Test loss 0.237, test accuracy 95%, the best so far, and it
finished more than ten times sooner than 300 epochs. Early stopping is
nearly free and almost always worth having. Just remember the best
epoch was chosen using validation data, which is why we report on a
separate test set.

## 5. Data augmentation

```
a point, its distance from centre              -> ([0.6, 0.8], 1.0)
rotated, its distance from centre              -> ([0.806, 0.591], 1.0)
random rotation of every batch, 300 epochs:
  train / test loss                            -> (0.093, 0.236)
  test accuracy                                -> 0.97
```

The best cure for too little data is more data. Sometimes you can make
it. **Data augmentation** means changing each training example in a way
that you *know* doesn't change its label.

For the ring, the label only depends on distance from the centre, so
spinning a point around the centre keeps its label. The script's
`rotate` function gives every point in every batch its own random
angle. The point (0.6, 0.8) moves to (0.806, 0.591), still exactly 1.0
from the centre.

```python
for xb, yb in loader:
    xb = rotate(xb)            # new, label-preserving versions every batch
    ...
```

The effect is the biggest yet: 97% test accuracy, test loss 0.236,
even after the full 300 epochs with no stopping. Each epoch the network
sees 60 points it has never seen before, so there's nothing fixed to
memorise.

The catch is the word *know*. The rotation works because it's true of
this problem. For photos, people flip, crop and nudge the colours: a
cat flipped left to right is still a cat. But a "6" rotated half a turn
is a "9". Exercise 4 shows what a wrong augmentation does.

## 6. Putting it together on the rides

```
train / val / test rides                       -> (824, 176, 176)
big net, no defences
  epochs run, best epoch                       -> (150, 7)
  train / test AUC                             -> (0.967, 0.837)
+ dropout 0.3, AdamW decay 0.1, patience 20
  epochs run, best epoch                       -> (21, 1)
  train / test AUC                             -> (0.853, 0.851)
```

Back to real data, with lesson 055's seeded split, and the same big
network (now with six inputs). No defences, 150 epochs: train AUC 0.967,
test 0.837. A gap of 0.13 is the memorising we saw on the ring.

With dropout, weight decay and early stopping together, the best
validation loss came after just the first epoch (26 steps of 32 rides),
and training stopped at epoch 21. Train and test AUC now agree, 0.853
and 0.851, and test is better than the undefended run.

Notice also what *didn't* happen: no defence pushed test AUC past
lesson 055's small 8-unit network (0.854). Six features about a bike
ride only carry so much information, and the small network had already
found most of it. Regularisation doesn't add knowledge; it stops a big
model from making things up. That's what lets you use big models
safely, and on problems with more to learn, like the images in lesson
057, that's where the gains come from.

## What you can do now

- Spot overfitting from training and validation loss, and notice
  overconfidence before accuracy moves.
- Add weight decay with AdamW and treat it as a dial to tune.
- Add `nn.Dropout` layers, and explain why `model.eval()` matters.
- Write early stopping with patience, saving and restoring
  `state_dict()`.
- Design a data augmentation that keeps labels true, and recognise one
  that doesn't.

## What to do now

1. Run `lesson.py`. Then in section 4 try a patience of 5 and of 100,
   and predict where each stops.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   combines two defences, then tries an augmentation that lies.
3. Next weekday, lesson 057: Convolutional networks, where augmentation
   and dropout really earn their keep on images. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

A sensible default for any new network: AdamW with a little weight
decay, a validation set, and early stopping. Reach for dropout and
augmentation when the gap between train and validation stays wide.
