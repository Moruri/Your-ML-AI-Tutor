# Lesson 060 - Mini-project: train and debug a real network

**Phase 3 - Deep learning foundations** | Week 8, Day 2 | Tuesday 2026-10-27

> **Goal:** Build, train, monitor and improve a classifier with proper
> logging and checkpoints.

Time: about 60 minutes. Same venv as lesson 051, with `torch` installed.

---

This is the last lesson of Phase 3, and it's a project rather than a new
idea. Eleven lessons gave us the parts: neurons, backprop, PyTorch,
`nn.Module`, activations, optimisers, DataLoaders, regularisation, and
the special layers for images, sequences and categories. Today we put
them together the way you would at work.

The job: given a ride's distance, duration, weather, bike type, hour and
day, say whether the rider is a **member** or a **casual** rider.

The model is the easy bit. The real subject today is the **routine**
around it: set the bar with baselines, check the plumbing before a long
run, log every epoch, save checkpoints you can reload, and open the
test set exactly once. We'll also break a run on purpose, because
training that fails quietly is the most common problem you'll face, and
the log is how you catch it.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-3-deep-learning/lesson-060-mini-project-train-and-debug-a-real-network/lesson.py
```

It writes `output/train_log.csv` and `output/best.pt` next to the
script. Every random step has a fixed seed, so your numbers will match.
It takes a few seconds.

## 1. The task, the split and the bar to beat

```
rides, input columns                           -> (1176, 7)
train / val / test                             -> (824, 176, 176)
share of members in train                      -> 0.697
baseline: always 'member', val accuracy        -> 0.648
baseline: logistic regression, val accuracy    -> 0.761
```

Seven inputs per ride: four numbers (distance, duration, temperature,
rain), electric or not, and the hour and day as categories. The split is
70/15/15 with a fixed seed: **train** to fit the weights, **validation**
to make every decision (when to stop, which settings to keep), and
**test**, locked away until the end.

Before writing a single layer, two baselines:

- **Always say "member".** 65% on validation, for free. Any model below
  that has learned worse than nothing.
- **Logistic regression** from lesson 033, with scaled numbers and
  one-hot hour and day: 76%. This is the bar a network has to clear to
  be worth its extra complexity.

## 2. Sanity checks before the long run

```
output shape for 824 rides                     -> (824,)
loss before training, and ln(2)                -> (0.744, 0.693)
loss after 300 steps on 16 rides               -> 0.0
trainable weights                              -> 471
```

The model, `RiderNet`, is small on purpose. Scaled numbers sit next to
an hour embedding (3 numbers) and a day embedding (2 numbers), from
lesson 059, and go into one hidden layer of 32 with dropout:

```python
x = torch.cat([(num - self.mean) / self.std, self.hour(hour), self.day(day)], dim=1)
return self.body(x).squeeze(-1)
```

One design choice to copy: the scaling `mean` and `std` are stored
**inside the model** with `register_buffer`. They're saved with the
weights and moved with them, so nobody can load the model later and
forget to scale the inputs the same way.

Then two checks that take a second and save hours:

1. **The first loss.** An untrained binary classifier should be unsure
   about everything, so its loss should be close to $\ln 2 \approx
   0.693$, the loss of a 50:50 guess. We get 0.744: close enough. A
   starting loss of 5 or 50 would mean something's off with the
   outputs or the labels.
2. **Overfit a tiny batch.** Can the model memorise 16 rides? With 471
   weights it should get the loss to zero, and it does. If it can't,
   don't bother with the full run: the model, the loss or the data
   pipeline is broken.

Both pass. Now, the real run.

## 3. A run that goes wrong, and reading the log

```
a run with a hidden bug:
epoch  train_loss  val_loss  val_acc  grad_norm
    1       0.711     0.643    0.716      0.539
    2       0.646     0.614    0.648      0.372
    5       0.615     0.629    0.648      0.281
   10       0.609      0.64    0.648      0.261
   20       0.598     0.662    0.648      0.264
   30       0.582     0.652    0.648      0.364
loss of just guessing the 0.697 share          -> 0.614
val accuracy of always saying 'member'         -> 0.648
```

First, a run with a bug in it. It doesn't crash. There's no error. The
loss even goes down a little. This is what makes deep learning bugs so
nasty: broken training usually *looks like* training.

Every epoch, `train` writes one row: the train loss, validation loss,
validation accuracy and the largest gradient norm. Read it like a
doctor reads a chart:

- **Validation accuracy is 0.648 from epoch 2 on, and never moves.**
  That's exactly the "always member" baseline. The model has learned to
  say "member" to everyone.
- **Train loss sits near 0.614**, which is the loss you get from
  predicting the class share (0.697) for every ride, without looking at
  the inputs at all. The small fall after that is the model starting to
  memorise noise, and validation loss rises while it does.

Together, they say: *the inputs carry no information about the labels*.
And that's the bug. The labels were shuffled separately from the
inputs, so ride 1's features sit next to ride 400's label. It's the
kind of thing that happens when you sort or filter a table in one place
and not the other, or `zip` two lists in different orders.

Some patterns worth recognising in a log:

| What you see | Likely cause |
|---|---|
| Loss stuck at the "guess the class share" value | labels and inputs don't line up, or inputs are all zero/constant |
| Loss is `nan` or shoots up | learning rate too high, or unscaled inputs |
| Huge gradient norms | the same; try clipping (lesson 058) or a lower learning rate |
| Train loss falls, validation rises | overfitting: regularise or stop earlier (lesson 056) |
| Both losses fall very slowly | learning rate too low, or a dead network (lesson 053) |

## 4. The real run: logging and checkpoints

```
the same run with the labels lined up:
epoch  train_loss  val_loss  val_acc  grad_norm
    1       0.683      0.62    0.716      0.719
    2        0.58     0.558    0.733      0.459
    5       0.464     0.502    0.767      0.313
   10       0.442     0.488    0.773      0.304
   20       0.426     0.479    0.773      0.339
   37       0.407     0.473    0.756      0.224
   57       0.395     0.482    0.761      0.271
epochs run, best epoch                         -> (57, 37)
files written                                  -> ['best.pt', 'train_log.csv']
```

Same code, labels fixed. Now the log looks healthy: both losses fall
fast, then train loss keeps creeping down while validation loss
bottoms out at epoch 37 and starts to rise. After 20 epochs without a
new best, early stopping ends the run at epoch 57.

Notice that validation *accuracy* peaked earlier, at epoch 10, than
validation *loss*. That's normal. Accuracy only counts which side of 0.5
a prediction lands on, so it's jumpy on 176 rides; loss also rewards
being confident when right and unsure when wrong. We choose by loss and
report accuracy.

Two files, both in `output/`:

- **`train_log.csv`**: every epoch's row. Open it in pandas or a
  spreadsheet and plot it. Keep it; when someone asks in three weeks
  "did you try a lower learning rate?", the log answers.
- **`best.pt`**: a **checkpoint**, saved every time validation loss
  hits a new best:

```python
torch.save({"epoch": epoch, "model": best_state,
            "optimizer": opt.state_dict(), "val_loss": val_loss}, ckpt_path)
```

Saving the optimiser too means you could carry on training from that
point, with AdamW's running averages intact. On a long run, checkpoints
are also your insurance against a crash at hour nine.

## 5. Reload, then open the test set once

```
checkpoint: epoch, val loss                    -> (37, 0.473)
reloaded model gives identical outputs         -> True
test accuracy: always 'member'                 -> 0.608
test accuracy: logistic regression             -> 0.75
test accuracy: the network                     -> 0.733
casuals caught, members caught                 -> ('32 of 69', '97 of 107')
median distance (km): wrong, right             -> (3.59, 2.38)
```

First we prove the checkpoint works: a fresh model, the saved weights
loaded in, and its outputs match the trained model's exactly. A
checkpoint you've never reloaded is a checkpoint you don't have.

Then, once, the test set. The network gets 73%, well above the 61% of
always saying "member".

And here's the honest bit: **logistic regression scores a little
higher, at 75%.** On 1,176 rides and seven columns, a straight line
already captures most of what there is (long, slow rides are usually
casual riders), and a network has nothing extra to find. On small
tables, the Phase 2 tools often win, and that's a result worth
reporting, not hiding. Networks pull ahead with more data, and with
data that a straight line can't handle: images, sound, text.

The last two lines show *where* the model struggles. It catches 97 of
107 members but only 32 of 69 casual riders, and the rides it gets
wrong are longer than the ones it gets right. A casual rider on a short
hop looks exactly like a member. If catching casual riders mattered,
lesson 043's threshold tuning is the next step.

## The routine, in one list

This is the checklist for every network from here on:

1. Split first, and lock the test set away.
2. Set baselines: the trivial one and the simplest sensible model.
3. Check the first loss, and overfit a tiny batch.
4. Log every epoch: train loss, validation loss, a metric, gradient norm.
5. Keep the best checkpoint by validation loss, and reload it to prove it works.
6. Change one thing at a time, judged on validation.
7. Open the test set once, at the end, and report what it says.

## What you can do now

- Plan a training run around baselines and a train/validation/test split.
- Run the sanity checks: the starting loss and the tiny-batch overfit.
- Read a training log and match its patterns to likely bugs.
- Save and reload checkpoints, with the scaling stored inside the model.
- Report a test result honestly, even when a simpler model wins.

## What to do now

1. Run `lesson.py`, then open `output/train_log.csv` and plot both
   losses against epoch with pandas and matplotlib (lesson 022).
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   asks you to improve the network using only the validation set.
3. Next weekday, lesson 061, the start of Phase 4: text as data. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

That's Phase 3 done. You've built a neuron by hand and grown it into
real networks, and you know how to tell when one is broken. Phase 4
takes these tools to the systems that made AI famous: language models,
attention and Transformers.
