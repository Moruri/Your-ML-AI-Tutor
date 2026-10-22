# Lesson 054 - Optimisers and learning-rate schedules

**Phase 3 - Deep learning foundations** | Week 7, Day 4 | Thursday 2026-10-22

> **Goal:** Compare SGD, momentum and Adam, and use schedulers to train
> faster and more stably.

Time: about 45 minutes. Same venv as lesson 051, with `torch` installed.

---

Every network so far has learned the same way: work out the gradient,
step a fixed distance downhill, repeat. Lesson 053 showed how delicate
that fixed distance is: too small crawls, too big breaks things. This
lesson is about the upgrades almost everyone uses instead:

| Tool | The idea in one line | In torch |
|------|---------------------|----------|
| **Momentum** | Keep part of the last step, so you build up speed in a steady direction | `torch.optim.SGD(..., momentum=0.9)` |
| **Adam** | Give every weight its own step size, based on its gradient history | `torch.optim.Adam(...)` |
| **Schedules** | Change the learning rate as training goes on | `torch.optim.lr_scheduler` |

Lesson 052 promised that swapping optimisers would be a one-line
change, because they all share `step()` and `zero_grad()`. Today we
cash that in and race them fairly.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-3-deep-learning/lesson-054-optimisers-and-learning-rate-schedules/lesson.py
```

It writes no files. Every random step has a fixed seed, so your numbers
will match. It takes about ten seconds.

## 1. A ravine, and plain gradient descent

```
start at w                                     -> (-4.0, 1.0)
starting loss                                  -> 33.0
gradient there                                 -> [-4.0, 50.0]
SGD, lr 0.01                                   -> 'reached in 448 steps'
SGD, lr 0.035                                  -> 'reached in 127 steps'
SGD, lr 0.041                                  -> 'blew up by step 109'
```

Before any network, a landscape small enough to picture. Two weights,
and a loss shaped like a long, narrow valley:

$$
\text{loss}(w) = \tfrac{1}{2}\left(w_0^2 + 50\, w_1^2\right)
$$

It's 50 times steeper across the valley ($w_1$) than along it ($w_0$).
Real loss landscapes are full of valleys like this, just in millions of
dimensions. We start at (-4, 1) and count the steps to get the loss
under 0.001.

Plain SGD has a problem. At the start the gradient is -4 in the gentle
direction and 50 in the steep one, and it uses one learning rate for
both. The steep direction sets the limit: above $2/50 = 0.04$, each
step across the valley overshoots further than the last, and at 0.041
it blows up. But at a safe 0.035, the gentle direction crawls, and it
takes 127 steps. At 0.01, 448.

## 2. Momentum: a heavy ball

```
SGD, lr 0.035, momentum 0.5                    -> 'reached in 58 steps'
SGD, lr 0.035, momentum 0.9                    -> 'reached in 87 steps'
same gradient 4 times: step sizes              -> [1.0, 1.9, 2.71, 3.439]
```

Picture a heavy ball rolling down the valley instead of a careful
walker. Momentum keeps a running **velocity**: each step is the new
gradient plus a fraction (the momentum, often 0.9) of the last step.

```python
velocity = momentum * velocity + gradient
w = w - lr * velocity
```

The last line shows why that helps. Feed the same gradient four times
with momentum 0.9 and the steps grow: 1, 1.9, 2.71, 3.44, heading for
10 times the plain step. Along the gentle floor of the valley, where the
gradient keeps pointing the same way, the ball speeds up. Across it,
where the gradient flips sign every step, the contributions cancel and
the zig-zag calms down.

Momentum 0.5 gets to the bottom in 58 steps instead of 127, at the same
learning rate. Momentum 0.9 takes 87: still faster than plain SGD, but
a heavier ball also overshoots the bottom and has to roll back. Like
the learning rate, it's a dial, and 0.9 is the usual starting point on
real networks, not a law.

## 3. Adam: a step size for every weight

```
gradient at the start                          -> (-4.0, 50.0)
SGD lr 0.035 would move by                     -> (0.14, -1.75)
Adam lr 0.1, first step moved by               -> [0.1, -0.1]
Adam, lr 0.1                                   -> 'reached in 95 steps'
Adam, lr 0.3                                   -> 'reached in 61 steps'
```

**Adam** (short for "adaptive moment estimation") goes after the
ravine's real problem: one step size for directions that need very
different ones. For every weight it keeps two running averages:

- the average gradient, which is momentum from section 2;
- the average *squared* gradient, a measure of how big that weight's
  gradients usually are.

Then it divides the first by the square root of the second. A weight
with big gradients gets its steps scaled down; a weight with small
ones gets them scaled up.

The first step shows it plainly. SGD would move 0.14 in the gentle
direction and 1.75 in the steep one, twelve times more. Adam moves both
by exactly 0.1, the learning rate. Each weight goes about `lr` per
step, whatever its gradient's size, which is why one Adam learning rate
suits most weights at once. At 0.3 it reaches the bottom in 61 steps,
without anyone needing to know the valley was 50 times steeper one way.

## 4. Racing optimisers on the ring

```
loss at steps                                  -> '100, 300, 1000'
SGD, lr 1.0                                    -> (0.359, 0.212, 0.169)
  accuracy on new points                       -> 0.945
SGD + momentum 0.9, lr 0.1                     -> (0.435, 0.212, 0.167)
  accuracy on new points                       -> 0.947
Adam, lr 0.01                                  -> (0.504, 0.224, 0.158)
  accuracy on new points                       -> 0.938
Adam, lr 0.05                                  -> (0.2, 0.156, 0.151)
  accuracy on new points                       -> 0.947
```

Back to lesson 052's `RingNet`. Same seed, same 1,000 steps, same
`fit` loop; only the optimiser line changes:

```python
optimiser = torch.optim.SGD(model.parameters(), lr=1.0)
optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
```

Notice momentum used a learning rate ten times smaller than plain SGD.
Momentum's steps build up to about $1/(1 - 0.9) = 10$ times the plain
step, so you divide the learning rate to match.

The honest result: on this small, friendly problem they all end up in
the same place, around 94-95% on new points. Adam at 0.05 gets there
fastest (0.2 at step 100). Plain SGD, with a learning rate we tuned in
earlier lessons, is perfectly respectable.

So why does nearly everyone reach for Adam first? Because of what you
*didn't* see: the tuning. Adam at 0.01 works acceptably on most
problems without a search, and a range of rates around it do too
(exercise 4 sweeps them).
Exercise 4 shows it rescuing lesson 053's ten-layer network, which SGD
couldn't train at all. With SGD, you have to find the right rate for
each new problem first.

## 5. Learning-rate schedules

```
lr shown at steps                              -> '0, 250, 500, 750, 1000'
StepLR, x0.1 every 400                         -> (0.05, 0.05, 0.005, 0.005, 0.0005)
CosineAnnealingLR, 1000 steps                  -> (0.05, 0.0427, 0.025, 0.0073, 0.0)
Adam 0.2, batches of 16, constant
  full loss at the end                         -> 0.183
  wobble over the last 100 steps               -> 0.145
  accuracy on new points                       -> 0.923
Adam 0.2, batches of 16, cosine
  full loss at the end                         -> 0.158
  wobble over the last 100 steps               -> 0.0
  accuracy on new points                       -> 0.938
```

Early in training you're far from good weights and big steps help.
Late on, you're close and big steps just bounce you around the
bottom. A **schedule** shrinks the learning rate as training goes on.
The two you'll see most:

- **`StepLR`** cuts the rate by a factor (here ×0.1) every so many
  steps: 0.05, then 0.005 from step 400, then 0.0005 from step 800.
- **`CosineAnnealingLR`** glides from the starting rate down to zero
  along half a cosine wave, over the number of steps you give it. No
  sudden drops, and only one number to choose: how long.

A scheduler wraps the optimiser, and you call its `step()` once per
training step, right *after* `optimiser.step()`:

```python
optimiser = torch.optim.Adam(model.parameters(), lr=0.2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=2000)
for step in range(2000):
    ...                       # zero_grad, forward, loss, backward
    optimiser.step()
    scheduler.step()
```

To see why it matters, the script trains a little differently: each
step learns from 16 random points instead of all 400. That's how real
training works (lesson 055 does it properly), and it makes every step
a bit noisy. With a constant, fairly high rate of 0.2, the model keeps
getting knocked about. Over the last 100 steps its loss on the full
data moves by 0.145, and it finishes at 0.183 with 92.3% on new points.
Same everything, plus a cosine schedule: the steps shrink to nothing,
the loss settles (wobble 0.0) at 0.158, and it scores 93.8%.

## 6. Same swap, the rides

```
SGD 0.5, 2000 steps (lesson 052)
  train / test AUC                             -> (0.864, 0.827)
Adam 0.01, 300 steps
  train / test AUC                             -> (0.852, 0.837)
AdamW 0.01 + decay 0.1, 300 steps
  train / test AUC                             -> (0.851, 0.841)
```

Last, lesson 052's casual-rider network with the same seed and split,
so the first line repeats lesson 052's result exactly: test AUC 0.827
after 2,000 SGD steps.

Adam reaches a test AUC of 0.837 in 300 steps, under a seventh of the
work. On a network this small that saves milliseconds; on one that
trains for a day, it saves most of the day.

The third line is **AdamW**, Adam with **weight decay**: every step
shrinks each weight slightly towards zero, the neural-network cousin
of lesson 041's Ridge penalty. Test AUC 0.841, matching lesson 050's
best, with the gap between train and test the smallest of the three.
AdamW is the default optimiser for most modern models, including the
Transformers of Phase 4. Lesson 056 digs into weight decay properly.

## What you can do now

- Explain why one learning rate struggles on a ravine-shaped loss.
- Describe momentum as a velocity, and why it pairs with a smaller rate.
- Explain what Adam adapts, and why it's an easy first choice.
- Swap optimisers in one line and compare them fairly with fixed seeds.
- Add `StepLR` or `CosineAnnealingLR`, and call `scheduler.step()` in
  the right place.
- Reach for AdamW when you want weight decay.

## What to do now

1. Run `lesson.py`. Then in section 2 try momentum 0.7 and 0.99, and
   predict the step counts before you look.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   lets Adam rescue lesson 053's failed deep network, and sweeps Adam's
   learning rate.
3. Next weekday, lesson 055: Datasets, DataLoaders and batching, the
   proper way to do what section 5 did by hand. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

A good default you can start almost anything with: AdamW at 0.001 to
0.01, with a cosine schedule over your total steps. Then measure,
change one thing, and measure again.
