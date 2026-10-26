# Lesson 058 - Recurrent networks and sequences

**Phase 3 - Deep learning foundations** | Week 8, Day 1 | Monday 2026-10-26

> **Goal:** Model ordered data with RNNs/GRUs and see where they
> struggle.

Time: about 45 minutes. Same venv as lesson 051, with `torch` installed.

---

Lesson 057's CNN cared about *where* things are in an image. Today's
data cares about *when*. Rides hour by hour, words in a sentence, a
heartbeat, a share price: shuffle the order and the meaning goes.

A **recurrent neural network** (RNN) reads a sequence one step at a
time and carries a small memory from each step to the next, called the
**hidden state**. One line describes it:

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

The new memory $h_t$ mixes the new input $x_t$ with the old memory
$h_{t-1}$. The same weights are used at every step, like a CNN's kernel
sliding across time instead of space.

We'll build one by hand, forecast next hour's rides with it, and then
break it, because the way plain RNNs fail is the reason the rest of
modern AI looks the way it does.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-3-deep-learning/lesson-058-recurrent-networks-and-sequences/lesson.py
```

It writes no files. Every random step has a fixed seed, so your numbers
will match. It takes about half a minute.

## 1. Rides as a sequence

```
hours, from, to                                -> (720, '2026-09-01 00:00:00', '2026-09-30 23:00:00')
rides in each hour of 1 September              -> [0, 0, 0, 0, 0, 0, 0, 1, 19, 1, 1, 0, 7, 6, 2, 0, 1, 15, 5, 1, 1, 0, 0, 0]
windows, shape (samples, time steps, features) -> (696, 24, 3)
targets, shape                                 -> (696,)
two sequences, their means                     -> (1.8, 1.8)
```

The script counts our 1,176 rides into the 720 hours of September,
zeros included. Look at the first day: nothing overnight, a rush at
8am, a bump at lunch, another rush at 5pm. That shape repeats, and it's
what we'll try to forecast.

Each training example is a **window**: 24 hours of history as the
input, and the hour straight after it as the target. Every hour in the
window has three features: its ride count (divided by 10, to keep
inputs small), and the hour of the day written as a sine and cosine, so
the model knows 23:00 sits right next to 00:00. That gives the shape
every torch recurrent layer expects with `batch_first=True`:
**(samples, time steps, features)**.

The last line is the reason we need a new kind of model. The sequences
`[0, 0, 8, 1, 0]` and `[8, 1, 0, 0, 0]` have the same mean, the same
values, but they tell different stories: a rush that just happened, or
one that was hours ago. Order is information.

## 2. A recurrent cell, by hand

```
hidden state after each step (by hand):
  step 0, input 0.0                            -> [-0.457, 0.67, 0.232]
  step 1, input 0.8                            -> [-0.007, 0.357, -0.012]
  step 2, input 0.1                            -> [-0.283, 0.546, 0.085]
  step 3, input 0.0                            -> [-0.092, 0.511, -0.001]
nn.RNN's final hidden state                    -> [-0.092, 0.511, -0.001]
nn.RNN weights: W_x, W_h, biases               -> ((3, 1), (3, 3), 6)
```

Here's the formula as a loop, using the weights from a small
`nn.RNN` with a hidden state of three numbers:

```python
h = torch.zeros(3)
for x in xs:
    h = torch.tanh(W_x @ x + W_h @ h + b)
```

Feed it four inputs and watch the memory change. The 0.8 at step 1
moves all three numbers, and its effect is still there two steps later
(step 3's state isn't the same as step 0's, even though both inputs
were zero). `nn.RNN` gives exactly the same final state; it's the same
loop, written in fast code.

The weights: $W_x$ is 3x1 (input to memory), $W_h$ is 3x3 (old memory
to new memory), and there are two bias vectors of three (torch keeps
two, though one would do). **The same weights at every step**, so a
sequence of 4 hours or 4,000 needs no extra weights.

## 3. Forecasting the next hour

```
train / val / test windows                     -> (384, 72, 240)
(MAE, RMSE) in rides per hour                  -> ''
  baseline: same as the hour before            -> (1.583, 3.085)
  baseline: same hour yesterday                -> (1.071, 2.032)
  RNN (best epoch 9)                           -> (1.368, 2.054)
  GRU (best epoch 29)                          -> (1.039, 1.729)
first test day                                 -> '2026-09-21'
  hours                                        -> [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  GRU predicted                                -> [3, 7, 4, 1, 2, 4, 4, 3, 1, 1, 5, 5, 0]
  actual                                       -> [0, 5, 4, 0, 0, 3, 5, 0, 0, 0, 8, 3, 1]
```

The model is short. The recurrent layer reads the whole window, and a
linear layer turns its last hidden state into one number:

```python
out, h = self.recurrent(x)        # out: every step, h: the last one
return self.head(h[-1]).squeeze(-1)
```

**The split is by time, never shuffled.** The first 16 days of windows
train, the next 3 validate, the last 10 test. If we shuffled, the model
would train on Thursday and be tested on Wednesday: forecasting the
past, with the future as a hint. Real forecasts never get that.

Before any network, two **baselines**. "Same as the hour before" is
poor, because rides come in rushes. "Same hour yesterday" is much
better, 1.07 rides off on average, because the daily pattern is so
strong. That's the bar to beat, and it's a high one.

The training loop is lesson 056's, with mean squared error as the loss.
Two scores: **MAE** (the average miss, in rides) and **RMSE** (which
squares the misses first, so big misses count extra; lesson 030 met
both).

The plain RNN doesn't clear the bar: it's worse than "same hour
yesterday" on MAE. The **GRU**, which we'll explain in section 5,
does: a little better on MAE (1.04) and clearly better on RMSE (1.73
against 2.03). It's reading the whole last day, so it can see if today
is busier or quieter than usual.

The day of predictions is honest about what that means. The GRU knows
8am and 5pm are busy, but with 1,176 rides in a whole month, any given
hour is a handful of people deciding to ride. It can't predict which
quiet hour will get zero. Small, noisy data is a hard place for any
model, and a baseline that's hard to beat is a sign you understand the
problem, not a failure.

## 4. Where plain RNNs struggle: long memory

```
Task: say whether the FIRST number was positive. The rest is noise.
sequences of length 10, test accuracy:
  RNN                                          -> 0.992
  GRU                                          -> 0.996
  GRU, remembering by default                  -> 0.998
sequences of length 100, test accuracy:
  RNN                                          -> 0.512
  GRU                                          -> 0.488
  GRU, remembering by default                  -> 0.998
```

To see the weakness clearly, a made-up task. Each sequence is random
numbers; the label is just "was the *first* one positive?". The model
has to notice the first number, then hold on to that one fact while
ignoring everything else.

With 10 steps, everyone gets it right. With 100 steps, the plain RNN
gets 51%: a coin flip. It has completely failed to learn that the
answer was at the very start.

The plain GRU fails too, which is the interesting part. A GRU *can*
remember: it has an **update gate**, a number between 0 and 1 for each
memory slot, that decides how much of the old memory to keep at each
step:

$$
h_t = z_t \cdot h_{t-1} + (1 - z_t) \cdot \tilde h_t
$$

When $z_t$ is near 1, the slot is copied forward almost untouched; near
0, it's replaced by the new candidate $\tilde h_t$ (which is the plain
RNN formula, with a second gate, the *reset* gate, inside it). But at
the start of training, the gates sit around 0.5, so the GRU forgets
half of everything at each step, just like the RNN, and never gets a
chance to learn to remember.

The script's `remember_by_default` sets the update gate's bias to 2.0,
so at the start each gate is about 0.88: keep most of the memory unless
there's a reason not to. One line of setup, and the GRU goes from 49%
to 99.8%. (The same trick for LSTMs, the GRU's older cousin, is
standard: their "forget gate" bias usually starts at 1.)

## 5. Why: the vanishing gradient

```
size of the gradient reaching the input at step 99, 90, 50, 0:
  RNN                                          -> (0.11, 3.8e-05, 3e-14, 3.7e-25)
  GRU                                          -> (0.083, 0.00027, 1.1e-09, 8.5e-17)
  GRU, remembering by default                  -> (0.021, 0.0041, 3e-08, 1.3e-06)
weights in RNN(1, 16), GRU(1, 16)              -> (304, 912)
```

Section 4's failure has a name: the **vanishing gradient**. Lesson 053
met it in deep networks, and an RNN unrolled over 100 steps *is* a
100-layer network, with the same weights in every layer.

The script asks backprop how much the output depends on each input.
For the plain RNN, the last input gets a gradient of 0.11. Ten steps
back, it's thousands of times smaller. By step 0 it's $3.7 \times
10^{-25}$, a number so small it's effectively zero. Backprop goes
through one $W_h$ multiply and one tanh slope per step, and if each of
those shrinks things even a little, a hundred of them shrink it to
nothing. (Exercise 5 does the sum.) Training can't learn from a signal
it can't see.

The remembering GRU's gradient at step 0 is $1.3 \times 10^{-6}$:
still small, but about ten billion times larger than the default GRU's,
and enough for training to find. When the update gate says "keep", the
memory passes straight through, with no multiply by $W_h$ and no tanh
to shrink it.

The cost of the gates: a GRU has three sets of weights where the RNN
has one (912 against 304). Worth it almost every time.

And this is where the story of modern AI turns. Even gated networks
must pass everything through a single memory, one step at a time, which
is slow and still leaky over thousands of steps. In Phase 4, lessons
062 and 063 meet **attention** and the **Transformer**, which skip the
chain entirely and let every step look directly at every other.

## What you can do now

- Explain why order matters, and shape sequence data as (samples, time
  steps, features).
- Write the RNN update by hand and match it to `nn.RNN`.
- Build a forecaster from `nn.RNN` or `nn.GRU` plus a linear head.
- Split time series by time, and compare against honest baselines.
- Explain the vanishing gradient, what a GRU's update gate does about
  it, and why its starting bias matters.

## What to do now

1. Run `lesson.py`. In section 4, try `remember_by_default` with a bias
   of 0.0 and of 4.0, and predict what happens to each first.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   asks how much the hour-of-day features were really worth.
3. Next weekday, lesson 059: Embeddings, turning categories into
   geometry. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

Sequence models taught the field to think about memory: what to keep,
what to forget, and how far back a model can really see. Those
questions don't go away with Transformers; they just get better
answers.
