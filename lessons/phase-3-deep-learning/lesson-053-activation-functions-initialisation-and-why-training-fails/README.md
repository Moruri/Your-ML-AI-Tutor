# Lesson 053 - Activation functions, initialisation and why training fails

**Phase 3 - Deep learning foundations** | Week 7, Day 4 | Thursday 2026-10-22

> **Goal:** Diagnose dead ReLUs, vanishing gradients and bad learning
> rates.

Time: about 45 minutes. Same venv as lesson 051, with `torch` installed.

---

In lesson 052 the five-line loop trained the ring network first time.
Enjoy that, because it won't always happen. Sooner or later you'll
press run and the loss will sit at the same number for a thousand
steps, or jump to `nan`, or drop a little and then stop while the
model predicts the same answer for everything.

The good news is that nearly every one of those failures comes from
one of three places:

| Where | The question it answers | What goes wrong |
|-------|------------------------|-----------------|
| The **activation function** | What does a neuron do after its weighted sum? | Its slope is tiny, so gradients fade, or zero, so neurons die |
| The **starting weights** | How big are the random numbers on day one? | Too small and the signal fades; too big and it explodes |
| The **learning rate** | How far does each step go? | Too small crawls; too big bounces and breaks things |

Today we break a network in each of those ways, on purpose, so you'll
recognise the symptoms when it happens by accident.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-3-deep-learning/lesson-053-activation-functions-initialisation-and-why-training-fails/lesson.py
```

It writes no files. Every random step has a fixed seed, so your numbers
will match. It takes about ten seconds.

## 1. Activation functions, side by side

```
inputs z                                       -> [-4.0, -1.0, 0.0, 1.0, 4.0]
sigmoid(z)                                     -> [0.018, 0.269, 0.5, 0.731, 0.982]
  its slope at z                               -> [0.018, 0.197, 0.25, 0.197, 0.018]
tanh(z)                                        -> [-0.999, -0.762, 0.0, 0.762, 0.999]
  its slope at z                               -> [0.001, 0.42, 1.0, 0.42, 0.001]
relu(z)                                        -> [0.0, 0.0, 0.0, 1.0, 4.0]
  its slope at z                               -> [0.0, 0.0, 0.0, 1.0, 1.0]
leaky_relu(z)                                  -> [-0.04, -0.01, 0.0, 1.0, 4.0]
  its slope at z                               -> [0.01, 0.01, 0.01, 1.0, 1.0]
gelu(z)                                        -> [-0.0, -0.159, 0.0, 0.841, 4.0]
  its slope at z                               -> [-0.001, -0.083, 0.5, 1.083, 1.001]
steepest sigmoid slope anywhere                -> 0.25
```

An activation function is the bend after each layer's weighted sum.
Without it, ten layers of `nn.Linear` collapse into one straight line
(lesson 050 showed why). Here are the five you'll meet most:

- **Sigmoid** squashes to between 0 and 1. Lovely for a final
  probability, poor inside a network: its slope is never above 0.25.
- **Tanh** squashes to between -1 and 1, slope up to 1. Better, but both
  ends go flat: at z = 4 the slope is 0.001.
- **ReLU** is `max(0, z)`. Slope exactly 1 for positive inputs, exactly
  0 for negative ones. Cheap, and the default in most networks.
- **LeakyReLU** is ReLU with a small slope (here 0.01) on the negative
  side, so it's never completely flat.
- **GELU** is a smooth ReLU; you'll see it inside Transformers in
  Phase 4.

The script gets each slope from autograd, and the slope is the number
to watch. Backprop (lesson 050) multiplies by it at every layer on the
way back. A slope near zero means "no message gets through here".

## 2. Vanishing gradients: ten layers deep

```
layers shown                                   -> '1st, 4th, 7th, 10th hidden, then output'
sigmoid: gradient size by layer                -> ['1.1e-10', '3.7e-07', '1.2e-04', '6.1e-02', '3.9e-01']
tanh: gradient size by layer                   -> ['1.7e-04', '1.8e-03', '9.3e-03', '7.2e-02', '1.1e-01']
relu: gradient size by layer                   -> ['1.2e-05', '9.8e-05', '1.8e-03', '3.4e-02', '5.1e-02']
sigmoid, 10 deep, 1000 steps: loss             -> 0.631
accuracy on new points                         -> 0.69
share of new points that are outside           -> 0.69
```

Now stack ten hidden layers of 32 units, built with the `deep_net`
helper, and measure how big each layer's gradient is before any
training. (The `1.1e-10` style is scientific notation: 1.1 × 10⁻¹⁰,
a decimal point followed by nine zeros.)

With sigmoid, the output layer gets a healthy 0.39. The first layer
gets 0.00000000011. Each layer on the way back multiplies by a slope of
at most 0.25, and ten of those multiplied together is under one in a
million, before the weights shrink it further. That's a **vanishing
gradient**.

Train it and you can see the effect. Loss 0.631, and 69% accuracy,
which sounds fine until you notice that 69% of the points are outside
the ring. The network has learned to say "outside" for everything. The
early layers never got a message, so they never learned anything.

The surprise is ReLU. Its slope is 1, so it should pass gradients
cleanly, yet its first layer gets only 0.000012. Something else is
shrinking the signal, and that's section 3.

## 3. Initialisation: how big should the starting weights be?

```
signal size after layers                       -> '1, 5 and 10 of 10, each 256 wide'
relu, std 0.01 (too small)                     -> (0.094, 0.0, 0.0)
relu, std 1.0 (too big)                        -> (9.401, 172152.078, 30696800256.0)
relu, Kaiming: sqrt(2 / 256)                   -> (0.831, 0.929, 0.893)
tanh, std 1.0: outputs stuck near +-1          -> 0.864
tanh, Xavier: (5/3) / sqrt(256)                -> (0.76, 0.654, 0.651)
nn.Linear's default weight spread              -> 0.036
Kaiming would want                             -> 0.088
relu, default init: first-layer grad, loss     -> ('1.2e-05', 0.631)
  accuracy on new points                       -> 0.69
relu, Kaiming init: first-layer grad, loss     -> ('2.6e-02', 0.075)
  accuracy on new points                       -> 0.957
tanh, Xavier init: first-layer grad, loss      -> ('1.8e-01', 0.267)
  accuracy on new points                       -> 0.925
```

Forget training for a moment. Push 1,000 random inputs through ten
random layers, each 256 wide, and watch how spread out the numbers are
(their standard deviation) after layers 1, 5 and 10.

- **Weights too small** (std 0.01): each layer shrinks the signal. By
  layer 5 it's 0.0. Nothing useful reaches the end, and on the way back
  nothing reaches the start.
- **Weights too big** (std 1.0): each layer multiplies it. By layer 10
  it's thirty billion. With tanh instead, 86% of outputs are pinned at
  ±1, right where section 1 showed the slope is flat.
- **Just right**: keep each layer's output about as spread out as its
  input. A weighted sum of `fan_in` inputs (here 256) grows by about
  $\text{std} \times \sqrt{\text{fan\_in}}$, and ReLU throws away half of
  it, so the fix is
  $\text{std} = \sqrt{2 / \text{fan\_in}}$. That's **Kaiming** (or
  **He**) initialisation, and it holds the signal near 0.9 all the way
  down. For tanh, the matching recipe is **Xavier** (or **Glorot**),
  with a small boost of 5/3 that torch calls the tanh "gain".

And here's the answer to section 2's puzzle. `nn.Linear`'s default
spread is $1/\sqrt{3 \times \text{fan\_in}}$, 0.036 for 256 inputs, a
good deal less than Kaiming's 0.088. Fine for one or two layers, which is
why lesson 052 never noticed. Over ten ReLU layers, it fades.

Swap in proper initialisation, two lines per layer:

```python
nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
nn.init.zeros_(layer.bias)
```

and the same ten-layer ReLU network that couldn't learn at all reaches
a loss of 0.075 and 95.7% on new points, as good as anything we've
managed on the ring.
The tanh version with Xavier gets 92.5%. Same model, same data, same
loop. Only the starting numbers changed.

## 4. Dead ReLUs

```
healthy start: dead units                      -> 0.094
after training, lr 0.5: dead units             -> 0.125
accuracy on new points                         -> 0.95
bias started at -3: dead units                 -> 1.0
after 2000 steps: dead units, loss             -> (1.0, 0.631)
accuracy on new points                         -> 0.69
same start, LeakyReLU(0.01): loss              -> 0.63
accuracy on new points                         -> 0.69
```

A ReLU unit is **dead** when it outputs 0 for every input you have. Its
slope is then 0 for every input too, so no gradient ever reaches its
weights, and it can never come back. The script counts them with
`dead_share`: the share of the 32 hidden units whose largest output
over all 400 points is exactly 0.

A few dead units are normal. A fresh network here starts with 9.4% dead,
ends with 12.5%, and still scores 95% on new points. It simply has
fewer useful units than you paid for.

Now push every bias to -3 before training. Each unit's weighted sum is
negative everywhere on the ring, so 100% are dead from step one. Two
thousand steps later, still 100%. The loss sits at 0.631, the "always
outside" answer from section 2. Nothing is broken in the code; there's
just no gradient to follow.

LeakyReLU's 0.01 slope means its units can't die outright, but from a
start this bad, that trickle of gradient is too slow to rescue it in
2,000 steps. The real fix is not to start there: sensible
initialisation, sensible learning rates (section 5) and scaled inputs
(section 6).

## 5. Learning rates: too small, about right, too big

```
lr 0.01: loss at steps 0, 10, 100, 999         -> (0.702, 0.68, 0.615, 0.507)
lr 0.01: dead units at the end                 -> 0.094
lr 1.0: loss at steps 0, 10, 100, 999          -> (0.702, 0.508, 0.215, 0.183)
lr 1.0: dead units at the end                  -> 0.125
lr 30.0: loss at steps 0, 10, 100, 999         -> (0.702, 1.855, 1.987, 5.154)
lr 30.0: dead units at the end                 -> 1.0
```

The same 32-unit ReLU network, three learning rates, 1,000 steps each:

- **0.01, too small.** The loss falls smoothly but slowly, still 0.507
  at the end. The symptom is a loss curve that looks healthy but
  hasn't finished.
- **1.0, about right.** Down to 0.215 by step 100 and 0.183 by the end.
- **30, too big.** The loss *rises*, to 1.855 after ten steps and 5.154
  by the end, worse than guessing. Huge steps fling the weights about,
  one of them pushes every unit negative, and 100% are dead. A
  learning rate that's too big is the most common cause of dead ReLUs.

In a deeper network, a rate that's too big can produce `nan` ("not a
number"): values so big they overflow. Exercise 4 makes it happen.
When you see `nan` in a loss, suspect the learning rate first.

Don't try to guess the right rate. Try a few powers of ten (0.001,
0.01, 0.1, 1) for a few hundred steps each, and keep the biggest one
whose loss falls steadily.

## 6. A health check, on the rides

```
biggest raw feature value                      -> 76.5
raw features, before: loss, grad, dead         -> (0.775, 3.69, 0.062)
raw features, after 500: loss, grad, dead      -> (0.529, 0.281, 0.875)
scaled features, before: loss, grad, dead      -> (0.662, 0.147, 0.0)
scaled features, after 500: loss, grad, dead   -> (0.432, 0.004, 0.0)
```

Here's the habit worth keeping. When training misbehaves, print three
numbers, which the script's `health` function does:

1. **The loss.** Is it falling, flat, rising, or `nan`?
2. **The gradient size in the first layer.** Near zero means vanishing;
   huge means exploding.
3. **The share of dead units.** More than about half is a problem.

Point it at the rides, predicting casual riders from the six features
of lesson 050, once with raw features and once with scaled ones.

The raw run starts with a first-layer gradient of 3.69, about 25 times
the scaled run's 0.147, because distances and durations reach 76.5
while the weekend flag is 0 or 1. Those big early steps kill 87.5% of
the hidden units, and the loss gets stuck at 0.529. Scaled, nothing
dies and the loss reaches 0.432. Its gradient has dropped to 0.004, but
that's fine: small gradients *with a falling loss* just mean it's
settling in. Small gradients with a stuck loss are the warning sign.

It's lesson 035's lesson again, now with the mechanism visible: scale
your inputs before you blame the model.

## What you can do now

- Compare sigmoid, tanh, ReLU, LeakyReLU and GELU by their slopes.
- Explain vanishing gradients, and spot them by printing each layer's
  gradient size.
- Choose Kaiming initialisation for ReLU and Xavier for tanh, and apply
  them with `nn.init`.
- Count dead ReLUs, and name their usual causes.
- Recognise a learning rate that's too small, too big, or about right.
- Run a three-number health check on any training run.

## What to do now

1. Run `lesson.py`. Then, in section 5, add `3.0` to the list of
   learning rates and see which way it goes.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   pushes a deep network until it produces `nan`, and tries GELU.
3. Next, lesson 054: smarter optimisers (momentum and Adam) and
   learning-rate schedules, which make section 5's choice a lot less
   delicate. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

Training failures look mysterious from the outside and boring from the
inside. Print the loss, the first layer's gradient and the dead share,
and the mystery usually turns into one of the three suspects from
today.
