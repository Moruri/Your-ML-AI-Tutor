# Lesson 049 - A neuron in pure NumPy

**Phase 3 - Deep learning foundations** | Week 7, Day 2 | Tuesday 2026-10-20

> **Goal:** Implement a single neuron, a loss, and gradient descent by
> hand.

Time: about 40 minutes. Needs the venv from lesson 013 (NumPy, pandas,
matplotlib, and scikit-learn for one comparison at the end). No PyTorch
yet; that arrives in lesson 051.

---

Welcome to Phase 3. "Neural network" sounds grand, but the building
block is small enough to write on a napkin. A **neuron** takes a few
numbers, multiplies each by a weight, adds them up with a bias, and
squashes the total into a value between 0 and 1. That's it. Deep
learning is mostly stacking lots of these and finding good weights.

Today we build one neuron by hand, then teach it. Teaching takes two
more pieces: a **loss**, one number that says how wrong the neuron is,
and **gradient descent**, a way to nudge the weights so the loss goes
down. Once you've written these three things yourself, nothing in
PyTorch will feel like magic.

## How to follow along

Venv active, `python` in the repo root. Same `rides_clean.csv` as Phase
2. Full script:

```bash
python lessons/phase-3-deep-learning/lesson-049-a-neuron-in-pure-numpy/lesson.py
```

It writes one picture, `output/loss_curve.png`, next to the script.
Every random step has a fixed seed, so your numbers will match. It takes
a couple of seconds.

## 1. The inputs, on one ruler

```
train rows / test rows                         -> (882, 294)
share of casual rides (train)                  -> 0.322
first three training rows, scaled              -> [[0.23, 0.548, 2.01], [0.037, 0.155, -0.498], [-0.004, -0.498, -0.498]]
column means after scaling                     -> [0.0, 0.0, 0.0]
column std devs after scaling                  -> [1.0, 1.0, 1.0]
```

The job is lesson 033's: tell casual riders from members. We give the
neuron three inputs: `duration_min`, `distance_km` and the `weekend`
flag. About a third of training rides are casual.

We do the split by hand this time, with a shuffled list of row numbers,
75% for training. Then we **standardise** each column: subtract its
mean, divide by its spread, using the training rows only (lesson 035).
Every column ends up centred on 0 with a spread of 1. That matters more
for gradient descent than it did for most of Phase 2, and section 6
shows why.

## 2. One neuron: weigh, add, squash

```
one ride's inputs x                            -> [0.23, 0.548, 2.01]
weights w, bias b                              -> ([1.0, 0.0, 0.5], -1.0)
z = x . w + b                                  -> 0.235
sigmoid(z), a probability                      -> 0.559
sigmoid at -5, 0, 5                            -> [0.007, 0.5, 0.993]
first five predictions with these weights      -> [0.559, 0.229, 0.222, 0.505, 0.257]
```

Here's the whole neuron:

$$
z = w_1 x_1 + w_2 x_2 + w_3 x_3 + b, \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

In code that's `sigmoid(X @ w + b)`. The `@` does the weighted sum for
every ride at once. The **sigmoid** $\sigma$ squashes any number
into the range 0 to 1: very negative becomes nearly 0, zero becomes
exactly 0.5, very positive becomes nearly 1. So $p$ reads as "the
chance this ride is casual".

The weights here are made up. This weekend ride gets 0.56. Is that
good? We need a way to score it.

## 3. A loss: one number for how wrong we are

```
loss with all weights 0 (always says 0.5)      -> 0.693
ln 2, for comparison                           -> 0.693
loss of the made-up weights from section 2     -> 0.468
cost of saying 0.9 when the truth is 1         -> 0.105
cost of saying 0.1 when the truth is 1         -> 2.303
cost of saying 0.01 when the truth is 1        -> 4.605
```

We use **log loss** (also called binary cross-entropy), which you met as
a metric in lesson 034. For each ride, if the truth is casual the cost
is $-\log p$; if it's a member the cost is $-\log(1 - p)$. Then
we average over all rides.

Look at the costs. Saying 0.9 when you're right costs 0.1. Saying 0.1
when you're wrong costs 2.3, and 0.01 costs 4.6. Log loss forgives
mild uncertainty and comes down hard on confident mistakes, which is
exactly what we want from a probability.

A neuron with all weights at zero says 0.5 for everything and scores
$\ln 2 \approx 0.693$. That's the number to beat. Our made-up
weights already do better, at 0.468.

## 4. The gradient, and a check that it's right

```
formula gradient for w                         -> [-0.052, -0.061, -0.005]
formula gradient for b                         -> -0.027
nudge-and-measure gradient for w               -> [-0.052, -0.061, -0.005]
biggest difference between the two             -> 3.8e-11
```

To improve, we need to know which way to turn each weight. The
**gradient** is that answer: for each weight, how much the loss changes
when the weight goes up a tiny bit. Negative means "turn this weight up
and the loss falls".

For a sigmoid neuron with log loss, the calculus tidies up beautifully:

$$
\frac{\partial L}{\partial w_j} = \frac{1}{n}\sum_i (p_i - y_i)\, x_{ij}, \qquad
\frac{\partial L}{\partial b} = \frac{1}{n}\sum_i (p_i - y_i)
$$

In words: the error on each ride (prediction minus truth), times that
ride's input, averaged. In code it's `X.T @ (p - y) / n`.

Don't take a formula on trust. The script checks it the slow, honest
way: nudge each weight up by 0.000001, then down, and measure how the
loss moves. The two answers agree to 11 decimal places. This
**gradient check** is the single best habit in deep learning, and
lesson 050 leans on it hard.

## 5. Gradient descent: take many small steps downhill

```
loss at step 0                                 -> 0.693
loss at step 10                                -> 0.471
loss at step 50                                -> 0.45
loss at step 100                               -> 0.45
loss at step 299                               -> 0.45
learned weights (duration, distance, weekend)  -> [0.962, 0.649, 0.531]
learned bias                                   -> -0.863
test accuracy                                  -> 0.769
test AUC                                       -> 0.849
test accuracy of always saying 'member'        -> 0.67
picture written to                             -> 'output/loss_curve.png'
```

Now the loop. Start with all weights at zero. Then, 300 times: compute
the gradient, and move every weight a small step the *other* way:

```python
w = w - lr * dw
b = b - lr * db
```

`lr` is the **learning rate**, the size of each step. Here it's 0.5.

The loss drops fast, from 0.693 to 0.471 in ten steps, then settles at
0.45. Open `output/loss_curve.png` to see that shape: a steep slide,
then a long flat floor. All three weights came out positive, so longer
rides, further rides and weekend rides all push towards "casual".

On the 294 test rides it gets 77% right, against 67% for always saying
"member", with an AUC of 0.85. Three inputs, four numbers learned, and
you wrote every line.

## 6. The learning rate, and a familiar face

```
lr=0.01: loss after 300 steps                  -> 0.502
lr=0.5: loss after 300 steps                   -> 0.45
lr=5.0: loss after 300 steps                   -> 0.45
lr=50.0: loss after 300 steps                  -> 2.983
lr=50.0: first eight losses                    -> [0.693, 2.724, 1.858, 0.956, 0.655, 2.537, 1.648, 0.761]
our weights after 300 steps                    -> [0.962, 0.649, 0.531]
our weights after 5000 steps                   -> [1.006, 0.605, 0.531]
scikit-learn LogisticRegression weights        -> [1.006, 0.605, 0.531]
```

The learning rate is the dial you'll tune most in Phase 3. At 0.01 the
neuron crawls and is still at 0.50 after 300 steps. At 50 it leaps right
over the valley and bounces from wall to wall: look at the loss jumping
2.7, 1.9, 1.0, 0.7, 2.5. It never settles. Somewhere in between (0.5
and 5 here) it just works.

Then the twist. After 300 steps our weights were close to scikit-learn's
but not equal. After 5000 they match to three decimals. Two things are
going on:

- **The last stretch is slow.** Duration and distance move together
  (lesson 047), so the loss barely changes if you trade a little of one
  weight for the other. That makes a long, flat-bottomed valley, and the
  loss stopped improving well before the weights finished drifting.
- **We've built this model before.** A single sigmoid neuron trained on
  log loss *is* logistic regression, lesson 033's model. Scikit-learn
  just uses a cleverer solver than plain gradient descent.

So a neuron isn't new. What's new is that you know exactly how it
learns, and that's the part that carries over when we stack them.

## What you can do now

- Write a neuron as a weighted sum, a bias and a sigmoid.
- Compute log loss and explain why it punishes confident mistakes.
- Derive the gradient for a sigmoid neuron and check it numerically.
- Train with gradient descent and read a loss curve.
- Spot a learning rate that's too small or too big.

## What to do now

1. Run `lesson.py` and open `output/loss_curve.png`. Where would you
   have stopped training, and why?
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   skips the scaling step to see what happens.
3. Next up, lesson 050: stack neurons into two layers and work out the
   gradient for all of them with backpropagation. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

A neuron weighs its inputs, adds a bias and squashes. A loss says how
wrong it is, the gradient says which way is downhill, and gradient
descent walks there in small steps. Everything in deep learning is built
on those three ideas.
