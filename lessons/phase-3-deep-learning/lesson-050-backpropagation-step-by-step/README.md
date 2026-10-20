# Lesson 050 - Backpropagation, step by step

**Phase 3 - Deep learning foundations** | Week 7, Day 2 | Tuesday 2026-10-20

> **Goal:** Derive and code the chain rule for a two-layer network
> without any framework.

Time: about 45 minutes. Same venv as lesson 049; still no PyTorch.

---

Lesson 049's neuron draws one straight line through the data and puts
"casual" on one side. Lots of problems don't split along a straight
line. The fix is to stack neurons: a **hidden layer** of neurons that
each look at the inputs their own way, and an output neuron that
combines what they found.

That raises one question, and it's the question this whole lesson is
about. With one neuron, we had a neat formula for the gradient. With two
layers, how much is a weight in the *first* layer to blame for the
final loss? **Backpropagation** is the answer: start at the loss, and
pass the blame backwards, layer by layer, using the chain rule from
school calculus. It sounds heavy. It's about ten lines of NumPy.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-3-deep-learning/lesson-050-backpropagation-step-by-step/lesson.py
```

It writes one picture, `output/decision_boundary.png`, next to the
script. Every random step has a fixed seed, so your numbers will match.
It takes about five seconds.

## 1. A problem one neuron can't solve

```
points / share inside the ring                 -> (400, 0.325)
one neuron: training accuracy                  -> 0.675
always saying 'outside': accuracy              -> 0.675
one neuron: weights                            -> [0.154, 0.124]
```

We start away from the rides, on a tiny made-up dataset where the
problem is easy to see. Scatter 400 points over a square; label a point
1 if it lands inside a circle of radius 1, otherwise 0. Flip 3% of the
labels at random, because real data is never perfectly clean.

Train lesson 049's neuron on it and it scores 67.5%, exactly what you'd
get by saying "outside" every time. Its weights are nearly zero. It
isn't broken; it's out of options. A single neuron can only say "this
side of a line" or "that side", and no straight line puts a circle's
inside on one side and its outside on the other.

## 2. Two layers: the forward pass

```
X shape (points, inputs)                       -> (400, 2)
W1 shape (inputs, hidden units)                -> (2, 8)
h shape, after tanh                            -> (400, 8)
W2 shape (hidden units, 1)                     -> (8, 1)
p shape, one probability per point             -> (400,)
parameters to learn                            -> 33
first three predictions, untrained             -> [0.575, 0.559, 0.499]
```

The network has two layers:

$$
h = \tanh(X W_1 + b_1), \qquad p = \sigma(h W_2 + b_2)
$$

The **hidden layer** is eight neurons side by side. Each has two
weights (one per input) and a bias, so `W1` is a 2-by-8 grid. Each draws
its own line through the square, and **tanh** squashes the result into
the range -1 to 1. That squash is the important bit: it bends the space.
Without it, two layers of weighted sums would collapse into one weighted
sum, and we'd be back to a single straight line.

The **output layer** is lesson 049's neuron, but its inputs are the
eight hidden values instead of the raw coordinates. Count it up:
2×8 + 8 + 8×1 + 1 = 33 numbers to learn. Getting from inputs to `p` is
called the **forward pass**. Keep an eye on the shapes; most bugs in
neural network code are shape bugs.

## 3. The chain rule, with one number at a time

```
x, w1, w2, true label y                        -> (2.0, 0.5, -1.0, 1.0)
h = tanh(w1 * x)                               -> 0.762
p = sigmoid(w2 * h)                            -> 0.318
loss = -log(p)                                 -> 1.145
dL/dw2 = (p - y) * h                           -> -0.519
dL/dw1 = (p - y) * w2 * (1 - h^2) * x          -> 0.573
nudge-and-measure dL/dw2                       -> -0.519
nudge-and-measure dL/dw1                       -> 0.573
```

Before matrices, shrink it right down: one input, one hidden unit, one
output, two weights. The input 2.0 goes in, becomes $h = 0.762$, then
$p = 0.318$. The truth is 1, so the loss is 1.145.

Now walk backwards. The **chain rule** says: if a change in $w_1$ moves
$h$, which moves $p$, which moves the loss, then the total effect is
those individual effects multiplied together. Each link is a slope you
already know:

| Step backwards | Local slope |
|----------------|-------------|
| loss to $z_2$ (sigmoid with log loss, from lesson 049) | $p - y$ |
| $z_2 = w_2 h$ to $w_2$ | $h$ |
| $z_2 = w_2 h$ to $h$ | $w_2$ |
| $h = \tanh(z_1)$ to $z_1$ | $1 - h^2$ |
| $z_1 = w_1 x$ to $w_1$ | $x$ |

So $\partial L / \partial w_2 = (p - y)\,h$ and
$\partial L / \partial w_1 = (p - y)\,w_2\,(1 - h^2)\,x$. Plug in the
numbers and you get -0.519 and 0.573. The nudge-and-measure check from
lesson 049 agrees on both.

Notice the order. To get $w_1$'s gradient, we reused the
$(p - y)$ we'd already worked out for $w_2$. That's the trick that makes
backprop fast: every layer's gradient is built from the one after it,
so nothing is computed twice.

## 4. The backward pass in code, and the gradient check

```
gradient shapes match parameter shapes         -> True
W2 gradient, first three (backprop)            -> [-0.034, -0.057, -0.022]
W2 gradient, first three (nudges)              -> [-0.034, -0.057, -0.022]
worst relative error, backprop vs nudges       -> 4.1e-08
same check, but forgetting tanh's slope        -> 1.0
```

Same five links, now with whole matrices. Here's the heart of
`backward()` in `lesson.py`:

```python
dz2 = (p - y).reshape(-1, 1) / n      # loss -> output, one row per point
dW2 = h.T @ dz2                       # output weights
dh  = dz2 @ W2.T                      # pass the blame back to the hidden layer
dz1 = dh * (1 - h ** 2)               # back through tanh
dW1 = X.T @ dz1                       # hidden weights
```

Line for line it's the table from section 3. The `.T` transposes are
there so the shapes line up: every gradient comes out exactly the shape
of the weight it belongs to, which is a free sanity check.

Then the real test. The script compares all 33 backprop gradients with
nudge-and-measure on 20 points. The worst relative gap is 0.00000004:
correct. Then it plants a classic bug, forgetting tanh's slope
$(1 - h^2)$, and the same check jumps to 1.0, meaning "completely
wrong". A gradient check catches in one second a bug that would
otherwise cost you an afternoon of "why won't it train?".

Why not always use the nudges? They need two extra forward passes per
weight: 66 here, and billions in a large network. Backprop gets every
gradient in one backward sweep. Check with nudges once, on a small
batch, then trust backprop.

## 5. Training the network

```
loss at step 0                                 -> 0.689
loss at step 100                               -> 0.361
loss at step 500                               -> 0.191
loss at step 1000                              -> 0.161
loss at step 2999                              -> 0.148
two-layer net: training accuracy               -> 0.955
on 400 fresh points from the same ring         -> 0.948
picture written to                             -> 'output/decision_boundary.png'
```

The training loop is lesson 049's, unchanged in spirit: forward, loss,
backward, step every weight against its gradient. Learning rate 1.0,
3000 steps, weights started small and random (more on why in the
exercises).

The loss falls from 0.69 to 0.15, and the network gets 95.5% of points
right. On 400 fresh points it's never seen, 94.8%, so it learned the
ring, not the particular points. With 3% of labels flipped on purpose,
about 97% is the best anyone could do.

Open `output/decision_boundary.png`. The black line is where the network
switches from "outside" to "inside", and the grey dashed circle is the
truth. It's a rounded blob, not a perfect circle: eight bent lines
glued together. More hidden units would make it rounder.

## 6. Back to the rides

```
one neuron, six inputs: test AUC               -> 0.844
two-layer net, 8 hidden: test AUC              -> 0.841
two-layer net: train AUC                       -> 0.861
```

Now the honest bit. Same rides as lesson 049, six scaled inputs
(duration, distance, weekend, hour, temperature, rain), same split. One
neuron gets a test AUC of 0.844. The two-layer network gets 0.841.

The extra layer bought nothing. On 882 rows and six tidy columns, a
straight line already captures most of the pattern, and the extra
weights mostly go into fitting the training rows a little more closely
(train AUC 0.861). Neural networks earn their keep when there's far
more data and far more shape to find, like the pixels and words coming
later in Phase 3 and in Phase 4. On tabular data like this, start with
Phase 2's tools.

## What you can do now

- Explain why one neuron can't learn a ring and two layers can.
- Write a two-layer forward pass and track every shape.
- Derive the backward pass with the chain rule, link by link.
- Code backprop in NumPy and verify it with a gradient check.
- Say honestly when a neural network is overkill.

## What to do now

1. Run `lesson.py` and open `output/decision_boundary.png`. Where does
   the blob bulge past the true circle, and can you see which noisy
   points pulled it there?
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   finds the fewest hidden units that can still draw the ring, and shows
   what happens if every weight starts at zero.
3. Next weekday, lesson 051: the same network in PyTorch, where autograd
   does today's backward pass for you. That's the first lesson that needs
   `torch` installed. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

Backpropagation is the chain rule, applied from the loss backwards: each
layer passes the blame to the one before it, multiplied by its own local
slope. Check it with nudges once, then trust it. Every deep learning
framework does exactly this, just faster.
