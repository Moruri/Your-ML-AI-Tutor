# Lesson 071 - Generative models: autoencoders and VAEs

**Phase 4 - Modern AI: NLP, vision, generative and LLM basics** | Week 9, Day 3 | Wednesday 2026-11-04

> **Goal:** Learn to compress and regenerate data, and sample new points
> from latent space.

Time: about 40 minutes. Same venv as lesson 051; nothing new to install.

---

Every model so far has looked at data and said something *about* it: a
label, a number, a box around a cat. Today the model makes data.

The idea is older and simpler than you might expect. Take a handwritten
digit, 64 pixels. Squeeze it through a narrow middle, just 8 numbers,
then ask a network to rebuild all 64 pixels from those 8. To succeed it
has to learn what digits are *like*. And once it knows that, you can
hand its second half 8 numbers of your own and see what it draws.

That's an **autoencoder**. With one change it becomes a **variational
autoencoder (VAE)**, and that change is the difference between a
network that can only copy and one that can invent.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-4-modern-ai/lesson-071-generative-models-autoencoders-and-vaes/lesson.py
```

It uses scikit-learn's built-in 8x8 digits (the same ones as lesson
057), so there's nothing to download. It trains four small networks,
writes four pictures to `output/`, and takes about 20 seconds.

## 1. Squeeze and rebuild: what PCA already does

```
numbers per digit image                        -> 64
rebuild error if we always guess the average   -> 0.073
PCA, squeezed to 2 numbers, rebuild error      -> 0.052
PCA, squeezed to 10 numbers, rebuild error     -> 0.019
PCA, squeezed to 30 numbers, rebuild error     -> 0.003
```

You've done this before. In lesson 045, PCA squeezed many columns into
a few and could turn them back again. The **rebuild error** (the mean
squared difference per pixel, pixels running from 0 to 1) tells you
what got lost on the way. Keep 30 numbers and almost nothing is lost;
keep 2 and you're barely better than drawing the average digit every
time.

PCA's limit is that it only squeezes along straight lines. Digits
don't sit on straight lines: a 7 doesn't become a 1 by fading.

## 2. An autoencoder: the same idea, learned and bendy

```
parameters in the autoencoder                  -> 34768
PCA, 8 numbers, rebuild error (test)           -> 0.024
autoencoder, 8 numbers, rebuild error (test)   -> 0.013
autoencoder, rebuild error (train)             -> 0.009
saved                                          -> 'output/rebuilds.png'
```

An autoencoder is two small networks back to back:

- The **encoder** takes 64 pixels down to 8 numbers, the **code**.
- The **decoder** takes those 8 numbers back up to 64 pixels.

Training is lesson 052's loop with a twist: the target *is the input*.
We show it a digit and score how far the rebuild is from that same
digit. There are no labels anywhere. The loss is binary cross-entropy
per pixel, the log loss from lesson 033, which suits values between 0 and 1.

With the same 8 numbers, it rebuilds held-out digits with about half
PCA's error, because its layers can bend. Open `output/rebuilds.png`:
real digits on top, rebuilt ones below. They're a little soft, but
clearly the same digits.

The set of all possible codes is called the **latent space** ("latent"
means hidden: these numbers are never in the data, the model makes them
up).

## 3. The latent space: where the codes live

```
one digit's 2-number code                      -> (0.718, -0.917)
its real label                                 -> 3
label guessed from 2-number autoencoder codes  -> 0.762
label guessed from 2-number VAE codes          -> 0.729
autoencoder codes: centre and spread           -> ((-1.24, -1.801), (3.466, 3.947))
VAE codes: centre and spread                   -> ((0.01, -0.024), (0.951, 0.985))
saved                                          -> 'output/latent.png'
```

Eight dimensions are hard to look at, so here we train a second pair of
models with only **2** numbers in the middle, and draw every test digit
as a dot. Open `output/latent.png`.

Neither model was told which digit is which, yet the dots arrange
themselves by digit: a simple logistic regression can guess the label
from just those two numbers about three times out of four. To squeeze
well, the model had to discover that a 0 and a 1 are different things.

Now compare where the codes sit. The autoencoder's are off-centre and
spread out, wherever training happened to put them. The VAE's sit in a
neat ball: centred on 0, spread about 1. That tidy ball is the whole
point of the next two sections.

## 4. Generating with an autoencoder: the empty gaps

```
real test digits the judge is 90%+ sure of     -> 0.844
rebuilt test digits, 90%+ sure                 -> 0.842
500 new codes, centred and spread like its own -> 0.668
```

How do you score a made-up digit? There's no right answer to compare
with. So we train a separate digit classifier on real digits and use it
as a **judge**. (It's a support vector machine from scikit-learn, a
classic model we haven't needed until now; treat it as a black box
that's good at digits.) If the judge is at least 90% sure what digit an
image shows, we call it digit-like. It's a rough measure, but a fair
one, because we use the same judge for everything.

Rebuilt digits pass as often as real ones. Now the generative move:
invent 500 brand-new codes, with the same centre and spread as the
codes the autoencoder actually uses, and decode them. Only two-thirds
pass.

The reason is that the autoencoder was only ever asked to rebuild
digits it saw. Nothing told it what to draw *between* those codes, so
its latent space is full of gaps, and a random point often lands in
one. You get a smudge.

## 5. The VAE: making the space smooth

```
rebuild error, VAE (test)                      -> 0.029
same, plain autoencoder (section 2)            -> 0.013
500 new codes from a bell curve, 90%+ sure     -> 0.77
digits the judge thinks it drew (0-9 counts)   -> [53, 36, 36, 45, 45, 31, 45, 39, 102, 68]
saved                                          -> 'output/new_digits.png'
```

A VAE changes two things:

1. **The encoder gives a fuzzy code, not an exact one.** It outputs a
   centre and a spread, and during training we decode a random point
   near that centre. So the decoder learns that a whole little *region*
   means "this 3", not a single point. Writing the random point as
   `centre + noise * spread` (the *reparameterisation trick*) keeps the
   gradients flowing back through the encoder.
2. **An extra term in the loss** (the *KL divergence*) gently pulls
   every code region towards a plain bell curve around 0. That's why
   section 3's VAE codes formed a neat ball.

Together they fill the gaps. To make a new digit you sample 8 numbers
from a plain bell curve (`np.random.standard_normal`), decode, done. 77%
pass the judge, almost as many as real digits and clearly more than the
autoencoder's 67%. Open `output/new_digits.png`: 32 digits that nobody
ever wrote.

Nothing is free. The VAE's rebuilds are about twice as far off, a
little blurrier, because it trades exact copying for a smooth space.
And look at the counts: it draws more 8s than anything else. A blurry
in-between of several digits often looks most like an 8, which is a
small, honest sign of where it's least sure.

## 6. Walking between two digits

```
plain pixel blend: judge's reading             -> [0, 0, 0, 0, 8, 8, 1, 1]
VAE walk: judge's reading                      -> [0, 0, 0, 8, 1, 1, 1, 1]
saved                                          -> 'output/walk.png'
```

Take a 0 and a 1. There are two ways to go from one to the other in 8
steps:

- **Blend the pixels.** Fade the 0 out while the 1 fades in.
- **Walk the latent space.** Encode both, take 8 evenly spaced points
  on the straight line between the two codes, and decode each.

Open `output/walk.png`. The top row is a double exposure: a ghost of a
0 with a 1 showing through. The bottom row changes *shape* instead: the
oval narrows, step by step, into a stroke. Every frame is a plausible
handwritten mark.

That's what a smooth latent space buys you. Moving a little in it
changes the output a little, in ways that make sense. It's how image
editors slide a face from "smiling" to "not smiling", and it's an idea
the next lesson builds on: Stable Diffusion does its work inside an
autoencoder's latent space, not on raw pixels.

## What you can do now

- Explain what an encoder, a decoder, a code and a latent space are.
- Train an autoencoder and compare its rebuild error with PCA's.
- Explain why a plain autoencoder is a poor generator: gaps.
- Describe the two changes that make a VAE, and the trade-off they
  bring (smooth space, blurrier rebuilds).
- Sample new data from a VAE and interpolate between two examples.
- Score generated data honestly when there's no right answer to compare
  with, using a separate judge model.

## What to do now

1. Run `lesson.py` and open all four pictures in `output/`. Then change
   `LATENT` from 8 to 2 or 32 and see what happens to the rebuilds and
   the new digits.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   turns the autoencoder into an anomaly detector.
3. Next, lesson 072: diffusion models, the idea behind today's image
   generators, built from scratch.

A plain autoencoder learns to squeeze and rebuild; a VAE also keeps its
squeezed space smooth, so any point you pick decodes into something
real.
