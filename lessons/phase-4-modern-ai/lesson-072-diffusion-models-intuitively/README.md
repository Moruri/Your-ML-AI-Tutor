# Lesson 072 - Diffusion models, intuitively

**Phase 4 - Modern AI: NLP, vision, generative and LLM basics** | Week 9, Day 3 | Wednesday 2026-11-04

> **Goal:** Understand noising and denoising, and run a small diffusion
> pipeline.

Time: about 45 minutes. Same venv as lesson 051; nothing new to install.

---

The image generators everyone talks about, Stable Diffusion and its
cousins, rest on one surprisingly modest trick:

1. Take real data and add a little noise. Then a little more. Keep
   going, a couple of hundred times, until nothing is left but static.
2. Train a network to undo **one small step** of that.
3. Start from fresh static and undo step after step. Data appears that
   was never in the training set.

That's a **diffusion model**. Nothing in it is magic, and today we
build the whole pipeline from scratch, small enough to run on a laptop.
First on 2-D points, where you can watch every step, then on the 8x8
digits from lesson 071, where we also learn to ask for a particular
digit.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-4-modern-ai/lesson-072-diffusion-models-intuitively/lesson.py
```

No downloads: the points come from scikit-learn's `make_moons` and the
digits from its built-in set. It writes four pictures to `output/` and
takes about 40 seconds. The timings in section 4 depend on your
machine; everything else should match exactly.

## 1. Adding noise, step by step

```
steps in the schedule                          -> 200
noise added at the first and last step         -> (0.0001, 0.05)
after step   0: original kept, likeness        -> (1.0, 1.0)
after step  10: original kept, likeness        -> (0.993, 0.987)
after step  30: original kept, likeness        -> (0.942, 0.903)
after step  60: original kept, likeness        -> (0.792, 0.697)
after step 100: original kept, likeness        -> (0.525, 0.421)
after step 199: original kept, likeness        -> (0.078, 0.06)
saved                                          -> 'output/noising.png'
```

The **noise schedule** decides how much static each step adds: a tiny
bit at first (0.0001), more later on (0.05). Each step shrinks the
picture slightly and mixes in fresh random noise, so the image fades
while the static grows.

"Original kept" is how much of the starting digit is left; "likeness"
is the correlation (lesson 024) between the noisy image and the
original. By step 100 it's half gone, and by step 199 there's only 6%
likeness left: pure static, as far as anyone can tell. Open
`output/noising.png` to watch a 9 dissolve.

A handy fact makes training cheap: you don't need to add noise 150
times to reach step 150. A short formula, `add_noise` in the script,
jumps straight there: keep a known share of the picture, fill the rest
with noise.

## 2. The learning task: guess the noise

```
training points (2-D moons)                    -> (4000, 2)
loss if you always guess 'no noise'            -> 1.0
loss, first 100 steps (average)                -> 0.507
loss, last 100 steps (average)                 -> 0.314
loss at step 5                                 -> 0.778
loss at step 50                                -> 0.563
loss at step 150                               -> 0.051
```

We start with 2-D points shaped like two interlocking crescents, the
"moons". Each training example is made like this:

1. Pick a real point and a random step `t`.
2. Make random noise and jump the point to step `t` with it.
3. Ask the network: *given this noisy point and the step number, what
   was the noise?*

The loss is the squared difference between the guess and the real
noise. That's it: a plain regression problem, the kind you've solved
since Phase 2. The network is a small multilayer network from lesson
052 that also gets the step number, as a learned embedding (lesson 059).

The loss never gets near zero, and the per-step numbers show why. At
step 150 the noisy point is nearly all noise, so "the noise" is easy
to guess. At step 5 the noise is tiny and hidden inside the real point;
the network has to know precisely where the moons are to spot it.

## 3. From static to data, one step at a time

```
pure static: share on a moon                   -> 0.26
after all 200 steps: share on a moon           -> 0.909
real held-out points, for comparison           -> 0.974
share of new points on the left moon's side    -> 0.509
saved                                          -> 'output/moons.png'
```

Now run it backwards. Start with 1,000 points of pure static. At each
step, ask the network for its noise guess, remove a share of it, and
(in the original recipe) add back a little fresh randomness so the
points don't all collapse onto one spot. Two hundred times.

Open `output/moons.png`: the blob tightens, then splits, then settles
into two crescents. We score it as the share of points within 0.05 of a
real training point. Static manages 26% by luck; the generated points
reach 91%, not far behind real held-out points at 97%. And the new
points split evenly between the two moons, so the model didn't just
learn one of them.

No single step does much. The model is never asked to draw a moon,
only to clean up a little. The shape comes from doing that again and
again.

## 4. Fewer, bigger steps

```
200 steps: share on a moon, time in ms         -> (0.909, 135)
 50 steps: share on a moon, time in ms         -> (0.92, 34)
 10 steps: share on a moon, time in ms         -> (0.826, 7)
  3 steps: share on a moon, time in ms         -> (0.229, 2)
  1 steps: share on a moon, time in ms         -> (0.04, 1)
```

Two hundred network calls per sample is slow when each call is a big
image model. Faster samplers skip along the schedule: with the noise
guess, you can estimate the clean point directly and jump to a much
earlier step. That's what the `steps` argument does (the method is
called DDIM), with no retraining.

Fifty steps are as good as 200, at a quarter of the cost. Ten still
mostly work. At three or one, the shortcut falls apart: one giant jump
can't do what many small ones do. Real image generators live in the
same trade-off, which is why their settings have a "steps" slider.

## 5. Diffusion on real digits

```
loss, last 100 steps (average)                 -> 0.148
distance to nearest training digit: static     -> 7.699
  new digits (no label asked)                  -> 3.117
  real held-out digits                         -> 2.099
closest any new digit gets to a training one   -> 1.81
closest any held-out digit gets                -> 0.992
saved                                          -> 'output/digits.png'
```

Same recipe, same code, now with 64 numbers per point instead of 2,
pixels scaled from -1 to 1. Open `output/digits.png`: 32 digits drawn
out of static, a bit rough round the edges but readable.

To score them we measure how far each image is from its nearest
training digit. Static is far away (7.7). Real held-out digits are
close (2.1), because real digits resemble each other. The new ones
land in between, much nearer the real thing (3.1).

The last two lines check something important: **is it just copying?**
A model that memorised its training set would produce digits sitting
right on top of training ones. The closest any new digit gets is 1.81,
further than some real held-out digits get (0.99). So these are new
digits, not copies. With huge models trained on scraped images, this
check really matters: they sometimes do reproduce training images
nearly exactly.

## 6. Asking for a particular digit

```
judge's accuracy on real held-out digits       -> 0.984
guidance 0.0: right digit, distance to real    -> (0.92, 2.609)
guidance 1.0: right digit, distance to real    -> (1.0, 2.326)
guidance 3.0: right digit, distance to real    -> (1.0, 2.781)
saved                                          -> 'output/asked_for.png'
```

Section 5's model was trained with a secret: alongside each noisy
digit, it was usually told the label, and 15% of the time it was told
"no label". So now it can denoise either way.

Ask for each digit 50 times and let a judge (a digit classifier that's
98% accurate on real digits) read the results:

- **Just the label (guidance 0):** the right digit 92% of the time.
- **Guidance 1:** at each step we compare the noise guess *with* the
  label to the guess *without* it, and push further in the direction
  the label points. Now every single one is the digit we asked for,
  and they look more like real digits, too.
- **Guidance 3:** still all correct, but pushed too hard: they drift
  further from real digits again. Too much guidance overcooks the
  picture.

This is **classifier-free guidance**, and it's the "guidance scale" in
every text-to-image tool. Open `output/asked_for.png` to see 0 to 9,
twice, made to order.

Now put the last three lessons together. Swap the label for a sentence
turned into a vector by a text encoder like CLIP (lesson 070). Run the
diffusion inside an autoencoder's compact latent space instead of on
raw pixels (lesson 071). Use a much bigger network and millions of
captioned images. That's Stable Diffusion. The loop you just ran is the
same loop.

## What you can do now

- Describe the forward process: a noise schedule that turns data into
  static, and the shortcut that jumps to any step.
- Explain the training task: guess the noise, a plain regression.
- Run the reverse process from static to data, and explain why many
  small steps beat one big one.
- Trade sampling steps for speed, and know where it breaks.
- Check generated data against held-out data, including for copying.
- Condition a diffusion model on a label and tune guidance.

## What to do now

1. Run `lesson.py` and open the four pictures in `output/`. Then try
   `steps=200` in section 5, or a guidance of 6 in section 6, and
   predict what you'll see first.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   turns a 3 into an 8, and shows how image editing with diffusion
   works.
3. Next weekday, lesson 073: evaluating AI systems, where the habit of
   checking generated output honestly becomes a whole toolkit.

Add noise until nothing is left, teach a network to take one small step
back, then walk all the way back from fresh static.
