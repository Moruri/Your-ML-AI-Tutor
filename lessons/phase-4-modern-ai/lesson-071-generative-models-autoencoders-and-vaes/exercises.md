# Lesson 071 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the
repo root:

```bash
cp lessons/phase-4-modern-ai/lesson-071-generative-models-autoencoders-and-vaes/lesson.py my_lesson_071.py
python my_lesson_071.py
```

`load_data`, `AutoEncoder`, `train`, `rebuild`, `codes_of` and `decode`
are there to reuse.

---

## 1. Why not a wider middle?

A colleague sets `LATENT = 64`, the same size as the input, and proudly
reports a rebuild error of almost zero. Why is that result worthless?

<details>
<summary>Check yourself</summary>

With 64 numbers in the middle there's no squeeze, so the network can
learn to pass each pixel straight through without learning anything
about what digits look like. A near-perfect rebuild proves nothing. The
narrow middle is the whole point: it forces the model to keep only
what matters. Choosing its width is the same trade-off as choosing how
many components to keep in PCA (lesson 045).

</details>

## 2. Autoencoder or VAE?

Which would you start with for each job?

1. Shrink 64-pixel scans to 8 numbers each, to store and search them.
2. Make 1,000 extra realistic-looking digits to test a form-reading app.
3. Let a designer drag a slider that morphs one digit style into another.

<details>
<summary>Check yourself</summary>

1. **Autoencoder.** You only need squeeze and rebuild, and it rebuilds
   more sharply (0.013 against the VAE's 0.029 in section 5).
2. **VAE.** You need to sample new codes, and its smooth bell-curve
   space is built for that (77% pass the judge, against 67%).
3. **VAE.** A slider is a walk through the latent space, and section 6
   showed the VAE's walk changes shape step by step instead of
   cross-fading pixels.

</details>

## 3. What does the KL term do?

You delete the KL term from the VAE's loss but keep the random noise
in the code. What do you expect to happen to the codes during training?

<details>
<summary>Check yourself</summary>

Nothing stops the encoder from spreading its codes far apart and
shrinking its spread towards zero, so the noise stops mattering. It
quietly turns back into a plain autoencoder, gaps and all, and sampling
from a bell curve around 0 no longer lands where the codes are. The KL
term is the pull that keeps the space compact and filled in.

</details>

## 4. Hands-on: an autoencoder as an anomaly detector

An autoencoder rebuilds well what looks like its training data, and
badly what doesn't. So the rebuild error is an "unusualness" score,
just like lesson 046's anomaly scores.

Train the plain autoencoder, then compute each image's rebuild error
for three sets: the real test digits, the same digits upside down, and
random static. Use the 99th percentile of the *training* errors as the
cut-off, and report how many of each set it flags.

Predict first: will it flag all the upside-down digits?

<details>
<summary>Expected results</summary>

```
cut-off 0.027
real test digits     mean error 0.013   flagged 0.051
upside-down digits   mean error 0.039   flagged 0.682
random static        mean error 0.214   flagged 1.000
```

Static is caught every time. Most upside-down digits are flagged, but
not all: an upside-down 0, 1 or 8 still looks like a perfectly normal
digit, so it *should* pass. About 5% of real test digits are flagged
too, a little more than the 1% you'd expect, because the model is
slightly better on data it trained on (lesson 031's train/test gap).
That's the false-alarm rate you'd tune, as in lesson 043.

</details>

<details>
<summary>One way to write it</summary>

In your copy, replace the `main()` call at the very bottom with:

```python
if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = load_data()
    ae = train(AutoEncoder(), X_tr)
    err = lambda X: ((rebuild(ae, X) - X) ** 2).mean(1)
    cut = np.quantile(err(X_tr), 0.99)
    print("cut-off", r3(cut))
    upside_down = X_te.reshape(-1, 8, 8)[:, ::-1].reshape(-1, 64).copy()
    static = np.random.default_rng(SEED).random(X_te.shape).astype(np.float32)
    for name, X in (("real test digits", X_te), ("upside-down digits", upside_down), ("random static", static)):
        e = err(X)
        print(f"{name:<20} mean error {r3(e.mean()):.3f}   flagged {r3((e > cut).mean()):.3f}")
```

</details>

## 5. (Optional) Asking for a particular digit

Today's VAE draws a random digit each time. How could you change it so
you can ask for "a 7, please"?

<details>
<summary>Check yourself</summary>

Give the label to both halves. Feed the encoder the image *and* its
label, and feed the decoder the code *and* the label. Then the code
only has to hold the style (slant, thickness), because the label
already says which digit it is. To draw a 7, sample a code from the
bell curve and decode it together with the label 7. This is called a
*conditional VAE*. Lesson 072 does the same trick with a diffusion
model, and swapping the label for a sentence is the heart of
text-to-image.

</details>

---

That's autoencoders and VAEs. If one thing sticks: *squeezing data
forces a model to learn what it's like, and keeping the squeezed space
smooth is what lets it make something new.*
