# Lesson 072 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the
repo root:

```bash
cp lessons/phase-4-modern-ai/lesson-072-diffusion-models-intuitively/lesson.py my_lesson_072.py
python my_lesson_072.py
```

`add_noise`, `Denoiser`, `train`, `predict_noise`, `sample` and the
schedule (`BETAS`, `ALPHAS`, `ALPHA_BAR`) are there to reuse.

---

## 1. Why not just one step?

A colleague asks: "Why bother with 200 steps? Train the network to turn
static into a digit in one go." What goes wrong?

<details>
<summary>Check yourself</summary>

From pure static, any digit is equally possible, so the best a
one-shot network can do under a squared loss is guess the *average* of
all of them: a grey blur (lesson 071's VAE had a milder version of the
same problem). Small steps avoid this. Each one only has to remove a
little noise, a question with a fairly clear answer, and the choices
pile up gradually into one specific digit. Section 4 shows the same
thing from the sampling side: at one or three steps, the result falls
apart.

</details>

## 2. Reading the loss

In section 2, the loss at step 5 was 0.778 but at step 150 only 0.051.
Does that mean the model is bad at early steps and should be retrained?

<details>
<summary>Check yourself</summary>

No. The two are different questions with different difficulty. At step
150 the noisy point is almost all noise, so guessing the noise is
almost the same as repeating the input. At step 5 the noise is tiny
and hidden inside the data, so the model has to know the data's shape
very precisely. Some loss there is unavoidable. Compare a loss with a
sensible baseline for the same question (here, 1.0 for "guess no
noise"), not with a different question's loss.

</details>

## 3. Guidance as a dial

A text-to-image tool offers a "guidance scale" from 1 to 20. From
section 6, what do you expect at the low and high ends?

<details>
<summary>Check yourself</summary>

Low guidance: more varied, more natural pictures that follow your
prompt loosely, sometimes ignoring part of it (our guidance 0 drew the
wrong digit 8% of the time). High guidance: follows the prompt very
closely, but the pictures get overcooked, less natural and less varied
(our guidance 3 drifted further from real digits). The useful setting
is in between, and the only honest way to choose it is to look at
results, which is lesson 073's topic.

</details>

## 4. Hands-on: turn a 3 into an 8

Diffusion can edit as well as create. Take a real 3, add noise up to
some step, then run the reverse process from that step while asking
for an **8** (guidance 1). Little noise keeps most of the 3; lots of
noise leaves the model free to draw an 8. Try starting at steps 30, 60,
100, 150 and 199, on 40 real test 3s. For each, report how often the
judge reads an 8, how often it still reads a 3, and how far the result
moved from the original.

Predict first: from which step does the 8 win?

<details>
<summary>Expected results</summary>

```
from step  30: reads as 8 0.050, still a 3 0.950, distance from the original 1.846
from step  60: reads as 8 0.650, still a 3 0.325, distance from the original 3.014
from step 100: reads as 8 0.950, still a 3 0.025, distance from the original 4.109
from step 150: reads as 8 1.000, still a 3 0.000, distance from the original 4.998
from step 199: reads as 8 1.000, still a 3 0.000, distance from the original 5.576
```

At step 30 the 3 survives: there's too little noise to change its
shape. Around step 60 it's a toss-up, and by 100 almost every one is
an 8. Notice the distance keeps growing: an 8 made from step 100 keeps
more of the original's position and thickness than one made from step
199, which ignores the 3 entirely.

That's how "image-to-image" and in-painting work in real tools: the
"strength" slider is the starting step. Low strength keeps your
picture; high strength mostly replaces it.

</details>

<details>
<summary>One way to write it</summary>

In your copy, replace the `main()` call at the very bottom with:

```python
if __name__ == "__main__":
    d = load_digits()
    X = (d.data / 8.0 - 1.0).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, d.target, test_size=0.25, random_state=SEED, stratify=d.target)
    judge = SVC().fit((X_tr + 1) / 2, y_tr)
    model, _ = train(Denoiser(64, 256, n_labels=10), torch.tensor(X_tr), iters=8000,
                     y=torch.tensor(y_tr), n_labels=10)
    threes = torch.tensor(X_te[y_te == 3][:40])
    ask = torch.full((len(threes),), 8)
    for start in (30, 60, 100, 150, 199):
        g = torch.Generator().manual_seed(SEED)
        x = add_noise(threes, torch.full((len(threes),), start), torch.randn(threes.shape, generator=g))
        ts = np.linspace(start, 0, 25).round().astype(int)
        for i, t in enumerate(ts):           # the same fast steps as sample()
            eps = predict_noise(model, x, int(t), ask, 10, guidance=1.0)
            x0_guess = (x - (1 - ALPHA_BAR[t]).sqrt() * eps) / ALPHA_BAR[t].sqrt()
            a_next = ALPHA_BAR[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
            x = a_next.sqrt() * x0_guess + (1 - a_next).sqrt() * eps
        got = judge.predict((x.clamp(-1, 1).numpy() + 1) / 2)
        moved = (x.clamp(-1, 1) - threes).norm(dim=1).mean()
        print(f"from step {start:>3}: reads as 8 {r3((got == 8).mean()):.3f}, "
              f"still a 3 {r3((got == 3).mean()):.3f}, distance from the original {r3(moved):.3f}")
```

</details>

## 5. (Optional) Where does the randomness come from?

Run section 3's sampler twice with the same starting static and you get
the same moons. Where, exactly, does a diffusion model's variety come
from, and what does that mean for the "seed" setting in image tools?

<details>
<summary>Check yourself</summary>

From the random numbers it's fed: the starting static, and (in the
200-step recipe) the fresh noise added at each step. The network
itself is deterministic. Same seed, same starting noise, same picture;
change the seed and you get a different picture for the same prompt.
That's why image tools let you fix the seed: to change one thing, such
as the prompt or the guidance, and see only that change's effect. It's
lesson 031's habit of fixing random states, for the same reason.

</details>

---

That's diffusion, intuitively. If one thing sticks: *a diffusion model
never learns to draw, only to clean up a little, and doing that again
and again from pure static is enough to make something new.*
