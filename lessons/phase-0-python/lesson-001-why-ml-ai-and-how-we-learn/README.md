# Lesson 001 - Why ML/AI, and how we'll learn

**Phase 0 - Python foundations for data work** | Week 1, Day 1 | Friday 2026-09-11

> **Goal:** understand the `data -> pattern -> prediction` mental model, and
> watch "learning" happen in a few lines of plain Python.

Time: about 30 minutes. No installs. You just need Python 3.10+.

---

Hi. Welcome. Grab a coffee.

You're at the start of a long walk, and I want the first step to feel easy.
So today there's almost no code to write. We'll talk about what machine
learning actually *is*, look at the road ahead, and run one small program that
shows the whole idea in miniature. That's it.

## Why bother with any of this?

You already use machine learning dozens of times a day without thinking about
it:

- Your phone unlocks by looking at your face. Nobody wrote a rule that says
  "if the nose is 3.2 cm from the chin, it's Sam." Something *learned* what
  your face looks like from examples.
- Your email hides spam. There's no fixed list of bad words. A model learned
  what spam tends to look like from millions of messages people flagged.
- Maps guesses your arrival time. It didn't memorise every route. It learned
  how traffic, time of day and distance relate from past trips.
- Your bank texts you "was this you?" when a purchase looks unusual *for you*.
- A weather app, a music recommendation, the autocomplete in your keyboard,
  the assistant that drafts an email. All learned behaviour.

Here's the common thread. In each case, writing the rules by hand would be
somewhere between painful and impossible. Instead, someone collected
**examples** and let a program **find the pattern** itself. Then the program
uses that pattern to **make a guess about new things it hasn't seen**.

That's machine learning. Everything else is detail.

And "AI" in the modern sense (chatbots, image generators, code assistants) is
the same idea pushed very far: models that learned patterns from enormous
amounts of text, images and code. Same mental model, much bigger scale.

## The one mental model to keep

Here it is. Say it out loud once so it sticks:

```
data  ->  pattern  ->  prediction
```

- **Data** is examples of the thing you care about. Emails and whether they
  were spam. Houses and what they sold for. Photos and what's in them.
- **Pattern** is the relationship hiding in those examples. "Longer trips take
  longer." "Emails with 'FREE!!!' are usually junk." The program's job is to
  find it.
- **Prediction** is using that pattern on something new. A house you haven't
  sold yet. An email that just arrived.

Whenever something in this course feels complicated (and some of it will),
come back to this line and ask: *what's the data, what's the pattern, what's
the prediction?* It works for a spreadsheet formula and it works for a
trillion-parameter language model.

## What "learning" actually means

People hear "the machine learns" and imagine something mysterious. It's not.
Here's the plain version:

1. Guess a rule.
2. Check how wrong the rule is on your examples.
3. Adjust the rule to be a bit less wrong.
4. Repeat until it's good enough.

That's it. "Learning" is just *searching for the rule with the smallest
error*. Fancy methods are fancy ways of searching faster. We'll meet them
later. Today we'll do the honest, slow version so you can see every moving
part.

## The demo: pizza delivery

Open `lesson.py` in this folder. Run it:

```bash
python lessons/phase-0-python/lesson-001-why-ml-ai-and-how-we-learn/lesson.py
```

The story: you run a small pizza place. You've written down, for a few recent
deliveries, how far the customer was (in km) and how long the delivery took
(in minutes). A customer 4.5 km away just ordered. How long should you tell
them?

**Data:** six past deliveries.

**Pattern:** we assume the rule looks like
`minutes = base + per_km * distance`. That's a guess about the *shape* of the
pattern. There's a fixed cost (boxing the pizza, finding the bike) and a
per-km cost (riding). We don't know the two numbers. Finding them is the
learning.

**Prediction:** once we have the numbers, plug in 4.5 km.

The script does four things, in order, and prints as it goes:

1. Shows the data.
2. Tries a hand-made guess (`10 + 4 * km`) and measures how wrong it is. The
   measure is boring on purpose: the average of "how many minutes off were
   we?" across all six deliveries.
3. **Learns** by trying a few thousand candidate rules and keeping the one
   with the smallest average error. This is steps 1-4 above, done with a
   simple loop.
4. Uses the winning rule to predict the new order, and shows why you should
   be careful predicting far outside the data you have.

Read the output. Then read the code. Most of it is printing; the heart of it
is a loop that looks like this:

```python
for base in candidate_bases:
    for per_km in candidate_rates:
        error = average_error(base, per_km, data)
        if error < best_error:
            best_error, best_rule = error, (base, per_km)
```

That loop *is* machine learning. Not a metaphor for it. When you later see
`model.fit(X, y)` in scikit-learn, or a neural network training for hours,
they are doing a much smarter version of exactly this: search for the rule
that makes the smallest error on the examples.

### Things worth noticing

- **We chose the shape of the rule.** A straight line. The program only found
  the two numbers. Choosing a good shape (a "model") is a human decision and a
  big part of this field.
- **The error never hits zero.** Real data is noisy. A rider hits a red light.
  A model that gets *exactly* zero error on its examples has usually memorised
  them rather than learned anything, which we'll talk about a lot in Phase 2.
- **Extrapolation is risky.** The rule was learned on 1-8 km trips. Asking it
  about a 40 km trip is asking it to guess outside anything it's seen. It will
  confidently give you a number. That number might be nonsense.

## The road ahead

Have a look at [the curriculum map](../../../curriculum/CURRICULUM.md). It's
long. Don't be put off. It's long because it's honest about what there is to
learn, not because any one step is big.

Here's the shape:

- **Phase 0 (now):** enough Python to stop thinking about Python. Standard
  library only.
- **Phase 1:** working with real data. Tables, plots, and the statistics
  intuition that keeps you from fooling yourself.
- **Phase 2:** classical machine learning. This is where today's loop becomes
  `fit`/`predict`, and where you learn to evaluate a model honestly.
- **Phase 3:** neural networks, from a hand-written neuron to PyTorch.
- **Phase 4:** modern AI. Language models, vision, generative models, and how
  to use them well.
- **Phase 5:** shipping things. Projects, production habits, and a capstone.

Two lessons a weekday. Weekends off. If a lesson doesn't click, redo it on the
weekend; there's no penalty for going slower than the schedule.

## What to do now

1. Run `lesson.py`. Read the output top to bottom.
2. Open `lesson.py` and read it. Don't worry about syntax you don't know yet;
   lesson 002 covers it. Just follow the story.
3. Do the exercises in [`exercises.md`](exercises.md). They're short.
4. Then move on to
   [Lesson 002 - Python warm-up for data people](../lesson-002-python-warm-up-for-data-people/README.md).

See you there.
