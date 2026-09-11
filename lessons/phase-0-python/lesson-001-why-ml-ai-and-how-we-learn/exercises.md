# Lesson 001 - Exercises

These are short and there's no code to write from scratch. The point is to
poke at `lesson.py` until the idea feels like yours. Change things, run it,
see what happens. You can't break anything.

Run the script from the repo root with:

```bash
python lessons/phase-0-python/lesson-001-why-ml-ai-and-how-we-learn/lesson.py
```

---

## 1. Spot the three parts

Without looking at the code, write one sentence each for what the **data**,
the **pattern** and the **prediction** were in the pizza demo.

Then pick *two* of these everyday systems and do the same for them. There's
no single right answer; the point is to practise seeing the shape.

- A music app recommending your next song
- A spam filter
- A weather app's "chance of rain"
- Your phone's autocorrect

<details>
<summary>Check yourself (spam filter)</summary>

**Data:** lots of past emails, each marked spam or not-spam by people.
**Pattern:** what spam emails tend to have in common (certain words, odd
senders, lots of links) versus normal emails.
**Prediction:** for a brand-new email, a guess at "spam" or "not spam".

</details>

## 2. Make the guess worse, then better

In `show_hand_made_guess()`, the hand-made rule is `base, per_km = 10, 4`.

1. Change it to `base, per_km = 0, 0` and run. What's the average error? Why
   does that number make sense? (Hint: what is the rule predicting for every
   delivery?)
2. Now try to beat the learned rule (`6.5 + 5.2 * km`, average error 0.57) by
   hand. Spend two minutes on it. Notice how it feels to do by hand what the
   loop does for you.

<details>
<summary>Check yourself</summary>

With `0, 0` the rule predicts 0 minutes for everything, so the error is just
the average of the actual times: (12+17+24+33+36+48)/6 = 28.33 minutes.

Beating 0.57 by hand is hard, and that's the lesson. Searching is tedious for
humans and trivial for computers. That's why we let the machine do it.

</details>

## 3. Break the search

In `learn()`, the search ranges are:

```python
for base in frange(0, 20, 0.5):
    for per_km in frange(0, 10, 0.1):
```

1. Change the `per_km` range to `frange(0, 3, 0.1)`. Run it. What happens to
   the best rule and its error? Why?
2. Put it back. Now change `0.1` to `1.0` in the `per_km` range. What happens?

Write one sentence about what these two experiments tell you about the
choices a *human* still makes even when the machine is "learning".

<details>
<summary>Check yourself</summary>

1. The best rule gets much worse (something like `14 + 2.9 * km`, error
   around 4.6 minutes). The true per-km cost is about 5, and we forbade the
   search from ever trying anything above 3, so it pushed `base` up to
   compensate as best it could. The machine can only find rules inside the
   space we let it look in.
2. The search is much coarser, so it settles on `7 + 5 * km` and a slightly
   higher error than before (0.58 vs 0.57). Fewer candidates, less precise
   answer. Here it barely mattered; on harder problems it can matter a lot.

The human still chooses: the shape of the rule, where to search, and how
finely. Those choices matter as much as the algorithm.

</details>

## 4. Add a data point

Add a seventh delivery to the `deliveries` list: `(3.0, 60)`. Maybe the rider
got a flat tyre. Run the script.

1. How much did the learned rule change?
2. How much did the average error change?
3. Is the new rule *better* or *worse* for predicting the next normal
   delivery? What would you do about this in real life?

<details>
<summary>Check yourself</summary>

The rule shifts a little (to about `7 + 5.1 * km`) and the error jumps a lot
(to almost 6 minutes; one point is off by ~40 minutes, and that gets averaged
in). The rule is probably slightly *worse* for normal deliveries now, because
it bent itself toward one weird example.

In real life you'd ask: is this point a mistake in the data, a rare event to
ignore, or a sign that our rule is missing something (like "did the bike
break?")? Deciding that is part of the job, and there's a whole lesson on
outliers and anomalies later in the course.

</details>

## 5. (Optional) A different measure of "wrong"

`average_error()` uses `abs(guess - actual)`. Change it to
`(guess - actual) ** 2` (squared error) and run again.

Does the best rule change? Look at exercise 4's flat-tyre point again with
squared error. Does the rule bend more or less toward it?

<details>
<summary>Check yourself</summary>

On the normal data the best rule barely moves. With the flat-tyre point in,
squared error gives something like `17 + 3.9 * km`, a big swing. Squaring
punishes big misses much more than small ones, so the rule bends *more*
toward the outlier. Neither choice is "correct". They encode different
opinions about which mistakes matter most. Picking the error measure (the
"loss function") is one of the quiet, important decisions in ML, and we'll
return to it many times.

</details>

---

Done? Go to
[Lesson 002 - Python warm-up for data people](../lesson-002-python-warm-up-for-data-people/README.md).
