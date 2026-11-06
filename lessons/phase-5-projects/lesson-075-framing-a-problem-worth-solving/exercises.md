# Lesson 075 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the
repo root:

```bash
cp lessons/phase-5-projects/lesson-075-framing-a-problem-worth-solving/lesson.py my_lesson_075.py
python my_lesson_075.py
```

`riders`, `train`, `test`, `p`, `rule`, `value`, `test_scale`, `COST`,
`VALUE` and `SAVE_RATE` are there to reuse.

---

## 1. Fill in the brief

A manager says: "We should use AI to predict which bikes will break
down." Write the eight lines of a brief (decision, who acts,
prediction, when, action cost, payoff, metric, baseline). Make up
sensible numbers. Which line was hardest?

<details>
<summary>Check yourself</summary>

One reasonable version: *Decision:* which bikes the workshop inspects
this week. *Who acts:* two mechanics, 30 bikes a week between them.
*Prediction:* will this bike be reported faulty in the next 14 days?
*When:* every Sunday night, using last week's ride data. *Action cost:*
20 minutes per inspection. *Payoff:* a breakdown on the road costs a
rider's goodwill and a 1,200-shilling pickup; an inspection catches
most faults. *Metric:* breakdowns per week, and inspections that found
something. *Baseline:* inspect the 30 bikes with the most kilometres
since their last service.

Most people find *who acts* or *payoff* hardest, because they require
asking someone. That's the point of the exercise: the gaps are
questions for the workshop, and a model can't answer them for you.

</details>

## 2. Is the threshold really 0.22?

A voucher costs 100 and saves a leaver 30% of the time, worth 1,500.
The break-even was 0.22. A colleague says "but if I send a voucher to
a rider who wasn't going to leave, I've wasted it, so shouldn't the
threshold be higher?" Are they right?

<details>
<summary>Check yourself</summary>

No: that waste is already counted. The threshold compares the cost of
the voucher (100, paid every time) with the *expected* gain, which is
P(leave) x 0.3 x 1,500. For a rider with P(leave) = 0.15 the gain is
67 against a cost of 100: don't send. At 0.22 they cancel exactly.
Wasted vouchers on riders who stay are inside the 100. What the model
can't tell you is whether a voucher causes a rider to stay. That's an
**uplift** question, and it needs an experiment: send vouchers to a
random half and compare. The 30% in the brief should come from such a
test, not from hope.

</details>

## 3. Spot the leak

You're predicting, every Monday, which riders will leave in the next 60
days. For each feature, say whether it's safe: (a) rides in the last
30 days, (b) total rides in the account's lifetime, as read from the
database today, (c) the rider's "customer segment", which the marketing
team assigns every quarter using the last 90 days' behaviour including
the future, (d) the number of support tickets in the last week.

<details>
<summary>Check yourself</summary>

(a) is safe: it's computed from history before the Monday. (d) is
safe, if you count only tickets before the Monday. (b) is a trap: "as
read today" includes rides after the prediction date, and for leavers
the total stops growing, so it leaks the answer when you train on
past Mondays. It would be safe only if recomputed as of each Monday.
(c) leaks too, since the segment was built with information from after
the moment you predict. The rule: every feature must be reproducible
"as of" the prediction date.

</details>

## 4. Hands-on: a less effective voucher

Suppose the real save rate turns out to be lower than 0.30. Loop over
`SAVE_RATE` in 0.05, 0.1, 0.2, 0.3, 0.5. For each, recompute the
break-even, and print the value of: a voucher for everyone, the rule,
and the model at its break-even threshold, plus how many riders the
model would message. Predict first: at which save rate does the best
policy become "send nothing"?

<details>
<summary>Check yourself</summary>

```
save rate 0.05: break-even 1.333, everyone -486375, rule -51500, model 0, messaged 0
save rate 0.10: break-even 0.667, everyone -372750, rule -3500, model 12875, messaged 527
save rate 0.20: break-even 0.333, everyone -145500, rule 92500, model 100500, messaged 1410
save rate 0.30: break-even 0.222, everyone 81750, rule 188500, model 240875, messaged 2125
save rate 0.50: break-even 0.133, everyone 536250, rule 380500, model 590500, messaged 3507
```

At 0.05 the break-even is above 1, which no probability can reach, so
the right policy is to send nothing and the model's value is zero.
The model also beats the rule at every rate, but by a small amount at
low save rates (12,875 against -3,500 at 0.10) and a fairly wide one at
0.30. Notice how much depends on one number from the brief that nobody
measured yet.

Add this at the end of your copy:

```python
heading("Exercise 4")
for rate in (0.05, 0.1, 0.2, 0.3, 0.5):
    SAVE_RATE = rate
    be = COST / (SAVE_RATE * VALUE)
    print(f"save rate {rate:.2f}: break-even {be:.3f}, "
          f"everyone {value(np.ones(len(test)))}, rule {value(rule)}, "
          f"model {value(p > be)}, messaged {int((p > be).sum() * test_scale)}")
```

`value` reads the global `SAVE_RATE`, which is why reassigning it in the
loop is enough.

</details>

## 5. Optional puzzle: the cap and the threshold

Both limits apply at once: marketing can send 400 messages, and each
should have P(leave) above the break-even. How would you pick riders?
What does it mean if the break-even rule picks 2,125 riders and the
cap allows 400?

<details>
<summary>Check yourself</summary>

Take the riders above the break-even, then keep only the 400 with the
highest probabilities: `np.argsort(-p)[:400]`, filtered with
`p > break_even`. When the break-even picks far more riders than the
cap allows, the cap is the bottleneck, not the model. The useful
question then is "what would it be worth to lift the cap?", and the
brief already holds the answer: every extra voucher above break-even is
worth money, so a longer message budget probably pays for itself.
Models are rarely the constraint; budgets and capacity are.

</details>
