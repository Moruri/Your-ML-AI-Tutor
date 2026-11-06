# Lesson 075 - Framing a problem worth solving

**Phase 5 - Projects and production intuition** | Week 9, Day 5 | Friday 2026-11-06

> **Goal:** Turn a vague request into a measurable ML problem, or decide
> it doesn't need ML at all.

Time: about 45 minutes. Same venv as lesson 013; nothing new to install.

---

Phase 5 is a change of mood. For four phases somebody handed you a
dataset and said "predict this column". Real work doesn't arrive like
that. It arrives as a sentence in a meeting:

> "Can we use AI to stop riders from leaving?"

Nothing in that sentence tells you what to predict, who will use the
answer, what a good result looks like, or whether a model is even the
right tool. Most failed ML projects fail right here, long before any
code: a strong model aimed at the wrong question. Today we work one
request all the way through. The setting is the bike-share scheme
again.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-5-projects/lesson-075-framing-a-problem-worth-solving/lesson.py
```

No downloads and no data file: the 6,000 riders are simulated inside
the script, and it runs in a few seconds. Every number below should
match yours exactly.

## 1. The request, and the brief behind it

```
the request                                    -> 'Can we use AI to stop riders from leaving?'
missing from the vague version                 -> ['who_acts', 'when', 'action_cost', 'payoff', 'metric', 'baseline']
missing from the written brief                 -> []
```

Before touching data, write a one-paragraph **brief**. The script keeps
it in a dictionary so a small function can tell you what's missing.
Eight questions, and the vague request answers two of them:

- **Decision:** what will someone do differently? Here, who gets a
  100-shilling ride voucher this week.
- **Who acts:** a person or a system, and what limits them? Marketing
  can message at most 400 riders a week.
- **Prediction:** the exact yes/no or number. "Will this rider take
  zero rides in the next 60 days?" is exact. "Churn" isn't.
- **When:** the moment of prediction, and only what's known then.
  Every Monday, using Monday's data.
- **Action cost:** a voucher costs 100 shillings, whether or not it
  works.
- **Payoff:** a kept rider is worth 1,500 shillings, and a voucher
  saves 3 in 10 of the riders who would have left.
- **Metric:** shillings gained per week, not accuracy.
- **Baseline:** what happens without a model. Message anyone who hasn't
  ridden for 21 days.

If you can't fill in a line, that's a question to take back to the
person who asked, and it's a better use of an afternoon than training
anything.

## 2. Some riders to practise on

```
riders                                         -> 6000
share who left                                 -> 0.253
```

The riders are invented, with five features known on the Monday:
rides in the last 30 days, days since the last ride, account age,
complaints, and whether they pay by M-Pesa. Whether each one **left**
(no ride in the next 60 days) comes from a rule we wrote, so we know
the truth. In real life you'd wait 60 days, or build the same table
from last year's Mondays. A quarter of riders leave. We keep 40% back
for testing, as in lesson 031.

## 3. Put money on it, then set the baselines

```
a voucher pays back only if P(leave) is above  -> 0.222
value: do nothing                              -> 0
value: voucher to everyone                     -> 81750
value: rule, no ride for 21+ days              -> 188500
riders the rule would message                  -> 995
ceiling: message exactly the leavers           -> 530250
```

Each voucher costs 100 and, for a rider who is going to leave, wins
back 0.3 x 1,500 = 450 on average. So sending one is worth it only if
the chance the rider leaves is above 100 / 450, which is **0.22**. This
isn't 0.5, and it's not a number the model gave us. It came from the
costs.

Every figure here is shillings per 6,000 riders, so the options are
comparable. Doing nothing earns nothing. A voucher for everyone makes
82,000, because leavers are common enough. The simple rule (message
anyone gone 21 days) does better, at 188,500, with fewer vouchers. The
ceiling, an oracle that messages exactly the leavers, shows the most
you could ever get: 530,000.

These are the **baselines**. A model has to beat the best of them, and
the rule is a strong one: it took no data scientist, only a sentence.
If your model's value is close to the rule's, ship the rule.

## 4. Now a model, judged by the decision

```
AUC (how well it ranks riders)                 -> 0.778
value: send when P(leave) > 0.5                -> 180500
value: send when P(leave) > break-even         -> 240875
riders messaged at that threshold              -> 2125
with a cap of 400: model top 400               -> 116375
with a cap of 400: longest-away 400            -> 110750
precision of the model's top 400               -> 0.869
```

A gradient-boosted model (lesson 040) trained on the five features
ranks riders decently: AUC 0.78. Then comes the part that's easy to
skip. Same model, two thresholds:

- Send when P(leave) is above 0.5 (the default, and lesson 043's point
  about thresholds): 180,500. That's *worse* than the rule. We left
  money on the table, because a 30% chance of leaving is still worth a
  voucher.
- Send when P(leave) is above 0.22, the break-even: 240,875, which
  beats the rule by about 52,000.

Same model; the threshold alone moves the value by 60,000. That's why
accuracy is the wrong score here: it treats a wasted voucher and a lost
rider as the same size of mistake. They aren't. One costs 100 and the
other 1,500.

The cap changes the question again. If marketing can only send 400
messages, what matters is **who's in the top 400**. The model's top 400
are 87% real leavers and bring in 116,375; just taking the 400 riders
who've been away longest brings in 110,750. A modest win, for a model.
Whether it's worth keeping that model running, and someone to look
after it, is a business question, and now you can put numbers on it.

## 5. A feature that cheats

```
AUC without the column                         -> 0.778
AUC with refund_requested                      -> 0.939
refund_requested, riders who stayed            -> 0
value on paper, send when P > break-even       -> 411750
```

The table has a column, `refund_requested`, filled in when a rider
closes their account and asks for a refund. Add it as a feature and the
AUC leaps to 0.94 and the value on paper to 411,750. Everyone is
delighted for about a day.

The giveaway is the line in the middle: among riders who stayed, the
column is zero for every one of them. It's *recorded because they left*.
On a real Monday it wouldn't be there yet. This is **leakage** (lesson
031's warning, in a different outfit), and framing prevents it: the
brief says *when* we predict, so every feature gets one question: "would
I know this on that Monday?"

Ask for each feature, and say it out loud in code review. The model
can't tell the difference, and a leaked score looks wonderful right up
to launch day.

## 6. When the honest answer is "don't build a model"

```
bikes, thefts                                  -> (5000, 20)
thefts in each test set (min, max)             -> (8, 8)
share of thefts caught in top 5% (min, max)    -> (0.0, 0.125)
average share caught                           -> 0.05
thefts at quiet stations (of bikes there)      -> (17, 2526)
thefts at busy stations (of bikes there)       -> (3, 2474)
```

A different request: "predict which bikes will be stolen". The
history is 5,000 bikes and 20 thefts. We split the bikes 20 different
ways, train a simple model each time, and flag the top 5%.

Look at the second line: each test set contains only 8 thefts, so
"share caught" can only be 0, 1/8, 2/8 and so on. Across the splits,
it ranges from 0 to 12.5%, and averages 5%, barely better than
flagging bikes at random. With so few examples, a model has little to
learn from and you have little to judge it with. Any score you'd report
would be mostly luck.

The last two lines are more useful than a model: 17 of the 20 thefts
happened at quiet stations (about 0.7% of bikes there, against 0.1% at
busy ones). A `groupby` takes a minute, and gives the operations team
something to act on today: move the overnight bikes, add lighting, or
lock them. You could also read all 20 stories one by one, and learn
more from that than from any score.

Reasons to skip ML, in rough order:

- **A rule already does the job** (service every 800 km).
- **Too few examples** of the thing you care about.
- **Nobody will act** on the prediction, or the action costs more than
  it saves.
- **No way to know if it worked**, because you never see the outcome.
- **A mistake is too costly** and a human has to check every case
  anyway. Then the model may be a ranking aid, not a decision-maker.

## What you can do now

- Turn a vague request into a written brief: decision, who acts,
  target, timing, costs, payoff, metric and baseline.
- Put a price on right and wrong outcomes, and work out the break-even
  probability from it.
- Set baselines (do nothing, treat everyone, a simple rule) that a model
  must beat.
- Choose a threshold from costs and capacity instead of defaulting to
  0.5.
- Catch a feature that wouldn't exist at prediction time.
- Recognise when there's too little data, or no decision, for a model
  to help.

## What to do now

1. Run `lesson.py`. Then change `BRIEF` into the vague version and see
   which questions come back unanswered.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   asks how the answer changes when a voucher is less effective.
3. Next, lesson 076: project structure, tests and reproducibility. A
   good question deserves a project you can re-run next month.

A model is one possible answer to a question. Write the question down
first, with the costs, and then check if it's the right answer.
