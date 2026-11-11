# Lesson 081 - Performance: making it fast enough

**Phase 5 - Projects and production intuition** | Week 10, Day 3 | Wednesday 2026-11-11

> **Goal:** Profile, batch, cache and quantise so the model fits the
> budget.

Time: about 45 minutes. Same venv as lesson 013; nothing new to install.

---

"Make it faster" is not a goal. It has no finish line, so it eats
afternoons. A goal sounds like this: **95% of requests answered in under
2 ms, and the nightly batch of 100,000 forecasts done in under a
second.** That's a **budget**. Once you have one, performance work
becomes a short loop:

1. Measure against the budget.
2. Profile to find where the time actually goes.
3. Fix the biggest thing, check the answers didn't change, measure again.
4. Stop as soon as you're inside the budget.

Today we run that loop on a deliberately naive version of the ride
service from lesson 079, then look at the last trick in the bag,
**quantisation**, on a bigger model.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-5-projects/lesson-081-performance-making-it-fast-enough/lesson.py
```

It takes about ten seconds. No downloads and no data file: the rides are
simulated in the same way as lessons 078 and 079. A small `output/`
folder appears next to the script, wiped every run.

One honest warning: **timings depend on your computer**, and even on
what else it's doing at that moment. Your milliseconds won't match the
ones below. The *ratios* should be similar (something 10 times faster
here will be roughly 10 times faster for you), and every number that
isn't a time should match, give or take the last decimal of the neural
network's error.

## 1. Set a budget, then measure the naive version

```
  budget: p95 per request (ms)                   -> 2.0
  naive: median, p95 (ms)                        -> (4.99, 5.84)
  within budget?                                 -> False
  answer for Market, 21 C, dry                   -> 41.9
```

`predict_naive` is what a first version often looks like, and there's
nothing *wrong* with it: it unpickles the model, reads the whole history
CSV, rebuilds every feature, picks out the last row for the station and
predicts. Correct, readable, and doing almost all of its work for
nothing, every single time.

We measure two numbers, not one. The **median** is the typical request.
The **p95** (95th percentile) is the slow end: 1 request in 20 is slower
than this. Users and calling programs feel the slow end, so budgets are
usually written as a p95 or p99, never as an average. An average hides
the occasional terrible request.

Write down the answer, too: 41.9 rides. Every speed-up from here has to
give the same answer, or it's not a speed-up, it's a bug.

## 2. Profile before you touch anything

```
  share of time, read_csv (%)                    -> 11
  share of time, build_features (%)              -> 58
  share of time, pickle.loads (%)                -> 0
  share of time, model.predict (%)               -> 10
  share of time, everything else (%)             -> 20
```

Guessing where time goes is a famous way to waste a day. Most people
would guess "the model", and the model is 10%. More than half the time
goes into `build_features`, recomputing lags and rolling averages for 90
days of history we already processed. Loading the pickle, which sounds
expensive, rounds to zero for a small ridge model.

The tool is `cProfile`, in the standard library:

```python
import cProfile, pstats
prof = cProfile.Profile()
prof.enable()
predict_naive("Market", 21.0, 0.0)
prof.disable()
pstats.Stats(prof).sort_stats("cumulative").print_stats(12)
```

You can also run a whole script under it with
`python -m cProfile -s cumulative lesson.py`. The lesson saves the full
top-12 table to `output/profile_naive.txt`. Open it: `cumtime` is the
time spent inside a function *including* everything it calls, which is
the column to read first.

## 3. Load once, keep what you need in memory

```
  load once: median, p95 (ms)                    -> (0.62, 0.69)
  plain numpy maths: median, p95 (ms)            -> (0.00092, 0.001)
  same answers?                                  -> (True, True)
  speed-up of numpy over naive, about            -> '5,416x'
```

The biggest fix is boring: do the expensive work **once, at start-up**,
not on every request. `predict_warm` loads the model once and keeps a
tiny dictionary of the latest `lag_1` and `avg_7` per station. That's
already well inside the budget, about 8 times faster.

What's left is mostly not maths. A ridge prediction is five
multiplications and an addition. The 0.62 ms is pandas building a
one-row DataFrame and scikit-learn checking it (column names, types,
shapes). Those checks are what you want during development, and they're
a fixed cost per call. `predict_lean` does the five multiplications with
numpy directly and lands under a microsecond, with exactly the same
answer.

Should you always do that? No. Hand-written maths has to be kept in step
with the model by hand; the day someone adds a feature and forgets, it
silently gives wrong answers. Here we were already inside the budget
with `predict_warm`, so in real life we'd have **stopped there**. It's
in the lesson to show where the time goes, not as a recommendation.

## 4. Batch: one call for many rows

```
  rows in the nightly batch                      -> 100000
  one row at a time, estimated (s)               -> 36.0
  one call for all rows (s)                      -> 0.004
  within the batch budget?                       -> True
  largest difference between the two             -> 1.4210854715202004e-14
```

Section 3 showed that every call has a fixed cost. **Batching** pays it
once for many rows. Predicting the nightly 100,000 rows one at a time
would take over half a minute; one call takes a few milliseconds,
thousands of times faster, and the answers agree to the 14th decimal
place (that tiny difference is floating-point rounding, not a change).

This is the same reason lesson 055 fed a network mini-batches, and why
GPUs exist: the hardware is fast at doing the same sum on lots of
numbers at once, and slow at starting and stopping. If your API gets
many small requests, some servers even **micro-batch**: wait a few
milliseconds, collect the requests that arrive, and predict them
together. That trades a little latency for a lot of throughput.

## 5. Cache answers people ask for again and again

```
  requests, distinct questions                   -> (2000, 21)
  cache hits, misses                             -> (1979, 21)
  time with cache vs without, about (s)          -> (0.05, 4.0)
  after new data: a miss, not a stale answer     -> True
```

Real traffic repeats itself. Here 2,000 requests ask only 21 different
questions ("Market, tomorrow" over and over). We pretend each forecast
costs 2 ms (a weather lookup, say) and wrap it in
`functools.lru_cache`, which remembers recent answers. 1,979 of the
2,000 requests are answered from memory.

The danger of a cache is a **stale answer**: yesterday's forecast served
after today's data arrived. The fix in the lesson is to make the data
version part of the key, `forecast(station, day, data_version)`. Once
the version changes, old entries are simply never asked for again. This
is lesson 078's fingerprints doing a second job. Two more rules worth
keeping: cache only functions that give the same output for the same
input, and give the cache a size limit (`maxsize`) so it can't eat all
your memory.

## 6. Quantise a bigger model: smaller, almost as good

```
  weights in the network                         -> 67072
  weight size in KB: float64, float32, int8      -> (524, 262, 66)
  test MAE: float64, float32, int8               -> (3.986, 3.986, 3.984)
  largest change in one prediction, int8         -> 0.114
  ridge test MAE, for comparison                 -> 4.0
  test rows                                      -> 636
  forward pass, float64 vs float32 (ms)          -> (0.68, 0.34)
```

Big models hit a different wall: memory and download size. A model's
weights are usually stored as 64-bit or 32-bit floats. **Quantisation**
stores them with fewer bits. We train a two-layer network (256 units
each, 67,072 weights) on two years of rides, then store its weights
three ways:

- **float32** halves the size and changes nothing you can see. Most deep
  learning already runs in float32 for this reason.
- **int8** is a quarter of float32. Each layer's weights are divided by
  one scale number so they fit between -127 and 127, rounded to whole
  numbers, and multiplied back when used. The error moved in the third
  decimal place and no single prediction changed by more than 0.12 of a
  ride.

That's the pattern you'll see with large language models too: a 7
billion-weight model is about 28 GB in float32 and about 7 GB in int8,
which is the difference between "needs a server" and "fits on a laptop".

Two honest footnotes. First, our int8 version saves *space*, not time:
numpy has no fast int8 maths, so we turn the numbers back into floats
before using them. Real speed-ups come from libraries with int8 kernels
(ONNX Runtime, PyTorch's quantisation tools, llama.cpp). Halving the
bits did make the float32 pass about twice as fast as float64, though,
because there's half as much memory to move. Second, always re-check
accuracy after quantising, on your test set, as we did. Some models
shrug it off; others lose more than you'd like.

And look at the ridge line: 4.0, as good as the network. On this data
the 67,000-weight network buys nothing over five numbers. **The cheapest
speed-up is a smaller model that's just as good**, which is why lesson
075 told you to beat a simple baseline before reaching for anything big.

## 7. Stop when it's fast enough

```
  request p95: naive vs now (ms)                 -> (5.84, 0.0012)
  request budget met?                            -> True
  batch budget met?                              -> True
```

Both budgets are met, so we stop. That's the hardest step for
enthusiastic people. Every optimisation costs something: more code,
more ways to go stale, harder debugging. Spend that only where the
budget asks for it.

In rough order of how often each one saves the day:

1. **Don't repeat work:** load once, precompute, keep features ready.
2. **Batch:** one call for many rows.
3. **Cache:** with the data version in the key.
4. **Shrink:** a smaller model, float32, then quantisation.
5. **Rewrite in something faster:** last, and rarely needed.

## What you can do now

- Turn "make it faster" into a budget with a median and a p95.
- Profile with `cProfile` and read `cumtime` before changing anything.
- Remove repeated work, batch predictions and cache safely by data
  version.
- Explain what quantisation does, what it saves, and how to check it
  didn't hurt.
- Know when to stop.

## What to do now

1. Run `lesson.py`, then open `output/profile_naive.txt`.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   profiles `predict_warm` and squeezes it under 0.1 ms without
   hand-written maths.
3. Next, lesson 082: the capstone begins. You'll choose and scope your
   own project.

Fast enough is a number. Find it, hit it, and go and do something else.
