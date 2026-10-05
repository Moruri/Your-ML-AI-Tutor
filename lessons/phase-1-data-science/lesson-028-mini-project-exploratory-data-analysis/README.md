# Lesson 028 - Mini-project: exploratory data analysis

**Phase 1 - Data science basics** | Week 5, Day 1 | Monday 2026-10-05

> **Goal:** Take a fresh dataset from raw CSV to a short written analysis
> with charts you'd show a colleague, using every Phase 1 tool in the
> order you'd actually reach for them.

Time: about 75 minutes. Needs the venv from lesson 013 (`pandas`, `numpy`,
`matplotlib`).

---

No coffee shop today. You've just been handed a month of rides from a
small city bike-share scheme and asked a vague question that every analyst
hears sooner or later: *"Can you have a look at this and tell us what's
going on?"*

That's what **exploratory data analysis** (EDA) is for. Not proving
anything, not building a model yet, just getting from "a CSV I've never
seen" to "here are three things worth knowing, here's how sure I am, and
here's what I'd check next". It's the part of the job that decides whether
the modelling in Phase 2 is built on solid ground.

The script follows a routine you can reuse on any new dataset:

1. Look before you touch.
2. Clean, and write down every change.
3. Ask a few specific questions, one chart each.
4. Check the claims that matter.
5. Write it up for someone who won't read your code.

## How to follow along

Venv active, `python` in the repo root. Two new data files sit next to the
script: `rides_september.csv` (one row per ride) and
`weather_september.csv` (one row per day). Full script:

```bash
python lessons/phase-1-data-science/lesson-028-mini-project-exploratory-data-analysis/lesson.py
```

It writes three PNG charts and `report.md` to `output/` next to the script.
Delete the folder and re-run any time. Resampling uses a seeded generator,
so your numbers will match.

## 1. First look: what did we actually get?

```
rows, columns                                  -> (1198, 8)
missing values per column:
    duration_min       9
    rating           216
bike_type spellings                            -> ['Classic', 'E-bike', 'classic', 'electric']
duration_min summary:
    50%        13.0
    max      1110.0
```

(Output trimmed. Run the script for the full version.)

Resist the urge to start charting. Spend five minutes with `head`, `isna`,
`unique` and `describe` (lesson 017) and write down everything that looks
off. Here that list is: station names with stray spaces and random lower
case, four spellings of two bike types, nine rides with no duration, lots
of missing ratings, and a maximum ride of 1,110 minutes. Nobody rides for
18 hours. That's a bike someone forgot to dock.

This list is the most valuable thing you make all day. Every item on it is
something that would quietly bend a result later.

## 2. Clean it, and keep a log of what you changed

```
- tidied station names and bike types
- dropped 7 exact duplicate rows
- dropped 9 rides with no duration
- dropped 6 rides over 180 minutes (bikes not docked)
- added date, hour, weekend and speed columns
- kept missing ratings as NaN (212 rides)
clean rows                                     -> 1176
```

The fixes are all from lesson 018: `str.strip().str.title()` for names,
`replace` for spellings, `drop_duplicates`, `dropna`, and a filter for the
impossible durations. The new habit is the **cleaning log**. Every change
gets one line, and the log goes into the final report. If someone later
asks "why do you have 1,176 rides when the export says 1,198?", you can
answer in ten seconds.

Notice two judgement calls. Missing ratings are **kept** as `NaN`, because
dropping those rides would throw away 18% of perfectly good trips just to
tidy a column we may not use. And `MAX_MINUTES = 180` is a named constant
at the top of the script, so the cut-off is visible and easy to argue
about, rather than buried in a line of code.

## 3. Who rides, and when?

```
average rides per day:
    weekday    42.4
    weekend    30.5
rider mix (share of rides):
    rider_type  casual  member
    weekday       0.26    0.74
    weekend       0.57    0.43
busiest weekday hour                           -> '8:00'
busiest weekend hour                           -> '14:00'
saved output/rides_by_hour.png
```

One question, one chart. Open `output/rides_by_hour.png`: weekdays have a
sharp morning spike, a lunchtime bump and an early-evening spike, which is
commuting. Weekends are a gentle hill after lunch, with more casual riders
than members.

Two details from lesson 022 are doing real work here. The y-axis is
**average rides per day**, not total rides, because there are 22 weekdays
and only 8 weekend days; plotting totals would make weekends look emptier
than they are. And the title states the finding, so the chart still makes
sense when someone pastes it into a slide without your explanation.

## 4. Members vs casual riders

```
ride duration (minutes):
                count  median  mean
    rider_type
    casual        381    20.3  23.1
    member        795    10.5  12.0
median gap, casual - member (min)              -> 9.8
95% bootstrap CI for that gap                  -> (8.3, 10.9)
saved output/duration_by_rider_type.png
```

Ride durations are right-skewed (a few long rides drag the mean up), so
lesson 023 says compare **medians**. Casual riders take rides about twice
as long. Is that solid? The bootstrap from lesson 026 puts the gap between
about 8 and 11 minutes, nowhere near zero. Members ride short, purposeful
trips; casual riders wander.

The histogram (`duration_by_rider_type.png`) overlays both groups on the
same bins with see-through bars, and the legend carries the medians, so
the chart and the numbers tell the same story.

## 5. Does rain keep people off the bikes?

```
days, wet days                                 -> (30, 14)
weekday rides per day, dry vs wet:
    dry    52.7
    wet    32.0
correlation, rain_mm vs rides (weekdays)       -> -0.74
dry - wet weekday gap (rides per day)          -> 20.7
permutation p-value                            -> 0.0002
saved output/rides_vs_rain.png
```

This needs the weather file, so it's a join (lesson 020): count rides per
day, then `merge` with the weather on `date`. `validate="1:1"` makes
pandas complain if either side has duplicate dates, which is a cheap way
to catch a broken join before it fools you.

Then the confounding trap from lesson 024. Weekends are quieter anyway, so
if a few weekends happened to be rainy, "rain" would get the blame for
"Saturday". The script compares **weekdays only**. Dry weekdays averaged
about 53 rides and wet ones 32, a correlation of -0.74 with rainfall, and
the permutation test from lesson 027 says chance alone almost never
produces a gap that size.

The scatter plot keeps weekdays and weekends as separate markers, so you
can see the pattern holds in both. Still, it's one month. Rainy days in
September were also colder (check `temp_c`), so rain may be sharing the
credit. That goes in the caveats.

## 6. Write it up

```
saved output/report.md
```

Open `output/report.md`. It's a one-page summary that someone who will
never run your code can read in two minutes:

- **What we found**: three numbered findings, each with a number, how sure
  we are, and the chart that shows it.
- **Caveats**: what could make us wrong.
- **What we cleaned**: the log from section 2, word for word.
- **Suggested next step**: one concrete action.

Every number in it comes straight from the analysis (the script builds the
text with f-strings), so if the data changes and you re-run, the report
can't drift out of date. That's the habit to keep: never retype a number
from your terminal into a document by hand.

## What you can do now

- Run a first-look pass on any new CSV and list what's wrong before
  fixing anything.
- Clean data with a written log of every change and every judgement call.
- Turn a vague brief into a few specific questions, with one honest chart
  each.
- Back up a headline claim with an interval or a test, and name the
  confounders you couldn't rule out.
- Write a short, numbers-first report straight from your script.

## What to do now

1. Run `lesson.py`, then open the three PNGs and `report.md` in
   `output/`. Read the report as if you were the scheme's manager. Is
   anything unclear?
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   adds a fourth finding to the report.
3. That's Phase 1 done. Next, lesson 029 starts Phase 2 with *what a model
   actually is*: features, targets and loss, then fitting a line by hand.
   See [PROGRESS.md](../../../curriculum/PROGRESS.md).

A good EDA isn't the cleverest chart. It's a short list of true things,
how sure you are of each, and what you'd check next.
