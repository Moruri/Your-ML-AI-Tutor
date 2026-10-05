# Lesson 028 - Exercises

This one is a mini-project, so the exercises are about judgement as much
as code: two quick checks, a hands-on fourth finding, a confounder hunt,
and an optional "your own data" run.

Activate your venv. For the hands-on tasks, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-028-mini-project-exploratory-data-analysis/lesson.py my_lesson_028.py
python my_lesson_028.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-028-mini-project-exploratory-data-analysis").resolve()
```

Every `section_...` function returns what it built, so you can reuse
`section_clean` to get the clean `rides` table.

---

## 1. Drop or keep?

For each problem, would you **drop the rows**, **fix them**, or **keep
them and note it**? One sentence of reasoning each.

- (a) 216 rides with no rating.
- (b) 6 rides lasting 600 to 1,500 minutes.
- (c) `"  Old Mill "` and `"old mill"` as station names.
- (d) A ride of 2.2 minutes.

<details>
<summary>Check yourself</summary>

- **(a) Keep and note.** The rest of each row is fine, and dropping 18% of
  rides would bias every other number. Just don't lean on ratings.
- **(b) Drop and log.** They're almost certainly bikes left undocked, not
  rides. Including them would wreck any mean duration.
- **(c) Fix.** It's the same station spelt badly; `strip().title()` makes
  it one.
- **(d) Keep.** Short, but plausible (someone changed their mind, or
  rode one block). There's no rule for "too short" without asking the
  scheme how it records rides, which is itself a good question to ask.

</details>

## 2. Totals vs per-day

A colleague plots *total* rides by hour for weekdays and weekends and
says "weekends are dead". What's wrong with the comparison, and how does
section 3 avoid it?

<details>
<summary>Check yourself</summary>

September 2026 has 22 weekdays and only 8 weekend days, so weekday totals
are nearly three times bigger just from having more days. Section 3
divides each hour's count by the number of days of that type, giving
**average rides per day**, which is a fair comparison.

</details>

## 3. Hands-on: a fourth finding

Electric bikes should be quicker. Are they, and by how much?

**a)** Using the clean `rides` table, find the median `speed_kmh` for
electric and classic bikes.

**b)** Bootstrap a 95% interval for the gap in medians (electric minus
classic, 2,000 resamples, `np.random.default_rng(28)`).

**c)** Make a chart that shows the two speed distributions, with a title
that states the finding.

**d)** Add a fourth numbered finding to the report in
`section_report`, built with an f-string like the other three.

Hints:

- `rides.loc[rides["bike_type"] == "electric", "speed_kmh"].to_numpy()`
- Copy the bootstrap from `section_members` and swap the columns.
- Overlapping `ax.hist(..., alpha=0.6)` with shared `bins` works well.

<details>
<summary>Expected results</summary>

```
a) electric median about 14.7 km/h, classic about 10.9 km/h
   (429 electric rides, 747 classic)
b) 95% interval for the gap about (3.6, 4.1) km/h
c) two clearly separated humps; a title like
   "Electric bikes are about 4 km/h faster"
d) something like:
   4. **E-bikes are quicker.** Median speed 14.7 km/h on electric vs
      10.9 km/h on classic bikes (gap about 3.6-4.1 km/h, 95% interval).
```

</details>

<details>
<summary>One way to write it</summary>

```python
raw = section_first_look()
rides, log = section_clean(raw)

rng = np.random.default_rng(28)
e = rides.loc[rides["bike_type"] == "electric", "speed_kmh"].to_numpy()
c = rides.loc[rides["bike_type"] == "classic", "speed_kmh"].to_numpy()
print("medians", round(float(np.median(e)), 1), round(float(np.median(c)), 1))

boot = np.array([
    np.median(rng.choice(e, len(e))) - np.median(rng.choice(c, len(c)))
    for _ in range(2000)
])
print("95% CI", np.percentile(boot, [2.5, 97.5]).round(1))

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.arange(4, 24, 0.5)
ax.hist(c, bins=bins, alpha=0.6, label="Classic")
ax.hist(e, bins=bins, alpha=0.6, label="Electric")
ax.set_title("Electric bikes are about 4 km/h faster")
ax.set_xlabel("Average speed of the ride (km/h)")
ax.set_ylabel("Number of rides")
ax.legend()
save(fig, "speed_by_bike_type.png")
```

</details>

## 4. Hunt the confounder

Rainy days had fewer rides. But open `weather_september.csv`: were rainy
days also different in some *other* way that could keep people off bikes?

**a)** Compare the average `temp_c` on wet and dry days.

**b)** Compute the correlation between `rain_mm` and `temp_c`.

**c)** Rewrite the third caveat in the report so it says exactly what you
found.

<details>
<summary>Check yourself</summary>

```
a) wet days averaged about 14.9 °C, dry days about 18.5 °C
b) correlation about -0.54: rainy days were noticeably colder
```

So "rain costs rides" is really "rainy, colder days cost rides", and one
month can't pull the two apart. A fair caveat: "Wet days were also about
3.6 °C colder on average, so some of the drop may be the cold rather than
the rain. More months of data would help separate them." Separating two
causes that move together is exactly what the regression models in Phase 2
start to help with.

</details>

## 5. (Optional) Your own data

Pick any CSV you care about: your bank export, a running log, a public
dataset from your city. Run the same five steps (first look, cleaning log,
three questions with a chart each, one check, a one-page `report.md`).
Keep it under an hour. The point is the routine, not perfection.

<details>
<summary>Check yourself</summary>

Swap your report with someone (or read it tomorrow morning). If they can
say what your three findings are, how sure you were, and what you'd do
next, without asking you anything, it worked.

</details>

---

That's Phase 1 done. If one thing sticks: *look first, log every change,
one question per chart, and never put a number in a report that you
didn't check.*
