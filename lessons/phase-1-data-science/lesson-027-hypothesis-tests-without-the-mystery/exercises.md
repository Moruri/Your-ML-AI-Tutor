# Lesson 027 - Exercises

Three quick checks, a hands-on borderline test, and an optional rewrite.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-1-data-science/lesson-027-hypothesis-tests-without-the-mystery/lesson.py my_lesson_027.py
python my_lesson_027.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-1-data-science/lesson-027-hypothesis-tests-without-the-mystery").resolve()
```

`load_orders`, `permutation_test`, `ORDERS_CSV` and `RNG` are there to
reuse.

---

## 1. Name the null

For each question, write the null hypothesis in one plain sentence.

- (a) Do oat-milk orders bring in more revenue than whole-milk orders?
- (b) Is Friday busier than Monday?
- (c) Did the new menu board change the average order?

<details>
<summary>Check yourself</summary>

- **(a)** Milk type makes no difference to average order revenue.
- **(b)** Friday and Monday have the same average number of orders.
- **(c)** Average order revenue is the same before and after the board.

The null is always the boring "nothing's going on" version. The test
asks how surprising the data would be if that were true.

</details>

## 2. Read the p-value

A test comparing two groups gives p = 0.03. Which statements are fair?

- (a) "There's a 3% chance the groups are really the same."
- (b) "If the groups were really the same, a gap this big would turn up
  about 3% of the time."
- (c) "The difference is large and important."
- (d) "We'd call this significant at the 0.05 level."

<details>
<summary>Check yourself</summary>

- **(a) Wrong.** That's the most common misreading. The p-value assumes
  the null is true; it can't also tell you the chance that it is.
- **(b) Fair.** That's the definition from section 5.
- **(c) Not shown.** p says nothing about size. You need the gap and an
  interval for that.
- **(d) Fair**, as a convention, as long as you remember 0.05 is a line
  people agreed on, not a law of nature.

</details>

## 3. Twenty slices

A colleague tests 20 different customer segments for "higher spend than
average" and proudly reports one with p = 0.04. What do you say?

<details>
<summary>Check yourself</summary>

With 20 tests on data where nothing is going on, the chance of at least
one p below 0.05 is about 64% (section 5). One hit out of 20 is roughly
what noise alone gives you. Ask for a fresh month of data to re-test that
one segment, or use a stricter line (a common rough fix is to divide 0.05
by the number of tests: 0.05 / 20 = 0.0025).

</details>

## 4. Hands-on: oat milk vs whole milk

**a)** Split order revenue into oat-milk and whole-milk orders. What is
the gap in mean revenue (oat minus whole)?

**b)** Run `permutation_test(oat, whole)`. What is the p-value?

**c)** Run Welch's t-test on the same groups. Do the two agree?

**d)** Bootstrap a 95% interval for the gap (2,000 resamples). Does it
include zero, or only just miss it?

**e)** In two sentences, tell the shop owner what you found. Would you
switch the whole menu to oat milk on this evidence?

Hints:

- `orders.loc[orders["milk"] == "oat", "revenue"]`
- `stats.ttest_ind(oat, whole, equal_var=False).pvalue`
- Copy the bootstrap pattern from `section_real_difference`.

<details>
<summary>Expected results</summary>

```
a) gap about £0.53 per order (oat £6.52, whole £5.99)
b) permutation p about 0.03 (one run of the pattern below gave 0.034)
c) Welch p about 0.044, so yes, they agree closely
d) an interval whose lower end only just clears zero; the pattern below
   gave about (0.04, 1.04), and other seeds can land on either side of 0
e) see below
```

</details>

<details>
<summary>One way to write it</summary>

```python
orders = load_orders(ORDERS_CSV)
oat = orders.loc[orders["milk"] == "oat", "revenue"]
whole = orders.loc[orders["milk"] == "whole", "revenue"]

gap, null, p = permutation_test(oat, whole)
print("gap", round(gap, 3), "permutation p", round(p, 4))
print("Welch p", stats.ttest_ind(oat, whole, equal_var=False).pvalue)

boot = np.array([
    RNG.choice(oat.to_numpy(), len(oat)).mean()
    - RNG.choice(whole.to_numpy(), len(whole)).mean()
    for _ in range(2000)
])
print("95% CI", np.percentile(boot, [2.5, 97.5]).round(3))
```

A fair summary: "Oat-milk orders brought in about 50p more on average in
September, and that's unlikely to be pure chance (p is about 0.04), but
the gap could be anywhere from a few pence to about a pound. It's worth
watching next month, not worth rebuilding the menu around." Before acting,
also ask *why* the gap exists: is oat priced higher on the menu, or do oat
customers buy more cups? A `groupby` on `price` and `quantity` (lesson 019)
answers that in a line or two.

</details>

## 5. (Optional) Fix the headline

A report says: "Morning customers spend significantly more (p = 0.14)."
Rewrite it so it's honest.

<details>
<summary>Check yourself</summary>

Something like: "Morning orders averaged about 28p more than later orders
in September, but a gap that size is common by chance (p = 0.14), so we
have no good evidence of a real difference yet." "Significantly" should
only appear when the test actually clears the line you chose in advance.

</details>

---

That's hypothesis testing done. If one thing sticks: *a p-value says how
often chance would fake a gap this big. It never says how big or how
important the gap is, so always report that too.*
