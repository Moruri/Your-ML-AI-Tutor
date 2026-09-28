# Lesson 012 - Exercises

This is a project lesson, so the exercises are the owner's follow-up
questions: three short ones to check you can read the report, a hands-on
set of extensions, and an optional design question. Predict first, then
run.

Work in a copy. From the repo root:

```bash
cp lessons/phase-0-python/lesson-012-mini-project-a-stdlib-data-report/lesson.py my_lesson_012.py
python my_lesson_012.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-0-python/lesson-012-mini-project-a-stdlib-data-report").resolve()
```

`load`, `Order`, `summarise`, `breakdowns`, `build_report` and `pounds` are
all there to reuse.

---

## 1. Read the report

From the printed report alone (no code):

- (a) The owner says "our average order is £5.54, so most customers spend
  about a fiver". What would you gently say back?
- (b) Which hour would you add a second barista for?
- (c) Card was 69% of revenue. Does that mean 69% of *customers* paid by
  card?

<details>
<summary>Check yourself</summary>

**(a)** The median is £3.90: half of all orders were £3.90 or less. A
typical customer buys one drink. The average is pulled up by a smaller
number of big multi-drink orders. "Most customers spend about £4, and
some groups spend a lot more" is closer to the truth.

**(b)** 08:00 to 09:00, with 51 cups, over half again as many as either
neighbouring hour. (07:00 and 09:00 are the next busiest, so really it's
a 7-10 morning rush with a peak at 8.)

**(c)** No. It's a share of *money*, not of *orders*. If card orders tend
to be bigger (or smaller), the two shares differ. Exercise 4 works out the
order share, and it's not the same number.

</details>

## 2. Why is this rejected?

Without running anything, which rule in `to_order` rejects each of these
normalised rows, and what does the message say?

```python
{"order_id": "B1", "date": "2026-09-14", "time": "08:15", "drink": "latte",
 "size": "large", "price": "3.80", "quantity": "1", "payment": "card"}

{"order_id": "B2", "date": "2026-09-14", "time": "8:15", "drink": "tea",
 "size": "medium", "price": "2.50", "quantity": "1", "payment": "contactless"}
```

<details>
<summary>Check yourself</summary>

**B1:** `price 3.80 doesn't match the menu (£4.30)`. A large latte is
£4.30. This is the rule that never fired on the real file; here it
catches a medium price typed against a large drink.

**B2:** `time '8:15' is not HH:MM`. `time.fromisoformat` wants two-digit
hours, so it's rejected before anyone looks at `contactless`. (Fix the
time and you'd get `payment 'contactless' is not card or cash`. First
failure wins, as in lesson 008.) Both are arguably *fixable* mess:
`8:15` clearly means `08:15`, and contactless is a kind of card payment.
If the till starts producing them, they belong in `normalise`, not in the
rejects file.

</details>

## 3. Blank lines and line numbers

Lesson 008 numbered rows with `enumerate(reader, start=2)`. This lesson
uses `reader.line_num`. The file has blank lines after rows 31 and 91.
If we'd used `enumerate`, what would the "mocha" reject (really line 59)
have been reported as, and why does it matter?

<details>
<summary>Check yourself</summary>

Line 58. `DictReader` skips the blank line after row 31 without giving
you a row for it, so `enumerate` counts one fewer than the file has, and
every reject after the first blank line is off by one (and by two after
the second). The owner opens line 58, finds a perfectly good order, and
stops trusting the rejects file. `line_num` counts what's really in the
file, blanks and all.

</details>

## 4. Hands-on: the owner's follow-up questions

The owner read the report and has questions. Answer each by adding to
your copy; reuse `load`, `summarise` and friends rather than re-reading
the file yourself.

**a) Sizes.** Revenue by size, largest first.

**b) Card vs cash, properly.** For each payment method: number of orders,
share of *orders*, mean spend and median spend.

**c) Week on week.** Revenue, orders and cups for week one (7-11 Sep) and
week two (14-18 Sep), and the percentage change in revenue.

**d) Most popular drink by orders.** Not revenue: which drink appears in
the most orders? Use `Counter`.

**e) Add it to the report.** Add a "Revenue by size" section to
`build_report`, in the same style as "Revenue by drink", with bars.

**f) A date filter.** Add `--start` and `--end` options (ISO dates,
optional) to `main()`, and filter orders to that range after loading.
Check that `--start 2026-09-14` gives week two's revenue.

Hints, if you want them:

- For (b), `[o.total_cents() for o in orders if o.payment == "card"]`
  gives you a list to hand to `summarise`.
- For (c), `o.day <= date(2026, 9, 11)` splits the weeks. Percentage
  change is `(new - old) / old`, and `f"{change:+.0%}"` shows the sign.
- For (f), `type=date.fromisoformat` works in `add_argument`: argparse
  will call it on the string and complain nicely if it fails.

<details>
<summary>Expected results</summary>

```
a) medium £334.60, small £189.50, large £157.40

b) card: 86 orders (70% of orders), mean £5.45, median £4.10
   cash: 37 orders (30% of orders), mean £5.74, median £3.80

c) week one: £369.90 from 64 orders
   week two: £311.60 from 59 orders, 93 cups
   change:   -16%

d) latte, in 41 orders (then cappuccino 31, flat white 22, tea 15, espresso 14)

f) --start 2026-09-14  ->  Revenue £311.60, Orders 59
```

(b) answers question 1(c): card was 69% of revenue and 70% of orders,
close but not the same, and the mean and median point in *opposite*
directions (cash has the higher mean, card the higher median). With only
37 cash orders, a couple of big ones can do that. That's worth knowing
before anyone builds a policy on "cash customers spend more".

(c) is a good one to be careful with in front of the owner: a 16% drop
sounds alarming, but it's two weeks, and week one included the busiest
Monday in the file. Lesson 026 is about how to tell a real change from
week-to-week wobble.

</details>

<details>
<summary>One way to write it</summary>

```python
from datetime import date

orders, rejects = load(RAW_CSV)

# a)
by_size = defaultdict(int)
for o in orders:
    by_size[o.size] += o.total_cents()
print(", ".join(f"{s} {pounds(c)}" for s, c in sorted(by_size.items(), key=lambda i: i[1], reverse=True)))

# b)
for method in ("card", "cash"):
    spend = [o.total_cents() for o in orders if o.payment == method]
    s = summarise(spend)
    print(f"{method}: {s['count']} orders ({s['count'] / len(orders):.0%} of orders), "
          f"mean {pounds(s['mean'])}, median {pounds(s['median'])}")

# c)
week_one = [o for o in orders if o.day <= date(2026, 9, 11)]
week_two = [o for o in orders if o.day >= date(2026, 9, 14)]
old = sum(o.total_cents() for o in week_one)
new = sum(o.total_cents() for o in week_two)
print(f"week one: {pounds(old)} from {len(week_one)} orders")
print(f"week two: {pounds(new)} from {len(week_two)} orders, {sum(o.quantity for o in week_two)} cups")
print(f"change:   {(new - old) / old:+.0%}")

# d)
print(Counter(o.drink for o in orders).most_common())

# e) inside build_report, after the "Revenue by drink" block:
#     by_size = defaultdict(int)
#     for o in orders:
#         by_size[o.size] += o.total_cents()
#     lines += ["", "Revenue by size"]
#     top = max(by_size.values())
#     for size, cents in sorted(by_size.items(), key=lambda item: item[1], reverse=True):
#         lines.append(f"  {size:<11} {pounds(cents):>9}  {bar(cents, top)}")

# f) in main():
#     parser.add_argument("--start", type=date.fromisoformat, default=None)
#     parser.add_argument("--end", type=date.fromisoformat, default=None)
#     ...
#     orders, rejects = step_validate(args.csv)
#     if args.start:
#         orders = [o for o in orders if o.day >= args.start]
#     if args.end:
#         orders = [o for o in orders if o.day <= args.end]
```

If you add the filter, the `assert`s at the end of `main()` will fail for
a filtered run, because they check the whole fortnight. Only run them when
no filter is set: `if args.csv == RAW_CSV and not (args.start or args.end):`.
Checks that are wrong for some inputs aren't checks; they're noise.

</details>

## 5. (Optional) What would you change first?

Suppose the owner loves the report and wants it every week, forever.
Name two things in `lesson.py` you'd change first, and why.

<details>
<summary>Check yourself</summary>

There's no single right answer. Good candidates:

- **Move the reusable parts into a module** (lesson 009): `Order`,
  `normalise`, `to_order`, `load`, `summarise`. The weekly script becomes
  a thin `main()` that imports them.
- **Take the menu from a file** instead of the code. Prices change; code
  shouldn't have to.
- **Stop hard-coding the checks.** The `assert`s pin this one fortnight's
  numbers. A weekly job needs checks that are true every week: rows in
  equals kept plus rejected, no duplicate IDs, revenue equals the sum of
  the drink lines.
- **Warn when rejects are high.** Seven out of 130 is fine; seventy would
  mean the till export format changed, and the report should say so
  loudly rather than quietly reporting on half the data.

Notice these are about *keeping it working as things change*, which is
Phase 5's whole subject. You'll be surprised how much of real data work
is exactly this.

</details>

---

That's Week 2, Day 5 done, and with it Phase 0. On Monday, Phase 1 begins
with lesson 013: setting up a real environment and installing your first
packages. If one thing sticks from today, let it be: *look, tidy,
validate, summarise, report, and always say how many rows you didn't
use*.
