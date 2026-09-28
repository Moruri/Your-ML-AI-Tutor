# Lesson 012 - Mini-project: a stdlib data report

**Phase 0 - Python foundations for data work** | Week 2, Day 5 | Friday 2026-09-18

> **Goal:** load a CSV, clean it, compute summary statistics, and print a
> tidy report with pure Python, using nothing you haven't already learned
> except one small, friendly standard library module.

Time: about 60 minutes. No installs. Python 3.10+.

---

This is the last lesson of Phase 0, and it's a project rather than a new
idea. The shop owner has sent two weeks of till exports, 7 to 18
September, glued into one file. The till has been updated since lesson 007
and now records the time and how people paid, and the file has all the
usual mess, plus some new kinds. The owner wants one page that answers the
questions owners actually ask: how much did we take, what sells, which
days and hours are busy, and how do people pay?

Everything you need, you've got:

| You'll use | From |
|------------|------|
| `csv.DictReader` / `DictWriter`, `pathlib`, an `output/` folder | 007 |
| String cleaning, f-string formatting, money in cents | 003 |
| Dicts, `Counter`, `defaultdict` for grouping | 005 |
| Small single-purpose functions | 006 |
| Validation that raises `ValueError` with a useful message | 008 |
| `argparse` and `if __name__ == "__main__":` | 009 |
| A frozen dataclass for a clean order | 010 |
| Looking at a file lazily, not all at once | 011 |

The one new thing is `statistics`, a standard library module for means,
medians and friends. It's introduced in step 4.

Try this before reading the code. Seriously: read the steps below, open
`orders_fortnight.csv`, and have a go at writing your own version first.
Then compare. The code in `lesson.py` is *one* good way, not the only way.

## How to follow along

```bash
python lessons/phase-0-python/lesson-012-mini-project-a-stdlib-data-report/lesson.py
```

The script writes two files next to itself: `output/report.txt` (the page
for the owner) and `output/rejected_rows.csv` (the rows a human needs to
look at).

## Step 1. Look before you trust

Before you write a single cleaning rule, look at the raw file. Not in a
spreadsheet, which tidies things up for you and hides the very problems
you're looking for, but as text:

```
     1 | Order ID, Date,Time,Drink ,Size,Price,Qty,Payment
     2 | A0001,2026-09-07,09:24, latte,Medium,3.80,4,card
     3 | A0002,2026-09-07,07:39,tea,small,2.00,4,card
     4 | A0003,2026-09-07,10:52,flat white,Small,3.40,2,card
  133 lines in total, 2 of them blank.
```

Four lines already tell you a lot. The header has stray spaces
(`" Date"`, `"Drink "`) and calls the quantity `Qty`. Values have leading
spaces and random capitals. Prices here are for *different sizes*: small
is 50p less than medium, large is 50p more. Scroll through the whole file
and you'll find `£` signs, `capp`, `flat-white`, `CASH`, and a few rows
that are just plain wrong.

`peek()` in `lesson.py` prints the first few lines with `enumerate` and a
`break`, lesson 011 style, so it never reads more than it shows. That
matters when the file is big. Make it a reflex: **look at the first lines
of every new file, as text, before anything else.**

## Step 2. Normalise what can be fixed

Some mess is just formatting. It changes how a value *looks*, not what it
*means*, and fixing it silently is the right thing to do. `normalise()`
does only that:

```python
COLUMNS = {"order id": "order_id", "date": "date", "time": "time", "drink": "drink",
           "size": "size", "price": "price", "qty": "quantity", "payment": "payment"}
DRINK_ALIASES = {"capp": "cappuccino", "flat-white": "flat white", "flatwhite": "flat white"}

def normalise(raw):
    row = {}
    for header, value in raw.items():
        if header is None:
            continue
        name = COLUMNS.get(header.strip().lower())
        if name:
            row[name] = (value or "").strip()
    row["drink"] = DRINK_ALIASES.get(row["drink"].lower(), row["drink"].lower())
    ...
```

Two ideas worth stealing:

- **A mapping for header names.** Rather than hoping the till never
  renames a column, `COLUMNS` translates whatever header it finds
  (stripped and lower-cased) into *our* names. When next month's export
  says `Quantity` instead of `Qty`, you add one entry to a dict, not an
  `if` somewhere in the middle of a function.
- **A mapping for aliases.** `DRINK_ALIASES.get(drink, drink)` means
  "translate it if we know a translation, otherwise leave it alone".
  Anything left over that still isn't on the menu gets rejected in step 3,
  so an unknown alias is caught, not guessed at.

`normalise` converts nothing and rejects nothing. It returns strings.
Keeping "tidy the text" separate from "judge the data" means each
function is short and each has one reason to change.

## Step 3. Validate into Orders, and record the rejects

The Order is lesson 010's dataclass with two new fields:

```python
@dataclass(frozen=True)
class Order:
    order_id: str
    day: date
    at: time
    drink: str
    size: str
    price_cents: int
    quantity: int
    payment: str

    def total_cents(self):
        return self.price_cents * self.quantity
```

`to_order(row)` is lesson 008's `validate_row`, grown up: each column is
converted in its own `try`, each rule is an `if`, and each failure raises
a `ValueError` that names the column and quotes the value. Two new rules
show how validation grows with the data:

- **The time must be a real time.** `time.fromisoformat("25:10")` refuses,
  and we say so.
- **The price must match the menu.** Given the drink and size, we know
  what the price *should* be. A mismatch means someone typed it wrong or
  the menu changed, and either way a human should know. (No row in this
  file breaks this rule, which is exactly what you hope. A rule that never
  fires is still earning its keep.)

The loader adds one thing we haven't done before, **duplicates**:

```python
if order.order_id in seen:
    raise ValueError(f"duplicate of {order.order_id} on line {seen[order.order_id]}")
```

Every order has an ID, and IDs are unique, so seeing one twice means the
till exported a row twice. Keeping both would count that money twice. A
dict from ID to line number catches it and says where the original is.

It also uses `reader.line_num` instead of `enumerate(..., start=2)`.
`DictReader` quietly skips blank lines, which throws a hand-counted line
number off; `line_num` is the true line in the file, so "line 72" is where
your editor will take you.

```
  123 good orders, 7 rejected:
    line  13  date '10/09/2026' is not YYYY-MM-DD
    line  44  duplicate of A0040 on line 43
    line  59  drink 'mocha' is not on the menu
    line  72  missing price
    line 107  time '25:10' is not HH:MM
    line 110  quantity 0 must be at least 1
    line 118  quantity 'two' is not a whole number
```

130 rows in, 123 kept, 7 rejected, and every one of the seven has a
reason a human can act on without reading code.

## Step 4. Summary statistics

Here's the new module. `statistics` does the arithmetic you'd otherwise
write by hand, and does it carefully:

```python
import statistics

spend = [order.total_cents() for order in orders]
statistics.mean(spend)          # the average
statistics.median(spend)        # the middle value when sorted
statistics.stdev(spend)         # how spread out the values are
statistics.quantiles(spend, n=4)    # the 25%, 50% and 75% marks
```

```
  Spend per order:
    count        123
    mean       £5.54
    median     £3.90
    stdev      £3.02
    min        £2.00
    max       £15.20
  Quartiles (25%, 50%, 75%): £3.40, £3.90, £7.50
```

What these mean in plain words:

- **Mean** is "total divided by count": £5.54 per order.
- **Median** is the middle order if you lined them all up by size. Half of
  orders were £3.90 or less.
- The mean is well above the median, and that's a story: most people buy
  one drink, but a few buy several at once, and those big orders drag the
  average up. If the owner asked "what does a *typical* customer spend?",
  the median is the more honest answer. Lesson 023 spends a whole lesson
  on this.
- **Standard deviation** is roughly "how far, on average, orders are from
  the mean". £3.02 is big next to £5.54, which says orders vary a lot.
- **Quartiles** split the orders into four equal-sized groups. A quarter
  of orders were under £3.40; a quarter were over £7.50.

`summarise()` wraps these in one dict. Note the guard on `stdev`: it
needs at least two values, and would raise `StatisticsError` otherwise.
An honest function says what it does with the edge case.

## Step 5. Break it down

Totals by drink, by day, by payment method, and cups by hour, all in one
pass over the orders with `defaultdict(int)` and `Counter`, lesson 005's
tiny databases:

```python
for order in orders:
    by_drink[order.drink] += order.total_cents()
    by_day[order.day] += order.total_cents()
    by_payment[order.payment] += order.total_cents()
    cups_by_hour[order.at.hour] += order.quantity
```

Because `order.day` is a real `date` and `order.at` a real `time`, the
grouping is free: `order.at.hour` is the hour as a number, and dates sort
in date order and format as `Monday 07 September` with
`f"{day:%A %d %B}"`. That's the payoff of converting types at the door:
everything downstream gets easier.

## Step 6. The report

`build_report` returns the whole report as one string. It doesn't print
and it doesn't write. `step_report` does both with the string it gets
back. That split means the same text can go to the screen, to a file, or
(one day) into an email, and you can test the report by looking at a
string.

```
COFFEE SHOP REPORT
Mon 07 Sep to Fri 18 Sep 2026
============================================
Revenue               £681.50
Orders                    123
Cups                      201
Average order           £5.54   (median £3.90)
Rows rejected               7   (see rejected_rows.csv)

Revenue by drink
  latte         £265.80  ########################
  cappuccino    £163.00  ###############
  flat white    £131.00  ############
  tea            £74.00  #######
  espresso       £47.70  ####
...
Cups by hour
  07:00    30  ##############
  08:00    51  ########################
  09:00    31  ###############
...
```

The bars are the only "chart" in Phase 0: `"#" * round(24 * value /
biggest)`. Crude, and remarkably effective. You can see the 8am rush at a
glance, and that the shop sells more lattes than tea and espresso put
together. Phase 1 gives you real charts; the idea (length in proportion
to value, labelled, sorted so the eye knows where to go) is the same.

The report also says how many rows were rejected, and where to find them.
A report that silently drops rows is a report nobody can trust. One that
says "7 rows need a look" is one the owner can.

## What you've built

Look at the shape of `main()`:

```python
step_look(args.csv)
step_normalise(args.csv)
orders, rejects = step_validate(args.csv)
stats = step_statistics(orders)
parts = step_breakdowns(orders)
step_report(orders, rejects, stats, parts)
```

Look, tidy, validate, summarise, break down, report. That's the skeleton
of almost every data analysis you'll ever do, whatever the tools. In
Phase 1, pandas will shrink several of these steps to a line or two, but
it won't change the order, and it won't do step 1 for you.

And every piece was plain Python: no installs, nothing magic. When pandas
does something in one line next week, you'll know roughly what it's doing
underneath, because you've written it yourself.

## What to do now

1. Run `lesson.py`, open `output/report.txt` and `output/rejected_rows.csv`,
   and check one number by hand (the Payment lines are easiest).
2. Do the exercises in [`exercises.md`](exercises.md). They extend the
   report the way a real owner would ask you to.
3. That's Phase 0 done. Lesson 013 (setting up a real environment) starts
   Phase 1 on Monday. See [PROGRESS.md](../../../curriculum/PROGRESS.md).

Two weeks ago, `print("hello")` was the plan. Today you turned a messy
file into a page someone could make decisions from. Have a good weekend.
You've earned it.
