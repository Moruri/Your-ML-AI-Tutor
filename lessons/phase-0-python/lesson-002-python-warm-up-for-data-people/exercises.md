# Lesson 002 - Exercises

The first three are quick checks. Exercise 4 is the real one: a small
hands-on task that uses everything from the lesson. Give it a proper go before
opening the answer.

Work in a new file so you don't lose the original. From the repo root:

```bash
cp lessons/phase-0-python/lesson-002-python-warm-up-for-data-people/lesson.py my_lesson_002.py
python my_lesson_002.py
```

(`my_lesson_002.py` is yours to mess up. Delete it when you're done, or keep
it; it's not part of the course.)

Because the copy is in a different folder, change the `CSV_PATH` line near
the top to point at the CSV directly:

```python
CSV_PATH = Path("lessons/phase-0-python/lesson-002-python-warm-up-for-data-people/coffee_orders.csv")
```

---

## 1. Strings vs numbers

Without running it, what does each line print? Then run them in `python` to
check.

```python
print("3" + "4")
print(3 + 4)
print("3" * 4)
print(float("3") + 4)
```

<details>
<summary>Check yourself</summary>

```
34          <- two strings glued together
7
3333        <- a string repeated 4 times
7.0         <- converted first, then added; float + int gives float
```

The first and third lines are exactly the kind of thing that goes wrong when
you forget CSV values are text.

</details>

## 2. The `.get()` pattern

Rewrite this so it doesn't need the `if`:

```python
counts = {}
for word in ["a", "b", "a", "c", "a"]:
    if word in counts:
        counts[word] = counts[word] + 1
    else:
        counts[word] = 1
print(counts)
```

<details>
<summary>Check yourself</summary>

```python
counts = {}
for word in ["a", "b", "a", "c", "a"]:
    counts[word] = counts.get(word, 0) + 1
print(counts)    # {'a': 3, 'b': 1, 'c': 1}
```

Same result, half the lines. (In lesson 005 you'll meet
`collections.Counter`, which makes this a one-liner.)

</details>

## 3. Read the sort

In the lesson we sorted with:

```python
sorted(totals.items(), key=lambda item: item[1], reverse=True)
```

1. What would change if you removed `reverse=True`?
2. What would change if you used `item[0]` instead of `item[1]`?

<details>
<summary>Check yourself</summary>

1. Smallest total first instead of biggest.
2. Sorted alphabetically by the *key* (the drink name) instead of by the
   total.

</details>

## 4. Hands-on: new questions, same CSV

The cafe owner has more questions. Answer each one by adding a few lines to
your copy of the script. The lesson's `load_orders`, `revenue_by`,
`line_total` and `print_ranked` functions are all there for you to use or
copy.

**a) Revenue by size.** Which size (small/medium/large) brings in the most
money?

**b) Cups sold.** How many cups were sold in total, across all rows? (Careful:
a row with `quantity` 3 is three cups.)

**c) Busiest day.** Which *date* sold the most cups? Note that's cups, not
money, so `revenue_by` won't do it directly. Write a similar function, or
generalise `revenue_by` so you can tell it what to add up.

**d) Average price per cup, per drink.** For each drink, total revenue divided
by total cups. Which drink has the highest average price?

Hints, if you want them:

- For (a) you should need exactly one new line.
- For (b) a plain `for` loop with a running total is fine. `sum(...)` with a
  generator is nicer but not required.
- For (c), the "running total per category" pattern works for *anything*
  you can add up, not just money. What if the thing you add is
  `row["quantity"]`?
- For (d) you need two dicts (revenue and cups) with the same keys, then a
  loop over one of them.

<details>
<summary>Expected results</summary>

```
a) by size      medium 44.50, large 19.80, small 15.40
b) cups sold    23
c) busiest day  2026-09-08 with 6 cups
d) avg per cup  cappuccino 4.00, flat white 3.90, latte 3.74, tea 2.70, espresso 2.20
```

</details>

<details>
<summary>One way to write it</summary>

```python
orders = load_orders(CSV_PATH)

# a) one line, same function, different column
print_ranked(revenue_by(orders, "size"), "by size")

# b) running total
cups = 0
for row in orders:
    cups += row["quantity"]
print(f"cups sold: {cups}")

# c) generalise the grouping: pass in what to add up
def total_by(rows, column, value_of):
    totals = {}
    for row in rows:
        totals[row[column]] = totals.get(row[column], 0) + value_of(row)
    return totals

cups_by_date = total_by(orders, "date", lambda row: row["quantity"])
print_ranked(cups_by_date, "cups by date")

# d) two dicts, one loop
revenue = revenue_by(orders, "drink")
cups_by_drink = total_by(orders, "drink", lambda row: row["quantity"])
avg = {}
for drink in revenue:
    avg[drink] = revenue[drink] / cups_by_drink[drink]
print_ranked(avg, "avg price per cup")
```

If yours looks different and gives the same numbers, yours is fine. There's
no style prize. (One wrinkle: `print_ranked` prints a `total` line at the
bottom, and for (d) that line is meaningless, since adding up averages tells
you nothing. Reusing a helper slightly outside its purpose is normal; just
know which numbers to ignore, or write a version without the total.) Notice how (c) turned `revenue_by` into something more
general by passing in a function. That trick, "pass in the behaviour", is
everywhere in Python data libraries, so it's worth staring at for a minute.

</details>

## 5. (Optional) Make it break, then make it clear

Open `coffee_orders.csv` and change one price to the word `free`. Run the
script. Read the error message from the *bottom* up. Which line is it
pointing at, and does the message make sense now that you know CSV values
are strings?

Put the file back the way it was afterwards (`git checkout` the file, or just
retype `3.80`).

<details>
<summary>Check yourself</summary>

You'll get a `ValueError: could not convert string to float: 'free'`,
pointing at the `float(row["price"])` line inside `load_orders`. That's
Python telling you exactly what went wrong and where. Lesson 008 is about
reading these calmly and deciding what to do about bad rows.

</details>

---

That's Day 1 done. Lessons 003 and 004 arrive Monday. Have a good weekend.
