# Lesson 002 - Python warm-up for data people

**Phase 0 - Python foundations for data work** | Week 1, Day 1 | Friday 2026-09-11

> **Goal:** use variables, lists, dicts, loops and functions to read a small
> CSV with the standard library and answer a real question about it.

Time: about 40 minutes. No installs. Python 3.10+.

---

Lesson 001 was mostly talking. This one is mostly typing.

We're going to cover the handful of Python pieces you'll use *constantly* in
data work. Not all of Python. Just the bits that show up every single day:
storing things, putting things in lists, looking things up in dicts, looping
over them, and wrapping the logic in functions. Then we'll read a real (tiny)
CSV file and answer a question a shop owner might actually ask.

If you already know Python, skim the first half and go straight to the CSV
section. If you're new, go slowly and type things out rather than copying.
Typing is how it goes into your hands.

## How to follow along

Open a terminal in the repo root and start Python's interactive mode:

```bash
python
```

You'll see `>>>`. Type the examples below there, one at a time, and look at
what comes back. Type `exit()` when you're done. Every snippet below works
as-is.

The full script for this lesson is `lesson.py` in this folder. Run it any
time to see everything together:

```bash
python lessons/phase-0-python/lesson-002-python-warm-up-for-data-people/lesson.py
```

## 1. Variables: giving things names

A variable is a name stuck on a value so you can refer to it later.

```python
drink = "latte"
price = 3.80
quantity = 2
total = price * quantity
print(total)          # 7.6
```

Three kinds of value you'll see all the time:

- **Strings** (`str`) are text: `"latte"`. Always in quotes.
- **Integers** (`int`) are whole numbers: `2`.
- **Floats** (`float`) are numbers with a decimal point: `3.80`.

That last one matters more than it seems. Data files are text, so when we read
`"3.80"` from a CSV it arrives as a *string*. You can't multiply a string by
2 and get 7.6. You have to convert it first:

```python
price_text = "3.80"
price = float(price_text)     # now it's a number
print(price * 2)              # 7.6
```

Forgetting this is the single most common beginner data bug. You'll do it. I
still do it. Now you know what the error means when it happens.

To build a sentence out of values, use an f-string. Put `f` before the quote
and your variables in curly braces:

```python
print(f"{quantity} x {drink} = {total:.2f}")   # 2 x latte = 7.60
```

The `:.2f` means "show 2 decimal places". You'll use it constantly for money
and metrics.

## 2. Lists: many things, in order

A list holds a sequence of values. Square brackets, commas between.

```python
prices = [3.80, 2.20, 4.20, 4.30]
print(len(prices))        # 4
print(prices[0])          # 3.8   <- first item. Python counts from 0.
print(prices[-1])         # 4.3   <- last item
print(sum(prices))        # 14.5
print(max(prices))        # 4.3
prices.append(2.50)       # add one to the end
print(prices)             # [3.8, 2.2, 4.2, 4.3, 2.5]
```

A list of numbers is a column. A list of lists is a table. That's most of data
work right there, at least until we meet pandas in Phase 1.

## 3. Dicts: looking things up by name

A dict (dictionary) maps **keys** to **values**. Curly braces, `key: value`.

```python
order = {"drink": "latte", "size": "medium", "price": 3.80}
print(order["drink"])           # latte
order["quantity"] = 2           # add a new key
print(order)
```

Why do we care so much? Because *one row of a CSV is a dict*: column name to
value. A whole CSV is a list of dicts. Once that clicks, reading data files
stops being scary.

Dicts are also perfect for counting and totalling. Here's the pattern you'll
use a hundred times, "add to a running total per category":

```python
sales = {}
sales["latte"] = sales.get("latte", 0) + 7.60
sales["latte"] = sales.get("latte", 0) + 3.80
sales["tea"] = sales.get("tea", 0) + 2.50
print(sales)     # {'latte': 11.399999999999999, 'tea': 2.5}
```

`sales.get("latte", 0)` means "give me the value for `latte`, or `0` if it's
not there yet". That one method saves you from writing `if key in dict` every
time.

And yes, that `11.399999999999999` is real. Computers store decimals in
binary and can't represent 0.1 exactly, so tiny errors creep in when you add
floats. It's harmless here (the `:.2f` formatting hides it), but it's why
you'll see money handled with care in later lessons. Lesson 003 is all about
this sort of thing.

## 4. Loops: doing something to every item

`for` walks through a list one item at a time:

```python
prices = [3.80, 2.20, 4.20]
for p in prices:
    print(f"{p:.2f}")
```

Note the four spaces of indentation. In Python, indentation *is* the
structure. Everything indented under the `for` line runs once per item.

Looping over a dict gives you keys and values together with `.items()`:

```python
sales = {"latte": 11.4, "tea": 2.5}
for drink, total in sales.items():
    print(f"{drink}: {total:.2f}")
```

And a very common data pattern, filter while looping:

```python
big_orders = []
for p in [3.80, 2.20, 4.20, 4.30]:
    if p > 4:
        big_orders.append(p)
print(big_orders)     # [4.2, 4.3]
```

## 5. Functions: naming a piece of logic

A function is a chunk of code with a name and, usually, some inputs and an
output.

```python
def line_total(price, quantity):
    return price * quantity

print(line_total(3.80, 2))    # 7.6
```

Why bother, when you could just write `price * quantity`? Three reasons that
matter for data work:

1. **You'll do the same thing many times.** Write it once, call it often.
2. **Names explain intent.** `line_total(p, q)` reads better than `p * q`
   buried in a loop.
3. **You can test it on its own.** When something's wrong in a 200-line
   script, small functions let you check each piece.

Rule of thumb: if you find yourself copy-pasting three lines, make a function.

## 6. Reading a CSV with the standard library

Here's the payoff. In this folder there's a file, `coffee_orders.csv`, with a
week of orders from a small cafe. Open it in a text editor. It looks like
this:

```
date,drink,size,price,quantity
2026-09-07,latte,medium,3.80,2
2026-09-07,espresso,small,2.20,1
...
```

CSV means "comma-separated values". First line is the column names (the
**header**), each following line is one row. That's it. It's the most common
data format in the world because it's so boring.

Python ships with a `csv` module. Its `DictReader` does exactly what the name
says: reads each row as a dict, using the header for keys.

```python
import csv

with open("lessons/phase-0-python/lesson-002-python-warm-up-for-data-people/coffee_orders.csv", newline="") as f:
    rows = list(csv.DictReader(f))

print(len(rows))        # 15
print(rows[0])          # {'date': '2026-09-07', 'drink': 'latte', 'size': 'medium', 'price': '3.80', 'quantity': '2'}
```

Three things to notice:

- `with open(...) as f:` opens the file and *closes it for you* when the
  indented block ends. Always use it.
- `newline=""` is a small incantation the `csv` module asks for. It avoids
  blank-line problems on Windows. Just always include it.
- **Every value is a string.** `'3.80'`, `'2'`. Remember section 1. Before you
  do maths, convert: `float(row["price"])`, `int(row["quantity"])`.

Now let's answer the question a cafe owner would actually ask: **which drink
made the most money this week?**

```python
import csv

def load_orders(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["price"] = float(row["price"])
        row["quantity"] = int(row["quantity"])
    return rows

def revenue_by_drink(rows):
    totals = {}
    for row in rows:
        money = row["price"] * row["quantity"]
        totals[row["drink"]] = totals.get(row["drink"], 0) + money
    return totals

orders = load_orders("lessons/phase-0-python/lesson-002-python-warm-up-for-data-people/coffee_orders.csv")
totals = revenue_by_drink(orders)

for drink, total in sorted(totals.items(), key=lambda item: item[1], reverse=True):
    print(f"{drink:<12} {total:>7.2f}")
```

Read that top to bottom. It's every idea from this lesson in one place:

- `load_orders` opens the file, reads rows as dicts, and converts the numeric
  columns. Notice the conversion happens *once*, right after loading. Do it
  there and the rest of your code never has to think about it.
- `revenue_by_drink` uses the "running total per category" dict pattern.
- The last loop sorts the dict by value. `sorted(..., key=lambda item:
  item[1], reverse=True)` is a mouthful, so let me unpack it: `totals.items()`
  gives `(drink, total)` pairs; `key=lambda item: item[1]` says "sort by the
  second thing in each pair"; `reverse=True` puts biggest first. You'll type
  this pattern so often it'll become automatic.

Run `lesson.py` to see the output. Then change the question. What about
revenue per *day*? Per *size*? Same shape, different key.

## What you can do now

- Store values, convert strings to numbers, print them nicely.
- Hold a column in a list and a row in a dict.
- Loop over both, and total or filter as you go.
- Wrap logic in a function so it has a name.
- Read any CSV file into a list of dicts and ask it a question.

Honestly, that's most of Phase 1's *thinking*. pandas will make it shorter and
faster, but the shape of the work (load, convert, group, total, sort) is what
you just did with your own hands.

## What to do now

1. Run `lesson.py` and read the output.
2. Do the exercises in [`exercises.md`](exercises.md). The main one asks you
   to answer a new question about the same CSV. Take your time.
3. That's the end of Week 1, Day 1. Lessons 003 and 004 arrive Monday. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Nice work today.
