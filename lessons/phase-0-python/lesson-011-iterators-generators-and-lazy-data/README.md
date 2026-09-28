# Lesson 011 - Iterators, generators and lazy data

**Phase 0 - Python foundations for data work** | Week 2, Day 5 | Friday 2026-09-18

> **Goal:** process data you can't (or don't want to) hold in memory, using
> generators and `itertools`, so that a file of a quarter of a million rows
> is no harder to handle than one of fifteen.

Time: about 45 minutes. No installs. Python 3.10+.

---

Every loader we've written so far does the same thing: read every row into
a list, then work on the list. For a week of coffee orders, fifteen rows,
that's perfect. For a year of orders from a busy shop, it's a quarter of a
million dicts sitting in memory at once, just so you can add up one column.
For the log files and datasets you'll meet later, it's the difference
between a script that finishes and one that makes your laptop fan scream
and then dies.

There's another way: ask for **one row at a time**, deal with it, and let
it go. Python is built around this idea, and you've been using it without
knowing. Every `for` loop you've written works this way underneath. Today
we open the lid, write our own one-at-a-time functions with a new keyword,
`yield`, and meet the small part of `itertools` that earns its keep.

By the end, `lesson.py` makes a made-up year of till data (250,000 rows,
about 8 MB) and totals it by drink using well under 1 MB of memory. Reading
it all into a list first takes about 90.

## How to follow along

Terminal in the repo root, `python`, type along at `>>>`. Full script:

```bash
python lessons/phase-0-python/lesson-011-iterators-generators-and-lazy-data/lesson.py
```

The script writes one file, `output/till_year.csv`, the first time it runs.
It takes a few seconds; after that it's reused. Delete it whenever you like.

## 1. What a `for` loop really does

```python
drinks = ["latte", "tea", "espresso"]
it = iter(drinks)
next(it)       # 'latte'
next(it)       # 'tea'
next(it)       # 'espresso'
next(it)       # StopIteration
```

`iter()` asks something for an **iterator**: an object whose only job is
to hand out the next item each time you call `next()` on it, and to raise
`StopIteration` when there's nothing left. That's the whole protocol.

And that's all a `for` loop is:

```python
for drink in drinks:
    print(drink)
```

means "call `iter(drinks)` once, then call `next()` on it over and over,
running the body each time, and stop quietly at `StopIteration`". Anything
you can call `iter()` on is an **iterable**: lists, tuples, strings
(letter by letter), dicts (key by key), files (line by line), `range`,
`csv.DictReader`, and everything we write today.

## 2. Iterators get used up

Here's a bug that has cost everyone an afternoon at least once:

```python
with open(ORDERS_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    first_pass = sum(1 for _ in reader)     # 15
    second_pass = sum(1 for _ in reader)    # 0
```

A list is an iterable you can loop over as often as you like; each `for`
gets a fresh iterator. A `DictReader`, a file, and a generator are
*themselves* iterators. They hand out each item exactly once, then they're
empty. The second loop doesn't fail. It just finds nothing, and your total
comes out as zero.

Even `2 in it` uses an iterator up as it searches. So the rule: **if you
need the data twice, either make a list (`rows = list(reader)`) or read the
file again.** Which one depends on whether it fits comfortably in memory,
and that's the choice the rest of this lesson is about.

## 3. Generator functions: `yield`

Put `yield` in a function and it becomes a **generator function**.
Calling it doesn't run it. It gives you a generator (an iterator) that runs
the body bit by bit, pausing at each `yield`:

```python
def countdown(n):
    print(f"(countdown starting from {n})")
    while n > 0:
        yield n
        n -= 1
    print("(countdown finished)")

gen = countdown(3)      # nothing printed at all
next(gen)               # prints "(countdown starting from 3)", returns 3
next(gen)               # 2
next(gen)               # 1
next(gen, "done")       # prints "(countdown finished)", returns 'done'
```

`yield n` means "hand `n` to whoever asked, then freeze right here, local
variables and all, until they ask again". `return` ends a function for
good; `yield` pauses it. When the body finishes, the generator raises
`StopIteration` and any `for` loop over it ends cleanly. (`next(gen,
"done")` returns a default instead of raising, which is handy when you're
poking at one by hand.)

Here's the one that matters:

```python
def read_orders(path):
    """Yield one clean order dict at a time. Never holds more than one row."""
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            yield {
                "date": raw["date"],
                "drink": raw["drink"],
                ...
                "quantity": int(raw["quantity"]),
            }
```

It looks almost like lesson 007's loader, but where that one did
`rows.append(...)` and `return rows` at the end, this one `yield`s each row
as soon as it's ready. The caller gets rows as a stream, one in memory at a
time, and the file stays open only while someone is still asking. Use it
exactly like a list in a `for` loop, or `sum(...)` over it.

## 4. Generator expressions vs list comprehensions

You know list comprehensions from lesson 004. Swap the square brackets for
round ones and you get a **generator expression**:

```python
as_list = [n * n for n in range(1_000_000)]    # 8,448,728 bytes, built now
as_gen = (n * n for n in range(1_000_000))     # 200 bytes, built on request
sum(as_list) == sum(as_gen)                    # True
```

Same answer. The list computed a million numbers and kept them all. The
generator expression kept a recipe and computed each number only when
`sum` asked for the next one.

When a generator expression is the only argument to a function, you can
drop its brackets: `sum(q for q in quantities)`, `any(d == "mocha" for d
in MENU)`, `max(len(d) for d in MENU)`. Use a list when you need to keep
the results, loop twice, index, or take a `len()`. Use a generator when you
only need to pass through once, which, for totals and counts, is most of
the time.

Several things you already use are lazy in the same way: `range` (so
`range(10**12)` is instant), `zip`, `enumerate`, `dict.items()`.

## 5. Pipelines of small lazy steps

Generators can take other generators as input, which lets you build a
pipeline out of tiny, single-purpose steps:

```python
def only_drink(rows, drink):
    for row in rows:
        if row["drink"] == drink:
            yield row

def with_total(rows):
    for row in rows:
        yield {**row, "total_cents": row["price_cents"] * row["quantity"]}

rows = read_orders(ORDERS_CSV)
priced_lattes = with_total(only_drink(rows, "latte"))
for row in priced_lattes:
    ...
```

Three lines of setup, and *nothing has happened yet*. When the `for` loop
asks for its first item, `with_total` asks `only_drink`, which asks
`read_orders`, which reads a line from the file. Each row travels the whole
pipeline before the next is read. That's lesson 006's "small functions,
one job each" carried over to data that flows: each step is easy to read
and to test on a tiny list, and the whole chain uses constant memory
however long the file is.

## 6. The `itertools` you'll actually use

The `itertools` module is full of tools for iterators. Five come up again
and again:

| Tool | Does | Example |
|------|------|---------|
| `islice(it, n)` | The first `n` items, lazily. Slicing for iterators. | Peek at the first two rows of a huge file. |
| `chain(a, b, ...)` | One stream from several, end to end. | Loop over four weekly files as if they were one. |
| `accumulate(it)` | Running totals. | `[2, 3, 4, 5, 8, ...]`: cups sold so far. |
| `groupby(it, key=...)` | Runs of neighbouring items with the same key. | Revenue per day, from a file sorted by date. |
| `count(start)` | 1, 2, 3, ... forever. | Numbering things, alongside `zip`. |

`groupby` needs a warning label. It groups **neighbours** only:
`groupby(["tea", "latte", "tea"])` gives three groups, not two. It's
perfect for a file already sorted by the key (till exports are usually in
date order) and wrong for anything else. When the data isn't sorted, use a
dict (lesson 005). It's also why `lesson.py` sums each group's rows
straight away: each group is itself a little iterator, used up the moment
`groupby` moves to the next one.

`count()` never ends, and that's fine: `zip` stops at the shortest input,
and `islice` stops when it's taken enough. Infinite iterators are safe as
long as something downstream knows when to stop.

## 7. Putting it together

`lesson.py` first *writes* the big file, with a generator:

```python
writer.writerows(fake_orders(BIG_ROWS))     # writerows takes any iterable
```

`fake_orders` yields 250,000 made-up orders (seeded, so everyone gets the
same numbers), and `DictWriter` writes each one as it arrives. The data is
never all in memory, even while it's being made.

Then it totals revenue by drink twice, once streaming through
`read_orders` and once with `list(read_orders(...))` first, and measures
the peak memory of each with the standard library's `tracemalloc`:

```
  output/till_year.csv: 8.4 MB, 250,000 rows
    flat white    £322,408.40
    latte         £318,958.10
    ...
    total        £1,342,450.20

  Peak memory, streaming with a generator:     0.05 MB
  Peak memory, list(...) of every row first:  90.34 MB
```

(Your memory numbers will differ a little. The gap won't.) Same totals, and
the streaming version needs a tiny fraction of the memory. More to the
point, its memory doesn't grow with the file. Make it ten times bigger and
the list version needs close to a gigabyte, while the streaming one still
needs almost nothing.

Notice also that an 8 MB file became 90 MB in a list. Python objects carry
overhead, and a dict per row is generous. That's one reason Phase 1's
NumPy and pandas exist: they store columns of numbers compactly. Even
then, "stream it" remains the answer when data is genuinely bigger than
memory.

## What you can do now

- Explain what `iter()`, `next()` and `StopIteration` do, and how a `for`
  loop uses them.
- Tell an iterable (loop as often as you like) from an iterator (used up
  after one pass), and avoid the empty-second-loop bug.
- Write a generator function with `yield`, and say how it differs from
  `return`.
- Choose between a list comprehension and a generator expression.
- Chain small generator steps into a pipeline that streams a file.
- Use `islice`, `chain`, `accumulate`, `groupby` (on sorted data) and
  `count`.

## What to do now

1. Run `lesson.py`. Then change `BIG_ROWS` to `1_000_000`, delete
   `output/till_year.csv`, and run it again. Watch which memory number
   moves.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   streams several weekly files as one.
3. Lesson 012, the Phase 0 mini-project, is this afternoon. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Lists are for data you want to keep. Generators are for data that's
passing through. Most data is passing through.
