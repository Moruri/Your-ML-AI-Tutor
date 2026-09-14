# Lesson 004 - Lists and tuples, properly

**Phase 0 - Python foundations for data work** | Week 2, Day 1 | Monday 2026-09-14

> **Goal:** slice, sort, unpack and comprehend lists without hesitating, and
> know when a tuple is the better fit.

Time: about 45 minutes. No installs. Python 3.10+.

---

On Friday you met lists: square brackets, `len`, `append`, index from zero.
Enough to get by. Today we go properly through the list toolkit, because
lists are the thing you'll hold data in until pandas turns up, and even after
that, half of what pandas does is "list operations, but faster".

We'll also meet the list's quieter sibling, the **tuple**, and talk about
when each one is the right choice. That's a small decision you'll make many
times a day, so it's worth making on purpose.

No data file today. The examples use a week of coffee orders typed straight
into the script, with prices in cents (you did lesson 003 an hour ago; let's
not undo it).

## How to follow along

Terminal in the repo root, `python`, type along at `>>>`. Full script:

```bash
python lessons/phase-0-python/lesson-004-lists-and-tuples-properly/lesson.py
```

## 1. Indexing and slicing

You know `prices[0]` and `prices[-1]`. Slicing gets you a *range* of items:

```python
prices = [380, 220, 420, 430, 390, 220, 370]
prices[0]        # 380          first
prices[-1]       # 370          last
prices[1:4]      # [220, 420, 430]   from index 1 up to (not including) 4
prices[:3]       # [380, 220, 420]   first three
prices[3:]       # [430, 390, 220, 370]   from index 3 to the end
prices[-2:]      # [220, 370]   last two
prices[::2]      # [380, 420, 390, 370]   every second item
prices[::-1]     # [370, 220, 390, 430, 420, 220, 380]   reversed
```

The shape is `list[start:stop:step]`. Leave any of them out and Python fills
in the obvious default. The one that trips everyone up: `stop` is *not
included*. `prices[1:4]` gives you indexes 1, 2 and 3. A useful way to hold
it in your head: `prices[:3]` is "the first 3", and `prices[3:]` is
"everything after the first 3", and together they make the whole list.

Two more things worth knowing:

- **Slices never crash.** `prices[5:100]` just gives you what's there. An
  index out of range (`prices[100]`) is an `IndexError`. Slices are forgiving;
  single indexes are not.
- **A slice is a new list.** Changing it doesn't change the original. This
  matters in a second.

## 2. Two names, one list

This one bites people who've been programming for years, so read it twice.

```python
a = [380, 220, 420]
b = a
b.append(999)
print(a)         # [380, 220, 420, 999]   <- a changed too!
```

`b = a` does not make a copy. It sticks a second name on the *same list*.
Change it through either name and both see it, because there's only one list.

If you want a separate list you can change independently, say so:

```python
b = a.copy()     # or: b = list(a)   or: b = a[:]
b.append(999)
print(a)         # [380, 220, 420]   <- untouched
```

Where this really gets you is functions. If you pass a list into a function
and the function `.append()`s to it or `.sort()`s it, the caller's list is
changed. Sometimes that's what you want. Usually it's a surprise. Rule of
thumb for now: **a function should either change the list it's given, or
return a new one, and its name should make it obvious which.**
`add_totals(rows)` sounds like it changes `rows`. `with_totals(rows)` sounds
like it hands you a new thing. Pick names that tell the truth.

## 3. Changing a list

The methods you'll actually use:

```python
queue = ["latte", "tea"]
queue.append("espresso")           # ['latte', 'tea', 'espresso']       one item on the end
queue.extend(["mocha", "tea"])     # [..., 'espresso', 'mocha', 'tea']   several items on the end
queue.insert(0, "flat white")      # ['flat white', 'latte', ...]        at a position (slow for big lists)
served = queue.pop(0)              # removes AND returns 'flat white'
last = queue.pop()                 # removes and returns the last item, 'tea'
queue.remove("tea")                # removes the FIRST 'tea' it finds
"mocha" in queue                   # True
queue.index("mocha")               # where it is
```

The classic mistake is `append` where you meant `extend`:

```python
queue.append(["mocha", "tea"])     # [..., ['mocha', 'tea']]   <- a list inside the list
```

`append` adds *one thing*. If the thing is a list, you get a nested list. Use
`extend` (or `queue += [...]`) to add the items themselves.

Notice too that `append`, `extend`, `insert` and `remove` all return `None`.
They change the list *in place* and give you nothing back. Writing
`queue = queue.append("x")` sets `queue` to `None` and ruins your afternoon.
`pop` is the exception: it removes *and* returns, which is why you assign it.

## 4. Sorting: `sorted` vs `.sort`, and `key=`

There are two ways to sort, and the difference is exactly section 2's
lesson:

```python
prices = [380, 220, 420]
ranked = sorted(prices)     # NEW list [220, 380, 420]; prices unchanged
prices.sort()               # prices itself is now [220, 380, 420]; returns None
```

`sorted()` works on anything you can loop over and always hands back a fresh
list. `.sort()` only exists on lists and changes them in place. When in
doubt, use `sorted()`. It's harder to shoot yourself with.

Both take `reverse=True`, and both take `key=`, which is the important one.
`key` is a function that says *what to sort by*. You met it on Friday with
`lambda item: item[1]`. Here it is on records:

```python
orders = [("latte", 380, 2), ("tea", 250, 1), ("cappuccino", 420, 1)]

sorted(orders, key=lambda o: o[1])                 # cheapest first
sorted(orders, key=lambda o: o[1] * o[2], reverse=True)   # biggest line total first
sorted(orders, key=lambda o: o[0])                 # alphabetical by name
```

`lambda o: o[1]` is a tiny unnamed function: "given an order `o`, give back
its second field". If a `key` gets longer than about that, give it a proper
name with `def` instead.

Two more things you'll use a lot:

- **Sort by several things.** Return a tuple from the key: `key=lambda o:
  (o[0], -o[1])` sorts by name, then by price *descending* within each name
  (the minus sign flips the order for numbers). Tuples compare element by
  element, first one wins, so this just works.
- **`min` and `max` take `key=` too.** `max(orders, key=lambda o: o[1] *
  o[2])` gives you the single biggest order without sorting everything.

Python's sort is **stable**: items that tie keep their original order. So
sorting by price then by drink (two separate sorts) does what you'd expect.

## 5. Unpacking: several names at once

You've been doing this since Friday without a name for it:

```python
for drink, total in sales.items():
```

That's **unpacking**: a two-item thing on the right, two names on the left,
one per item. It works anywhere:

```python
order = ("latte", 380, 2)
drink, price, qty = order          # three fields, three names
a, b = 1, 2
a, b = b, a                        # swap. No temporary variable.
first, *rest = [380, 220, 420]     # first=380, rest=[220, 420]
*most, last = [380, 220, 420]      # most=[380, 220], last=420
```

The `*rest` form is "everything left over, as a list". Handy for "the
header row, then the data rows".

Unpacking is what makes two built-in helpers pleasant. `enumerate` gives you
the position alongside each item; `zip` walks two lists in step:

```python
for i, drink in enumerate(["latte", "tea", "mocha"], start=1):
    print(f"{i}. {drink}")            # 1. latte / 2. tea / 3. mocha

drinks = ["latte", "tea"]
cups = [7, 12]
for drink, n in zip(drinks, cups):
    print(f"{drink}: {n} cups")
```

If you ever catch yourself writing `for i in range(len(things))` just to get
at `things[i]`, you want `enumerate`. If you're indexing two lists with the
same `i`, you want `zip`.

## 6. List comprehensions, kept readable

On Friday, filtering looked like this:

```python
big = []
for p in prices:
    if p > 400:
        big.append(p)
```

A list comprehension says the same thing in one line, in roughly the order
you'd say it out loud:

```python
big = [p for p in prices if p > 400]
```

Read it as: "a list of `p`, for each `p` in `prices`, if `p > 400`". The
three parts are: **what to keep** (`p`), **where from** (`for p in prices`),
**on what condition** (`if p > 400`, optional). You can transform as you go:

```python
pounds = [p / 100 for p in prices]                     # convert every item
names = [o[0].title() for o in orders]                 # pull a field from each record
totals = [price * qty for _, price, qty in orders]     # unpack inside the comprehension
```

That last one unpacks each order into three names and ignores the first (`_`
is the conventional name for "I don't need this").

Now the important part. Comprehensions are for *simple* transforms and
filters. If you need two `for`s, an `if`/`else` inside the expression, or
you find yourself squinting, **write the loop**. A four-line loop that a
tired colleague can read beats a one-liner that makes them stop and think.
The test is: can you say it out loud in one breath? If not, it's a loop.

One cousin worth knowing: `sum(price * qty for _, price, qty in orders)`.
Same shape, round brackets, no list built in between. Use it when you're
feeding the result straight into `sum`, `max`, `any` or `all`.

## 7. When a tuple is the better fit

A tuple looks like a list with round brackets:

```python
point = (3.5, 7.2)
order = ("2026-09-14", "latte", "medium", 380, 2)
```

You can index it, slice it, loop over it and unpack it exactly like a list.
The one difference: **you can't change it**. No `append`, no `order[3] =
400`. That sounds like a limitation. It's actually the point. Choose a tuple
when:

**The thing has a fixed shape.** An order is always date, drink, size,
price, quantity, in that order. A tuple says "this is a record with five
known fields", where a list says "this is a collection that might grow". Use
tuples for rows and lists for columns. When you see a list *of* tuples,
that's a table.

**You need a dict key.** Dict keys must be unchangeable, so lists can't be
keys but tuples can. This is how you group by two things at once:

```python
by_day_and_drink = {}
for date, drink, size, price, qty in orders:
    key = (date, drink)
    by_day_and_drink[key] = by_day_and_drink.get(key, 0) + price * qty
```

Friday's "running total per category" pattern, with a two-part category.
You'll do this constantly.

**A function wants to return more than one thing.**

```python
def cheapest_and_dearest(prices):
    return min(prices), max(prices)

lo, hi = cheapest_and_dearest(prices)
```

That `return a, b` is quietly a tuple, and the unpacking on the next line
takes it apart again. It's the standard Python way to hand back two values,
and it's why unpacking matters so much.

One trap: a tuple with a single item needs a trailing comma. `(380)` is just
the number 380 in brackets. `(380,)` is a one-item tuple. You'll hit this
about once and then never again.

And if you take one sentence away from this section: **a list is for many
of the same kind of thing; a tuple is for one thing with several parts.**

## 8. Putting it together: a week of orders

Here's Friday's CSV again as a list of tuples, prices in cents:

```python
orders = [
    ("2026-09-07", "latte",      "medium", 380, 2),
    ("2026-09-07", "espresso",   "small",  220, 1),
    ("2026-09-07", "cappuccino", "large",  420, 1),
    # ... 15 rows in lesson.py
]
```

And the questions a shop owner asks, each answered with today's tools:

```python
# Line total for every order, unpacking inside the comprehension.
line_totals = [price * qty for _, _, _, price, qty in orders]

# The three biggest orders. sorted() gives a new list; slice the top.
top3 = sorted(orders, key=lambda o: o[3] * o[4], reverse=True)[:3]
for i, (date, drink, size, price, qty) in enumerate(top3, start=1):
    print(f"{i}. {date} {drink:<11} {size:<7} £{price * qty / 100:>6.2f}")

# Distinct drinks, in the order we first saw them.
seen = []
for _, drink, *_ in orders:
    if drink not in seen:
        seen.append(drink)

# Revenue by (date, drink), using a tuple as the key.
by_day_and_drink = {}
for date, drink, _, price, qty in orders:
    by_day_and_drink[(date, drink)] = by_day_and_drink.get((date, drink), 0) + price * qty
```

Look at how much of that is unpacking. `for i, (date, drink, ...) in
enumerate(...)` unpacks twice: the `(index, item)` pair from `enumerate`,
then the five fields inside the item. `for _, drink, *_ in orders` grabs just
the second field and throws the rest away. Once this reads naturally, most
Python data code reads naturally.

Run `lesson.py` to see the output, then change a question. Cheapest three
orders? Distinct sizes? Revenue by `(drink, size)`?

## What you can do now

- Slice any part of a list, forwards, backwards or every nth item.
- Explain why `b = a` doesn't copy, and make a real copy when you need one.
- Pick the right one of `append`/`extend`/`insert`/`pop`/`remove`.
- Sort by anything with `key=`, and choose `sorted()` over `.sort()` unless
  you mean to change the list.
- Unpack tuples, swap values, use `enumerate` and `zip`.
- Write a comprehension for a simple filter or transform, and *not* write one
  when it would be unreadable.
- Choose a tuple for records, dict keys and multiple return values.

## What to do now

1. Run `lesson.py` and read it next to this page.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   gives you a fresh list of orders and four questions.
3. That's Week 2, Day 1. Lessons 005 (dictionaries, properly) and 006
   (functions that don't lie) arrive tomorrow. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Good work. Two lessons in a day is a real amount of new material; if the
comprehension section felt fast, that's normal, and tomorrow uses them
gently.
