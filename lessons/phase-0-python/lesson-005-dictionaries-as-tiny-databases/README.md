# Lesson 005 - Dictionaries as tiny databases

**Phase 0 - Python foundations for data work** | Week 2, Day 2 | Tuesday 2026-09-15

> **Goal:** group, count and look things up with dicts, `collections.Counter`
> and `defaultdict`, so that "how many of each?" and "what's the total per X?"
> become three-line habits instead of ten-line puzzles.

Time: about 45 minutes. No installs. Python 3.10+.

---

Morning. Yesterday you held a week of coffee orders as a list of tuples: a
table, one row per order. You also, almost in passing, wrote this twice:

```python
by_day_and_drink[key] = by_day_and_drink.get(key, 0) + price * qty
```

That line is a dict doing the job of a database: take a key, find the row for
it, update it. You've been using dicts since Friday without stopping to look
at them properly. Today we stop and look.

The title is not a metaphor I'm stretching for effect. A database table with
a primary key is, at heart, "give me the key, I'll give you the row". That is
exactly what a dict does, only it lives in memory and has no opinions about
disks. Most of the questions a shop owner asks about their data ("which drink
sells most?", "what did Tuesday make?", "which orders were Chloe's?") are a
dict plus a loop. By the end of today you'll be able to answer all of them
without thinking about the mechanics, which frees you up to think about the
question.

We'll use yesterday's `ORDERS` again, prices in cents.

## How to follow along

Terminal in the repo root, `python`, type along at `>>>`. Full script:

```bash
python lessons/phase-0-python/lesson-005-dictionaries-as-tiny-databases/lesson.py
```

## 1. A dict is a lookup table

Here's the menu board, medium size, in cents:

```python
MENU = {
    "espresso": 220,
    "tea": 250,
    "cappuccino": 370,
    "latte": 380,
    "flat white": 390,
}
```

Curly braces, `key: value` pairs, commas between. The keys here are drink
names; the values are prices. Look one up with square brackets:

```python
MENU["latte"]          # 380
len(MENU)              # 5
"tea" in MENU          # True     <- checks KEYS, not values
"mocha" in MENU        # False
MENU["mocha"]          # KeyError: 'mocha'
```

That last one matters. Asking for a key that isn't there is an **error**, not
`None`, not `0`. Python assumes that if you ask for `MENU["mocha"]` with no
fallback, you believed mocha was on the menu, and being wrong about that is
something you'd want to know. Section 2 is how to ask when you're *not* sure.

Two rules about what can be a key:

- **Keys must be unchangeable.** Strings, numbers and tuples work. Lists
  don't (`TypeError: unhashable type: 'list'`). That's why yesterday's
  two-part key was a tuple `(date, drink)` and not a list.
- **Keys are unique.** Assign to a key that already exists and you overwrite
  the old value. There is no "two rows for latte". If you need several things
  under one key, the *value* has to be a list (section 5).

Values can be anything at all: numbers, strings, lists, other dicts. That
flexibility is the whole reason dicts show up everywhere.

One more thing worth knowing: dicts remember the order you put things in.
Loop over `MENU` and you get espresso first, flat white last, every time.
(This has been guaranteed since Python 3.7. If a book tells you dicts are
unordered, it's an old book.)

## 2. `.get`: asking politely

```python
MENU.get("latte")                    # 380
MENU.get("mocha")                    # None        <- no error
MENU.get("mocha", 0)                 # 0           <- your choice of fallback
MENU.get("mocha", "not on the menu") # 'not on the menu'
```

`.get(key, default)` says "give me the value for this key, or `default` if
there isn't one". Leave the default out and it's `None`.

This is the engine behind the counting pattern you've been using since
Friday. Spelled out once, slowly:

```python
cups = {}
for _, drink, _, _, qty in ORDERS:
    cups[drink] = cups.get(drink, 0) + qty
```

First time we meet `"latte"`, `cups.get("latte", 0)` is `0`, we add the
quantity, and store it. Every later time, `.get` finds the running total and
we add to it. No `if drink in cups` needed.

So when do you use `[]` and when `.get`? A rule that has served me well:

- Use `MENU[key]` when a missing key is a **bug** and you want to hear about
  it loudly. Looking up a price for a drink that *must* be on the menu, say.
- Use `MENU.get(key, default)` when a missing key is **normal** and you know
  what to do instead. Counting, grouping, "use this if we've never seen it".

Reaching for `.get` everywhere feels safe but it's how errors go quiet: a
typo in a column name silently gives you `None` and your totals are wrong
with no traceback. Let `[]` shout when shouting is the right response.

## 3. Changing a dict, and looping over one

```python
menu = dict(MENU)              # a copy, so the original stays clean
menu["mocha"] = 400            # add a key (or overwrite it if it exists)
menu["tea"] = 260              # overwrite
removed = menu.pop("espresso") # remove AND return the value, 220
menu.pop("hot choc", None)     # remove if present; None instead of an error if not
menu.update({"hot chocolate": 350, "tea": 270})   # add/overwrite several at once
del menu["mocha"]              # remove; KeyError if it isn't there
```

Yesterday's lesson applies word for word: `menu = MENU` is two names for one
dict, and `dict(MENU)` (or `MENU.copy()`) is a real copy. If a function
changes a dict you passed it, you'll see the changes.

Looping over a dict gives you keys by default. Three variations:

```python
for drink in MENU:                    # keys
for cents in MENU.values():           # values
for drink, cents in MENU.items():     # both, unpacked. Use this one.
```

`.items()` with unpacking is what you'll write nine times out of ten. It
gives you `(key, value)` pairs, and pairs are what you want to sort,
print, or filter:

```python
sorted(MENU.items(), key=lambda item: item[1], reverse=True)
# [('flat white', 390), ('latte', 380), ('cappuccino', 370), ('tea', 250), ('espresso', 220)]

sorted(MENU)          # just the keys, alphabetical
```

You can't sort a dict in place; you sort its items into a list. If you want
a sorted dict, wrap it: `dict(sorted(MENU.items(), key=...))`.

And, because you learned list comprehensions yesterday, here's the dict
version. Same idea, curly braces, `key: value` before the `for`:

```python
{drink: cents / 100 for drink, cents in MENU.items()}    # every price in pounds
{d: c for d, c in MENU.items() if c < 300}               # the cheap drinks
{cents: drink for drink, cents in MENU.items()}          # flipped: price -> drink
```

That last one, flipping keys and values, is handy for reverse lookups. Just
remember that if two drinks had the same price, one would overwrite the
other, because keys are unique.

## 4. Counting: `Counter`

Take the drink from every order:

```python
drinks = [drink for _, drink, *_ in ORDERS]
```

You know how to count these by hand now (`counts.get(d, 0) + 1`). It's so
common that Python ships a dict that does it for you:

```python
from collections import Counter

orders_per_drink = Counter(drinks)
# Counter({'latte': 5, 'espresso': 3, 'cappuccino': 3, 'flat white': 2, 'tea': 2})

orders_per_drink.most_common(2)     # [('latte', 5), ('espresso', 3)]
orders_per_drink["latte"]           # 5
orders_per_drink["mocha"]           # 0    <- missing key is 0, not an error
orders_per_drink.total()            # 15   <- sum of all counts
```

A `Counter` *is* a dict (everything from sections 1 to 3 works on it), with
three niceties: give it anything you can loop over and it counts the items;
`.most_common(n)` hands you the top `n` as sorted pairs; and a missing key is
`0`. That last one is exactly what you want when counting, and exactly what
you *don't* want when looking up prices, so use `Counter` for counts and a
plain dict for everything else.

Counting *rows* is often not the question, though. Five latte orders doesn't
tell you how many lattes were poured. For that you add the quantities, and a
`Counter` is still the tidiest place to put them:

```python
cups = Counter()
for _, drink, _, _, qty in ORDERS:
    cups[drink] += qty
# Counter({'latte': 8, 'cappuccino': 5, 'espresso': 4, 'flat white': 4, 'tea': 2})
```

`cups[drink] += qty` works even the first time you see a drink, because a
missing key is `0`. Compare it with the `.get(drink, 0) + qty` version from
section 2 and notice you've stopped saying the same thing twice.

Counters also do arithmetic, which is more useful than it sounds:

```python
last_week = Counter({"latte": 6, "espresso": 5, "cappuccino": 4, "flat white": 5, "tea": 2})

cups - last_week      # Counter({'latte': 2, 'cappuccino': 1})       what grew
last_week - cups      # Counter({'espresso': 1, 'flat white': 1})    what shrank
cups + last_week      # both weeks combined
```

Subtraction keeps only positive results, so `this - last` is literally "what
went up" and `last - this` is "what went down". Two lines, and you've got a
week-on-week report.

## 5. Grouping: `defaultdict`

Counting is "one number per key". Grouping is "a *list* of things per key":
all the orders on each date, all the sizes each drink was sold in. By hand:

```python
by_date = {}
for order in ORDERS:
    date = order[0]
    if date not in by_date:
        by_date[date] = []          # first time: make an empty list
    by_date[date].append(order[1])  # every time: append to it
```

Three lines of "is it there yet? if not, make one" before you get to the
line you actually cared about. `defaultdict` removes them:

```python
from collections import defaultdict

by_date = defaultdict(list)
for date, drink, *_ in ORDERS:
    by_date[date].append(drink)

by_date["2026-09-08"]      # ['latte', 'flat white', 'espresso']
```

`defaultdict(list)` means "when I ask for a key that isn't there, call
`list()` to make an empty list, store it under that key, and hand it to me".
So `by_date[date].append(...)` just works on a brand new key. The thing in
the brackets is a *function that makes the default*, not the default itself:
`list`, not `[]`; `int`, not `0`.

`defaultdict(int)` is the same trick for running totals. `int()` is `0`, so:

```python
revenue = defaultdict(int)
for _, drink, _, price, qty in ORDERS:
    revenue[drink] += price * qty
```

You now have three ways to write a running total: `.get(k, 0) + x`,
`Counter`, and `defaultdict(int)`. They all work. `Counter` says "I'm
counting things" to whoever reads it next; `defaultdict(int)` says "I'm
summing things"; plain `.get` says nothing in particular. Pick the one that
tells the truth about what you're doing. (That sentence is most of lesson
006.)

**The trap.** A `defaultdict` creates a key the moment you *read* it:

```python
len(revenue)          # 5
revenue["mocha"]      # 0     <- looks like a harmless lookup...
len(revenue)          # 6     <- ...but mocha is now in the dict
```

You checked the revenue for a drink nobody ordered and it appeared in your
report with £0.00. This is the single most common `defaultdict` bug. Two
habits avoid it: use a `defaultdict` only while *building*, and once you're
done, convert it with `dict(revenue)` so later lookups behave like a normal
dict again. Or look up with `.get` / `in`, which don't trigger the default.

## 6. Nested dicts vs tuple keys

Yesterday you grouped by two things with a tuple key:

```python
flat = defaultdict(int)
for date, drink, _, price, qty in ORDERS:
    flat[(date, drink)] += price * qty

flat[("2026-09-08", "latte")]     # 430
```

The other shape is a dict *of* dicts. Date on the outside, drink on the
inside:

```python
nested = defaultdict(dict)
for date, drink, _, price, qty in ORDERS:
    nested[date][drink] = nested[date].get(drink, 0) + price * qty

nested["2026-09-08"]               # {'latte': 430, 'flat white': 1170, 'espresso': 440}
nested["2026-09-08"]["latte"]      # 430
sum(nested["2026-09-08"].values()) # 2040    <- Tuesday's total, for free
```

Same facts, two shapes. Which to choose depends on the questions you'll
ask:

- **Flat, tuple keys** is easier to build and trivial to sort ("top five
  date-drink combinations"). It's also what a spreadsheet or a database
  table looks like: one row per combination.
- **Nested** is better when you keep asking for "everything about one
  date": one lookup and you have that day's whole dict to total, sort, or
  print.

Neither is wrong. Pick the one that makes your most common question a
one-liner, and don't be afraid to build the other from it if a new question
comes along. Going from flat to nested is a four-line loop.

A word of caution about depth: two levels is fine, three is getting hard to
read, four means you want a different structure (a list of dicts, or, from
lesson 010, a small class). When you find yourself writing
`data[a][b][c][d]`, stop and reconsider.

## 7. Lookups as joins

Here's the "tiny database" idea paying off. `ORDERS` knows what was sold.
A second, separate dict knows what *kind* of drink each one is:

```python
CATEGORY = {
    "espresso": "black coffee",
    "latte": "milky coffee",
    "flat white": "milky coffee",
    "cappuccino": "milky coffee",
    "tea": "tea",
}
```

Neither table has "revenue by category" in it. Joining them is one `.get`
per row:

```python
cups_by_category = Counter()
revenue_by_category = defaultdict(int)
for _, drink, _, price, qty in ORDERS:
    category = CATEGORY.get(drink, "unknown")
    cups_by_category[category] += qty
    revenue_by_category[category] += price * qty
```

```
milky coffee    17 cups   £65.50
black coffee     4 cups    £8.80
tea              2 cups    £5.40
```

Seventeen of twenty-three cups need milk. That's the kind of fact the owner
actually wants, and it wasn't in either dict until you joined them. When we
get to pandas, this is `merge`; in SQL it's `JOIN`. The idea is the same:
one table's value is another table's key.

The `"unknown"` fallback is deliberate. If a new drink shows up that isn't
in `CATEGORY`, it lands in a bucket you can see, rather than crashing your
report or (worse) vanishing.

The same trick cleans data. Staff type `"flat-white"`, `"CAPP "`, and
`"Flatwhite"`. A dict of *the spellings you've seen* to *the spelling you
want* fixes them:

```python
ALIASES = {"flat-white": "flat white", "flatwhite": "flat white", "capp": "cappuccino"}

key = raw.strip().lower()          # lesson 003's normalise-first habit
clean = ALIASES.get(key, key)      # translate if known, else leave alone
```

`ALIASES.get(key, key)` reads as "look it up; if it's not there, it was
already fine". You'll write that one line more often than almost anything
else in this course. Every time you meet a new misspelling in a real file,
you add one entry to the dict rather than one `if` to the code.

## 8. Putting it together: five questions, five dicts

One pass over the orders, building four things at once:

```python
cups = Counter()
revenue = defaultdict(int)
by_date = defaultdict(list)
sizes_per_drink = defaultdict(Counter)      # a dict whose values are Counters
for date, drink, size, price, qty in ORDERS:
    cups[drink] += qty
    revenue[drink] += price * qty
    by_date[date].append((drink, price * qty))
    sizes_per_drink[drink][size] += qty
```

Then each question is a short loop over the right dict:

```python
# Most popular drinks
for drink, n in cups.most_common():
    print(f"{drink:<12} {n:>3}")

# Revenue per drink, biggest first
for drink, cents in sorted(revenue.items(), key=lambda item: item[1], reverse=True):
    print(f"{drink:<12} £{cents / 100:>6.2f}")

# Each day's total and its best-selling line
for date in sorted(by_date):
    day_total = sum(cents for _, cents in by_date[date])
    best_drink, best_cents = max(by_date[date], key=lambda item: item[1])
    print(f"{date}  £{day_total / 100:>6.2f}   best line: {best_drink}")

# Which sizes does each drink sell in?
for drink in sorted(sizes_per_drink):
    print(f"{drink:<12} {dict(sizes_per_drink[drink])}")
```

`defaultdict(Counter)` deserves a second look. Its values are Counters, so
`sizes_per_drink["latte"]["medium"] += 1` creates the inner Counter on
first sight of `"latte"` and the `"medium"` count on first sight of that.
Two levels of "make it if it's missing" in one line. That's the nested shape
from section 6, built without a single `if`.

Run `lesson.py` and read the output next to this. Then change a question.
Revenue by size? Average cups per order per drink? Which day sold the most
tea? Each one is a dict you already know how to build.

## What you can do now

- Build a dict, look things up with `[]`, and explain why a missing key is an
  error.
- Choose between `[]` and `.get(key, default)` on purpose.
- Add, overwrite, remove and update keys; loop with `.items()`; sort a dict
  by its values.
- Write a dict comprehension for a simple transform or filter.
- Count anything with `Counter`, get the top `n` with `.most_common`, and
  compare two Counters with `-`.
- Group into lists or running totals with `defaultdict`, and avoid the
  "reading creates a key" trap.
- Choose between tuple keys and nested dicts, and say why.
- Join two small tables with `.get`, and clean spellings with an alias dict.

## What to do now

1. Run `lesson.py` and read it next to this page.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one is a
   loyalty card log with the usual human spelling in it.
3. Then move on to
   [Lesson 006 - Functions that don't lie](../lesson-006-functions-that-dont-lie/README.md),
   where we take the loops you've been writing and give them proper names.
