# Lesson 006 - Functions that don't lie

**Phase 0 - Python foundations for data work** | Week 2, Day 2 | Tuesday 2026-09-15

> **Goal:** write small, named, single-purpose functions with default
> arguments and clear return values, so that a function's name is a promise
> the rest of your code can rely on.

Time: about 45 minutes. No installs. Python 3.10+.

---

You've been writing functions since Friday. `load_orders`, `parse_price_cents`,
`cheapest_and_dearest`, and whatever you called things in the exercises.
Nobody has told you yet what makes a function *good*, which is a bit like
teaching someone to drive and never mentioning mirrors.

Here's the whole lesson in one sentence: **a function should do what its name
says, only that, and hand back what it promises.** When a function does that,
you can read the name and move on. When it doesn't, you have to read the body,
and then the body of everything it calls, and now you're debugging at eleven
at night.

Yesterday's rule of thumb (`add_totals(rows)` sounds like it changes `rows`;
`with_totals(rows)` sounds like it gives you a new thing) was the first hint
of this. Today we make it a habit. Everything in this lesson is small. None
of it is hard. All of it is the difference between code you trust and code
you tiptoe around.

Same `ORDERS`, prices in cents.

## How to follow along

Terminal in the repo root, `python`, type along. Full script:

```bash
python lessons/phase-0-python/lesson-006-functions-that-dont-lie/lesson.py
```

## 1. Print is for people, return is for code

The single most common beginner confusion, so let's have it out now.

```python
def describe_order_badly(drink, cents):
    print(f"{drink}: £{cents / 100:.2f}")

label = describe_order_badly("latte", 380)     # prints "latte: £3.80"
print(label)                                    # None
```

The function *showed* you something on the screen. But it *handed back*
nothing, so `label` is `None`. You can't store, compare, sort, or reuse
something that went to the screen. It's gone.

```python
def describe_order(drink, cents):
    return f"{drink}: £{cents / 100:.2f}"

label = describe_order("latte", 380)    # 'latte: £3.80'
label.upper()                           # 'LATTE: £3.80'
labels = [describe_order(d, c) for d, c in MENU.items()]   # MENU from lesson 005
```

Now the caller decides what happens: print it, put it in a list, write it to
a file. **`print` is for a human looking at a screen. `return` is for the
next line of code.** A function that computes something should `return` it,
and one function, near the end, prints.

Related: a `def` with no `return` statement returns `None`. So does a bare
`return` with nothing after it. If you ever see `None` where you expected a
number, the first thing to check is whether you forgot to return.

## 2. One job, named honestly

```python
def line_total(order):
    """The value of one order line, in cents."""
    _, _, _, price, qty = order
    return price * qty

def revenue_by_drink(orders):
    """Total cents per drink, as a plain dict. Does not change orders."""
    totals = defaultdict(int)
    for order in orders:
        totals[order[1]] += line_total(order)
    return dict(totals)

def is_large(order):
    """True if the order is a large size."""
    return order[2] == "large"
```

Read the names as promises. `line_total(order)` gives you a number.
`revenue_by_drink(orders)` gives you a dict, drink to cents. `is_large(order)`
gives you `True` or `False`. You know all that without reading a body. That's
the goal.

Some naming habits that pay off every day:

- **Start with a verb, or make it a noun that says what comes back.**
  `parse_price_cents`, `load_orders`, `format_row` (verbs); `line_total`,
  `revenue_by_drink` (nouns that name the result).
- **`is_`/`has_` for yes/no questions.** `is_large`, `has_milk`. The reader
  knows they'll get a boolean.
- **Put the unit in the name if it could be ambiguous.** `parse_price_cents`
  saved you a bug in lesson 003 that `parse_price` would have hidden.
- **The "and" test.** If describing the function needs the word "and"
  ("it loads the file *and* totals it *and* prints"), it's two or three
  functions. Split it.

And the docstring: one line, in quotes, first thing inside the function.
Say what comes back and anything surprising. `revenue_by_drink`'s docstring
says "as a plain dict" and "does not change orders" because those are the
two things a caller might wonder. You don't need an essay. You need the
sentence a colleague would ask you for.

How small is small? A function should fit on your screen without scrolling,
and you should be able to say what it does in one breath. `line_total` is
three lines. That's not too small. Most good functions are shorter than
people expect.

## 3. Return values that don't surprise

A function should return **the same kind of thing every time**. Here's one
that doesn't:

```python
def parse_quantity_liar(text):
    cleaned = text.strip()
    if cleaned == "":
        return                    # None, by accident
    if not cleaned.isdigit():
        return "bad quantity"     # a string?!
    return int(cleaned)           # an int
```

Three inputs, three *types* of answer. The caller does
`total += parse_quantity_liar(x)` and it works on Monday's data and crashes
on Tuesday's when a `"bad quantity"` string turns up. The bug is in this
function, but the error appears somewhere else, later. That's what "lying"
costs you.

The honest version, from lesson 003's exercises:

```python
def parse_quantity(text):
    """Whole-number text -> int. Anything else -> None."""
    cleaned = text.strip()
    if not cleaned.isdigit():
        return None
    return int(cleaned)
```

Two possible answers, an `int` or `None`, both in the docstring. `None` is
the right thing to return for "there is no answer here"; it's honest, it's
easy to check (`if qty is None`), and it can't be mistaken for a real value.
Returning `0` or `-1` or `""` to mean "missing" is how missing values leak
into totals.

Two more habits that keep return values honest:

**Guard clauses.** Check for the bad case first and `return` early, so the
main path isn't buried inside an `else`. `parse_quantity` above does this:
one `if` that bails out, then the real work unindented. Functions read
better top to bottom when the exits come first.

**Several answers? Return a tuple.** You did this yesterday:

```python
def cheapest_and_dearest(orders):
    return min(orders, key=lambda o: o[3]), max(orders, key=lambda o: o[3])

lo, hi = cheapest_and_dearest(ORDERS)
```

Two or three related values as a tuple is idiomatic. Once you're up to four
or five, a dict (or a small class, lesson 010) with named fields is kinder
to the reader than `result[3]`.

## 4. Default arguments, and the one trap

Give a parameter a default and the caller can leave it out:

```python
def format_money(cents, symbol="£", decimals=2):
    return f"{symbol}{cents / 100:.{decimals}f}"

format_money(380)                    # '£3.80'
format_money(380, symbol="$")        # '$3.80'
format_money(380, decimals=0)        # '£4'
format_money(380, "€", 1)            # '€3.8'   positional works too, but read on
```

Defaults are how a function stays simple to call in the common case and
still flexible in the rare one. Put the parameters people always supply
first, and the tunable ones with sensible defaults after.

When you call, **use keywords for anything past the first argument or two.**
`top_n(revenue, n=1)` reads like a sentence. `top_n(revenue, 1)` makes the
reader go and look up what the second argument is. You've been doing this
already with `sorted(orders, key=..., reverse=True)`: `key` and `reverse`
are defaults you override by name. Now you know how to write functions that
work the same way.

**The trap.** Never use a list or dict as a default value:

```python
def add_cup_trap(drink, cups=[]):       # DON'T
    cups.append(drink)
    return cups

morning = add_cup_trap("latte")         # ['latte']
afternoon = add_cup_trap("tea")         # ['latte', 'tea']   <- what?
```

The `[]` is created **once**, when Python reads the `def`, not once per
call. Every call that doesn't pass `cups` shares that single list, and
`append` keeps filling it. This one has bitten every Python programmer
alive. The fix is a small ritual you'll write on autopilot:

```python
def add_cup(drink, cups=None):
    if cups is None:
        cups = []
    cups.append(drink)
    return cups
```

Default to `None`, make the fresh list inside. Strings, numbers, `None` and
tuples are all fine as defaults because they can't be changed.

## 5. Don't change what you were given

Lesson 004, section 2: passing a list to a function passes *the list*, not
a copy. So this function lies in a way that's genuinely hard to spot:

```python
def top_three_sneaky(prices):
    prices.sort(reverse=True)
    return prices[:3]

prices = [380, 220, 420, 430, 390]
top_three_sneaky(prices)         # [430, 420, 390]   looks right...
prices                           # [430, 420, 390, 380, 220]   ...but your list is rearranged
```

The name says "give me the top three". It doesn't say "and reorder your data
while I'm at it". If `prices` was in date order, that order is gone, and the
next function that assumed date order is now wrong, and nothing points back
here.

```python
def top_three(prices):
    return sorted(prices, reverse=True)[:3]
```

`sorted()` builds a new list; the caller's is untouched. Python's own
library models the rule exactly: `list.sort()` changes the list and returns
`None`; `sorted()` returns a new list and changes nothing. **Do one or the
other, never both, and let the name say which.** If a function *is* meant to
change its argument (`remove_refunds(orders)`), give it a verb that says so,
and have it return `None` so nobody thinks they got a copy.

A function that changes nothing outside itself and always gives the same
answer for the same inputs is called **pure**. `line_total` is pure.
`format_money` is pure. Pure functions are the easiest code there is: you
can call them in any order, as many times as you like, test them with a
single line, and reason about them without knowing anything else in the
program. Most of your data-cleaning helpers can and should be pure.

Not everything can be. Printing changes the screen, saving changes a file,
`ORDERS.append` changes a list. Those are called **side effects**, and
they're fine, as long as they live in a few clearly named functions
(`print_report`, `save_orders`) at the edges, and the functions doing the
thinking stay pure in the middle.

## 6. Pass it in, hand it back

```python
discount_rate = 0.10

def discounted_sneaky(cents):
    return round(cents * (1 - discount_rate))     # reaches outside for discount_rate
```

This works. It also can't be understood, tested, or reused without knowing
about a variable somewhere else in the file, and it silently changes
behaviour if anyone edits that variable. Reading module-level *constants*
(like `ORDERS` in `lesson.py`, in capitals, never reassigned) is acceptable.
Reaching out for values that might change is how lies begin.

```python
def discounted(cents, rate=0.10):
    return round(cents * (1 - rate))
```

Everything the function needs comes in through the parameters. Everything
it produces goes out through `return`. A function like that is a sealed
box, and sealed boxes are the only kind you can stack.

The flip side: names you create inside a function stay inside it. A variable
`inner` made in `make_label()` doesn't exist after the function returns.
That's a feature. It means you can reuse short names (`total`, `row`,
`cents`) in every function without them colliding.

## 7. `assert`: a one-line honesty check

You've been checking your functions by running them and eyeballing output.
Here's a way to make Python do the eyeballing:

```python
assert line_total(("d", "latte", "medium", 380, 2)) == 760
assert parse_quantity(" 3 ") == 3
assert parse_quantity("x") is None
assert top_three([1, 5, 3, 4, 2]) == [5, 4, 3]
assert format_money(1250) == "£12.50"
```

`assert something` does nothing if `something` is true, and stops the
program with an `AssertionError` if it isn't. Add a message after a comma
and it'll tell you why:

```python
assert format_money(380) == "£3.80", "format_money should always show 2 decimals"
```

A handful of these under a function, with the cases you'd otherwise check by
hand (the normal one, the empty one, the weird one), is the cheapest
insurance in programming. Run the file, silence means good. When you change
the function next month, the asserts tell you in a second whether you broke
its promise.

This only works if your functions **return** things. You can't `assert` what
a `print` showed. Which is the real reason section 1 matters: a function
that returns its answer can be checked; a function that prints it can only
be watched. Proper tests with a proper tool arrive in lesson 076; for now,
`assert` under your helpers is the habit to start.

## 8. Putting it together: a report made of small, honest pieces

Here's the week-of-orders revenue report, rebuilt from functions that each
do one thing:

```python
def format_row(drink: str, cents: int, star_from_cents: int) -> str:
    """One line of the report. Adds a star for big earners."""
    star = " *" if cents >= star_from_cents else ""
    return f"{drink:<12} {pounds(cents):>7}{star}"

def report_lines(orders, top: int = 5, star_from_cents: int = 2000) -> list[str]:
    """Build the report as a list of strings. Nothing printed, nothing changed."""
    revenue = revenue_by_drink(orders)
    lines = [f"{'drink':<12} {'revenue':>7}"]
    for drink, cents in top_n(revenue, n=top):
        lines.append(format_row(drink, cents, star_from_cents))
    lines.append(f"{'total':<12} {pounds(sum(revenue.values())):>7}")
    lines.append(f"* = {pounds(star_from_cents)} or more")
    return lines

def print_report(orders, top: int = 5, star_from_cents: int = 2000) -> None:
    """The one function that talks to a human. It only prints."""
    for line in report_lines(orders, top=top, star_from_cents=star_from_cents):
        print(line)
```

Things to notice:

- **`pounds`, `line_total`, `revenue_by_drink`, `top_n`, `format_row`** are
  all pure. Each can be tested with an `assert`. `print_report` is the only
  one with a side effect, and it does nothing *but* print.
- **`report_lines` returns a list of strings rather than printing.** That
  means you can test the report (`assert lines[1].startswith("latte")`),
  write it to a file, or email it, without touching the code that builds it.
- **Defaults make the common call short**: `print_report(ORDERS)`. Keywords
  make the unusual call clear: `print_report(ORDERS[:6], top=2,
  star_from_cents=1000)`.
- **`ORDERS` is exactly as it was afterwards.** Nothing sorted it, nothing
  appended to it.

Those `: str` and `-> list[str]` bits are **type hints**: notes saying what
goes in and what comes out. Python doesn't enforce them (pass a float and
nothing stops you), but readers and editors use them, and writing `-> None`
on `print_report` is a nice way of promising "this returns nothing, don't
assign it". They're optional in this course. I'll use them when they make a
signature clearer and skip them when they'd just add noise.

Run `lesson.py`, then try: change `top_n` to accept `n=None` meaning "all of
them". Which functions have to change? (One. That's the point.)

## What you can do now

- Explain the difference between printing and returning, and choose
  `return` for anything another line of code will use.
- Name a function so the name is a promise, and split it when the promise
  needs an "and".
- Return one consistent kind of thing, use `None` for "no answer", use a
  tuple for two or three answers, and put the exits first.
- Give parameters sensible defaults, call with keywords, and never use a
  list or dict as a default.
- Leave arguments unchanged unless the name says otherwise; know what pure
  means and why it's worth aiming for.
- Pass in what a function needs instead of reaching outside for it.
- Write `assert` checks under a function to keep it honest.

## What to do now

1. Run `lesson.py` and read it next to this page.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   hands you a working-but-ugly script and asks you to turn it into
   functions that don't lie.
3. That's Week 2, Day 2. Lessons 007 (files, paths and CSV round-trips) and
   008 (errors, and what to do about them) arrive tomorrow. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Today was two lessons about *shape*: how to arrange data (dicts) and how to
arrange code (functions). Neither one is flashy. Both are what make tomorrow's
file handling feel easy instead of fiddly.
