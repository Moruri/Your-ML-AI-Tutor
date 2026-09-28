# Lesson 009 - Exercises

Three quick checks, a hands-on task where you write a second script on top
of `coffee_tools`, and an optional question about where code belongs.
Predict first, then run.

This time you'll work *inside* the lesson folder, because the point is to
import `coffee_tools.py` from a file sitting next to it. From the repo
root:

```bash
cd lessons/phase-0-python/lesson-009-modules-scripts-and-main
```

Any new files you make there are yours; delete them when you're done if
you'd like the folder tidy.

---

## 1. What prints?

Make a file called `noisy.py` in the lesson folder with exactly this in it:

```python
print("noisy is being run, __name__ =", __name__)

def hello():
    return "hello from noisy"

if __name__ == "__main__":
    print("noisy's main part")
```

Predict the output of each, then try them:

- (a) `python noisy.py`
- (b) `python -c "import noisy"`
- (c) `python -c "import noisy; import noisy; print(noisy.hello())"`

<details>
<summary>Check yourself</summary>

```
(a) noisy is being run, __name__ = __main__
    noisy's main part

(b) noisy is being run, __name__ = noisy

(c) noisy is being run, __name__ = noisy
    hello from noisy
```

The top-level `print` runs every time the file runs, whether it was
started or imported. That's why modules shouldn't have one. The `if`
block runs only in (a). And in (c) the first line appears **once**, not
twice: the second `import` got the cached module from `sys.modules`
without running the file again.

</details>

## 2. Which import?

For each, write the import line you'd use, and say why:

- (a) You need `Path` in lots of places in one script.
- (b) You need `pounds`, `validate_row`, `load_orders_carefully` and
  `revenue_by_drink` from `coffee_tools`.
- (c) You need one function called `load` from each of two modules,
  `till` and `stock`.

<details>
<summary>Check yourself</summary>

**(a)** `from pathlib import Path`. One name, used constantly, and `Path`
is clear on its own.

**(b)** `import coffee_tools` (or `import coffee_tools as ct`). Four names
is a lot to list, and `coffee_tools.validate_row` tells a reader where to
look. Listing all four with `from coffee_tools import ...` is also fine;
it's a judgement call. `import *` is not.

**(c)** `import till` and `import stock`, then `till.load()` and
`stock.load()`. With `from till import load` and `from stock import load`,
the second would quietly replace the first. The module name is doing
useful work here: it says *which* load.

</details>

## 3. The mysterious AttributeError

A friend has these two files in one folder:

```python
# random.py  (their dice experiment)
print("rolling...")
```

```python
# picker.py
import random
print(random.choice(["latte", "tea"]))
```

`python picker.py` prints `rolling...` and then
`AttributeError: module 'random' has no attribute 'choice'`. What happened,
and what's the fix?

<details>
<summary>Check yourself</summary>

Python searched the script's own folder first (section 3), found *their*
`random.py`, ran it (hence `rolling...`), and handed that back as
`random`. Their file has no `choice`, so: `AttributeError`.

Fix: rename `random.py` to something like `dice_experiment.py`, and delete
any `__pycache__/random.*.pyc` left behind. Never name a file after a
module you use. `python -c "import random; print(random.__file__)"` shows
which file Python actually found, which settles this sort of mystery in
one line.

</details>

## 4. Hands-on: a second script, zero copies

The shop owner wants a different report: revenue **per day**, not per
drink, and sometimes for one drink only. Write it as a new script,
`daily.py`, in the lesson folder. It must not copy any function from
`coffee_tools.py`. Import them.

**a)** Start `daily.py` with the imports and a `revenue_by_day(orders)`
function that returns `{date: cents}`. Put it in `daily.py` for now (it's
new, so there's nothing to copy).

**b)** Give it a command line with `argparse`:

- `--csv` (a `Path`, default `orders.csv` next to the script; use
  `Path(__file__).resolve().parent` like `lesson.py` does);
- `--drink` (a string, default `None`, meaning "all drinks").

If `--drink` is given, keep only orders for that drink before totalling.
Lower-case it, so `--drink Latte` works.

**c)** Print one line per day in date order, then a total, using
`pounds`. Keep all the work in `main()` and call it under
`if __name__ == "__main__":`.

**d)** Check it from the command line:

```bash
python daily.py
python daily.py --drink latte
python daily.py --drink mocha
```

**e)** Prove it can be imported without printing anything:
`python -c "import daily; print(daily.revenue_by_day([]))"` should print
only `{}`.

Hints, if you want them:

- `from coffee_tools import load_orders_carefully, pounds` is all you need
  from the module.
- `orders = [row for row in orders if row["drink"] == args.drink.lower()]`
  filters. Only do it `if args.drink:`.
- Dates are ISO strings, so `sorted(totals.items())` puts them in date
  order for free (lesson 007).

<details>
<summary>Expected output</summary>

```
$ python daily.py
2026-09-07   £14.00
2026-09-08   £20.40
2026-09-09   £13.70
2026-09-10   £12.70
2026-09-11   £18.90
total        £79.70

$ python daily.py --drink latte
2026-09-07    £7.60
2026-09-08    £4.30
2026-09-09    £3.80
2026-09-10    £6.60
2026-09-11    £7.60
total        £29.90

$ python daily.py --drink mocha
total         £0.00
```

`--drink mocha` printing a £0.00 total is honest here: mocha genuinely
sold nothing, because it's not on the menu. If you'd like it louder,
check `args.drink.lower() in MENU` and call `parser.error(...)`, which
prints a usage message and exits with code 2, argparse style.

</details>

<details>
<summary>One way to write it</summary>

```python
"""Revenue per day, optionally for one drink. Uses coffee_tools; copies nothing."""

import argparse
from pathlib import Path

from coffee_tools import load_orders_carefully, pounds

HERE = Path(__file__).resolve().parent


def revenue_by_day(orders):
    """Total cents per date, as a plain dict."""
    totals = {}
    for row in orders:
        totals[row["date"]] = totals.get(row["date"], 0) + row["price_cents"] * row["quantity"]
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description="Revenue per day.")
    parser.add_argument("--csv", type=Path, default=HERE / "orders.csv")
    parser.add_argument("--drink", default=None, help="only this drink (default: all)")
    args = parser.parse_args(argv)

    orders, _rejects = load_orders_carefully(args.csv)
    if args.drink:
        orders = [row for row in orders if row["drink"] == args.drink.lower()]

    totals = revenue_by_day(orders)
    for day, cents in sorted(totals.items()):
        print(f"{day}  {pounds(cents):>7}")
    print(f"{'total':<10}  {pounds(sum(totals.values())):>7}")


if __name__ == "__main__":
    main()
```

Notice how short it is. All the careful work (validation, rejects,
formatting money) is already done and tested in `coffee_tools`. The new
script only says what's *new*. And `_rejects` with a leading underscore is
a convention for "I know this exists and I'm deliberately not using it".

</details>

## 5. (Optional) Which file does it belong in?

`revenue_by_day` lives in `daily.py`. Should it move to `coffee_tools.py`?
What about the `argparse` setup? And `print`?

<details>
<summary>Check yourself</summary>

**`revenue_by_day`: probably yes, eventually.** It's a small, general tool
with no printing and no command line: exactly what a module is for. The
moment a second script wants it, move it (and only then; moving things
"in case" makes modules full of code nobody uses).

**`argparse` setup: no.** Options belong to a particular program. A module
that read the command line would do surprising things to whatever script
imported it.

**`print`: no.** Modules return values; scripts decide what to show and
how. That's what lets `report` in `lesson.py` and `daily.py` present the
same numbers differently.

A useful test: *could two different scripts want this exact thing?* If
yes, module. If it's about what *this* program shows or accepts, script.

</details>

---

That's the morning of Week 2, Day 4 done. Lesson 010 is this afternoon.
If one thing sticks, let it be: *definitions at the top level, work inside
`main()`, and `if __name__ == "__main__":` at the bottom*.
