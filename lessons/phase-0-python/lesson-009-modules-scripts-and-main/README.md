# Lesson 009 - Modules, scripts and `if __name__ == "__main__"`

**Phase 0 - Python foundations for data work** | Week 2, Day 4 | Thursday 2026-09-17

> **Goal:** split code across files, import your own helpers, and run
> scripts from the command line with arguments, so that the useful
> functions you write live in one place and every script can borrow them.

Time: about 45 minutes. No installs. Python 3.10+.

---

Count how many times you've written `pounds()` this week. Lesson 005, 006,
007, 008: every `lesson.py` has its own copy, and so does every exercise
file you made with `cp`. `validate_row` has been copied three times now.
If you found a bug in it tomorrow, you'd have to remember every file it
lives in and fix each one. You wouldn't. Nobody does.

The fix is small and it's one of the most useful things in Python: put the
helpers in their own file, and *import* them. That file is called a
**module**, and you've been using modules since lesson 002 (`import csv` is
exactly this, with a file someone else wrote). Today you write your own.

Along the way you'll finally find out what the line at the bottom of every
lesson means:

```python
if __name__ == "__main__":
    main()
```

And you'll turn a script into a proper little command-line tool, the kind
you run with options: `python lesson.py --top 2`.

## How to follow along

This lesson has one extra file. Besides `lesson.py` there's
`coffee_tools.py`: the menu, `pounds`, `validate_row`,
`load_orders_carefully` and `revenue_by_drink`, gathered from the last four
lessons into one place. Open it; there's nothing in it you haven't seen.
Then run:

```bash
python lessons/phase-0-python/lesson-009-modules-scripts-and-main/lesson.py
python lessons/phase-0-python/lesson-009-modules-scripts-and-main/lesson.py --top 2
python lessons/phase-0-python/lesson-009-modules-scripts-and-main/coffee_tools.py
```

The script writes no files. (You may see a `__pycache__/` folder appear
next to `coffee_tools.py`. That's Python saving a pre-read copy of the
module so the next import is quicker. It's ignored by git; delete it
whenever you like.)

## 1. A module is just a `.py` file

That's the whole definition. Any file ending in `.py` is a module, and its
name is the file name without `.py`. So `coffee_tools.py` is the module
`coffee_tools`, and from a script in the same folder:

```python
import coffee_tools

coffee_tools.pounds(7970)          # '£79.70'
coffee_tools.MENU["latte"]         # 380
```

`import coffee_tools` does three things: finds the file, runs it from top
to bottom once, and gives you a single name, `coffee_tools`, through which
you can reach everything the file defined. Functions, constants, anything
assigned at the top level. The dot means "the thing called `pounds` that
lives inside `coffee_tools`".

Look at what `coffee_tools.py` does when it runs: it defines some functions
and a dict. It doesn't print, doesn't read files, doesn't do any work. A
good module is a shelf of tools, not a program that goes off the moment you
touch it. Section 4 is how you give it a program anyway, safely.

## 2. Three ways to import

```python
import coffee_tools                      # use as coffee_tools.pounds(...)
from coffee_tools import pounds          # use as pounds(...)
import coffee_tools as ct                # use as ct.pounds(...)
```

All three reach the very same function object (`pounds is
coffee_tools.pounds` is `True`); only the name you type changes. How to
choose:

- **`import module`** when you use several things from it, or when the name
  alone would be confusing. `coffee_tools.validate_row` tells a reader
  exactly where to look.
- **`from module import name`** for one or two things you use a lot, with
  names that are clear on their own: `from pathlib import Path`,
  `from collections import Counter`. You've been doing this for a week.
- **`import module as short`** for long names, or names everyone
  abbreviates. You'll meet `import numpy as np` and `import pandas as pd`
  in Phase 1, and every Python programmer on earth writes them that way.

And the one to avoid: `from coffee_tools import *`. It pours every name in
the module into your file. Now `pounds` works, but nobody reading your code
can tell where it came from, and if two modules both define `load`, the
second silently replaces the first.

Imports go at the **top of the file**, standard library first, then
installed packages, then your own modules, each group separated by a blank
line. That's a convention, not a rule, but it means anyone can see at a
glance what a file depends on. `lesson.py` follows it.

## 3. Where Python looks for modules

When you write `import coffee_tools`, Python searches a list of folders,
in order, for `coffee_tools.py`. The list is `sys.path`, and the first
entry is the **folder of the script you ran**. That's why this works from
the repo root, from inside the lesson folder, from anywhere: Python always
looks next to `lesson.py` first.

After that come the standard library's folders, then the ones where
installed packages live (Phase 1). The first match wins.

That order has a famous trap. Save an experiment as `csv.py` or
`random.py`, put it next to a script that does `import csv`, and *your*
file wins. You get `AttributeError: module 'csv' has no attribute
'DictReader'`, which makes no sense until you remember the search order.
**Never name your files after modules you use.** If you see a baffling
`AttributeError` on a module you know well, check for a file with the same
name in your folder.

Second thing worth knowing: a module runs **once per program**. After the
first `import`, Python keeps it in `sys.modules` and every later import
hands back the same object. So a module with a `print` at the top prints
once however many files import it, and changing `coffee_tools.py` while a
program is running changes nothing until you run it again.

## 4. `__name__`, and the line at the bottom of every lesson

Python gives every module a variable called `__name__` (two underscores
each side; people say "dunder name"). Its value depends on *how the file
was started*:

| How the file runs | `__name__` is |
|-------------------|---------------|
| `python coffee_tools.py` (you ran it) | `"__main__"` |
| `import coffee_tools` (someone imported it) | `"coffee_tools"` |

That's all the famous line is:

```python
if __name__ == "__main__":
    main()
```

"If I'm the file that was run, do the program. If I've been imported, just
provide the functions and keep quiet." One file, two jobs.

`coffee_tools.py` uses it for a self-check:

```python
if __name__ == "__main__":
    # Only runs for `python coffee_tools.py`, never on `import coffee_tools`.
    print(f"coffee_tools self-check (__name__ is {__name__!r})")
    assert pounds(7970) == "£79.70"
    ...
    print("  all good")
```

Run `python coffee_tools.py` and you get `all good`. Import it and you get
nothing. `lesson.py` proves this by running the file both ways.

Why it matters: without that `if`, importing a module would run its whole
program. Every lesson in this course can be imported by another script
(your exercises, later) because each one keeps its work inside `main()`
and only calls `main()` under that line. Make it a habit: **top level for
definitions, `main()` for doing things, and the `if` at the bottom.**

## 5. Command-line arguments: `sys.argv`, then `argparse`

Scripts get more useful when you can change what they do without editing
them. Words typed after the script name arrive in `sys.argv`:

```bash
python lesson.py --top 2
```

```python
sys.argv        # ['lesson.py', '--top', '2']
```

A list of strings: the script name, then everything else, split on
spaces. You *could* dig through that by hand, but you'd be writing error
messages for `--top three` and `--top` with nothing after it for the rest
of your life. The standard library's `argparse` does it for you:

```python
def build_parser():
    parser = argparse.ArgumentParser(description="Weekly revenue report for the coffee shop.")
    parser.add_argument("--csv", type=Path, default=ORDERS_CSV,
                        help="till export to read (default: orders.csv next to this script)")
    parser.add_argument("--top", type=int, default=3,
                        help="how many drinks to show (default: 3)")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the total line")
    return parser

args = build_parser().parse_args()
args.top        # 2, an int, not the string '2'
args.quiet      # False, because you didn't type --quiet
```

What you get for those few lines:

- **Types.** `type=int` converts `"2"` to `2`, and refuses `three` with a
  clear message. `type=Path` gives you a `Path`, ready for lesson 007's
  tricks.
- **Defaults.** Leave an option out and you get the default, so
  `python lesson.py` still works exactly as before.
- **Flags.** `action="store_true"` makes an on/off switch: `--quiet`
  present means `True`.
- **`--help` for free.** Try `python lesson.py --help`. Every `help=` string
  turns up there, neatly formatted.

When something's wrong, argparse prints a usage line and an error, and
exits with code `2`. (Exit codes are how programs tell the terminal
whether they worked: `0` for fine, anything else for not.) `lesson.py`
shows that without actually quitting, by catching the exit.

A handy trick for testing: `parse_args` takes a list, so
`parser.parse_args(["--top", "2"])` lets you try arguments without typing
them in a terminal. `main(argv=None)` in `lesson.py` uses the same idea:
`None` means "read the real command line", and a list means "pretend".

## 6. Putting it together

`lesson.py` is now three things at once: it imports its tools from
`coffee_tools`, it takes options from the command line, and it can itself
be imported without anything running. The last section is just:

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    ...
    total, cups = report(args.csv, args.top, args.quiet)
```

```
$ python lesson.py
    latte          £29.90
    cappuccino     £20.00
    flat white     £15.60
    (others)       £14.20
    total          £79.70   (23 cups, 15 orders, 0 rejected)

$ python lesson.py --top 2 --quiet
    total          £79.70   (23 cups, 15 orders, 0 rejected)
```

Same £79.70 as every day this week, but the functions that produced it
live in a different file, and there's exactly one copy of each. That's
the shape of every real project: a handful of modules that each do one
job, and a thin script on top that wires them together.

## What you can do now

- Say what a module is (a `.py` file) and what `import` does (find it, run
  it once, give you a name for it).
- Choose between `import x`, `from x import y` and `import x as y`, and
  explain why `from x import *` is a bad idea.
- Explain why Python finds `coffee_tools.py` next to your script, and why
  a file called `csv.py` breaks `import csv`.
- Say what `__name__` is when a file is run versus imported, and write the
  `if __name__ == "__main__":` line knowing what it does.
- Add options to a script with `argparse`: typed values, defaults, a flag,
  and `--help`.

## What to do now

1. Run `lesson.py` a few ways: no options, `--top 1`, `--quiet`, `--help`,
   and `--top three` to see the refusal.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   builds a second script on top of `coffee_tools`.
3. Lesson 010 (a little bit of classes) is this afternoon. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

From today, when you write a function you'll want again, it goes in a
module. Future you, fixing a bug once instead of four times, says thanks.
