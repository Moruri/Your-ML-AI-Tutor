# Lesson 013 - Setting up a real environment

**Phase 1 - Data science basics** | Week 3, Day 1 | Monday 2026-09-21

> **Goal:** create a virtual environment, install the Phase 1 packages,
> and know how to check what's installed, so that "it works on my
> machine" becomes "it works, and I know why".

Time: about 40 minutes. First real install. Python 3.10+ and an internet
connection.

---

Welcome to Phase 1. For two weeks, everything we've done has used only
the standard library: the tools that come in the box with Python. That was
deliberate. You now know what a CSV loader, a group-by and a summary
statistic look like from the inside, because you wrote them.

From here on we'll use **packages**: code other people wrote, published,
and maintain, which you download and install. NumPy for fast arrays of
numbers, pandas for tables, matplotlib for charts, SciPy for statistics.
They're the reason Python is the language of data science, and they'll
turn some of last week's twenty-line functions into one line.

But installing things is where a lot of beginners get stuck, and it's
rarely their fault. The errors are confusing, the advice online
contradicts itself, and "just `pip install` it" works right up until it
doesn't. So this lesson is about doing it *properly*, once, and
understanding it well enough that when something goes wrong, you can
work out why.

## How to follow along

This lesson is mostly terminal commands, and `lesson.py` is a health
check you run before and after. It uses only the standard library, so it
works even when nothing is installed. From the repo root:

```bash
python lessons/phase-1-data-science/lesson-013-setting-up-a-real-environment/lesson.py
```

Run it now, *before* you change anything, and keep the output. You'll
run it again at the end and compare. It writes no files and installs
nothing.

## 1. Which Python is this?

Many computers have more than one Python: one the operating system uses,
one you installed, perhaps one that came with another tool. When you type
`python`, you get whichever one your terminal finds first, and it isn't
always the one you think.

```bash
python --version                                    # or python3 --version
python -c "import sys; print(sys.executable)"       # the actual file being run
```

`sys.executable` is the full path of the Python that's running. Section 1
of `lesson.py` prints it. **When anything to do with packages is
confusing, this is the first thing to check**, because the most common
cause of "I installed it but Python can't find it" is installing into one
Python and running another.

On macOS and Linux, `python` may not exist and `python3` is the name; on
Windows, `py` is often the most reliable. Use whichever gives you 3.10 or
newer. Inside a virtual environment (next section), plain `python` always
means the right one, which is one of the nicest things about them.

## 2. Virtual environments: one project, one box

A **virtual environment** ("venv") is a folder containing its own copy of
the Python launcher and its own, initially empty, place for packages.
When it's *activated*, `python` and `pip` in that terminal refer to the
venv's versions, and anything you install goes into that folder and
nowhere else.

Why bother? Because different projects need different versions of
things. This course pins `pandas>=2.2,<3`. Another project on your
machine might need pandas 1.5. Install both into your system Python and
one of them breaks. Give each project its own venv and they never meet.
It also means you can delete the whole thing and start again in thirty
seconds, which takes the fear out of experimenting.

From the repo root:

```bash
python -m venv .venv
```

That creates a folder called `.venv` in the repo. (The name is a
convention; the dot hides it on macOS and Linux, and it's already in
`.gitignore`, so it'll never be committed.) Then activate it:

```bash
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows (Command Prompt or PowerShell)
```

Your prompt changes to start with `(.venv)`. That's how you know it's on.
Check:

```bash
python -c "import sys; print(sys.executable)"
# .../Your-ML-AI-Tutor/.venv/bin/python
```

Activation only lasts for that terminal window. Open a new terminal and
you'll need to activate again; type `deactivate` to switch off. Forgetting
to activate is the second most common setup mistake, and section 2 of
`lesson.py` exists to catch it: inside a venv, `sys.prefix` (the current
environment) differs from `sys.base_prefix` (the Python it was built
from).

If PowerShell refuses to run the activate script with a message about
execution policies, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try
again. It's a Windows safety setting, not a problem with your setup.

## 3. Installing packages with pip

`pip` is Python's package installer. It downloads packages from the
Python Package Index (PyPI) and puts them in the current environment's
`site-packages` folder. With the venv active:

```bash
python -m pip install --upgrade pip           # a fresh venv's pip is often old
python -m pip install -r requirements.txt
```

`-r requirements.txt` means "install everything listed in this file".
Open it: the active lines look like `pandas>=2.2,<3`, which means "any
version 2.2 or newer, but not 3". Pinning a range like that keeps the
lessons working: new bug fixes are welcome, but a major version (which may
change how things work) has to be chosen on purpose.

Notice `python -m pip` rather than plain `pip`. They usually do the same
thing, but `python -m pip` guarantees you're using the pip that belongs
to *this* `python`. It sidesteps the one-Python-installs-another-runs
problem completely. Make it a habit.

The install downloads a fair amount (NumPy, pandas, matplotlib, SciPy and
their dependencies). It takes a minute or two. Lots of output scrolls
past; what matters is the last line, which should start with
`Successfully installed`.

If you see `error: externally-managed-environment`, you're not in a venv:
your operating system is protecting its own Python from being changed.
It's a good error. Activate the venv and try again.

## 4. Checking what's installed

Three commands you'll use for the rest of your Python life:

```bash
python -m pip list                  # every package in this environment, with versions
python -m pip show pandas           # details of one: version, location, what it needs
python -m pip freeze                # the exact versions, in requirements.txt format
```

`pip show pandas` includes a `Location:` line. It should be inside your
`.venv` folder. If it isn't, you've installed somewhere else.

From inside Python, the standard library can tell you too:

```python
import importlib.metadata
importlib.metadata.version("pandas")    # '2.2.3' or similar
```

And most packages carry their own version:

```python
import numpy as np
np.__version__
```

That's what section 4 of `lesson.py` does for every Phase 1 package: it
reads `requirements.txt`, looks up the installed version of each, and
checks it's inside the pinned range. Section 5 then imports each one,
because *installed* and *importable* are almost the same thing, and the
rare times they aren't are exactly when you need to know.

## 5. When it goes wrong

Nearly every setup problem is one of these:

| You see | Usually means | Fix |
|---------|---------------|-----|
| `ModuleNotFoundError: No module named 'pandas'` | Not installed in *the Python you're running*. | Is the venv active? `python -m pip install -r requirements.txt`. |
| It's installed (`pip list` shows it) but import still fails | `pip` and `python` belong to different Pythons. | Use `python -m pip`, and check `sys.executable`. |
| `externally-managed-environment` | You're installing into the system Python. | Activate the venv. |
| `command not found: python` | That name isn't on this system. | Try `python3` or `py`. |
| Your editor says the import is missing, the terminal runs fine | The editor is using a different interpreter. | Point the editor at `.venv` (search its docs for "select interpreter"). |
| Something's just weird | A broken or confused environment. | Delete `.venv`, create it again, reinstall. Two minutes. |

That last row is worth taking seriously. A venv is disposable. When it's
in a strange state, don't spend an hour fixing it. Delete the folder and
make a new one. That's what they're for.

## 6. Putting it together

With the venv created, activated and filled, run the health check again:

```bash
python lessons/phase-1-data-science/lesson-013-setting-up-a-real-environment/lesson.py
```

```
6. Summary: what to do next
---------------------------
  [ok] Python 3.10+
  [ok] inside a virtual environment
  [ok] Phase 1 packages installed at the right versions

  All set for Phase 1. Lesson 014 is waiting.
```

Compare with your first run. `sys.executable` now points inside `.venv`,
`sys.prefix` and `sys.base_prefix` differ, and every package has a
version next to it. If anything still says `[!!]`, the summary tells you
the exact commands to run.

Your daily routine from now on: open a terminal, `cd` to the repo,
activate the venv, run lessons. Deactivate (or just close the terminal)
when you're done.

## What you can do now

- Find out which Python is running with `sys.executable`, and explain why
  that's the first thing to check when packages go missing.
- Create a virtual environment with `python -m venv .venv`, activate it,
  and tell from your prompt (or `sys.prefix`) that it's active.
- Install from `requirements.txt` with `python -m pip install -r`, and read
  a version pin like `>=2.2,<3`.
- Check what's installed with `pip list`, `pip show`, `pip freeze` and
  `importlib.metadata.version`.
- Diagnose the half-dozen setup errors that cover nearly every problem,
  and know that deleting and recreating a venv is a legitimate fix.

## What to do now

1. Create the venv, install the requirements, and get `lesson.py` to all
   ticks. If you get stuck, section 5's table is the place to start.
2. Do the exercises in [`exercises.md`](exercises.md). They break the
   setup on purpose so you can see what each problem looks like.
3. Lesson 014 (NumPy arrays: why not just lists?) is this afternoon, and
   it needs today's setup. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Setting up environments isn't glamorous. But the person on a team who
can calmly fix "it can't find pandas" is worth a lot, and as of today,
that's you.
