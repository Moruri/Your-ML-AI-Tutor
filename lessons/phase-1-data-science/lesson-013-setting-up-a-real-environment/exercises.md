# Lesson 013 - Exercises

Three quick checks, a hands-on task where you break your setup on purpose
(safely) and fix it, and an optional question about sharing environments.
Predict first, then run.

All of these are done in the terminal, from the repo root. The hands-on
task makes a *second*, throwaway venv, so your real `.venv` is never at
risk.

---

## 1. Read the prompt

Someone pastes this into a chat and says "pandas is broken":

```
$ python -c "import pandas"
ModuleNotFoundError: No module named 'pandas'
$ pip list | grep pandas
pandas     2.2.3
```

- (a) Is pandas installed?
- (b) What's the most likely explanation?
- (c) Which two commands would you ask them to run next?

<details>
<summary>Check yourself</summary>

**(a)** Yes, *somewhere*. `pip list` found it.

**(b)** `pip` and `python` belong to different Pythons. `pip` installed
pandas into one environment; `python` is running another. There's no
`(.venv)` at the start of their prompt, so they probably haven't activated
the venv, and `pip` and `python` happen to point to two different system
installs.

**(c)** `python -c "import sys; print(sys.executable)"` to see which
Python runs, and `pip --version`, which prints the Python pip belongs to.
They'll differ. The fix: activate the venv and use
`python -m pip install -r requirements.txt`, so pip and python can't
disagree.

</details>

## 2. Read the pins

Which of these versions satisfy `pandas>=2.2,<3`?

`2.1.4`, `2.2.0`, `2.2.3`, `2.10.1`, `3.0.0`, `3.0.0rc1`

<details>
<summary>Check yourself</summary>

`2.2.0`, `2.2.3` and `2.10.1` do.

- `2.1.4` is older than 2.2.
- `2.10.1` is *newer* than 2.2, not older: versions compare part by part
  as numbers, so 10 beats 2. (As text, `"2.10" < "2.2"`, which is why you
  never compare versions as strings. `parse_version` in `lesson.py` turns
  them into tuples of numbers for exactly this reason.)
- `3.0.0` fails `<3`.
- `3.0.0rc1` is a *release candidate* for 3.0, and pip's rules exclude it
  too (and by default pip doesn't install pre-releases at all).

</details>

## 3. Where did it go?

With your venv active, run:

```bash
python -m pip show numpy
```

Find the `Location:` line. Then deactivate (`deactivate`) and run the same
command again. What changed, and what does that tell you?

<details>
<summary>Check yourself</summary>

Active: the location is inside your repo's `.venv/lib/.../site-packages`.

Deactivated: either `WARNING: Package(s) not found: numpy` (the system
Python doesn't have it), or a *different* location with possibly a
different version (the system Python has its own copy).

Either way, it shows that each environment has its own packages, and
which one you get depends entirely on which Python is active. Reactivate
before you carry on.

</details>

## 4. Hands-on: break it, then fix it

You'll create a throwaway venv called `.venv-scratch`, check it, break
it in two common ways, and read the health check each time. Nothing here
touches your real `.venv`.

(Add `.venv-scratch/` to your *own* mental list of things to delete
afterwards. It isn't covered by `.gitignore`, so don't commit it.)

**a) An empty venv.** Create and activate it:

```bash
python -m venv .venv-scratch
source .venv-scratch/bin/activate          # Windows: .venv-scratch\Scripts\activate
python lessons/phase-1-data-science/lesson-013-setting-up-a-real-environment/lesson.py
```

Predict before running: which lines are `[ok]` and which are `[!!]`?

**b) The wrong version.** Still in `.venv-scratch`, install a pandas that
breaks the pin, then run the health check again:

```bash
python -m pip install "pandas<2.2"
```

What does section 4 say about pandas? What about numpy, which you didn't
ask for?

**c) Fix it properly.**

```bash
python -m pip install -r requirements.txt
```

Run the check again. Did pip upgrade pandas? Read the last lines of pip's
output to see.

**d) Freeze it.** Save the exact versions:

```bash
python -m pip freeze > scratch-freeze.txt
```

Open the file. How many lines are there, compared with the four Phase 1
packages you asked for? Why?

**e) Throw it away.**

```bash
deactivate
rm -rf .venv-scratch scratch-freeze.txt           # Windows: rmdir /s .venv-scratch & del scratch-freeze.txt
```

Then reactivate your real `.venv` and run the health check one last time.
Still all ticks? Good. That's what "disposable" means.

<details>
<summary>What you should see</summary>

```
a) [ok] Python 3.10+
   [ok] inside a virtual environment
   [!!] numpy / pandas / matplotlib / scipy: not installed
   The venv is active (section 2 says yes) but completely empty.

b) [!!] pandas      wanted >=2.2,<3       have 2.1.4  (outside >=2.2,<3)
   [ok] numpy       ... have 1.26.x   (or similar)
   numpy arrived without being asked: pandas depends on it, so pip
   installed it too. matplotlib and scipy are still missing.

c) "Successfully installed ... pandas-2.x.x ..." with a newer pandas.
   pip saw the installed version didn't meet the pin and upgraded it.
   All ticks.

d) Twenty-odd lines, not four. pip freeze lists EVERY package in the
   environment, including the dependencies of your dependencies
   (python-dateutil, pytz, pillow, contourpy, ...).
```

The exact version numbers will depend on the day you do this. (Very new
Pythons sometimes can't install old versions like `pandas<2.2` at all,
because nobody built them for that Python. If step (b) fails with a long
build error, that's why. Skip to (c); you've still seen what a pin
mismatch looks like in section 4 of the lesson.)

</details>

## 5. (Optional) requirements.txt or pip freeze?

This repo's `requirements.txt` has ranges (`pandas>=2.2,<3`). `pip freeze`
gives exact versions (`pandas==2.2.3`) for *everything*. When would you
want each?

<details>
<summary>Check yourself</summary>

**Ranges** (like this repo) suit something many people install at
different times, on different computers and Python versions. Each person
gets the newest versions that fit, including bug fixes, and the upper
bound stops a surprise major version from breaking the lessons.

**Exact pins from `pip freeze`** suit "this must behave identically every
time": a deployed model, an experiment you need to reproduce exactly next
year, a colleague who must see the same numbers as you. The catch is that
exact pins age: in two years some of those versions won't install on a
new Python.

Real projects often keep both: a short, human-written file of ranges for
what the project *needs*, and a machine-written "lock" file of exact
versions for what it was *tested with*. You'll see this again in lesson
076, when we talk about reproducibility.

</details>

---

That's the morning of Week 3, Day 1 done. This afternoon, lesson 014 uses
your new environment for its first real package: NumPy. If one thing
sticks, let it be: *activate the venv, use `python -m pip`, and when
something's weird, check `sys.executable` first*.
