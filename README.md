# Your ML / AI Tutor

Friendly Python lessons that take you from "I've never written code" to
building, evaluating and shipping machine learning and AI systems. One
curriculum, taught in order, two short lessons every weekday.

**Today's lessons (Friday 2026-09-25, Week 3 Day 5):**

- [Lesson 021 - Dates and time series basics](lessons/phase-1-data-science/lesson-021-dates-and-time-series-basics/README.md)
- [Lesson 022 - Plotting that tells the truth](lessons/phase-1-data-science/lesson-022-plotting-that-tells-the-truth/README.md)

New here? Start at
[Lesson 001 - Why ML/AI, and how we'll learn](lessons/phase-0-python/lesson-001-why-ml-ai-and-how-we-learn/README.md)
and work forward; the numbering is the path.

The full road map is in [curriculum/CURRICULUM.md](curriculum/CURRICULUM.md).
What's been published so far is in [curriculum/PROGRESS.md](curriculum/PROGRESS.md).

## What this is

A complete, phased curriculum covering machine learning, data science and
modern AI, all in Python:

- **Phase 0** - Python foundations for data work (standard library only)
- **Phase 1** - Data science basics: NumPy, pandas, plotting, statistics intuition
- **Phase 2** - Classical ML: supervised and unsupervised learning, evaluation, pipelines
- **Phase 3** - Deep learning foundations, from a hand-written neuron to PyTorch
- **Phase 4** - Modern AI: NLP, vision, generative models and LLM basics
- **Phase 5** - Projects and production intuition, ending in a capstone

Every lesson is a folder with three files:

| File | What it's for |
|------|---------------|
| `README.md` | The lesson itself. Read this first. It explains *why* before *how*. |
| `lesson.py` | A runnable script that walks through the same ideas in code. |
| `exercises.md` | A few short exercises, with hints and a way to check yourself. |

## Who it's for

- Absolute beginners who want a calm, structured path and don't want to be
  talked down to.
- People who already code but never got around to the maths/ML side.
- Anyone who has bounced off tutorials that jump from "hello world" to
  "here's a 400-line notebook" with nothing in between.

You need curiosity and about 30-45 minutes per lesson. That's it.

## How the cadence works

- **Monday to Friday, two lessons a day.** Lesson `001` and `002` on day one,
  `003` and `004` the next weekday, and so on.
- **No weekend content.** Rest, or redo an exercise that didn't land.
- **Strict order.** Each lesson assumes you did the one before it. If you
  join late, start at `001` and go at your own pace; the numbering is the
  path, the dates are just when each one was written.
- **Phases build on phases.** Phase 0 uses only the standard library, so you
  can start today with nothing but Python installed.

[PROGRESS.md](curriculum/PROGRESS.md) shows what's published and what's
coming next.

## How to run a lesson

You need **Python 3.10 or newer**. Check with:

```bash
python --version      # or: python3 --version
```

Then, from the repo root:

```bash
git clone https://github.com/Moruri/Your-ML-AI-Tutor.git
cd Your-ML-AI-Tutor

# Run lesson 001
python lessons/phase-0-python/lesson-001-why-ml-ai-and-how-we-learn/lesson.py

# Run lesson 002
python lessons/phase-0-python/lesson-002-python-warm-up-for-data-people/lesson.py

# Run today's lessons (Phase 1: inside the virtual environment, see below)
python lessons/phase-1-data-science/lesson-021-dates-and-time-series-basics/lesson.py
python lessons/phase-1-data-science/lesson-022-plotting-that-tells-the-truth/lesson.py
```

Every `lesson.py` also runs from inside its own folder; the scripts find
their own data files wherever you launch them from. From lesson 007 on,
some scripts write files; those go in an `output/` folder next to the
script (ignored by git), so you can delete it and re-run at any time.

Phase 0 lessons need **no installs**. From Phase 1 on (lesson 013 walks you
through it), create a virtual environment and install from
[`requirements.txt`](requirements.txt):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Scripts are meant to be read as much as run. Open `lesson.py` next to the
lesson's `README.md`, run it, change a number, run it again.

## Repo layout

```
lessons/
  phase-0-python/
    lesson-001-why-ml-ai-and-how-we-learn/
      README.md
      lesson.py
      exercises.md
    lesson-002-python-warm-up-for-data-people/
      README.md
      lesson.py
      exercises.md
      coffee_orders.csv
    lesson-003-numbers-strings-and-the-things-that-bite/
      README.md
      lesson.py
      exercises.md
    lesson-004-lists-and-tuples-properly/
      README.md
      lesson.py
      exercises.md
    lesson-005-dictionaries-as-tiny-databases/
      README.md
      lesson.py
      exercises.md
    lesson-006-functions-that-dont-lie/
      README.md
      lesson.py
      exercises.md
    lesson-007-files-paths-and-csv-round-trips/
      README.md
      lesson.py
      exercises.md
      orders_raw.csv
    lesson-008-errors-and-what-to-do-about-them/
      README.md
      lesson.py
      exercises.md
      orders_messy.csv
    lesson-009-modules-scripts-and-main/
      README.md
      lesson.py
      exercises.md
      coffee_tools.py
      orders.csv
    lesson-010-a-little-bit-of-classes/
      README.md
      lesson.py
      exercises.md
      orders.csv
    lesson-011-iterators-generators-and-lazy-data/
      README.md
      lesson.py
      exercises.md
      orders.csv
    lesson-012-mini-project-a-stdlib-data-report/
      README.md
      lesson.py
      exercises.md
      orders_fortnight.csv
  phase-1-data-science/
    lesson-013-setting-up-a-real-environment/
      README.md
      lesson.py
      exercises.md
    lesson-014-numpy-arrays-why-not-just-lists/
      README.md
      lesson.py
      exercises.md
      orders.csv
    lesson-015-numpy-indexing-broadcasting-and-reductions/
      README.md
      lesson.py
      exercises.md
      orders.csv
    lesson-016-randomness-you-can-reproduce/
      README.md
      lesson.py
      exercises.md
    lesson-017-pandas-series-and-dataframes/
      README.md
      lesson.py
      exercises.md
      orders_september.csv
    lesson-018-cleaning-data-missing-values-types-and-duplicates/
      README.md
      lesson.py
      exercises.md
      orders_september_raw.csv
    lesson-019-group-aggregate-pivot/
      README.md
      lesson.py
      exercises.md
      orders_september.csv
    lesson-020-joining-tables-and-reshaping/
      README.md
      lesson.py
      exercises.md
      orders_september.csv
      menu.csv
      targets_wide.csv
    lesson-021-dates-and-time-series-basics/
      README.md
      lesson.py
      exercises.md
      orders_september.csv
    lesson-022-plotting-that-tells-the-truth/
      README.md
      lesson.py
      exercises.md
      orders_september.csv
curriculum/
  CURRICULUM.md      # every lesson, grouped by phase, with a one-line goal
  PROGRESS.md        # what's published and what's next
requirements.txt     # pinned deps for Phase 1 on (Phase 0 needs none)
```

## A note on tone

These lessons are written the way a patient friend would explain things over
coffee: plainly, with real examples, and without pretending anything is
simpler or harder than it is. If a sentence ever reads like a textbook, that's
a bug. Open an issue.
