# Your ML / AI Tutor

Friendly Python lessons that take you from "I've never written code" to
building, evaluating and shipping machine learning and AI systems. One
curriculum, taught in order, two short lessons every weekday.

**Today's lessons (Tuesday 2026-10-06, Week 5 Day 2):**

- [Lesson 029 - What a model actually is](lessons/phase-2-classical-ml/lesson-029-what-a-model-actually-is/README.md)
- [Lesson 030 - Linear regression with scikit-learn](lessons/phase-2-classical-ml/lesson-030-linear-regression-with-scikit-learn/README.md)

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

# Run today's lessons (Phase 2: inside the virtual environment, see below)
python lessons/phase-2-classical-ml/lesson-029-what-a-model-actually-is/lesson.py
python lessons/phase-2-classical-ml/lesson-030-linear-regression-with-scikit-learn/lesson.py
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

## A note on tone

These lessons are written the way a patient friend would explain things over
coffee: plainly, with real examples, and without pretending anything is
simpler or harder than it is. If a sentence ever reads like a textbook, that's
a bug. Open an issue.
