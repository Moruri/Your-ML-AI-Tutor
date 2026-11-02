# Lesson 068 - Retrieval-augmented generation (RAG)

**Phase 4 - Modern AI: NLP, vision, generative and LLM basics** | Week 9, Day 1 | Monday 2026-11-02

> **Goal:** Ground an LLM's answers in your own documents and evaluate
> whether it helped.

Time: about 45 minutes. Same venv as lesson 064, with `torch` and
`transformers` installed.

---

Ask a chat model about *your* prices, *your* refund rules or *your*
opening hours, and it has a problem: it never saw them. It was trained
on the public internet, months ago. What it does next is the dangerous
part. It doesn't say "I don't know". It writes a fluent, confident,
made-up answer, often called a **hallucination**.

**Retrieval-augmented generation**, RAG, is the standard fix, and you
already have both halves:

1. **Retrieve**: search your documents for the passages that answer
   the question (lesson 067).
2. **Augment**: paste those passages into the prompt as notes (lesson
   066).
3. **Generate**: ask the model to answer from the notes only.

The model doesn't learn anything new; no training happens. It just
gets an open-book exam instead of a closed-book one. Today we build it
for the bike-share's help centre, and, as always, measure whether it
actually helped.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-4-modern-ai/lesson-068-retrieval-augmented-generation-rag/lesson.py
```

It reads `help_centre.md`, seven short made-up help articles for the
course's made-up bike-share service, and writes no files. Open it and
skim it first: it's the "company knowledge" the model has never seen.
It uses lesson 064's embedding model and lesson 066's chat model,
`Qwen/Qwen2.5-0.5B-Instruct`, both already in your cache if you ran
those lessons. Replies are greedy, so your numbers should match. It
takes about two minutes.

We test with eight rider questions, each paired with the fact a correct
answer must contain (`QUESTIONS`), plus two questions the help centre
doesn't cover (`NOT_COVERED`).

## 1. Asking without notes

```
question                                       -> "How much is one ride if I'm back within half an hour?"
reply                                          -> "If you're back within half an hour of your ride, it typically costs $1.50 to $2.00 per ride on most bike-sharing systems like Bluejay or Lime."
question                                       -> "I'm a student. What does a month of riding cost me?"
reply                                          -> "A month of riding a bike-share system can cost anywhere from $10 to $50 per day, depending on your location and the type of plan you have. You'll also need to consider any additional fees such as parking fines or"
correct facts, no notes                        -> '1 / 8'
```

Fluent, polite, in dollars, and wrong. "$10 to $50 per day" for a
*month* isn't even consistent with itself.

Look closely at the one "correct": our checker, `correct`, just asks
whether the expected fact appears in the reply, and "$1.50" contains
"50". A fluke. Simple automatic checks are cheap and useful, but they
can be fooled; read a sample of replies by eye too. Lesson 073 is all
about evaluating AI systems properly.

## 2. Your documents, in chunks

```
articles                                       -> 7
chunks (one per paragraph)                     -> 16
one chunk                                      -> 'Paying with M-Pesa: If you are charged twice for the same ride, the duplicate charge is reversed automatically within 3 working days. You do not need to contact us.'
longest chunk (words)                          -> 41
```

We don't search whole documents; we search **chunks**. `load_chunks`
splits each article into paragraphs and glues the article's title to
the front of each one.

- **Small chunks** are about one thing, so the search can land on the
  exact paragraph that answers, and the prompt stays short.
- **Too small**, and a chunk loses its context. "It costs KSh 1,500"
  means nothing without knowing *what* costs that. The title prefix is
  a cheap way to keep each chunk understandable on its own.

Real systems split long documents into pieces of a few hundred words,
often overlapping a little so no sentence is cut off from its context.
Paragraphs are a natural boundary when your documents have them.

## 3. Retrieve: find the passages that matter

```
question                                       -> "There's no space to park at the station. What should I do?"
  0.504  Full or empty docks: If you arrive at a station and every dock
  0.363  Full or empty docks: If a station has no bikes, the app's map
  0.282  Late returns and lost bikes: Bikes must be docked by midnight.
right article in the top 1                     -> '8 / 8'
right article in the top 3                     -> '8 / 8'
```

This is lesson 067's search, pointed at 16 chunks. "No space to park"
finds "every dock is full", with no shared words.

Before building anything on top, we check this step alone. **Retrieval
is the ceiling of a RAG system**: if the right passage isn't in the
notes, the model can only guess or refuse. Here search finds the right
article for all eight questions, even with just one result, so any
mistakes from now on belong to the generation step.

## 4. Augment and generate

```
question                                       -> "How much is one ride if I'm back within half an hour?"
reply with notes                               -> "I don't know."
correct facts, no notes (section 1)            -> '1 / 8'
correct facts, with retrieved notes            -> '5 / 8'
```

`rag_prompt` builds lesson 066's kind of prompt: a spec in the system
message, the retrieved chunks pasted underneath as notes, and the
question as the user message. The spec says three things:

- Answer **using only the notes**.
- Quote numbers **exactly** as the notes give them.
- If the notes don't contain the answer, reply exactly `I don't know.`

Five out of eight, up from a fluke one. Same model, same questions, no
training; only three paragraphs of our own text in the prompt.

But look at the example. The answer, KSh 50, was right there in the
notes, and the model said "I don't know". That's the opposite of a
hallucination: an **over-refusal**. The escape hatch we gave it is
useful, and a small model sometimes takes it when it shouldn't. Hold
that thought for section 6.

## 5. When the answer isn't there

```
question                                       -> 'Can I take a bike on the commuter train?'
best search score                              -> 0.362
reply                                          -> "I don't know."
said it doesn't know                           -> True
question                                       -> 'Who founded the bike-share company?'
best search score                              -> 0.379
reply                                          -> "I don't know."
said it doesn't know                           -> True
lowest best score, covered questions           -> 0.319
```

A RAG system also has to cope with questions the documents don't
answer. Search will still return its top three chunks, relevant or
not, so the model always gets *some* notes.

Here it did the right thing both times. Two warnings, though:

- **A score threshold wouldn't have caught these.** Lesson 067
  suggested refusing when the best search score is low. But these
  off-topic questions scored 0.36 and 0.38, *higher* than one real
  question (0.32). With short, similar chunks the scores are close
  together, so a threshold is a helpful extra check, not a guarantee.
- **Always test both kinds of question.** A system that says "I don't
  know" to everything passes this section perfectly. Section 4 already
  caught ours refusing a question it could answer. You need answerable
  and unanswerable questions in your test set, side by side.

## 6. Evaluate the parts separately

```
question                                     found  k=1  k=3
How much is one ride if I'm back within half   yes  yes   no
I'm a student. What does a month of riding c   yes   no   no
I paid twice for one trip. When do I get my    yes  yes  yes
There's no space to park at the station. Wha   yes  yes  yes
What's the fine if I keep a bike overnight?    yes  yes   no
How old do I have to be to sign up?            yes  yes  yes
Until what time in the evening can I reach s   yes  yes  yes
How do I tell you a bike has a flat tyre?      yes  yes  yes
correct with 1 note                            -> '7 / 8'
correct with 3 notes                           -> '5 / 8'
```

A RAG system has two parts that fail in different ways, so score them
separately:

- **found**: did retrieval put the right article in the notes?
- **k=1** and **k=3**: was the final answer right, with one note or
  three?

Every "found" is yes, so every wrong answer is a generation problem.
And a surprise: **one note beats three**, 7 to 5. With three notes, the
overnight question was answered with the *other* paragraph in the same
article (the KSh 25,000 lost-bike charge), and the half-hour price got
the over-refusal from section 4. Extra notes are extra chances to get
distracted, and small models are easily distracted.

That doesn't mean "always use one". With one note, the student
question got only the single-ride paragraph and the model invented a
monthly price from it ("approximately KSh 480"); with three, it found
the monthly pass but missed the student discount. Bigger models handle
more notes much better. The number of chunks, the chunk size and the
prompt are all settings you tune, by measuring, exactly like a
model's hyperparameters in Phase 2.

When something's wrong, this table tells you where to look: "found =
no" means fix the search (chunking, the embedding model, hybrid search
from lesson 067); "found = yes" but a wrong answer means fix the prompt,
the number of notes, or use a stronger model.

## What you can do now

- Explain why chat models hallucinate about your own data, and how RAG
  helps without any training.
- Split documents into self-contained chunks.
- Retrieve chunks for a question and check retrieval on its own.
- Write a grounded prompt with an "I don't know" escape hatch.
- Test with both answerable and unanswerable questions.
- Score retrieval and generation separately, and tune the number of
  notes by measuring.

## What to do now

1. Run `lesson.py`. Then add a fact to `help_centre.md` (say, a new
   station opening) and a question about it to `QUESTIONS`, and rerun.
   No retraining needed: that's the point of RAG.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   tries to fix the student question.
3. Next weekday, lesson 069: tool use and agents, where the model stops
   just reading notes and starts calling Python functions, and we see
   where agent loops go wrong.

RAG gives a model an open book: retrieve the right page, tell it to
answer only from that page, and check both the page and the answer.
