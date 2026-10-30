# Lesson 066 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv (with `transformers` installed, as in lesson 064).
For the hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-4-modern-ai/lesson-066-prompting-as-programming/lesson.py my_lesson_066.py
cp lessons/phase-4-modern-ai/lesson-066-prompting-as-programming/feedback.csv .
python my_lesson_066.py
```

The script reads `feedback.csv` from its own folder, which is why the
second line copies it next to your copy. `load_model`, `ask`,
`evaluate`, `parse_word`, `make_few_shot`, `clear_prompt` and
`answer_scores` are there to reuse.

---

## 1. Why test on 40 messages, not 1?

A colleague tries a new prompt on one message, gets the right answer,
and wants to ship it. What do you tell them?

<details>
<summary>Check yourself</summary>

One reply tells you almost nothing. In section 2 the clear prompt
answered `bike` five times in a row, and it would look perfect on any
bike message, yet it was right only a third of the time overall. A
prompt is a model, and like any model from Phase 2 it has to be
measured on a set of examples it wasn't tuned on. Forty is small; for
anything that matters, use more, and keep them fixed so you can compare
prompts fairly.

</details>

## 2. Where does the spec go?

Why put the instructions in the **system** message rather than at the
start of every user message?

<details>
<summary>Check yourself</summary>

The system message is the place chat models are trained to treat as
standing rules for the whole conversation, so it tends to carry more
weight, and it stays put while user messages change. It also keeps
your instructions apart from user input, which makes the prompt easier
to read and test, and makes it clearer to the model which text is the
spec and which is data. (It is not a security boundary on its own;
section 6 showed an injection can still get through.)

</details>

## 3. Usable versus accurate

Section 4's JSON prompt had 97.5% usable replies but 75% accuracy.
Section 5's scored prompt had 100% usable replies. Does 100% usable
mean the program is correct?

<details>
<summary>Check yourself</summary>

No. Usable only means the reply has the right *shape*. Scoring
guarantees a valid topic, not the right one: section 5 was still wrong
on 15% of messages. Validation protects your code from crashing on
nonsense; only an evaluation on labelled data tells you whether the
answers are any good. You need both.

</details>

## 4. Hands-on: more examples

Section 3 used one example per topic. Try two per topic (8 examples),
shuffled so the topics aren't in a neat order, with the same parser,
on the same 40 test messages. Predict first. Then, for comparison,
score the zero-shot prompt from section 2 (no examples) the way section
5 does.

<details>
<summary>Expected results</summary>

```
clear prompt, generated (section 2):    usable 0.575, accuracy 0.325
clear prompt, scored:                   usable 1.0,   accuracy 0.5
4 examples, generated (section 3):      usable 0.975, accuracy 0.825
8 examples, generated:                  usable 0.95,  accuracy 0.875
```

More examples help a bit more. Scoring alone fixes the format problem
of the zero-shot prompt, and lifts accuracy from 0.325 to 0.5, but it
can't supply what the examples teach about where one topic ends and the
next begins. Examples did most of the work here. Each extra example
also makes every prompt longer, so slower and, on a paid API, more
expensive. Measure where the gains flatten out.

</details>

<details>
<summary>One way to write it</summary>

In your copy, replace the `main()` call at the very bottom with:

```python
if __name__ == "__main__":
    df = pd.read_csv(HERE / "feedback.csv")
    train, test = train_test_split(df, test_size=40, random_state=66, stratify=df["topic"])
    tokenizer, model = load_model()
    shots = train.groupby("topic").sample(2, random_state=66).sample(frac=1, random_state=66)
    valid, acc, _ = evaluate(tokenizer, model, make_few_shot(shots), test, parse_word)
    show("8 examples: usable, accuracy", (valid, acc))
    picks = []
    for text in test["text"]:
        scores = answer_scores(tokenizer, model, clear_prompt(text))
        picks.append(max(scores, key=scores.get))
    show("clear prompt, scored: accuracy", round((pd.Series(picks, index=test.index) == test["topic"]).mean(), 3))
```

</details>

## 5. (Optional) The prompt is your code, so version it

Your team's prompt lives in a string in the code. Someone "improves"
the wording, and a week later customers complain. What would you have
wanted in place to catch this before it shipped?

<details>
<summary>Check yourself</summary>

The same things you'd want for any code: the prompt kept in version
control with a clear history, a test suite like `CASES` (including the
cases that once broke), and a fixed evaluation set with a recorded
score, all run automatically on every change. Note the model's name
and version too: the same prompt can behave differently on a newer
model, so changing the model is a change that needs the tests. Lesson
073 comes back to evaluating models properly.

</details>

---

That's prompting as programming. If one thing sticks: *a prompt that
hasn't been tested on real examples is a prompt you don't understand
yet.*
