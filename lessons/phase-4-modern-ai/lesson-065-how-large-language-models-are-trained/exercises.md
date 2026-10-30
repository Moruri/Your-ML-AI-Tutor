# Lesson 065 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv (with `torch` installed, as in lesson 051). For the
hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-4-modern-ai/lesson-065-how-large-language-models-are-trained/lesson.py my_lesson_065.py
cp lessons/phase-4-modern-ai/lesson-065-how-large-language-models-are-trained/feedback.csv .
python my_lesson_065.py
```

The script reads `feedback.csv` from its own folder, which is why the
second line copies it next to your copy. `section_task`,
`section_pretrain`, `to_example`, `finetune` and `judge` are there to
reuse.

---

## 1. Why the causal mask?

During pretraining, what would go wrong if we left out the causal mask
in `TinyGPT.forward`?

<details>
<summary>Check yourself</summary>

Every position could look at the characters after it, including the
very one it's meant to predict. The training loss would fall almost to
zero, because the model would just learn to copy the next character
from its input. Then, when generating, there's no "next character" to
peek at yet, and it would write nonsense. The mask makes training match
how the model is used: only the past is visible.

</details>

## 2. Base model or chat model?

You download a model called `something-7B` and a sibling called
`something-7B-Instruct`. You ask each "What's the capital of Kenya?".
What do you expect from each, and why?

<details>
<summary>Check yourself</summary>

The **base** model has only been pretrained. It continues text, so it
might answer, or it might carry on with more quiz questions ("What's the
capital of Uganda? What's the capital of...") because that's what
usually follows a question like this on the web. The **Instruct** model
has been through instruction tuning (and usually preference tuning), so
it answers: Nairobi. Same architecture, same pretraining, different
last stages, just like our section 2 and section 4 models.

</details>

## 3. Why the extra term in DPO?

In section 5 we add the plain answer loss on the chosen replies to the
DPO loss. Look at what DPO actually rewards: a bigger *gap* between
chosen and rejected. What's the cheapest way for the model to grow that
gap, and why is it a problem?

<details>
<summary>Check yourself</summary>

The gap grows just as well by pushing the rejected answer's probability
down as by pushing the chosen one up, and pushing down is easy: make
the whole reply less likely. Taken far enough, the chosen answer gets
less likely too, and the model drifts away from sensible replies
altogether. When we tried DPO on its own here, topic accuracy fell
from 0.62 to between 0.14 and 0.44, depending on the settings. The extra term says "and keep the
chosen answer likely", which anchors it. The frozen reference model is
another anchor of the same kind.

</details>

## 4. Hands-on: just write better examples?

If the problem was rude examples, why not drop them? Train the
instruction-tuned model on only the polite examples (every other one of
the 600, so 300), with the same settings: `torch.manual_seed(65)`, 600
steps, learning rate `1e-3`. Then `judge` it on the same 100 test
messages. Predict first: how polite, and how accurate on topics,
compared with sections 4 and 5?

<details>
<summary>Expected results</summary>

```
section 4, 600 mixed examples:          right topic 0.62, polite 0.55
section 5, plus preference tuning:      right topic 0.6,  polite 0.98
300 polite examples only:               right topic 0.49, polite 0.99
```

Clean examples fix politeness just as well. But we threw away half our
examples, and the rude ones still carried useful information about
topics, so topic accuracy drops. Preference tuning kept everything the
model learned and only changed the behaviour people disliked. In real
life there's another reason too: you rarely know in advance what's
wrong with your examples. Comparing two answers is how you find out.

</details>

<details>
<summary>One way to write it</summary>

In your copy, replace the `main()` call at the very bottom with:

```python
if __name__ == "__main__":
    df = pd.read_csv(HERE / "feedback.csv")
    text, tok = section_task(df)
    base, _ = section_pretrain(df, tok)
    demo, test = df.iloc[:600], df.iloc[700:]
    pairs = [to_example(t, c, POLITE) for t, c in zip(demo["text"][::2], demo["topic"][::2])]
    torch.manual_seed(65)
    sft = finetune(copy.deepcopy(base), tok, pairs, steps=600, lr=1e-3)
    show("300 polite examples: right topic, polite", judge(sft, tok, test))
```

</details>

## 5. (Optional) Where do the facts come from?

Our tiny model "knows" that Station Road and Uni Square exist, but it
was never told. Big chat models know a lot about the world. Which of the
three stages gives them most of that knowledge, and what does that
suggest about asking them for facts?

<details>
<summary>Check yourself</summary>

Pretraining. The facts are whatever patterns were common in the text
it read, stored only as next-token probabilities. Instruction and
preference tuning mostly teach format and behaviour, not new facts.
That has two consequences: a model can't know about anything after its
training text was collected, and for rare facts it may produce
something that *sounds* right with nothing behind it. When accuracy
matters, give the model the facts in the prompt instead. That's
retrieval-augmented generation, lesson 068.

</details>

---

That's how a chat model is made. If one thing sticks: *it's the same
next-token model all the way through; each stage just changes what text
it learns from.*
