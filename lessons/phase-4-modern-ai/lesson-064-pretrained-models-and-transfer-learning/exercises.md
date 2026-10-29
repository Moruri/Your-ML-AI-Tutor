# Lesson 064 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv (with `transformers` installed, as in the README).
For the hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-4-modern-ai/lesson-064-pretrained-models-and-transfer-learning/lesson.py my_lesson_064.py
cp lessons/phase-4-modern-ai/lesson-064-pretrained-models-and-transfer-learning/feedback.csv .
python my_lesson_064.py
```

The script reads `feedback.csv` from its own folder, which is why the
second line copies it next to your copy. `load_data`, `tfidf_baseline`,
`embed`, `fine_tune`, `predict`, `r3` and `show` are there to reuse.

---

## 1. The matching tokenizer

Why must you always load the tokenizer that belongs to the model, and
not some other one?

<details>
<summary>Check yourself</summary>

The model learned what each token *id* means. In this model's
tokenizer, id 5,338 is "charged"; in another tokenizer that id could be
any word at all, so the wrong tokenizer feeds the model nonsense. It won't
crash; it'll just quietly perform badly, which is worse. Loading both
with the same name, as `from_pretrained(MODEL)` does, keeps them in
step.

</details>

## 2. Why such a small learning rate?

Training from scratch, as in lesson 063's tiny Transformer, used a
learning rate of `3e-3`. Fine-tuning here uses `1e-4`. Why?

<details>
<summary>Check yourself</summary>

The pretrained weights already encode a lot of useful knowledge. Big
steps would overwrite it with whatever fits 20 examples, a problem
called **catastrophic forgetting**, and you'd be back near the
random-weights result. Small steps nudge the model towards your task
while keeping most of what it knew. Typical fine-tuning learning rates
are around `1e-5` to `1e-4`.

</details>

## 3. Frozen or fine-tuned?

Frozen embeddings scored 0.905 and fine-tuning 0.925. Give one reason
you might still pick the frozen approach.

<details>
<summary>Check yourself</summary>

A few good ones: it's much cheaper to train (a logistic regression
fits in milliseconds, with no GPU); one frozen model can serve many
tasks at once, since you just add another small classifier per task;
and with very few labels, there's less risk of the big model
overfitting them. Fine-tuning usually wins on accuracy when you have
enough labels to make it worth the cost.

</details>

## 4. Hands-on: more labels, fewer labels

Re-run the comparison with `PER_TOPIC = 2` (8 labels) and then
`PER_TOPIC = 20` (80 labels). For each, train the TF-IDF baseline and a
fine-tuned model (same seed, `torch.manual_seed(640)`, same 20 steps)
and measure accuracy on the same 200 test messages. Predict first:
does the gap grow or shrink with more labels?

<details>
<summary>Expected results</summary>

```
 8 labels:  TF-IDF 0.65   fine-tuned 0.74
20 labels:  TF-IDF 0.785  fine-tuned 0.925   (the lesson)
80 labels:  TF-IDF 0.95   fine-tuned 1.0
```

Both improve with labels, and the fine-tuned model is ahead at every
size. But look at TF-IDF at 80 labels: 0.95, from a model of a few
hundred numbers that trains in a blink. The pretrained model's
advantage is biggest exactly where labels are scarce. As labels pile
up, the cheap baseline closes in, which is the trade-off from section
6.

</details>

<details>
<summary>One way to write it</summary>

In your copy, replace the `main()` call at the very bottom with:

```python
if __name__ == "__main__":
    for k in (2, 20):
        PER_TOPIC = k            # load_data reads this setting
        few, test, topics = load_data()
        vec, clf = tfidf_baseline(few, test)
        tfidf = r3((clf.predict(vec.transform(test["text"])) == test["topic"]).mean())
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        torch.manual_seed(640)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(topics))
        model = fine_tune(model, tokenizer, few, topics)
        preds = pd.Series(predict(model, tokenizer, test["text"], topics), index=test.index)
        show(f"{4 * k} labels: TF-IDF, fine-tuned", (tfidf, r3((preds == test["topic"]).mean())))
```

</details>

## 5. (Optional) Leaky test sets

Our model was pretrained by someone else, on text we never looked at.
Why could that be a problem for evaluating it honestly, and what would
you do about it?

<details>
<summary>Check yourself</summary>

If the pretraining text happened to include your test examples (or
near-copies of them), the model may have effectively seen the answers,
and your test score would look better than real-world performance.
The rule from Phase 2 still holds: test data must be unseen. With public
benchmarks this "contamination" is a real worry for large models. The
safest protection is a test set built from your *own* fresh data,
collected after the model was trained. Our feedback messages are made
up for this course, so we're safe here. Lesson 073 comes back to this.

</details>

---

That's transfer learning. If one thing sticks: *start from a model that
already knows language, adapt it gently with the few labels you have,
and always compare it with a simple baseline.*
