# Lesson 064 - Pretrained models and transfer learning

**Phase 4 - Modern AI: NLP, vision, generative and LLM basics** | Week 8, Day 4 | Thursday 2026-10-29

> **Goal:** Load a pretrained model from Hugging Face and fine-tune it on
> a small task.

Time: about 50 minutes. Same venv as lesson 051, plus one new install
(below) and a one-off download of about 90 MB.

---

Lesson 063 ended with GPT-2's 124 million parameters. Training numbers
like that from scratch takes huge amounts of text, weeks of hardware and
a budget most of us don't have. The good news: you almost never need to.

People who did train these models have shared them. A model trained on
a billion sentences has already learned a great deal about language:
that "refund" is about money, that "log in" is about an app, that
"wobbly" and "loose" mean similar things. **Transfer learning** means
taking that model and teaching it *your* small task on top. The
pretrained knowledge transfers; you only supply the last bit.

Today we use lesson 061's rider-feedback messages again, but with a
twist. Instead of 600 labelled examples, we pretend we only have
**20**, five per topic. Labelling is the slow, expensive part of real
projects, so this is the situation you'll actually be in.

## Setting up

This lesson needs Hugging Face's `transformers` library. With your venv
active:

```bash
pip install "transformers>=4.40,<5"
```

(It's also in [`requirements.txt`](../../../requirements.txt) now, so
`pip install -r requirements.txt` works too.)

The first time you run the script it downloads the model,
`sentence-transformers/all-MiniLM-L6-v2`, from the
[Hugging Face Hub](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
into a cache folder in your home directory (`~/.cache/huggingface`).
That needs an internet connection once; every later run uses the cache.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-4-modern-ai/lesson-064-pretrained-models-and-transfer-learning/lesson.py
```

It reads [`feedback.csv`](feedback.csv), the same 800 made-up messages
as lesson 061, and writes no files. Seeds are fixed, so your numbers
should match; timings in sections 4 and 6 will depend on your machine.
It takes about half a minute.

## 1. Only twenty labels

```
labelled training messages                     -> 20
per topic                                      -> {'app': 5, 'bike': 5, 'payment': 5, 'station': 5}
test messages                                  -> 200
TF-IDF + logistic regression: accuracy         -> 0.785
words it knows                                 -> 124
```

First, the baseline. Lesson 061's TF-IDF plus logistic regression
scored 0.99 with 600 labels. Train it on 20 and it drops to 0.785.

The reason is in the last line. From 20 messages it learns only 124
words. Any word outside that list is simply invisible to it: a message
about a "refund" means nothing if "refund" never came up. TF-IDF knows
only what you show it. A pretrained model arrives knowing far more.

## 2. Loading a pretrained model

```
model                                          -> 'sentence-transformers/all-MiniLM-L6-v2'
parameters (millions)                          -> 22.7
layers, heads, numbers per token               -> (6, 12, 384)
vocabulary size                                -> 30522
tokens                                         -> ['[CLS]', 'honestly', ',', 'i', 'was', 'charged', 'twice', 'via', 'm', '-', 'pe', '##sa', '.', 'so', 'annoying', '.', '[SEP]']
output shape (batch, tokens, numbers)          -> (1, 17, 384)
```

Two lines load everything:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
```

The "Auto" classes read the model's settings from the Hub and build the
right architecture for it. Every pretrained model comes as this pair:

- The **tokenizer** turns text into token ids, exactly the way the model
  saw text during its training. You must use the matching one.
- The **model** is a Transformer encoder: six of lesson 063's blocks,
  12 heads, 384 numbers per token, 22.7 million parameters.

Look at the tokens. This tokenizer splits text into **subwords**: common
words are one token, and rare ones get broken into pieces it does know.
"M-pesa" isn't in its 30,522-token vocabulary, so it becomes `m`, `-`,
`pe`, `##sa` (the `##` means "glued to the piece before"). Nothing is
ever unknown; at worst it's spelled out in pieces. `[CLS]` and `[SEP]`
are special tokens added at the start and end.

The output is one 384-number vector per token, each one shaped by
attention over the whole message.

## 3. Frozen features: use it as it comes

```
similarity: billed twice vs m-pesa twice       -> 0.289
similarity: billed twice vs flat tyre          -> 0.087
same pair with TF-IDF vectors                  -> 0.0
frozen embeddings + logistic regression        -> 0.905
```

The simplest way to use a pretrained model is to not train it at all.
Average its token vectors (ignoring padding, thanks to the
`attention_mask`) and you get one vector per message, an **embedding**
of its meaning. The model stays **frozen**: we never change a weight.

"My card was billed two times" and "m-pesa took the money twice" share
no words, so TF-IDF says their similarity is 0. The pretrained model
sees two complaints about being overcharged and puts them closer
together than either is to a flat tyre.

Then do what you'd do with any features (lesson 033): fit a logistic
regression on the 20 labelled embeddings. Accuracy jumps from 0.785 to
0.905. Same 20 labels; better features.

This "embed, then use a simple model" pattern is cheap and powerful.
Lesson 067 builds a search tool on it.

## 4. Fine-tuning, and what pretraining is worth

```
new layer added on top                         -> 'Linear(in_features=384, out_features=4, bias=True)'
pretrained body, untrained head: accuracy      -> 0.245
same model from random weights: accuracy       -> 0.71
fine-tuned pretrained model: accuracy          -> 0.925
seconds for both training runs                 -> 20
```

**Fine-tuning** goes one step further: let training adjust *every*
weight, not just a classifier on top.

```python
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=4)
```

This loads the same pretrained body and adds a fresh linear layer that
maps its 384 numbers to our 4 topics. That layer starts random, so
before training the model guesses: 0.245, chance level. Then it's an
ordinary PyTorch training loop from lesson 052, 20 steps over the 20
examples, with a small learning rate (`1e-4`) so we nudge the
pretrained weights rather than wreck them.

Now the experiment that shows what pretraining is worth. Build the
*same architecture* from **random weights** and train it the same way:

- From random weights: 0.71. It can't even beat TF-IDF. Twenty
  examples are nowhere near enough to learn language from scratch.
- From pretrained weights: 0.925, the best result in this lesson.

Identical code, data and training. The only difference is where the
weights started. That gap is transfer learning.

## 5. Words it was never shown

```
the right topics                               -> ['payment', 'station', 'app', 'bike']
TF-IDF predicts                                -> ['bike', 'app', 'payment', 'bike']
fine-tuned model predicts                      -> ['payment', 'station', 'app', 'app']
```

Four brand-new messages, each built around a word missing from the 20
training examples: "refund still not received", "every dock near the
mall was full", "I can't log into my account", "the handlebar is loose".

TF-IDF gets one right. The fine-tuned model gets three: it knows a
refund is about payment and logging in is about the app, without ever
being shown either. It also gets "handlebar" wrong, which is worth
remembering. Twenty labels is still very few, and pretrained models
make mistakes that look silly to us. Always check on real examples.

## 6. What you pay for it

```
TF-IDF model size (numbers)                    -> 500
fine-tuned model size (millions of numbers)    -> 22.7
milliseconds per message: TF-IDF, fine-tuned   -> (0.01, 5.8)
```

Nothing's free. The TF-IDF model is 500 numbers. The fine-tuned one is
22.7 million, tens of thousands of times bigger, and on this laptop it's
hundreds of times slower per message (your timings will differ). This
one is *small* by modern standards, too.

So which should you use? It depends, and that's the honest answer:

- **Few labels, varied wording:** pretrained wins clearly, as today.
- **Lots of labels, simple text:** lesson 061's baseline hit 0.99 in a
  second. A Transformer might add little, at a real cost in speed,
  memory and hosting.
- **In between:** frozen embeddings plus a simple classifier is often
  the sweet spot.

Measure both on your own test set, then choose. Never assume the big
model wins.

## What you can do now

- Explain transfer learning: start from a model trained on lots of
  data, then adapt it to your task.
- Load a tokenizer and model from Hugging Face with the `Auto` classes,
  and read their tokens and outputs.
- Use a frozen model as a feature extractor with mean pooling.
- Fine-tune a pretrained model with a new classification head.
- Show what pretraining is worth by comparing with random weights.
- Weigh accuracy against size and speed when choosing a model.

## What to do now

1. Run `lesson.py`. In section 4, change `steps=20` to `steps=5` and to
   `steps=60`, and watch what happens to the fine-tuned accuracy.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   changes how many labels you get.
3. Next weekday, lesson 065: how large language models are trained, from
   pretraining on the internet to instruction tuning. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Don't start from zero when someone has already done the expensive part.
Borrow a pretrained model, add a small head, fine-tune it gently on the
little data you have, and check it honestly against a simple baseline.
