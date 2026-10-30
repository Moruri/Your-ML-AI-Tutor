# Lesson 065 - How large language models are trained

**Phase 4 - Modern AI: NLP, vision, generative and LLM basics** | Week 8, Day 5 | Friday 2026-10-30

> **Goal:** Understand pretraining, instruction tuning and preference
> tuning at the intuition level.

Time: about 45 minutes. Same venv as lesson 051, with `torch` installed.

---

Lesson 063 ended with three families of Transformer, and promised that
the decoders, the GPT kind, would get their own lesson. Here it is.

Every chat model you've used was made in three stages:

1. **Pretraining.** Read an enormous pile of text and learn one thing:
   guess the next token. This is where almost all the compute goes, and
   where the model picks up language and facts.
2. **Instruction tuning.** Show it worked examples of requests and good
   answers, written by people, so it learns to *answer* rather than just
   carry on writing.
3. **Preference tuning.** Show it pairs of answers and which one people
   liked better, so it leans towards the better behaviour.

The real versions cost millions. Today we run all three in miniature on
a laptop: a 113,000-parameter model that reads our rider messages one
character at a time. It's a toy, but each stage is the real recipe, and
you'll see each one change the model's behaviour in its own way.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-4-modern-ai/lesson-065-how-large-language-models-are-trained/lesson.py
```

It reads `feedback.csv` (the same 800 made-up rider messages as lessons
061 and 064, copied into this folder) and writes no files. Every random
step has a fixed seed, so your numbers will match. It takes about two
minutes.

## 1. The only task: guess the next token

```
characters of text                             -> 53550
vocabulary size (distinct characters)          -> 40
One sentence is many small exercises:
  'the '               -> next is 'a'
  'the app '           -> next is 'c'
  'the app crashe'     -> next is 'd'
loss of a blind guess, ln(vocab)               -> 3.689
```

Real models use subword tokens like the ones you met in lesson 064. To
keep things small we use the simplest tokenizer there is: **one token
per character**, lower-cased. That gives a vocabulary of 40.

The training task is the same either way. Take any text, and at every
position ask: given everything so far, what comes next? One sentence of
15 characters is 14 little exercises.

Notice what's missing: **labels**. The text is its own answer key. That
is the whole reason pretraining can scale; any text anyone ever wrote is
training data, with no one paid to label it.

A model that knows nothing gives every character equal odds, which is a
cross-entropy loss (the one our Phase 3 classifiers used) of ln(40), about 3.69. That's our
starting line.

## 2. Pretraining a tiny GPT

```
parameters                                     -> 113448
before training, it writes                     -> 'hdv02 coubw-kron!e0 2an:tswxvuoy3al1svub ((v!xixfwok!idbthv-'
held-out loss at steps 0, 500, 1000, 1500      -> [(0, 3.817), (500, 0.422), (1000, 0.322), (1500, 0.304)]
after training, 'the ' continues               -> 'service though.'
after training, 'my ' continues                -> 'account was charged for 3 hours instead of 20 minutes. thank'
after training, 'the app ' continues           -> "won't load."
characters it has read                         -> 6144000
```

`TinyGPT` is lesson 063 almost unchanged: a token embedding, a learned
position embedding, two Transformer blocks, a final layer norm and a
linear layer that scores all 40 characters. The one important addition
is lesson 062's **causal mask**, so position 10 can only look at
positions 0 to 9. Without it the model could peek at the answer.

We train on the first 700 messages and keep the last 100 hidden to
measure honestly. Each step takes 32 random windows of 128 characters,
and the loss is the average next-character loss across every position
in every window, so one step is about 4,000 exercises at once.

The held-out loss drops from 3.8 to 0.3, and the writing goes from
random keys to things a rider might actually say. Our messages come
from a small set of patterns, so it learns them fast; real text is far
less predictable, and real losses stay much higher.

Look at what it learned without being told: spelling, punctuation,
station names, and the rhythm of a complaint ("...so annoying.",
"...otherwise fine."). Look also at what it didn't learn: it can't
answer a question or follow an instruction. It only ever continues
text. A pretrained model, often called a **base model**, is a brilliant
autocomplete and nothing more.

## 3. Sampling: the model gives probabilities, not words

```
after 'the ', temperature 0.5: top 4           -> [('r', 0.474), ('s', 0.196), ('m', 0.1), ('d', 0.059)]
after 'the ', temperature 1.0: top 4           -> [('r', 0.24), ('s', 0.154), ('m', 0.11), ('d', 0.085)]
after 'the ', temperature 2.0: top 4           -> [('r', 0.13), ('s', 0.104), ('m', 0.088), ('d', 0.077)]
temperature 0.5, three tries                   -> ['service though.', 'rain. please fix!', 'rain. otherwise fine.']
temperature 1.0, three tries                   -> ['service though.', 'market. :(', 'rain. otherwise fine.']
temperature 2.0, three tries                   -> ['suck at uni square hadnh', 'map in the app 8am, and ', 'notifications is so byid']
```

The model never outputs a word. It outputs a probability for each
possible next token, and `generate` rolls weighted dice to pick one,
then repeats. That's why the same prompt can give different replies.

**Temperature** divides the scores before the softmax:

- Below 1, the favourite gets more likely (`r` goes from 0.24 to 0.47).
  Replies get safer and more repetitive.
- Above 1, the odds flatten out. At 2.0 the replies stop being words.

Temperature 0 is the extreme case: always take the top choice, which is
called **greedy** decoding. Lesson 066 uses it so its numbers are the
same every run.

## 4. Instruction tuning: learn from worked examples

```
one worked example                             -> 'q: station road had a broken dock on monday. otherwise fine.\na: station, sorry about that!\n'
examples, and share with the polite reply      -> (600, 0.5)
question                                       -> 'Quick note: Station Road was blocked by cars at rush hour, and the saddle is broken. Otherwise fine.'
base model answers                             -> 'the front brake squeaks on bi'
instruction-tuned model answers                -> 'app, not our fault.'
100 new questions: right topic, polite         -> (0.62, 0.55)
```

Now we want a model that does a job: read a rider's message and reply
with its topic and a short response. Ask the base model, and it just
writes another complaint.

So we write **worked examples** in a fixed format, `q:` then the
message, `a:` then the answer, and keep training on them. That's
instruction tuning, also called **supervised fine-tuning** or SFT. Two
details matter:

- **Loss on the answer only.** `answer_loss` scores just the characters
  after `a:`. We want the model to learn how to reply, not how to write
  more questions.
- **Same model, same next-token task.** Nothing about the architecture
  changes. We only changed what text it practises on.

After 600 examples the model answers in the right format every time,
and picks the right topic 62% of the time on 100 messages it has never
seen. That's well short of lesson 061's TF-IDF classifier, because our
model is tiny, and a message that names two problems (like this one,
where the answer should be `station`) easily confuses it.

But look at the politeness: 55%. We wrote our examples like a real team
would, with different people writing them, and half of them signed off
with `not our fault.` The model copied that faithfully. **An
instruction-tuned model does what its examples did**, the good and the
bad.

## 5. Preference tuning: learn which answer people liked

```
one preference pair: prompt                    -> 'q: station road had a broken dock on monday. otherwise fine.\na:'
  chosen, rejected                             -> (' station, sorry about that!\n', ' station, not our fault.\n')
log-prob gap, chosen minus rejected: before    -> -0.793
log-prob gap, chosen minus rejected: after     -> 79.08
100 new questions: right topic, polite         -> (0.6, 0.98)
```

Writing perfect examples is slow and expensive. *Comparing* two answers
and picking the better one is quick, even for non-experts. So the third
stage learns from **preference pairs**: a prompt, a chosen answer and a
rejected one.

We use **DPO**, Direct Preference Optimisation (2023), because it fits
in a dozen lines. `seq_logprob` measures how likely the model finds a
whole answer. DPO compares the gap between chosen and rejected for the
model being trained against the same gap for a frozen copy, the
**reference** model, and pushes the first gap up:

```python
gap = logp(chosen) - logp(rejected)            # model being trained
ref_gap = logp_ref(chosen) - logp_ref(rejected)  # frozen copy
loss = -logsigmoid(beta * (gap - ref_gap))
```

Measuring against the frozen copy keeps the model anchored to what it
already knew. We also add the plain answer loss on the chosen replies.
On its own, DPO happily pushes the rejected answer so far down that
everything else gets pulled out of shape with it (exercise 3 asks why);
this small extra term is a common fix in practice.

After 40 steps the gap goes from slightly negative to 79, meaning the
rude reply is now astronomically less likely. On 100 new messages the
model is polite 98% of the time, and still gets the topic right 60% of
the time, about where it was. Nobody wrote a single new answer.

The older method you'll read about is **RLHF**, reinforcement learning
from human feedback. It uses the same pairs to train a separate
**reward model** that scores answers, then trains the chat model to get
high scores. More moving parts, same goal.

## 6. The same recipe, at real scale

```
our model: parameters, tokens seen             -> (113448, 6144000)
our tokens per parameter                       -> 54.2
GPT-3 (2020): parameters, tokens               -> ('175 billion', '300 billion')
GPT-3 tokens per parameter                     -> 1.7
Chinchilla (2022): parameters, tokens          -> ('70 billion', '1.4 trillion')
Chinchilla tokens per parameter                -> 20.0
```

Our model saw about 6 million tokens (characters, here, counting
overlaps between windows). GPT-3 had 175 billion parameters and was
trained on 300 billion tokens, about 1.7 tokens per parameter.

The **Chinchilla** paper (2022) asked how to split a fixed compute
budget between model size and training text. Its answer: GPT-3-era
models were too big for the text they saw, and roughly **20 tokens per
parameter** was the better balance. Its 70-billion-parameter model,
trained on 1.4 trillion tokens, beat much larger models. Since then,
many models have trained well past 20 tokens per parameter, because a
smaller model that's been trained longer is cheaper to *run*.

The stages scale up the same way they worked here:

| Stage | Data | Ours | Real |
|-------|------|------|------|
| Pretraining | Raw text, no labels | 700 messages | Trillions of tokens from the web, books and code |
| Instruction tuning | Written examples | 600 | Tens of thousands to millions |
| Preference tuning | Chosen vs rejected pairs | 600 | Hundreds of thousands |

## What you can do now

- Explain next-token prediction, and why it needs no labels.
- Say what a base model can and can't do.
- Describe how temperature changes sampling, and what greedy means.
- Explain instruction tuning, and why the loss covers only the answer.
- Explain preference tuning, what DPO compares, and why it keeps a
  reference model.
- Put rough numbers on the scale of real training runs.

## What to do now

1. Run `lesson.py`. In section 2, change `steps, losses = 1500, []` to
   `300` and watch how much worse the samples and the later stages get.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   asks whether careful examples could have replaced preference tuning.
3. Next, lesson 066: prompting as programming, where we take a real
   (small) chat model and write prompts the way we'd write code: with a
   spec, examples, checked output and tests.

A chat model is a next-token predictor that first read everything, then
copied good examples, then learned which answers people prefer. Each
stage uses the same simple loss on different data.
