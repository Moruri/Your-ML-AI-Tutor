# Lesson 059 - Embeddings: turning categories into geometry

**Phase 3 - Deep learning foundations** | Week 8, Day 2 | Tuesday 2026-10-27

> **Goal:** Learn dense representations for words and IDs, and inspect
> what they capture.

Time: about 40 minutes. Same venv as lesson 051, with `torch` installed.

---

Networks eat numbers. But a lot of what we care about isn't a number:
the day of the week, a station name, a customer ID, a word. These are
**categories**, and so far we've had one way to feed them in: one-hot
columns, from lesson 035.

Today we meet the better way. An **embedding** gives every category a
short list of numbers that the network *learns*, so each category
becomes a point in space. Train it on a task, and categories that behave
alike drift close together. The geometry ends up meaning something, and
you can read it.

This is one of the most important ideas in modern AI. Every word, every
token that goes into a large language model starts life as a row in an
embedding table.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-3-deep-learning/lesson-059-embeddings-turning-categories-into-geometry/lesson.py
```

It writes no files. Every random step has a fixed seed, so your numbers
will match. It takes a few seconds.

## 1. Why one-hot runs out of road

```
Monday, one-hot                                -> [1, 0, 0, 0, 0, 0, 0]
distance Mon-Tue, Mon-Sat, Sat-Sun             -> (1.414, 1.414, 1.414)
one-hot columns for a 50,000-word vocabulary   -> 50000
embedding columns for the same vocabulary      -> 64
```

One-hot gives Monday its own column and puts a 1 in it. It's honest:
it doesn't pretend Monday is "less than" Tuesday, the way coding days as
0 to 6 would. But look at the distances. Every pair of days is exactly
$\sqrt{2}$ apart. To one-hot, Saturday is as different from Sunday as it
is from Monday. Whatever the model learns about Saturday tells it
nothing about Sunday.

The second problem is size. A vocabulary of 50,000 words means 50,000
columns, all zeros but one. An embedding of 64 numbers per word is a
normal size, and it's dense: every number carries information.

## 2. `nn.Embedding` is a lookup table

```
weight shape (categories, dimensions)          -> (7, 2)
emb([Mon, Sat, Sat])                           -> [[-0.778, 0.403], [-0.981, -1.135], [-0.981, -1.135]]
one_hot @ weight gives the same rows           -> True
weights in Embedding(50_000, 64)               -> 3200000
```

```python
emb = nn.Embedding(num_embeddings=7, embedding_dim=2)
emb(torch.tensor([0, 5, 5]))      # rows 0, 5 and 5 of emb.weight
```

That's all an embedding is: a weight matrix with one row per category.
Give it category numbers and it hands back their rows. Saturday twice
gives the same row twice.

The third line is the important one. Multiplying a one-hot vector by the
weight matrix picks out exactly one row, so **an embedding is a linear
layer on one-hot inputs**, done without building the giant vector of
zeros. That's why it trains with the same backprop as everything else:
the gradient simply flows into the rows that were used.

The rows start random, like any weights. Training is what gives them
meaning. One catch: 3.2 million weights for a big vocabulary is a lot,
and each row only learns from the examples where its word appears.

## 3. Embeddings for IDs: when do people ride?

```
hours: train / test                            -> (504, 216)
baseline: average for that hour, test MAE      -> 1.095
embedding model, test MAE                      -> 0.953
learned day vectors (2 numbers each):
  Mon                                          -> [-0.844, 0.28]
  Tue                                          -> [-0.619, 0.394]
  Wed                                          -> [-1.181, 0.433]
  Thu                                          -> [-1.052, 0.081]
  Fri                                          -> [-0.805, -0.101]
  Sat                                          -> [0.603, -0.878]
  Sun                                          -> [0.119, -1.031]
closest to Sat                                 -> ['Sun', 'Fri']
closest to Tue                                 -> ['Mon', 'Fri']
closest to 08h                                 -> ['17h', '09h', '12h']
closest to 12h                                 -> ['13h', '09h', '08h']
closest to 03h                                 -> ['00h', '22h', '06h']
```

Back to our rides. The script counts rides in each of September's 720
hours (lesson 058's series) and gives the model just two facts about
each hour: the **day of the week** and the **hour of the day**, both as
category numbers. Each gets its own embedding with 2 numbers per
category, and the two vectors go side by side into a small network:

```python
x = torch.cat([self.day(day), self.hour(hour)], dim=1)
return self.head(x).squeeze(-1)
```

The output is the log of the expected number of rides, and the loss is
`nn.PoissonNLLLoss`, the standard choice for counts (it's built for
"how many times did this happen", where values are whole numbers, often
zero). We train on the first three weeks and test on the rest.

The baseline uses the average count for each hour, ignoring the day,
and misses by 1.10 rides per hour. The embedding model gets 0.95,
because it can tell a Tuesday 8am from a Sunday 8am.

Now the good part: **read the embeddings**. All five weekdays sit on the
left, with a negative first number. Saturday and Sunday sit off on their
own, and they're each other's closest neighbour. Nobody told the model
which days are the weekend. It worked that out because weekend riding
looks different (no 8am rush, a slow spread through the day), and the
cheapest way to predict that was to put those days near each other.

The hours did the same. 8am's nearest neighbour is 5pm, ten hours away
on the clock but the same kind of hour: a commuter rush. 12pm sits with
1pm, the lunch bump. 3am sits with midnight and 10pm: hours when nobody
rides. Distance in embedding space is about *behaviour*, not about the
label. (The embeddings start out tiny and close together on purpose,
with `std=0.1`, so the final positions come from training rather than
from where the random start happened to put them.)

## 4. Embeddings for words

```
sentences, vocabulary size                     -> (3000, 26)
two sentences                                  -> ['we buy bread in mombasa', 'we take the matatu to nakuru']
(word, neighbour) training pairs               -> 42184
most similar words (cosine similarity):
  cat                                          -> [('goat', 0.972), ('dog', 0.963), ('horse', 0.947)]
  mango                                        -> [('beans', 0.977), ('rice', 0.789), ('bread', 0.769)]
  kisumu                                       -> [('nairobi', 0.991), ('mombasa', 0.99), ('nakuru', 0.986)]
  matatu                                       -> [('train', 0.871), ('bike', 0.857), ('bus', 0.839)]
average similarity: same group, different      -> (0.933, -0.164)
```

Words work the same way, with one twist: what's the task? There's no
label for "what does *mango* mean". The trick, from a 2013 method called
**word2vec**, is to make one up. Take every word, and ask the model to
predict the words near it:

```python
pairs = (word, neighbour) for every neighbour within 2 words
logits = self.out(self.emb(word))           # a score for every word
loss = F.cross_entropy(logits, neighbour)
```

This is called **skip-gram**. The script makes 3,000 little sentences
from templates ("the *animal* eats *food*", "we take the *ride* to
*place*"), so we know the true groups. The model never sees the groups;
it only sees which words sit next to which.

To compare word vectors, we use **cosine similarity**: the cosine of the
angle between two vectors, from 1 (same direction) through 0 to -1
(opposite). After five passes through the pairs, *cat*'s closest words
are the other animals, *mango*'s are the other foods, the towns find the
towns and *matatu* finds the other ways to travel. On average, words in
the same group score 0.93; words from different groups score below zero.

The idea behind it is old and simple: *you shall know a word by the
company it keeps*. Words used in the same places end up with the same
neighbours to predict, so the cheapest solution is to give them similar
vectors. Real word embeddings are trained the same way, on billions of
sentences, and that's how "Nairobi" ends up near "Kampala".

## 5. What an embedding captures, and what it doesn't

```
a row that training never touched              -> [0.919, -0.981, -1.135]
cosine similarity, straight-line distance      -> (0.999, 2.002)
```

Three limits worth knowing.

**Unseen categories keep their random start.** A row only moves when
its category appears in training. A brand-new station, or a word that
never showed up, has meaningless numbers. Real systems keep a special
"unknown" row, or break rare words into smaller pieces (lesson 061).

**"Close" depends on how you measure.** The vectors `[1, 0]` and
`[3, 0.1]` point the same way (cosine 0.999) but sit 2 apart. Word
vectors are usually compared by cosine, because their length mostly
tracks how common the word is. For section 3's day vectors, straight-line
distance was the natural choice. Use whichever matches how the vectors
are used inside the model.

**An embedding learns the data, including its flaws.** It captures
whatever patterns help with the task. If the training text pairs
certain jobs with certain genders, the embeddings will too. They're a
mirror of the data, not of the truth, so look at them before trusting
them.

## What you can do now

- Explain why one-hot treats every pair of categories as equally
  different, and what an embedding does instead.
- Use `nn.Embedding`, and explain why it equals one-hot times a weight
  matrix.
- Build a model that embeds IDs (days, hours, stations) next to other
  inputs.
- Train word vectors with skip-gram, and find similar words with cosine
  similarity.
- Inspect learned embeddings, and know their limits.

## What to do now

1. Run `lesson.py`. In section 4, change the number of passes from 5 to
   1 and see how much less tidy the groups get.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   asks what happens when each day gets just one number.
3. Next weekday, lesson 060: the Phase 3 mini-project, training and
   debugging a real network. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Embeddings turn "which one" into "where". Once things have a place in
space, you can measure, compare and search them, and most of modern AI
is built on that.
