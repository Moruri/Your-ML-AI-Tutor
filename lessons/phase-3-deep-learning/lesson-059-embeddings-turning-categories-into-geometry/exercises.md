# Lesson 059 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-3-deep-learning/lesson-059-embeddings-turning-categories-into-geometry/lesson.py my_lesson_059.py
python my_lesson_059.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-3-deep-learning/lesson-059-embeddings-turning-categories-into-geometry").resolve()
```

`hourly_table`, `RideRate`, `nearest`, `SkipGram`, `most_similar`,
`DAYS`, `r3` and `show` are there to reuse.

---

## 1. Counting weights

A shop has 20,000 products and you give each one an embedding of 32
numbers. How many weights is that? How many columns would one-hot need?

<details>
<summary>Check yourself</summary>

20,000 x 32 = 640,000 weights, in a table of 20,000 rows. One-hot needs
20,000 columns, every one of them zero except one. The embedding is
also *shared knowledge*: two products bought by the same people end up
with similar rows, so what the model learns about one helps with the
other. One-hot columns never share anything.

</details>

## 2. Lookup or multiply?

Section 2 showed `one_hot @ emb.weight` equals `emb(idx)`. Why does
every library use the lookup instead?

<details>
<summary>Check yourself</summary>

Speed and memory. For a batch of 64 words and a vocabulary of 50,000,
the one-hot matrix has 3.2 million entries, nearly all zero, and the
multiply does 50,000 x 64 sums per word to pick out a single row. The
lookup just copies 64 numbers. Same answer, same gradients, a tiny
fraction of the work.

</details>

## 3. Reading the map

In section 3, 8am's closest hour was 5pm, not 7am or 9am. Is the model
confused?

<details>
<summary>Check yourself</summary>

No, it's right. The embedding is trained to predict ride counts, so
"close" means "similar riding", not "close on the clock". 8am and 5pm
are both commuter rushes on weekdays and quiet on weekends. 7am, by
contrast, is nearly empty every day. If you wanted the model to know
the clock as well, you'd add lesson 058's sine and cosine hour
features next to the embedding.

</details>

## 4. Hands-on: one number per day

Section 3 gave each day 2 numbers. Train the same model with **one**
number per day and hour: `RideRate(dim=1)`, seed 59, the same
three-week split, `Adam` with `lr=0.02`, `PoissonNLLLoss(log_input=True)`,
400 full-batch steps. Report the test MAE and the seven day values.
Predict first: will a single number be enough to separate the weekend?

<details>
<summary>Expected results</summary>

```
test MAE (dim=1)  -> 0.932      (dim=2 gave 0.953)
day values        -> Mon -0.406, Tue -0.397, Wed -0.345, Thu -0.329,
                     Fri -0.271, Sat 0.646, Sun 1.312
```

One number is plenty, and it scores slightly better here. The five
weekdays bunch together at one end of a line, Saturday and Sunday sit
far off at the other, and Friday is the weekday leaning closest to the
weekend. With only 21 days to learn from, the smaller model has less
room to fit noise. Embedding size is a hyperparameter like any other:
pick it by validation, and start small when there's little data. A
common rule of thumb for categories is something near the fourth root
of the number of categories, rounded up, and words need much more.

</details>

<details>
<summary>One way to write it</summary>

```python
day, hour, y = hourly_table()
tr, te = slice(0, 21 * 24), slice(21 * 24, None)
torch.manual_seed(59)
model = RideRate(dim=1)
opt = torch.optim.Adam(model.parameters(), lr=0.02)
loss_fn = nn.PoissonNLLLoss(log_input=True)
for epoch in range(400):
    opt.zero_grad()
    loss_fn(model(day[tr], hour[tr]), y[tr]).backward()
    opt.step()
with torch.no_grad():
    pred = model(day[te], hour[te]).exp()
print(r3((pred - y[te]).abs().mean()))
print({d: r3(v) for d, v in zip(DAYS, model.day.weight[:, 0])})
```

</details>

## 5. (Optional) The famous analogy

Word vectors trained on huge amounts of text famously satisfy
*king - man + woman ≈ queen*. In words, not code, why might subtracting
vectors do that?

<details>
<summary>Check yourself</summary>

If "king" and "queen" are used in the same contexts except for the
ones that go with gender ("he", "she", "his", "her"), and "man" and
"woman" differ in those same contexts, then the difference between the
vectors captures roughly the same direction in both pairs. Subtracting
"man" removes that direction, adding "woman" puts the other one in,
and the nearest word left is "queen". It works some of the time, not
always, and it's a vivid reminder from section 5: those directions
come from the text, flaws and all.

</details>

---

That's embeddings. If one thing sticks: *an embedding is a learned
table with one vector per category; training puts categories that
behave alike close together, so the geometry tells you what the model
learned.*
