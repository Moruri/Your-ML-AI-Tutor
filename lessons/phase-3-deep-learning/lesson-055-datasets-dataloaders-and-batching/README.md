# Lesson 055 - Datasets, DataLoaders and batching

**Phase 3 - Deep learning foundations** | Week 7, Day 5 | Friday 2026-10-23

> **Goal:** Feed data to a model efficiently, shuffle it correctly, and
> split it cleanly.

Time: about 40 minutes. Same venv as lesson 051, with `torch` installed.

---

Lesson 054's schedule section did something new: instead of learning
from all 400 ring points at every step, it picked 16 at random with
`torch.randint`. That's how almost every real network trains, and I
promised we'd do it properly. Doing it by hand gets messy quickly: you
have to remember to shuffle, to cover every example, to handle the
leftovers at the end. torch splits the job into two small tools:

| Tool | Its one job | You write |
|------|-------------|-----------|
| **`Dataset`** | Say how many examples there are, and hand back example number `i` | `__len__` and `__getitem__` |
| **`DataLoader`** | Shuffle the examples and serve them in batches, epoch after epoch | Nothing: you just configure it |

By the end you'll have the training loop you'll use for the rest of the
course: `for epoch ...: for xb, yb in loader: ...`.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-3-deep-learning/lesson-055-datasets-dataloaders-and-batching/lesson.py
```

It writes no files. Every random step has a fixed seed, so your numbers
will match. It takes about ten seconds.

## 1. Why batches at all

```
weights in RingNet                             -> 33
examples                                       -> 400
batch size: average error vs full gradient     -> ''
  batch of 1                                   -> 0.902
  batch of 16                                  -> 0.193
  batch of 64                                  -> 0.1
  batch of 256                                 -> 0.032
full gradient's own size                       -> 0.225
```

The "true" gradient is the one you get from all the data. With 400
points that's cheap. With 4 million images it isn't: one step would mean
pushing every image through the network. So instead we estimate the
gradient from a **mini-batch**, a random handful of examples.

The script measures how good that estimate is. It takes lesson 052's
`RingNet` at its starting weights, works out the full gradient, then
compares it with gradients from 200 random batches of each size. The
numbers are the average distance between the two.

A single example (batch of 1) is wildly off: an error of 0.9 when the
real gradient is only 0.225 long. It's pointing almost anywhere. A batch
of 16 is much better, and 64 better again. Notice the pattern: going
from 16 to 64 examples, four times more, roughly halves the error. Noise
shrinks with the *square root* of the batch size, the same rule behind
lesson 026's confidence intervals, so each halving costs four times the
work.

That's the trade. Small batches: cheap, noisy steps, lots of them.
Big batches: accurate, expensive steps, fewer of them. Somewhere between
16 and 256 is where most people start, and the noise turns out to be
partly helpful, as section 6 shows.

## 2. A Dataset is two methods

```
len(ds)                                        -> 1176
ds[0] features                                 -> [16.5, 4.59, 0.0, 7.0, 21.4, 0.0]
ds[0] label                                    -> 1.0
ds[-1] label                                   -> 0.0
TensorDataset(ring): len, ring[0]              -> (400, [0.92, 1.068], 0.0)
```

A `Dataset` is any object that answers two questions. Here's the rides
one, doing the feature prep from lesson 052 once, up front:

```python
from torch.utils.data import Dataset

class RidesDataset(Dataset):
    def __init__(self, csv_path):
        rides = pd.read_csv(csv_path, parse_dates=["started_at"])
        ...                                   # add weekend and hour
        self.X = torch.tensor(rides[RIDE_FEATURES].to_numpy(dtype=float), dtype=torch.float32)
        self.y = torch.tensor((rides["rider_type"] == "casual").to_numpy(dtype=float),
                              dtype=torch.float32)

    def __len__(self):
        return len(self.y)                    # how many examples?

    def __getitem__(self, i):
        return self.X[i], self.y[i]           # give me example i
```

`ds[0]` is the first ride: 16.5 minutes, 4.59 km, a weekday, 7am,
21.4°C, dry, and a casual rider (label 1). That's the whole contract.
For a folder of photos, `__getitem__` would open image `i` from disk
and turn it into a tensor, which is how Dataset copes with data far
bigger than memory: it only loads what's asked for.

When your data is already in tensors, you don't even need a class.
`TensorDataset(X, y)` wraps them and does the same job, as the ring
line shows.

## 3. DataLoader: batches, epochs and the last batch

```
one batch: features shape                      -> (32, 6)
one batch: labels shape                        -> (32,)
batches per epoch (len(loader))                -> 37
first and last batch sizes                     -> (32, 24)
examples seen in one epoch                     -> 1176
with drop_last=True: batches                   -> 36
```

Hand the Dataset to a `DataLoader` and loop over it:

```python
from torch.utils.data import DataLoader

loader = DataLoader(ds, batch_size=32, shuffle=True)
for xb, yb in loader:          # xb: (32, 6), yb: (32,)
    ...
```

The loader asks the Dataset for 32 examples and stacks them into one
tensor of shape `(32, 6)` plus labels of shape `(32,)`. One full pass
through every example is an **epoch**: 37 batches here, covering all
1,176 rides exactly once.

1,176 isn't a multiple of 32: it's 36 × 32 + 24, so the last batch has
just 24 rides. Usually that's fine. If a tiny last batch would cause
trouble (a batch of 1 can upset some layers you'll meet later),
`drop_last=True` skips it. With shuffling on, different rides get
skipped each epoch, so nothing is lost for long.

## 4. Shuffle, or learn the order

```
share of 1s per batch, unshuffled              -> [0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 1.0, 1.0]
shuffle=False, 60 epochs: training loss        -> 0.444
  accuracy on new points                       -> 0.702
  share predicted as 1                         -> 0.363
shuffle=True, 60 epochs: training loss         -> 0.23
  accuracy on new points                       -> 0.91
  share predicted as 1                         -> 0.265
true share of 1s on new points                 -> 0.31
```

Data often arrives sorted: by date, by customer, by label. Here's the
classic bug. The script sorts the ring so all the 0s come first and all
the 1s last, then trains the same `RingNet` twice for 60 epochs, with
`shuffle=False` and with `shuffle=True`. Nothing else changes.

The first line shows the problem: batch after batch of pure 0s, then
pure 1s. Unshuffled, the model spends most of each epoch learning
"everything is a 0", then gets dragged to "everything is a 1", every
epoch, forever. It never settles: training loss 0.444, 70% on new
points, and it ends each epoch leaning towards 1 (it predicts 1 for 36%
of points; the truth is 31%).

Shuffled, every batch is a fair sample of the whole, each step points
roughly the right way, and the same model reaches 91%.

Rule of thumb: `shuffle=True` for the training loader, always.
`shuffle=False` for validation and test loaders, where you're only
measuring and order can't matter.

## 5. Splitting cleanly

```
train / val / test sizes                       -> (824, 176, 176)
same seed, same split                          -> True
overlap between train and test                 -> 0
scaling stats from train only: mean            -> [15.78, 3.2, 0.21, 12.87, 17.62, 2.45]
casual share: train / val / test               -> (0.337, 0.284, 0.301)
```

`random_split` does lesson 031's train/test split for Datasets, with as
many parts as you like:

```python
train_ds, val_ds, test_ds = random_split(
    ds, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(55)
)
```

Three things to get right:

- **Seed the generator.** Same seed, same split, every run. Without it,
  every run tests on different rides and you can't compare experiments.
- **Split before anything learns.** The parts don't overlap, and
  nothing from val or test should shape the model. Validation is for
  your choices (epochs, learning rate); test is touched once, at the
  end.
- **Scale with training numbers only.** Lesson 036's rule still holds.
  The mean and spread used to standardise come from the 824 training
  rides, then get applied to all three parts.

The casual share differs a little between parts (33.7%, 28.4%, 30.1%).
With only 176 rides in a part, that's normal luck. On small data,
expect your test score to wobble by a point or two for the same reason.

## 6. The full loop on the rides, and batch size

```
epochs for every run                           -> 15
batch 8: steps, test AUC                       -> (1545, 0.855)
batch 32: steps, test AUC                      -> (390, 0.854)
batch 128: steps, test AUC                     -> (105, 0.853)
full batch: steps, test AUC                    -> (15, 0.803)
```

Everything together. To apply the training-split scaling, the script
wraps each part in a tiny `Standardised` Dataset whose `__getitem__`
scales the features on the way out. Then it's the loop you'll write
from now on:

```python
for epoch in range(15):
    model.train()
    for xb, yb in loader:
        optimiser.zero_grad()
        loss_fn(model(xb), yb).backward()
        optimiser.step()
```

Same network as lesson 052, Adam at 0.01, 15 epochs each, only the
batch size changes. With batches of 8, 32 or 128, the test AUC lands at
0.853 to 0.855, a touch better than lesson 054's best of 0.841 on a
different split. One giant batch of all 824 training rides gets only 15
steps in 15 epochs and is still a long way behind at 0.803.

That's the real reason mini-batches won: for the same pass over the
data, you get many more updates. Batch 32 made 390 steps from the same
15 epochs that gave full-batch just 15. On a GPU, bigger batches run
faster per example, so in practice you choose the biggest batch that
still learns well and fits in memory, and adjust the learning rate to
match.

## What you can do now

- Explain why we train on mini-batches, and how noise shrinks with
  batch size.
- Write a `Dataset` with `__len__` and `__getitem__`, or use
  `TensorDataset`.
- Use `DataLoader` with `batch_size`, `shuffle` and `drop_last`, and
  work out batches per epoch.
- Spot an unshuffled training loader before it costs you a day.
- Split with `random_split` and a seeded generator, and scale with
  training numbers only.
- Write the standard epoch-and-batch training loop.

## What to do now

1. Run `lesson.py`. Then in section 6 try a batch size of 2, and
   predict the step count before you look.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   compares batch sizes at a fixed number of steps, and splits the rides
   by time instead of at random.
3. Next, lesson 056: Regularising neural nets, using the validation
   split from today to fight overfitting. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

From here on, every network in this course trains through a DataLoader.
Two lines to remember: `shuffle=True` for training, and a seeded
generator for every split.
