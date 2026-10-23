# Lesson 055 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, work in a copy. From the repo
root:

```bash
cp lessons/phase-3-deep-learning/lesson-055-datasets-dataloaders-and-batching/lesson.py my_lesson_055.py
python my_lesson_055.py
```

Change the `HERE = ...` line near the top so the copy finds the data:

```python
HERE = Path("lessons/phase-3-deep-learning/lesson-055-datasets-dataloaders-and-batching").resolve()
```

`make_ring`, `RingNet`, `RidesDataset`, `Standardised`, `train_epochs`,
`predict_proba`, `accuracy`, `auc`, `r3` and `show` are there to reuse.

---

## 1. Counting batches

The rides have 1,176 examples. With `batch_size=50`, how many batches
are in one epoch, and how big is the last one? How many with
`drop_last=True`?

<details>
<summary>Check yourself</summary>

1,176 = 23 × 50 + 26, so 24 batches, the last one holding 26 rides.
With `drop_last=True`, 23. In general, `len(loader)` is the number of
examples divided by the batch size, rounded up (or down with
`drop_last`).

</details>

## 2. Buying accuracy

Section 1 found a batch of 16 gave a gradient error of about 0.19, and
a batch of 64 about 0.10. Roughly what batch size would you need to get
the error down to 0.05?

<details>
<summary>Check yourself</summary>

Noise shrinks with the square root of the batch size. Halving it again
means four times the examples: about 256. Section 1 measured 0.032 at
256, a bit better than the rule predicts, because each batch here is a
big slice of only 400 points (sampling without replacement helps once
the batch is most of the data). On a big dataset the rule holds more
closely.

</details>

## 3. Shuffling the test set

A teammate sets `shuffle=True` on the test loader too, "to be safe".
Does it change the test AUC? Is there any reason not to?

<details>
<summary>Check yourself</summary>

It doesn't change the score: AUC and accuracy look at every test
example once, and don't care about order. But there's no gain, it costs
a little time, and it makes it harder to line predictions up with the
original rows when you want to look at the mistakes. Shuffle the
training loader; leave validation and test in order.

</details>

## 4. Hands-on: fixed steps, and a split by time

**a)** Section 6 gave every batch size the same number of *epochs*.
Now give them the same number of *steps* instead. On the ring
(`TensorDataset(*make_ring())`), train `RingNet()` with Adam at 0.01 for
exactly 2,000 steps with batch sizes 4, 16, 64 and 400 (seed 54 before
building each model, loader generator seeded with 55; keep looping
over epochs until 2,000 steps are done). Print the final loss on all
400 training points and the accuracy on `make_ring(seed=51)`.

**b)** The rides are sorted by start time. Instead of a random split,
train on the *first* 824 rides and test on the *last* 176
(`range(1000, 1176)`), using `torch.utils.data.Subset`. Keep everything
else from section 6 with batch size 32. What test AUC do you get, and
what does a time split tell you that a random one can't?

<details>
<summary>Expected results</summary>

```
a) batch 4:   loss 0.262, accuracy on new 0.928
   batch 16:  loss 0.216, accuracy on new 0.923
   batch 64:  loss 0.198, accuracy on new 0.935
   batch 400: loss 0.152, accuracy on new 0.94
b) time split, batch 32: test AUC 0.865
   casual share: train 0.318, test 0.341
```

(a) Flip section 6 around and the result flips too. With a fixed number
of *steps*, bigger batches win, because each step sees more data and
points more truly downhill. So the real question is never "which batch
size is best?" but "best for what budget?". Per epoch, small batches
learn more; per step, big ones do. On a GPU a step of 64 costs barely
more than a step of 4, which is why people go as big as memory allows.

(b) 0.865, close to the random split's 0.854; here, the last week of
rides looks much like the first ones. That won't always be true. A time
split asks the question you'll actually face once the model is
deployed: "trained on the past, how does it do on the future?". When
the data has a date and you'll predict forward in time, split by time.

</details>

<details>
<summary>One way to write it</summary>

```python
X, y = make_ring()
X_new, y_new = make_ring(seed=51)
ring = TensorDataset(X, y)
loss_fn = nn.BCEWithLogitsLoss()
for batch in (4, 16, 64, 400):
    torch.manual_seed(54)
    model = RingNet()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loader = DataLoader(ring, batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(55))
    steps = 0
    while steps < 2000:
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
            steps += 1
            if steps == 2000:
                break
    with torch.no_grad():
        final = loss_fn(model(X), y)
    show(f"batch {batch}", (r3(final), accuracy(predict_proba(model, X_new), y_new)))

from torch.utils.data import Subset
ds = RidesDataset(RIDES_CSV)
train_idx, test_idx = list(range(824)), list(range(1000, 1176))
tr = torch.tensor(train_idx)
mean, std = ds.X[tr].mean(dim=0), ds.X[tr].std(dim=0)
train_s = Standardised(Subset(ds, train_idx), mean, std)
test_s = Standardised(Subset(ds, test_idx), mean, std)
X_te, y_te = next(iter(DataLoader(test_s, batch_size=len(test_s))))
torch.manual_seed(52)
model = RingNet(n_inputs=6, n_hidden=8)
loader = DataLoader(train_s, batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(55))
train_epochs(model, loader, torch.optim.Adam(model.parameters(), lr=0.01), 15)
show("time split, test AUC", auc(predict_proba(model, X_te), y_te))
```

</details>

## 5. (Optional) Batches of dictionaries

Real datasets often return more than `(x, y)`: an ID, a mask, several
inputs. Make a Dataset of length 10 whose `__getitem__(i)` returns
`{"x": torch.tensor([float(i), 2.0 * i]), "y": torch.tensor(float(i % 2))}`.
Predict what one batch from `DataLoader(..., batch_size=4)` looks like,
then check.

<details>
<summary>Check yourself</summary>

```
{'x': (4, 2), 'y': (4,)}    y = [0.0, 1.0, 0.0, 1.0]
```

A dictionary of batched tensors, keyed the same way. The DataLoader
stacks matching pieces from each example, whether they come as a tuple,
a list or a dict, so `batch["x"]` is a `(4, 2)` tensor. That's how
you'll see text models fed in Phase 4, with keys like `input_ids` and
`attention_mask`.

</details>

---

That's Datasets and DataLoaders. If one thing sticks: *a Dataset says
what one example is, a DataLoader shuffles and batches, and every split
gets a seed.*
