# Lesson 057 - Convolutional networks

**Phase 3 - Deep learning foundations** | Week 8, Day 1 | Monday 2026-10-26

> **Goal:** Understand convolutions and pooling, then classify small
> images with a CNN.

Time: about 45 minutes. Same venv as lesson 051, with `torch` installed.

---

Every network so far has taken a flat row of numbers. That's fine for a
bike ride, where "distance" and "hour" have no sense of being next to
each other. An image is different. A pixel means almost nothing on its
own; what matters is the pattern it makes with its neighbours: an edge,
a curve, a corner.

A **convolutional network** (CNN) is built around that idea. It needs
just two new layers:

| Layer | What it does | In torch |
|-------|--------------|----------|
| **Convolution** | Slides a small grid of weights over the image, looking for one pattern everywhere | `nn.Conv2d(in_channels, out_channels, kernel_size=3)` |
| **Pooling** | Shrinks the result, keeping the strongest response in each patch | `nn.MaxPool2d(2)` |

Our images are scikit-learn's handwritten digits: 1,797 tiny 8x8
greyscale pictures of the numbers 0 to 9. They come with scikit-learn,
so there's nothing to download.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-3-deep-learning/lesson-057-convolutional-networks/lesson.py
```

It writes no files. Every random step has a fixed seed, so your numbers
will match. It takes about ten seconds.

## 1. An image is a grid of numbers

```
images, shape                                  -> (1797, 1, 8, 8)
pixel values run from, to                      -> (0.0, 1.0)
label of image 0                               -> 0
image 0, drawn with characters:
        ..##::
        ####++##..
        ##    ++::
      ..++    ::::
      ..::    ::::
      ..++    ++::
        ##..++++
        ..##++
its top row, as numbers                        -> [0.0, 0.0, 0.312, 0.812, 0.562, 0.062, 0.0, 0.0]
```

To a computer, a picture is a grid of brightness values. The originals
run from 0 to 16; the script divides by 16 so they run from 0 to 1, the
same "keep inputs small" habit as standardising in lesson 055.

Look at the shape: `(1797, 1, 8, 8)`. That's **(images, channels,
height, width)**, the order every torch image layer expects. One channel
because these are greyscale. A colour photo has three (red, green,
blue), so a batch of 32 phone photos might be `(32, 3, 224, 224)`.

## 2. A convolution, by hand

```
kernel (finds dark-to-light, left to right)    -> [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
input 8x8 -> output shape                      -> (6, 6)
row 3 of the output                            -> [1.94, -0.81, -1.88, 1.81, 1.38, -1.81]
torch's conv2d gives the same numbers          -> True
with padding=1, output shape                   -> (8, 8)
```

A **kernel** (or filter) is a small grid of weights, here 3x3. To
convolve, put it on the top-left corner of the image, multiply each
weight by the pixel under it, add up the nine products, and write the
total down. Slide one pixel right and do it again. Then the next row.

```python
for i in range(h - k + 1):
    for j in range(w - k + 1):
        out[i, j] = (img[i:i + k, j:j + k] * kernel).sum()
```

That's the whole operation. This kernel subtracts the left column from
the right one, so it gives a big positive number where ink appears as
you move right (a stroke's left edge), a big negative number where it
disappears (the right edge), and about zero on plain paper or in the
middle of a stroke. Row 3 of the output crosses the "0" and shows
exactly that: positive where the loop's left stroke starts, negative
where it ends, then positive and negative again for the right stroke.

A 3x3 kernel can't sit centred on the border pixels, so an 8x8 image
gives a 6x6 output. `padding=1` adds a ring of zeros round the edge
first, keeping the output 8x8. Torch's `F.conv2d` does the same sum,
just much faster.

Here's the important bit. In a CNN, nobody designs the kernels. Their
nine numbers are weights, learned by backprop like any other. And the
**same nine weights are used at every position**. A pattern detector
learned in one corner works in every corner, for free.

## 3. Pooling

```
4x4 feature map, row by row                    -> [[1, 0, 0, 0], [0, 0, 3, 0], [2, 0, 0, 0], [0, 0, 0, 0]]
after max_pool2d(2)                            -> [[1, 3], [2, 0]]
same map, shifted one pixel right              -> [[0, 1, 0, 0], [0, 0, 0, 3], [0, 2, 0, 0], [0, 0, 0, 0]]
pooled again                                   -> [[1, 3], [2, 0]]
```

A convolution's output is called a **feature map**: how strongly the
pattern showed up at each position. **Max pooling** cuts the map into
2x2 patches and keeps only the largest value in each. The map is now a
quarter of the size, and it answers a slightly vaguer question: not "is
there an edge at exactly this pixel?" but "is there an edge somewhere
round here?".

That vagueness is useful. Shift everything one pixel right and the
pooled result here doesn't change at all, because each value stayed
inside its patch. (A shift that crosses a patch boundary would change
it; pooling gives tolerance, not total immunity.)

## 4. Two networks, and their weights

```
plain network (flatten, 64 hidden), weights    -> 4810
CNN (two conv + pool blocks), weights          -> 6090
first conv layer, weight shape                 -> (16, 1, 3, 3)
one image's shape, layer by layer through the CNN:
  input                                        -> (1, 1, 8, 8)
  after Conv2d                                 -> (1, 16, 8, 8)
  after MaxPool2d                              -> (1, 16, 4, 4)
  after Conv2d                                 -> (1, 32, 4, 4)
  after MaxPool2d                              -> (1, 32, 2, 2)
  after Flatten                                -> (1, 128)
  after Linear                                 -> (1, 10)
```

The contenders are a plain network (flatten the 64 pixels, one hidden
layer of 64, ten outputs) and a small CNN:

```python
nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128, 10),
)
```

Follow the shape down the list, because it's the classic CNN pattern.
The first conv layer learns 16 different kernels, so one image becomes
16 feature maps. Pooling halves the height and width. The second conv
layer looks at all 16 maps at once and produces 32 new ones, built from
the first layer's patterns: edges combine into curves and corners. Pool
again, flatten, and a normal linear layer makes the final call between
ten classes. **Channels go up while the picture gets smaller.**

The weight shape `(16, 1, 3, 3)` reads as: 16 kernels, each looking at
1 input channel through a 3x3 window. That's only 160 weights,
including biases, and it doesn't depend on the image size at all. On a
224x224 photo, a plain linear layer with 64 units would need over three
million weights for its first layer alone; this one still needs 160.
That sharing is what makes CNNs workable on real photos.

## 5. Training both on the digits

```
train / val / test images                      -> (1079, 359, 359)
plain network
  best epoch                                   -> 24
  test accuracy                                -> 0.969
  test accuracy, images shifted 1 pixel right  -> 0.432
CNN
  best epoch                                   -> 21
  test accuracy                                -> 0.981
  test accuracy, images shifted 1 pixel right  -> 0.688
```

Both use lesson 056's sensible defaults: AdamW with a little weight
decay, a validation set, and early stopping. Ten classes means one new
piece. The last layer gives ten scores (logits), and
`nn.CrossEntropyLoss` is the ten-class version of the
`BCEWithLogitsLoss` we've been using: it turns the scores into
probabilities with a softmax and punishes low probability on the true
class. The prediction is simply the class with the highest score,
`model(X).argmax(dim=1)`.

On the test images as they are, the CNN wins, 98.1% against 96.9%. Not
a huge gap: these digits are small, centred and tidy, and a plain
network handles them well.

The last line in each block is the telling one. The script moves every
test digit one pixel to the right, something a person wouldn't even
notice, and asks again. The plain network collapses to 43%. Each of its
weights is tied to one exact pixel, so if the ink moves, it's looking in
the wrong place. The CNN holds on much better, at 69%, because its
kernels find strokes wherever they are, and pooling blurs exact
positions.

But 69% is not 98%. The CNN's last linear layer still cares about
roughly where things are on a tiny 2x2 grid, and on an 8x8 image one
pixel is a big move. The fix is one you already know from lesson 056:
data augmentation. Exercise 4 shows how far it goes.

## 6. What the CNN still gets wrong

```
test images it got wrong                       -> (7, 359)
each mistake as (true, predicted)              -> [(2, 1), (3, 7), (5, 6), (5, 8), (6, 8), (8, 1), (8, 6)]
first mistake: true label 6, predicted 8
          ++##++
        ..##++##..
        ##++  ::
      ..##::
      ::####::
        ##++##..
        ++####::
          ::##
```

Seven mistakes out of 359. Always look at them before you celebrate a
number. That "6" has a closed loop at the bottom and a stroke curling
over the top; at 8x8 pixels, calling it an 8 is a fair guess. Mistakes
that make sense are reassuring. Mistakes that look like nonsense (a
clear "1" read as "7" again and again) would mean something is wrong
with the data or the model.

## What you can do now

- Explain how a convolution slides a kernel over an image, and compute
  one by hand.
- Use `padding` to keep the size, and `MaxPool2d` to halve it.
- Read and predict shapes through a conv, pool, flatten, linear stack.
- Explain why sharing kernel weights makes CNNs small and tolerant of
  where things are.
- Train a ten-class classifier with `CrossEntropyLoss` and `argmax`.

## What to do now

1. Run `lesson.py`. In section 2, swap the kernel for its transpose
   (`vertical.T`) and see which edges it finds now.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   teaches the CNN to shrug off a shifted digit.
3. Next weekday, lesson 058: Recurrent networks and sequences, where
   the order of things matters as much as the things themselves. See
   [PROGRESS.md](../../../curriculum/PROGRESS.md).

Convolutions are how nearly every image model sees, from phone cameras
to medical scans. When we meet vision again in Phase 4, the networks
will be much deeper, but each layer will still be doing the
multiply-and-add you did by hand today.
