# Lesson 070 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv (with `transformers` and `pillow` installed). For
the hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-4-modern-ai/lesson-070-computer-vision-with-pretrained-models/lesson.py my_lesson_070.py
python my_lesson_070.py
```

`load_photos`, `classify`, `clip_scores`, `detect` and `iou` are there
to reuse.

---

## 1. Forgetting the processor

A colleague feeds photos to ResNet-18 by resizing them to 224x224 and
dividing the pixels by 255, skipping the processor. Nothing crashes.
What goes wrong?

<details>
<summary>Check yourself</summary>

The pixels end up between 0 and 1, but the model was trained on values
shifted by the ImageNet mean and divided by its standard deviation,
roughly -2 to 2.6 (section 1). Every input now looks unusually dark
and washed out compared with anything it saw in training, and the
predictions get quietly worse. No error, just worse answers, which is
the most dangerous kind of bug. Use the model's own processor.

</details>

## 2. Classifier, CLIP or detector?

Which of today's three models would you start with for each job?

1. Riders photograph a damaged bike; sort the photo into "flat tyre",
   "broken chain", "bent wheel" or "other".
2. Count how many bikes are parked in a photo of a station.
3. Check that an uploaded photo shows a bike at all.

<details>
<summary>Check yourself</summary>

1. **CLIP, zero-shot**, to start: your own labels, no training, and an
   "other" label for photos that fit none (section 3). If you gather a
   few hundred labelled photos later, fine-tuning will likely beat it.
2. **The detector**: counting means finding each object. "bicycle" is
   one of COCO's 80 labels. Watch out for duplicate boxes (section 4).
3. Either works: CLIP with "a photo of a bicycle" against "a photo of
   something else", or the detector, checking for any "bicycle" box
   above your threshold. ResNet-18 could help a little too: its 1,000
   labels include "mountain bike" and "tandem bicycle".

</details>

## 3. Reading IoU

Two boxes: `[0, 0, 100, 100]` and `[50, 0, 150, 100]`. What's their
IoU? Work it out by hand, then check with `iou`.

<details>
<summary>Check yourself</summary>

They overlap in a 50x100 strip: area 5,000. Each box is 10,000, so the
union is 10,000 + 10,000 - 5,000 = 15,000. IoU = 5,000 / 15,000 =
0.333. Half of each box overlaps, but the IoU is only a third, because
it divides by the combined area. A common rule treats IoU above 0.5 as
"the same object".

</details>

## 4. Hands-on: what is the wild cat?

Both ResNet-18 and CLIP called the wild cat something it isn't
("lynx", and just "a cat"). Score it with CLIP against these labels,
then again with a fourth label added:

1. `["a house cat", "a lynx", "a tiger"]`
2. `["a house cat", "a lynx", "a tiger", "a Pallas's cat"]`

Predict first: what happens to "a lynx"?

<details>
<summary>Expected results</summary>

```
['a house cat', 'a lynx', 'a tiger'] [0.056, 0.904, 0.04]
['a house cat', 'a lynx', 'a tiger', "a Pallas's cat"] [0.001, 0.008, 0.0, 0.991]
```

With three labels, CLIP is 90% sure it's a lynx. Add the right answer
and "lynx" collapses to under 1%: it's a Pallas's cat, a small wild cat
from Central Asia, at 99%. Same photo, same model; only the list
changed.

That's section 3's warning in numbers. A high zero-shot score means
"best of the labels you offered", never "correct". CLIP knew the
answer all along, but only once we offered it.

</details>

<details>
<summary>One way to write it</summary>

In your copy, replace the `main()` call at the very bottom with:

```python
if __name__ == "__main__":
    photos = load_photos()
    clip, cproc = CLIPModel.from_pretrained(CLIP).eval(), CLIPProcessor.from_pretrained(CLIP)
    for labels in (["a house cat", "a lynx", "a tiger"],
                   ["a house cat", "a lynx", "a tiger", "a Pallas's cat"]):
        row = clip_scores(clip, cproc, [photos["wild cat"]], labels)[0]
        print(labels, [r3(p) for p in row])
```

</details>

## 5. (Optional) A smarter "I'm not sure"

In section 6 the black-and-white cat was called "Egyptian cat" at 82%,
more confident than the original. If you were building the damaged-
bike sorter from exercise 2, how could you decide when to send a photo
to a person instead of trusting the model?

<details>
<summary>Check yourself</summary>

A single probability isn't enough, as section 6 shows, but a few
cheap signals help. Ask whether the top two labels are close (a 0.48
versus 0.45 split is a coin toss). Run the model on a couple of
variants of the same photo (slightly cropped, flipped) and see whether
the answer changes; if it does, it's unsure. Check image quality
directly (too dark, too small) before classifying at all. Then, as
always, pick the cut-off by measuring on a set of real, labelled rider
photos, and send everything below it to a person. Lesson 073 goes much
further into evaluating AI systems honestly.

</details>

---

That's computer vision with pretrained models. If one thing sticks: *a
borrowed model can only choose from the answers it was given, so check
that the right answer is on its list, and test it on the photos your
users really take.*
