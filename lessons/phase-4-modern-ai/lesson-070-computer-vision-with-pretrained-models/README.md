# Lesson 070 - Computer vision with pretrained models

**Phase 4 - Modern AI: NLP, vision, generative and LLM basics** | Week 9, Day 2 | Tuesday 2026-11-03

> **Goal:** Classify, detect and embed images with off-the-shelf vision
> models.

Time: about 40 minutes. Same venv as lesson 064, plus `pillow` for
opening images.

---

In lesson 057 you trained a small convolutional network on 8x8 digits.
It worked, and it was also about as far as you'll get training vision
models from scratch on a laptop. The models people actually use were
trained on millions of photos, for days, on many GPUs.

The good news: you don't have to do that. Just as lesson 064 borrowed a
text model, you can download a vision model someone else trained and
use it in a few lines. Today we borrow three, each answering a
different question about a photo:

- **What is this?** An *image classifier*, which picks from a fixed
  list of labels.
- **Which of my words fits best?** A *zero-shot* model, CLIP, which
  scores a photo against any labels you write.
- **What's in it, and where?** An *object detector*, which draws boxes.

Then we turn photos into vectors and search them with words, and
finish by testing how much the models' confidence is worth.

## How to follow along

Venv active, `python` in the repo root. Install Pillow once (it's in
`requirements.txt` from today), then run the script:

```bash
pip install "pillow>=10.3,<13"
python lessons/phase-4-modern-ai/lesson-070-computer-vision-with-pretrained-models/lesson.py
```

It downloads seven sample photos from the Hugging Face hub's
documentation images, pinned to one version so they never change: two
cats on a sofa, a wild cat in the snow, a bee, a mosque, a plate of
baklava, a cubist painting of a squirrel, and a screenshot of a web
page showing a tiger. It also downloads three models, about 670 MB in
all, cached after the first run:

- `microsoft/resnet-18`, a classifier trained on ImageNet (1,000
  labels).
- `openai/clip-vit-base-patch32`, CLIP, trained on 400 million images
  with their captions.
- `hustvl/yolos-tiny`, a small detector trained on COCO (80 kinds of
  everyday object).

It writes one file, `output/detections.png`. Once everything is cached
it runs in under a minute.

## 1. An image is a grid of numbers

```
size (width, height)                           -> (640, 480)
as an array                                    -> 'shape (480, 640, 3), dtype uint8'
one pixel (red, green, blue)                   -> (240, 75, 151)
what the model gets                            -> (1, 3, 224, 224)
value range after preprocessing                -> (-2.118, 2.64)
processor's mean (per colour)                  -> [0.485, 0.456, 0.406]
```

A colour photo is a 3-D array: height, width, and three colour
channels, each a whole number from 0 to 255. The pixel in the middle
of the cat photo is a pink (lots of red, some blue): the blanket.

The model doesn't want that, though. It wants a batch of exactly
224x224 images, channels first, with each colour shifted and scaled by
the **mean and standard deviation of the photos it was trained on**.
That's why the values now run from about -2 to 2.6.

Every pretrained model has its own expected size and scaling, and gets
quietly worse if you get them wrong. So every model ships with a
**processor** (`AutoImageProcessor`) that does it for you. Always use
the one that matches the model, just as lesson 064 always used the
model's own tokenizer.

## 2. Classify with a fixed list of labels

```
labels the model knows                         -> 1000
is 'baklava' one of them?                      -> False
  cats on a sofa       tabby                  0.604
  wild cat             lynx                   0.654
  bee                  bee                    0.747
  mosque               mosque                 0.990
  baklava              burrito                0.479
  squirrel painting    shield                 0.181
  tiger web page       web site               0.908
```

ResNet-18 is a convolutional network, lesson 057's idea, scaled up
and trained on ImageNet's 1.2 million photos. The softmax gives a
probability for each of its 1,000 labels; we print the top one.

- **On its list, it's good.** Bee, mosque, tabby cat: all right.
- **Off its list, it's still sure of something.** "Baklava" isn't one
  of its labels, so it can't say it. It picks the nearest thing it
  knows, a burrito. The wild cat is a Pallas's cat, also not on the
  list; "lynx" is a reasonable near miss.
- **A painting isn't a photo.** The squirrel gets "shield" at 18%. Low
  confidence is useful: it's the model telling you it's lost.
- **It labels pixels, not your intentions.** The "tiger" photo is
  really a screenshot of a web page with a tiger in it, and "web site"
  is honestly the right answer.

A fixed-label classifier is perfect when its labels are your labels.
When they aren't, you'd normally fine-tune it on your own classes
(lesson 064's recipe works for images too), or try the next section.

## 3. Zero-shot: bring your own labels

```
our labels                                     -> ['a cat', 'a tiger', 'an insect', 'a building', 'food', 'a painting']
  cats on a sofa       a cat                  0.957
  wild cat             a cat                  0.663
  bee                  an insect              0.980
  mosque               a building             0.906
  baklava              food                   0.999
  squirrel painting    a painting             0.582
  tiger web page       a tiger                0.968
tiger web page, with a 'screenshot' label      -> ('a screenshot of a web page', 0.713)
```

CLIP was trained on images *and their captions*, to put a photo and
its caption close together. So it can score a photo against any text
you write. We turn each label into a caption ("a photo of food"), and
CLIP tells us which caption fits each photo best. No training, no
labelled data: you just change the list. That's **zero-shot
classification**.

Seven out of seven look right. But look at the tiger web page: "a
tiger", 97% sure. That's not wrong, exactly, but it isn't the whole
truth either. Add "a screenshot of a web page" to the list and that
wins instead.

The catch: CLIP's probabilities only compare **the labels you gave
it**, and always add up to 1. Leave out the right label and it can't
tell you; it puts its confidence on the best of the rest. So include
an "other" kind of label (a screenshot, a blurry photo, "something
else") when your real photos might need it.

## 4. Detect objects: what, and where

```
  bed      0.502  box (x0, y0, x1, y1) = [0, 1, 640, 475]
  cat      0.979  box (x0, y0, x1, y1) = [11, 54, 313, 471]
  remote   0.974  box (x0, y0, x1, y1) = [42, 72, 178, 120]
  remote   0.991  box (x0, y0, x1, y1) = [46, 73, 179, 119]
  remote   0.908  box (x0, y0, x1, y1) = [336, 79, 368, 192]
  cat      0.934  box (x0, y0, x1, y1) = [337, 18, 638, 373]
objects found at threshold 0.3                 -> 7
objects found at threshold 0.5                 -> 6
objects found at threshold 0.9                 -> 5
largest overlap between two remote boxes       -> 0.924
saved                                          -> 'output/detections.png'
```

A detector answers two questions at once: *what* (a label and a score)
and *where* (a box, given as left, top, right and bottom pixel
positions). Open `output/detections.png` to see the boxes drawn on the
photo.

Two cats and two remotes, all found. Then two oddities:

- **The same remote, twice.** Two remote boxes overlap almost
  perfectly. We measure that with **IoU**, intersection over union:
  the shared area divided by the combined area, from 0 (no overlap) to
  1 (identical boxes). 0.92 means it's one remote. Real pipelines
  remove such duplicates with a step called non-maximum suppression:
  keep the highest-scoring box and drop others that overlap it a lot.
- **The bed.** The sofa is boxed as "bed" at 0.502, just over our
  threshold.

The **threshold** is your choice, and it's the precision and recall
trade-off from lesson 034 again. At 0.9 we keep 5 objects and lose the
bed; at 0.3 we get 7, one more doubtful box. A cycle-lane safety
camera wants high recall (don't miss a person); an app that auto-tags
photos wants high precision (don't tag nonsense).

## 5. Images as vectors: similarity and search

```
one image as a vector                          -> (512,)
most similar pair of photos                    -> ('cats on a sofa', 'wild cat', 0.633)
bee vs mosque                                  -> 0.497
search: 'something sweet to eat'               -> 'baklava'
search: 'a place of worship'                   -> 'mosque'
search: 'an insect on a flower'                -> 'bee'
search: 'modern art'                           -> 'squirrel painting'
```

CLIP can also give you the vector it uses inside, an **embedding**:
512 numbers per image. Normalise them and lesson 067's whole toolkit
applies, cosine similarity and nearest neighbours included.

- The two closest photos are the two of cats. Nobody told CLIP they
  were cats; similar images just land close together. This is how
  "find similar photos" and duplicate detection work.
- And because CLIP puts *text* in the same space, words can search
  photos. "Something sweet to eat" finds the baklava, though the word
  "baklava" appears nowhere. That's semantic search, with photos as the
  documents.

## 6. How sure is it, really?

```
  original               tabby                  0.604
  upside down            tabby                  0.734
  much darker            Egyptian cat           0.881
  black and white        Egyptian cat           0.817
  remote only (crop)     pencil box             0.519
  tiny, then blown up    fox squirrel           0.081
```

Same cat photo, changed the way real photos get changed, through
ResNet-18 again:

- **Upside down, it's *more* confident.** Darker and in black and
  white, it switches to "Egyptian cat" and gets *more* confident
  still. Those probabilities are numbers from a softmax, not a measure
  of how right the model is.
- **The crop of a remote** becomes "pencil box", though "remote
  control" is one of its 1,000 labels.
- **Blurred to 24x18 pixels and blown back up,** it guesses "fox
  squirrel" at 8%. Here, at least, the low score honestly says "I'm
  lost".

The habit to take away: a model's confidence is not a guarantee. Before
you trust a vision model, test it on photos like the ones your users
will really send: dark, tilted, cropped, blurry.

## What you can do now

- Load a pretrained vision model and its processor, and explain what
  the processor does.
- Classify images with a fixed-label model, and recognise when the
  right answer isn't on its list.
- Run zero-shot classification with CLIP, using your own labels.
- Detect objects, read the boxes, measure overlap with IoU, and choose
  a threshold.
- Embed images, find similar ones, and search them with text.
- Stress-test a model's confidence with realistic changes to a photo.

## What to do now

1. Run `lesson.py` and open `output/detections.png`. Then add a photo
   of your own to `PHOTOS` (a local path works too: swap
   `hf_hub_download(...)` for the path) and see what each model makes
   of it.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   finds out what the wild cat really is.
3. Next weekday, lesson 071: generative models, autoencoders and VAEs,
   where we stop labelling data and start making new data.

Borrow a pretrained model, feed it through its own processor, and
check its answers against what your users will actually show it.
