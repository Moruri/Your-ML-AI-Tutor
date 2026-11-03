"""
Lesson 070 - Computer vision with pretrained models

Photos are just grids of numbers, and people have already trained excellent
models on millions of them. Today we borrow three: one that classifies an
image into a fixed list of 1,000 things, one that matches images to any words
you like, and one that finds and boxes objects. Then we use images as vectors
for search, and poke at how easily a confident model can be fooled.

Run it with (inside the venv from lesson 013, with torch and transformers
installed as in lessons 051 and 064, plus pillow):

    pip install "pillow>=10.3,<13"
    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

It downloads seven small sample photos from the Hugging Face hub (pinned to
one version, so they never change) and three models: ResNet-18 (about 45 MB),
CLIP (about 600 MB) and YOLOS-tiny (about 25 MB). Everything is cached after
the first run. It writes one file, output/detections.png. After the downloads
it takes under a minute on a laptop CPU.
"""

from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from transformers import (AutoImageProcessor, AutoModelForImageClassification, AutoModelForObjectDetection,
                          CLIPModel, CLIPProcessor)
from transformers.utils import logging as hf_logging

torch.set_num_threads(1)              # same numbers on every machine
hf_logging.set_verbosity_error()      # keep the library's chatter out of the lesson output

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

PHOTO_REPO = "huggingface/documentation-images"
PHOTO_VERSION = "118cfd2ad096f09570e4bf761648cd34293bc709"   # pinned, so the photos never change
PHOTOS = {
    "cats on a sofa": "coco_sample.png",
    "wild cat": "pipeline-cat-chonk.jpeg",
    "bee": "bee.jpg",
    "mosque": "mosque.jpg",
    "baklava": "baklava.jpg",
    "squirrel painting": "image_of_squirrel_painting.png",
    "tiger web page": "tiger_image.png",
}

CLASSIFIER = "microsoft/resnet-18"           # trained on ImageNet's 1,000 labels
CLIP = "openai/clip-vit-base-patch32"        # trained on 400 million image-caption pairs
DETECTOR = "hustvl/yolos-tiny"               # trained to box 80 kinds of object (COCO)


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def r3(x):
    return round(float(x), 3) + 0.0


def load_photos():
    return {name: Image.open(hf_hub_download(PHOTO_REPO, file, repo_type="dataset", revision=PHOTO_VERSION))
            .convert("RGB") for name, file in PHOTOS.items()}


# ---------------------------------------------------------------------------
# 1. An image is a grid of numbers
# ---------------------------------------------------------------------------

def section_pixels(photos, processor):
    heading("1. An image is a grid of numbers")
    img = photos["cats on a sofa"]
    arr = np.asarray(img)
    show("size (width, height)", img.size)
    show("as an array", f"shape {arr.shape}, dtype {arr.dtype}")
    show("one pixel (red, green, blue)", tuple(int(v) for v in arr[240, 320]))
    x = processor(img, return_tensors="pt")["pixel_values"]
    show("what the model gets", tuple(x.shape))
    show("value range after preprocessing", (r3(x.min()), r3(x.max())))
    show("processor's mean (per colour)", processor.image_mean)
    print("  Every pretrained model expects its own size and scaling. The")
    print("  processor that ships with the model does it; always use it.")


# ---------------------------------------------------------------------------
# 2. Classify with a fixed list of labels
# ---------------------------------------------------------------------------

def classify(model, processor, img, k=1):
    with torch.no_grad():
        probs = model(**processor(img, return_tensors="pt")).logits.softmax(-1)[0]
    p, i = probs.topk(k)
    return [(model.config.id2label[int(j)].split(",")[0], r3(v)) for v, j in zip(p, i)]


def section_classify(photos, model, processor):
    heading("2. Classify with a fixed list of labels")
    show("labels the model knows", len(model.config.id2label))
    show("is 'baklava' one of them?", any("baklava" in v.lower() for v in model.config.id2label.values()))
    for name, img in photos.items():
        label, p = classify(model, processor, img)[0]
        print(f"    {name:<20} {label:<22} {p:.3f}")
    print("  Right when the answer is on its list. When it isn't (baklava), it")
    print("  still has to pick one of its 1,000: a burrito. And it saw the tiger")
    print("  web page for what it is, a web page. It labels pixels, not intent.")


# ---------------------------------------------------------------------------
# 3. Zero-shot: bring your own labels
# ---------------------------------------------------------------------------

def clip_scores(clip, cproc, images, labels):
    batch = cproc(text=[f"a photo of {l}" for l in labels], images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        return clip(**batch).logits_per_image.softmax(-1)


def section_zero_shot(photos, clip, cproc):
    heading("3. Zero-shot: bring your own labels")
    labels = ["a cat", "a tiger", "an insect", "a building", "food", "a painting"]
    show("our labels", labels)
    probs = clip_scores(clip, cproc, list(photos.values()), labels)
    for name, row in zip(photos, probs):
        print(f"    {name:<20} {labels[int(row.argmax())]:<22} {r3(row.max()):.3f}")
    web = photos["tiger web page"]
    more = labels + ["a screenshot of a web page"]
    row = clip_scores(clip, cproc, [web], more)[0]
    show("tiger web page, with a 'screenshot' label", (more[int(row.argmax())], r3(row.max())))
    print("  No training, just words. But the probabilities only compare YOUR")
    print("  labels: leave out the right one and the model can't say so.")


# ---------------------------------------------------------------------------
# 4. Detect objects: what, and where
# ---------------------------------------------------------------------------

def iou(a, b):
    """Overlap of two boxes, from 0 (apart) to 1 (identical)."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return w * h / (area(a) + area(b) - w * h)


def detect(model, processor, img, threshold):
    with torch.no_grad():
        out = model(**processor(img, return_tensors="pt"))
    res = processor.post_process_object_detection(out, threshold=threshold, target_sizes=[img.size[::-1]])[0]
    return [(model.config.id2label[int(l)], r3(s), [round(float(v)) for v in b])
            for s, l, b in zip(res["scores"], res["labels"], res["boxes"])]


def section_detect(photos, model, processor):
    heading("4. Detect objects: what, and where")
    img = photos["cats on a sofa"]
    found = detect(model, processor, img, threshold=0.5)
    for label, s, box in sorted(found, key=lambda f: f[2][0]):
        print(f"    {label:<8} {s:.3f}  box (x0, y0, x1, y1) = {box}")
    for t in (0.3, 0.5, 0.9):
        show(f"objects found at threshold {t}", len(detect(model, processor, img, t)))
    remotes = [box for label, _, box in found if label == "remote"]
    pairs = [(r3(iou(a, b)), a, b) for i, a in enumerate(remotes) for b in remotes[i + 1:]]
    show("largest overlap between two remote boxes", max(p[0] for p in pairs))
    OUT.mkdir(exist_ok=True)
    drawn = img.copy()
    pen = ImageDraw.Draw(drawn)
    for label, s, box in found:
        pen.rectangle(box, outline=(255, 40, 40), width=3)
        pen.text((box[0] + 4, box[1] + 4), f"{label} {s:.2f}", fill=(255, 255, 0))
    drawn.save(OUT / "detections.png")
    show("saved", "output/detections.png")
    print("  Two cats, two remotes, so why three remote boxes? Two of them")
    print("  overlap almost completely: the same remote, found twice. The")
    print("  threshold trades missed objects for false ones; you choose it.")


# ---------------------------------------------------------------------------
# 5. Images as vectors: similarity and search
# ---------------------------------------------------------------------------

def section_embed(photos, clip, cproc):
    heading("5. Images as vectors: similarity and search")
    names = list(photos)
    with torch.no_grad():
        img_vecs = clip.get_image_features(**cproc(images=list(photos.values()), return_tensors="pt"))
    img_vecs = torch.nn.functional.normalize(img_vecs, dim=1)
    show("one image as a vector", tuple(img_vecs[0].shape))
    sims = (img_vecs @ img_vecs.T).numpy()
    np.fill_diagonal(sims, -1)
    i, j = np.unravel_index(sims.argmax(), sims.shape)
    show("most similar pair of photos", (names[i], names[j], r3(sims[i, j])))
    show("bee vs mosque", r3(sims[names.index("bee"), names.index("mosque")]))
    queries = ["something sweet to eat", "a place of worship", "an insect on a flower", "modern art"]
    with torch.no_grad():
        txt = clip.get_text_features(**cproc(text=queries, return_tensors="pt", padding=True))
    txt = torch.nn.functional.normalize(txt, dim=1)
    for q, row in zip(queries, (txt @ img_vecs.T).numpy()):
        show(f"search: {q!r}", names[int(row.argmax())])
    print("  Lesson 067's semantic search, but the documents are photos. CLIP")
    print("  puts words and images in the same space, so text can find pictures.")


# ---------------------------------------------------------------------------
# 6. How sure is it, really?
# ---------------------------------------------------------------------------

def section_robust(photos, model, processor):
    heading("6. How sure is it, really?")
    img = photos["cats on a sofa"]
    w, h = img.size
    variants = {
        "original": img,
        "upside down": img.rotate(180),
        "much darker": ImageEnhance.Brightness(img).enhance(0.25),
        "black and white": ImageOps.grayscale(img).convert("RGB"),
        "remote only (crop)": img.crop((40, 70, 180, 120)),
        "tiny, then blown up": img.resize((24, 18)).resize((w, h)),
    }
    for name, v in variants.items():
        top = classify(model, processor, v, k=1)[0]
        print(f"    {name:<22} {top[0]:<22} {top[1]:.3f}")
    print("  A probability is the model's confidence, not a guarantee. Test on")
    print("  photos like the ones your users will really send.")


def main():
    print("=" * 66)
    print("  Lesson 070: Computer vision with pretrained models")
    print("=" * 66)
    photos = load_photos()
    processor = AutoImageProcessor.from_pretrained(CLASSIFIER)
    classifier = AutoModelForImageClassification.from_pretrained(CLASSIFIER).eval()
    clip, cproc = CLIPModel.from_pretrained(CLIP).eval(), CLIPProcessor.from_pretrained(CLIP)
    det_proc = AutoImageProcessor.from_pretrained(DETECTOR)
    detector = AutoModelForObjectDetection.from_pretrained(DETECTOR).eval()
    section_pixels(photos, processor)
    section_classify(photos, classifier, processor)
    section_zero_shot(photos, clip, cproc)
    section_detect(photos, detector, det_proc)
    section_embed(photos, clip, cproc)
    section_robust(photos, classifier, processor)
    print()
    print("Borrow a pretrained model, feed it through its own processor, and")
    print("check its answers against what your users will actually show it.")
    print()


if __name__ == "__main__":
    main()
