"""
Lesson 064 - Pretrained models and transfer learning

Training a language model from scratch takes mountains of text. Somebody has
already done that and shared the result. Today we download a small pretrained
model from Hugging Face, use it as a ready-made feature extractor, then
fine-tune all of it on our rider-feedback messages, with only 20 labelled
examples. Then we compare it honestly with lesson 061's TF-IDF baseline.

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051 and transformers installed as in this lesson's README):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

It reads feedback.csv (the same made-up messages as lesson 061) from this
folder and writes no files. The first run downloads the model (about 90 MB)
into Hugging Face's cache in your home folder; later runs reuse it. Every
random step has a fixed seed, so your numbers should match the README. It
takes about half a minute on a laptop CPU.
"""

import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

torch.set_num_threads(1)              # same numbers on every machine
hf_logging.set_verbosity_error()      # keep the library's chatter out of the lesson output

HERE = Path(__file__).resolve().parent
MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # small, free, and good with short sentences
PER_TOPIC = 5                                       # labelled examples per topic: 20 in total


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def r3(x):
    if torch.is_tensor(x):
        x = x.detach()
    return round(float(x), 3)


def load_data():
    df = pd.read_csv(HERE / "feedback.csv")
    train, test = train_test_split(df, test_size=200, random_state=64, stratify=df["topic"])
    few = train.groupby("topic").sample(PER_TOPIC, random_state=64)
    return few, test, sorted(df["topic"].unique())


# ---------------------------------------------------------------------------
# 1. Only twenty labels
# ---------------------------------------------------------------------------

def tfidf_baseline(few, test):
    vec = TfidfVectorizer()
    clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(few["text"]), few["topic"])
    return vec, clf


def section_few(few, test):
    heading("1. Only twenty labels")
    show("labelled training messages", len(few))
    show("per topic", few["topic"].value_counts().to_dict())
    show("test messages", len(test))
    vec, clf = tfidf_baseline(few, test)
    acc = (clf.predict(vec.transform(test["text"])) == test["topic"]).mean()
    show("TF-IDF + logistic regression: accuracy", r3(acc))
    show("words it knows", len(vec.vocabulary_))
    print("  Lesson 061 scored 0.99 with 600 labels. With 20, a word it never saw")
    print("  in training is just invisible. Labels are the expensive part.")
    return vec, clf


# ---------------------------------------------------------------------------
# 2. Loading a pretrained model
# ---------------------------------------------------------------------------

def section_load():
    heading("2. Loading a pretrained model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    text = "honestly, i was charged twice via m-pesa. so annoying."
    batch = tokenizer(text, return_tensors="pt")
    show("model", MODEL)
    show("parameters (millions)", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))
    show("layers, heads, numbers per token",
         (model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size))
    show("vocabulary size", tokenizer.vocab_size)
    show("tokens", tokenizer.convert_ids_to_tokens(batch["input_ids"][0]))
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    show("output shape (batch, tokens, numbers)", tuple(hidden.shape))
    print("  Six of lesson 063's blocks, trained on about a billion sentence")
    print("  pairs by someone else. 'm-pesa' isn't in its vocabulary, so the")
    print("  tokenizer splits it into pieces it does know: m, -, pe, ##sa.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# 3. Frozen features: use it as it comes
# ---------------------------------------------------------------------------

def embed(tokenizer, model, texts, batch_size=64):
    """One vector per text: the average of its token vectors (padding ignored)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            out.append((hidden * mask).sum(1) / mask.sum(1))
    return torch.cat(out)


def section_frozen(tokenizer, model, few, test):
    heading("3. Frozen features: use it as it comes")
    pair = ["my card was billed two times", "m-pesa took the money twice", "the front tyre is flat"]
    e = embed(tokenizer, model, pair)
    show("similarity: billed twice vs m-pesa twice", r3(F.cosine_similarity(e[0], e[1], dim=0)))
    show("similarity: billed twice vs flat tyre", r3(F.cosine_similarity(e[0], e[2], dim=0)))
    t = torch.tensor(TfidfVectorizer().fit_transform(pair).toarray())
    show("same pair with TF-IDF vectors", r3(F.cosine_similarity(t[0], t[1], dim=0)))
    clf = LogisticRegression(max_iter=1000).fit(embed(tokenizer, model, few["text"]).numpy(), few["topic"])
    acc = (clf.predict(embed(tokenizer, model, test["text"]).numpy()) == test["topic"]).mean()
    show("frozen embeddings + logistic regression", r3(acc))
    print("  The two payment messages share no words, yet the model puts them")
    print("  close together. A plain classifier on top, same 20 labels, does")
    print("  far better than TF-IDF. Not one weight of the model changed.")


# ---------------------------------------------------------------------------
# 4. Fine-tuning, and what pretraining is worth
# ---------------------------------------------------------------------------

def fine_tune(model, tokenizer, few, topics, steps=20, lr=1e-4):
    batch = tokenizer(list(few["text"]), padding=True, return_tensors="pt")
    labels = torch.tensor([topics.index(t) for t in few["topic"]])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        loss = model(**batch, labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def predict(model, tokenizer, texts, topics):
    with torch.no_grad():
        logits = model(**tokenizer(list(texts), padding=True, return_tensors="pt")).logits
    return [topics[i] for i in logits.argmax(1)]


def section_finetune(tokenizer, few, test, topics):
    heading("4. Fine-tuning, and what pretraining is worth")
    torch.manual_seed(640)
    scratch = AutoModelForSequenceClassification.from_config(
        AutoConfig.from_pretrained(MODEL, num_labels=len(topics)))
    torch.manual_seed(640)
    tuned = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(topics))
    show("new layer added on top", str(tuned.classifier))
    acc = lambda m: r3((pd.Series(predict(m, tokenizer, test["text"], topics), index=test.index) == test["topic"]).mean())
    show("pretrained body, untrained head: accuracy", acc(tuned))
    start = time.perf_counter()
    fine_tune(scratch, tokenizer, few, topics)
    show("same model from random weights: accuracy", acc(scratch))
    fine_tune(tuned, tokenizer, few, topics)
    show("fine-tuned pretrained model: accuracy", acc(tuned))
    show("seconds for both training runs", round(time.perf_counter() - start))
    print("  Identical architecture, identical 20 examples, identical training.")
    print("  From random weights it can't even beat TF-IDF; from pretrained")
    print("  weights it beats everything here. That gap is transfer learning.")
    return tuned


# ---------------------------------------------------------------------------
# 5. Words it was never shown
# ---------------------------------------------------------------------------

def section_new_words(tuned, tokenizer, vec, clf, topics):
    heading("5. Words it was never shown")
    new = ["refund still not received",
           "every dock near the mall was full",
           "I can't log into my account",
           "the handlebar is loose"]
    truth = ["payment", "station", "app", "bike"]
    show("the right topics", truth)
    show("TF-IDF predicts", list(clf.predict(vec.transform(new))))
    show("fine-tuned model predicts", predict(tuned, tokenizer, new, topics))
    print("  'refund', 'mall', 'log', 'handlebar': none of them appear in")
    print("  the 20 examples. The fine-tuned model gets three of four, TF-IDF")
    print("  one. Neither is perfect: twenty labels is still very few.")


# ---------------------------------------------------------------------------
# 6. What you pay for it
# ---------------------------------------------------------------------------

def section_costs(tuned, tokenizer, vec, clf, test, topics):
    heading("6. What you pay for it")
    texts = list(test["text"])
    start = time.perf_counter()
    clf.predict(vec.transform(texts))
    fast = time.perf_counter() - start
    start = time.perf_counter()
    predict(tuned, tokenizer, texts, topics)
    slow = time.perf_counter() - start
    show("TF-IDF model size (numbers)", int(clf.coef_.size + clf.intercept_.size))
    show("fine-tuned model size (millions of numbers)", round(sum(p.numel() for p in tuned.parameters()) / 1e6, 1))
    show("milliseconds per message: TF-IDF, fine-tuned", (round(1000 * fast / len(texts), 2), round(1000 * slow / len(texts), 1)))
    print("  Better with few labels, but tens of thousands of times bigger and")
    print("  hundreds of times slower. With plenty of labels and simple text,")
    print("  the baseline might be all you need. Measure both, then choose.")


def main():
    print("=" * 66)
    print("  Lesson 064: Pretrained models and transfer learning")
    print("=" * 66)
    few, test, topics = load_data()
    vec, clf = section_few(few, test)
    tokenizer, model = section_load()
    section_frozen(tokenizer, model, few, test)
    tuned = section_finetune(tokenizer, few, test, topics)
    section_new_words(tuned, tokenizer, vec, clf, topics)
    section_costs(tuned, tokenizer, vec, clf, test, topics)
    print()
    print("Borrow a model that already understands language, add a small head,")
    print("and fine-tune it on the little data you have.")
    print()


if __name__ == "__main__":
    main()
