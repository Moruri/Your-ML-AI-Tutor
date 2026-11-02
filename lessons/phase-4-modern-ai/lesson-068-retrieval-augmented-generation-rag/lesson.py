"""
Lesson 068 - Retrieval-augmented generation (RAG)

A language model only knows what was in its training data, and when it
doesn't know something it tends to make up a confident answer anyway. RAG is
the fix used almost everywhere: search your own documents for the passages
that matter (lesson 067), paste them into the prompt (lesson 066), and ask the
model to answer from them. Today we build one, and measure whether it helped.

Run it with (inside the venv from lesson 013, with torch and transformers
installed as in lessons 051 and 064):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

It reads help_centre.md (made-up help articles for the course's made-up
bike-share service) from this folder and writes no files. It uses two models
you already have from lessons 064 and 066: the small embedding model and the
small chat model (about 1 GB, cached after the first download). Replies are
generated greedily, so your numbers should match the README. It takes two or
three minutes on a laptop CPU.
"""

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

torch.set_num_threads(1)              # same numbers on every machine
hf_logging.set_verbosity_error()      # keep the library's chatter out of the lesson output

HERE = Path(__file__).resolve().parent
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # lesson 064 and 067
CHAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"                # lesson 066
DONT_KNOW = "I don't know."


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def r3(x):
    return round(float(x), 3) + 0.0


class Embedder:
    """Lesson 067's embedder: texts in, unit-length vectors out."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL).eval()

    def __call__(self, texts):
        batch = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        return F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1).numpy()


class Chat:
    """Lesson 066's chat model: a list of messages in, a greedy reply out."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL).eval()

    def __call__(self, messages, max_new_tokens=48):
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


# Questions a rider might ask, the fact a correct answer must contain, and the
# article that holds it. None means the help centre doesn't cover it.
QUESTIONS = [
    ("How much is one ride if I'm back within half an hour?", "50", "Prices and passes"),
    ("I'm a student. What does a month of riding cost me?", "1,500", "Prices and passes"),
    ("I paid twice for one trip. When do I get my money back?", "3 working days", "Paying with M-Pesa"),
    ("There's no space to park at the station. What should I do?", "15", "Full or empty docks"),
    ("What's the fine if I keep a bike overnight?", "500", "Late returns and lost bikes"),
    ("How old do I have to be to sign up?", "16", "Accounts and safety"),
    ("Until what time in the evening can I reach support?", "9pm", "Contacting support"),
    ("How do I tell you a bike has a flat tyre?", "spanner", "Reporting a broken bike"),
]
NOT_COVERED = [
    "Can I take a bike on the commuter train?",
    "Who founded the bike-share company?",
]


def correct(reply, fact):
    return fact.lower().replace(" ", "") in reply.lower().replace(" ", "")


def says_dont_know(reply):
    return "don't know" in reply.lower() or "do not know" in reply.lower()


# ---------------------------------------------------------------------------
# 1. Asking without notes
# ---------------------------------------------------------------------------

def closed_book(question):
    return [{"role": "system", "content": "You are the support assistant for a bike-share service. "
                                          "Answer the rider's question in one or two sentences."},
            {"role": "user", "content": question}]


def section_closed(chat):
    heading("1. Asking without notes")
    replies = [chat(closed_book(q)) for q, _, _ in QUESTIONS]
    for (q, fact, _), reply in list(zip(QUESTIONS, replies))[:2]:
        show("question", q)
        show("reply", reply)
    score = sum(correct(r, f) for r, (_, f, _) in zip(replies, QUESTIONS))
    show("correct facts, no notes", f"{score} / {len(QUESTIONS)}")
    print("  The model has never seen our price list, so it can't know. It")
    print("  answers anyway, in dollars, fluently. The one 'correct' is a fluke:")
    print("  '$1.50' happens to contain '50'. Simple checkers can be fooled too.")
    return score


# ---------------------------------------------------------------------------
# 2. Your documents, in chunks
# ---------------------------------------------------------------------------

def load_chunks():
    """Split the help centre into paragraphs, each tagged with its article title."""
    text = (HERE / "help_centre.md").read_text(encoding="utf-8")
    chunks = []
    for part in re.split(r"^## ", text, flags=re.M)[1:]:
        title, body = part.split("\n", 1)
        for para in body.strip().split("\n\n"):
            chunks.append({"title": title.strip(), "text": f"{title.strip()}: {' '.join(para.split())}"})
    return chunks


def section_chunks():
    heading("2. Your documents, in chunks")
    chunks = load_chunks()
    show("articles", len({c["title"] for c in chunks}))
    show("chunks (one per paragraph)", len(chunks))
    show("one chunk", chunks[4]["text"])
    show("longest chunk (words)", max(len(c["text"].split()) for c in chunks))
    print("  Small chunks keep each one about a single thing, so search can find")
    print("  exactly the paragraph that answers. The title stays attached so a")
    print("  chunk still makes sense on its own.")
    return chunks


# ---------------------------------------------------------------------------
# 3. Retrieve: find the passages that matter
# ---------------------------------------------------------------------------

def retrieve(embed, index, chunks, question, k=3):
    scores = index @ embed([question])[0]
    top = np.argsort(-scores)[:k]
    return [chunks[i] for i in top], scores[top]


def section_retrieve(embed, chunks):
    heading("3. Retrieve: find the passages that matter")
    index = embed([c["text"] for c in chunks])
    q = QUESTIONS[3][0]
    found, scores = retrieve(embed, index, chunks, q)
    show("question", q)
    for c, s in zip(found, scores):
        print(f"    {r3(s):.3f}  {c['text'][:62]}")
    for k in (1, 3):
        hits = sum(any(c["title"] == art for c in retrieve(embed, index, chunks, q, k)[0])
                   for q, _, art in QUESTIONS)
        show(f"right article in the top {k}", f"{hits} / {len(QUESTIONS)}")
    print("  Lesson 067's search, pointed at our own documents. Before going")
    print("  further, check this step alone: if the right passage isn't found,")
    print("  no prompt can save the answer.")
    return index


# ---------------------------------------------------------------------------
# 4. Augment and generate
# ---------------------------------------------------------------------------

RAG_SPEC = ("You are the support assistant for a bike-share service. Answer the rider's "
            "question using only the notes below. Quote prices, times and numbers exactly "
            f"as the notes give them. If the notes don't contain the answer, reply exactly: {DONT_KNOW}")


def rag_prompt(question, found):
    notes = "\n".join(f"- {c['text']}" for c in found)
    return [{"role": "system", "content": f"{RAG_SPEC}\n\nNotes:\n{notes}"},
            {"role": "user", "content": question}]


def answer(chat, embed, index, chunks, question, k=3):
    found, scores = retrieve(embed, index, chunks, question, k)
    return chat(rag_prompt(question, found)), found, scores


def section_generate(chat, embed, index, chunks, closed_score):
    heading("4. Augment and generate")
    q = QUESTIONS[0][0]
    reply, _, _ = answer(chat, embed, index, chunks, q)
    show("question", q)
    show("reply with notes", reply)
    replies = [answer(chat, embed, index, chunks, q)[0] for q, _, _ in QUESTIONS]
    score = sum(correct(r, f) for r, (_, f, _) in zip(replies, QUESTIONS))
    show("correct facts, no notes (section 1)", f"{closed_score} / {len(QUESTIONS)}")
    show("correct facts, with retrieved notes", f"{score} / {len(QUESTIONS)}")
    print("  Same model, same questions. The only change is three paragraphs of")
    print("  our own text in the prompt, chosen by search for each question.")
    return replies


# ---------------------------------------------------------------------------
# 5. When the answer isn't there
# ---------------------------------------------------------------------------

def section_not_covered(chat, embed, index, chunks):
    heading("5. When the answer isn't there")
    best_known = [retrieve(embed, index, chunks, q, 1)[1][0] for q, _, _ in QUESTIONS]
    for q in NOT_COVERED:
        reply, _, scores = answer(chat, embed, index, chunks, q)
        show("question", q)
        show("best search score", r3(scores[0]))
        show("reply", reply)
        show("said it doesn't know", says_dont_know(reply))
    show("lowest best score, covered questions", r3(min(best_known)))
    print("  It admitted it both times. But these off-topic questions scored")
    print("  higher in search than one real question, so a score threshold alone")
    print("  couldn't catch them, and section 4 showed the opposite failure: 'I")
    print("  don't know' with the answer right there. Test both kinds of question.")


# ---------------------------------------------------------------------------
# 6. Evaluate the parts separately
# ---------------------------------------------------------------------------

def section_evaluate(chat, embed, index, chunks, replies_k3):
    heading("6. Evaluate the parts separately")
    replies_k1 = [answer(chat, embed, index, chunks, q, k=1)[0] for q, _, _ in QUESTIONS]
    print(f"    {'question':<44} {'found':>5} {'k=1':>4} {'k=3':>4}")
    for (q, fact, art), r1, r3_ in zip(QUESTIONS, replies_k1, replies_k3):
        found = any(c["title"] == art for c in retrieve(embed, index, chunks, q, 3)[0])
        mark = lambda ok: "yes" if ok else "no"
        print(f"    {q[:44]:<44} {mark(found):>5} {mark(correct(r1, fact)):>4} {mark(correct(r3_, fact)):>4}")
    show("correct with 1 note", f"{sum(correct(r, f) for r, (_, f, _) in zip(replies_k1, QUESTIONS))} / {len(QUESTIONS)}")
    show("correct with 3 notes", f"{sum(correct(r, f) for r, (_, f, _) in zip(replies_k3, QUESTIONS))} / {len(QUESTIONS)}")
    print("  'found' is retrieval, 'k=1' and 'k=3' are the final answers. A wrong")
    print("  answer with 'found = yes' is a generation problem; with 'no', fix")
    print("  the search first. Measure each part, or you won't know what to fix.")


def main():
    print("=" * 66)
    print("  Lesson 068: Retrieval-augmented generation (RAG)")
    print("=" * 66)
    embed, chat = Embedder(), Chat()
    closed_score = section_closed(chat)
    chunks = section_chunks()
    index = section_retrieve(embed, chunks)
    replies = section_generate(chat, embed, index, chunks, closed_score)
    section_not_covered(chat, embed, index, chunks)
    section_evaluate(chat, embed, index, chunks, replies)
    print()
    print("Retrieve the right passages, put them in the prompt, ask for answers")
    print("from the notes only, and measure retrieval and answers separately.")
    print()


if __name__ == "__main__":
    main()
