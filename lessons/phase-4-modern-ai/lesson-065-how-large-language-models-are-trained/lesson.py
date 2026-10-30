"""
Lesson 065 - How large language models are trained

A chat model is made in three stages. Pretraining: read a mountain of text and
learn to predict the next token. Instruction tuning: learn from examples
written by people what a good answer to a request looks like. Preference
tuning: learn from pairs of answers which one people liked better.

Today we do all three, in miniature, on one laptop CPU. Our "mountain" is the
800 made-up rider messages from lesson 061, read one character at a time, and
our model is lesson 063's Transformer block with a causal mask. It's tiny and
it writes badly, but every stage is the real recipe.

Run it with (inside the venv from lesson 013, with torch installed as in
lesson 051):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

It reads feedback.csv from this folder and writes no files. Every random step
has a fixed seed, so your numbers should match the README. It takes about a
two minutes on a laptop CPU.
"""

import copy
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)          # same numbers on every machine

HERE = Path(__file__).resolve().parent
BLOCK = 128                       # how many characters the model can look back over
POLITE, RUDE = " sorry about that!", " not our fault."


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def r3(x):
    return round(float(x), 3)


# ---------------------------------------------------------------------------
# 1. The only task: guess the next token
# ---------------------------------------------------------------------------

class Chars:
    """The simplest tokenizer: one token per character."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.index = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s):
        return [self.index[c] for c in s]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def section_task(df):
    heading("1. The only task: guess the next token")
    text = "\n".join(df["text"].str.lower()) + "\n"
    tok = Chars(text + "qa:" + POLITE + RUDE)
    show("characters of text", len(text))
    show("vocabulary size (distinct characters)", len(tok.chars))
    s = "the app crashed"
    print("  One sentence is many small exercises:")
    for i in (4, 8, 14):
        print(f"    {s[:i]!r:<20} -> next is {s[i]!r}")
    show("loss of a blind guess, ln(vocab)", r3(math.log(len(tok.chars))))
    print("  No labels needed: the text itself is the answer key. That's why")
    print("  pretraining can use any text at all, and as much of it as exists.")
    return text, tok


# ---------------------------------------------------------------------------
# 2. Pretraining a tiny GPT
# ---------------------------------------------------------------------------

class TinyGPT(nn.Module):
    """Lesson 063's blocks, with a causal mask so each position sees only the past."""

    def __init__(self, vocab, d=64, heads=4, depth=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(BLOCK, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        n = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(n))
        mask = nn.Transformer.generate_square_subsequent_mask(n)
        return self.head(self.norm(self.blocks(x, mask=mask, is_causal=True)))


def batches(data, size, gen):
    starts = torch.randint(len(data) - BLOCK - 1, (size,), generator=gen)
    x = torch.stack([data[s:s + BLOCK] for s in starts])
    y = torch.stack([data[s + 1:s + BLOCK + 1] for s in starts])
    return x, y


def lm_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


@torch.no_grad()
def generate(model, tok, prompt, n=60, temperature=0.8, seed=0, stop="\n"):
    gen = torch.Generator().manual_seed(seed)
    ids = tok.encode(prompt)
    for _ in range(n):
        logits = model(torch.tensor([ids[-BLOCK:]]))[0, -1] / temperature
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=gen).item()
        ids.append(nxt)
        if tok.chars[nxt] == stop:
            break
    return tok.decode(ids)[len(prompt):].rstrip("\n")


def section_pretrain(df, tok):
    heading("2. Pretraining a tiny GPT")
    join = lambda part: torch.tensor(tok.encode("\n".join(part["text"].str.lower()) + "\n"))
    train, held = join(df.iloc[:700]), join(df.iloc[700:])   # the last 100 messages stay unseen
    torch.manual_seed(65)
    model = TinyGPT(len(tok.chars))
    show("parameters", sum(p.numel() for p in model.parameters()))
    show("before training, it writes", generate(model, tok, "the "))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(65)
    hx, hy = batches(held, 64, torch.Generator().manual_seed(1))
    steps, losses = 1500, []
    for step in range(steps + 1):
        if step % 500 == 0:
            with torch.no_grad():
                losses.append((step, r3(lm_loss(model, hx, hy))))
        if step == steps:
            break
        x, y = batches(train, 32, gen)
        loss = lm_loss(model, x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    show("held-out loss at steps 0, 500, 1000, 1500", losses)
    for i, start in enumerate(["the ", "my ", "the app "]):
        show(f"after training, '{start}' continues", generate(model, tok, start, seed=i))
    show("characters it has read", steps * 32 * BLOCK)
    print("  From a blind guess to plausible rider-speak, with no labels at all.")
    print("  It has learned spelling, place names and the shape of a complaint.")
    print("  It has not learned to *do* anything; it only continues text.")
    return model, steps * 32 * BLOCK


# ---------------------------------------------------------------------------
# 3. Sampling: the model gives probabilities, not words
# ---------------------------------------------------------------------------

def section_sampling(model, tok):
    heading("3. Sampling: the model gives probabilities, not words")
    prompt = "the "
    with torch.no_grad():
        logits = model(torch.tensor([tok.encode(prompt)]))[0, -1]
    for t in (0.5, 1.0, 2.0):
        p = F.softmax(logits / t, -1)
        top = p.topk(4)
        show(f"after 'the ', temperature {t}: top 4",
             [(tok.chars[i], r3(v)) for v, i in zip(top.values, top.indices)])
    for t in (0.5, 1.0, 2.0):
        show(f"temperature {t}, three tries",
             [generate(model, tok, prompt, n=24, temperature=t, seed=s) for s in range(3)])
    print("  Every reply is a series of weighted dice rolls. Low temperature")
    print("  sharpens the odds (safe, repetitive); high temperature flattens")
    print("  them (lively, then nonsense). Chat models usually sit near 0.7-1.")


# ---------------------------------------------------------------------------
# 4. Instruction tuning: learn from worked examples
# ---------------------------------------------------------------------------

def to_example(text, topic, reply):
    """One training conversation. Returns (prompt, answer)."""
    return f"q: {text.lower()}\na:", f" {topic}," + reply + "\n"


def answer_loss(model, tok, pairs):
    """Loss on the answer characters only: we teach replies, not questions."""
    total = 0.0
    for prompt, answer in pairs:
        ids = tok.encode(prompt + answer)[-BLOCK - 1:]
        x, y = torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])
        logits = model(x)[0]
        k = len(answer)
        total = total + F.cross_entropy(logits[-k:], y[0, -k:])
    return total / len(pairs)


def finetune(model, tok, pairs, steps, lr, batch=16, seed=65):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(len(pairs), (batch,), generator=gen)
        loss = answer_loss(model, tok, [pairs[i] for i in idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def ask(model, tok, text, seed=0):
    prompt, _ = to_example(text, "", "")
    return generate(model, tok, prompt, n=30, seed=seed).strip()


def judge(model, tok, df, seeds=1):
    """Share of replies with the right topic, and share that are polite."""
    right = polite = 0
    for seed in range(seeds):
        for text, topic in zip(df["text"], df["topic"]):
            reply = ask(model, tok, text, seed=seed)
            right += reply.startswith(topic + ",")
            polite += reply.endswith(POLITE.strip())
    n = len(df) * seeds
    return r3(right / n), r3(polite / n)


def section_sft(base, tok, df):
    heading("4. Instruction tuning: learn from worked examples")
    demo, test = df.iloc[:600], df.iloc[700:]
    replies = [POLITE if i % 2 == 0 else RUDE for i in range(len(demo))]
    pairs = [to_example(t, c, r) for t, c, r in zip(demo["text"], demo["topic"], replies)]
    show("one worked example", "".join(pairs[0]))
    show("examples, and share with the polite reply", (len(pairs), r3(sum(r == POLITE for r in replies) / len(pairs))))
    msg = test["text"].iloc[0]
    show("question", msg)
    show("base model answers", ask(base, tok, msg))
    torch.manual_seed(65)
    sft = finetune(copy.deepcopy(base), tok, pairs, steps=600, lr=1e-3)
    show("instruction-tuned model answers", ask(sft, tok, msg))
    right, polite = judge(sft, tok, test)
    show("100 new questions: right topic, polite", (right, polite))
    print("  Six hundred written examples, and the model now answers instead")
    print("  of rambling. But it copied everything, including the half of our")
    print("  demonstrators who were rude. It does what the examples did.")
    return sft, demo, test


# ---------------------------------------------------------------------------
# 5. Preference tuning: learn which answer people liked
# ---------------------------------------------------------------------------

def seq_logprob(model, tok, prompt, answer):
    ids = tok.encode(prompt + answer)[-BLOCK - 1:]
    logits = model(torch.tensor([ids[:-1]]))[0]
    k = len(answer)
    logp = F.log_softmax(logits[-k:], -1)
    return logp.gather(1, torch.tensor(ids[-k:]).unsqueeze(1)).sum()


def dpo_loss(policy, ref, tok, triples, beta=0.1):
    """Direct Preference Optimisation: raise chosen over rejected, relative to a frozen copy."""
    losses = []
    for prompt, chosen, rejected in triples:
        with torch.no_grad():
            ref_gap = seq_logprob(ref, tok, prompt, chosen) - seq_logprob(ref, tok, prompt, rejected)
        gap = seq_logprob(policy, tok, prompt, chosen) - seq_logprob(policy, tok, prompt, rejected)
        losses.append(-F.logsigmoid(beta * (gap - ref_gap)))
    return torch.stack(losses).mean()


def section_dpo(sft, tok, demo, test):
    heading("5. Preference tuning: learn which answer people liked")
    triples = []
    for text, topic in zip(demo["text"], demo["topic"]):
        prompt, chosen = to_example(text, topic, POLITE)
        triples.append((prompt, chosen, to_example(text, topic, RUDE)[1]))
    show("one preference pair: prompt", triples[0][0])
    show("  chosen, rejected", triples[0][1:])
    ref = copy.deepcopy(sft).eval()
    policy = copy.deepcopy(sft)
    with torch.no_grad():
        p, c, r = triples[0]
        before = r3(seq_logprob(policy, tok, p, c) - seq_logprob(policy, tok, p, r))
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(65)
    for _ in range(40):
        batch = [triples[int(i)] for i in torch.randint(len(triples), (16,), generator=gen)]
        # the preference push, plus "keep imitating the chosen answer" so it doesn't drift
        loss = dpo_loss(policy, ref, tok, batch) + answer_loss(policy, tok, [(p, c) for p, c, _ in batch])
        opt.zero_grad()
        loss.backward()
        opt.step()
    p, c, r = triples[0]
    with torch.no_grad():
        after = r3(seq_logprob(policy, tok, p, c) - seq_logprob(policy, tok, p, r))
    show("log-prob gap, chosen minus rejected: before", before)
    show("log-prob gap, chosen minus rejected: after", after)
    show("100 new questions: right topic, polite", judge(policy, tok, test))
    print("  Nobody wrote a single new answer. People only picked the better of")
    print("  two, and the model moved towards that behaviour while keeping its")
    print("  topics. RLHF does the same job with a separate reward model.")


# ---------------------------------------------------------------------------
# 6. The same recipe, at real scale
# ---------------------------------------------------------------------------

def section_scale(model, tokens_seen):
    heading("6. The same recipe, at real scale")
    params = sum(p.numel() for p in model.parameters())
    show("our model: parameters, tokens seen", (params, tokens_seen))
    show("our tokens per parameter", round(tokens_seen / params, 1))
    show("GPT-3 (2020): parameters, tokens", ("175 billion", "300 billion"))
    show("GPT-3 tokens per parameter", round(300 / 175, 1))
    show("Chinchilla (2022): parameters, tokens", ("70 billion", "1.4 trillion"))
    show("Chinchilla tokens per parameter", round(1400 / 70, 1))
    print("  Chinchilla showed GPT-3-era models were undertrained: for a fixed")
    print("  compute budget, about 20 tokens per parameter works better than")
    print("  a bigger model on less text. Many newer models go far beyond 20.")


def main():
    print("=" * 66)
    print("  Lesson 065: How large language models are trained")
    print("=" * 66)
    df = pd.read_csv(HERE / "feedback.csv")
    text, tok = section_task(df)
    base, tokens_seen = section_pretrain(df, tok)
    section_sampling(base, tok)
    sft, demo, test = section_sft(base, tok, df)
    section_dpo(sft, tok, demo, test)
    section_scale(base, tokens_seen)
    print()
    print("Predict the next token on everything, imitate good examples, then")
    print("lean towards what people prefer. That's how a chat model is made.")
    print()


if __name__ == "__main__":
    main()
