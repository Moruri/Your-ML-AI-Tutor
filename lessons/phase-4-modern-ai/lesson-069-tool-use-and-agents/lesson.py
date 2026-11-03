"""
Lesson 069 - Tool use and agents

A language model can only write text. But if the text it writes is a request
like "please call ride_price with minutes=70", your code can run that function
and hand the result back. That's tool use, and putting it in a loop (ask,
call, read, ask again) is what people call an agent. Today we build a small
one for the bike-share's support desk, then go looking for the ways it breaks.

Run it with (inside the venv from lesson 013, with torch and transformers
installed as in lessons 051 and 064):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

It needs no data file and writes no files. It uses lesson 066's small chat
model, Qwen/Qwen2.5-0.5B-Instruct (about 1 GB, cached after the first
download). Replies are greedy, so your numbers should match the README. It
takes about three minutes on a laptop CPU.
"""

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema
from transformers.utils import logging as hf_logging

torch.set_num_threads(1)              # same numbers on every machine
hf_logging.set_verbosity_error()      # keep the library's chatter out of the lesson output

CHAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # lesson 066


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


# ---------------------------------------------------------------------------
# The tools: ordinary Python functions with honest docstrings
# ---------------------------------------------------------------------------

# Made-up live data: (bikes available, free docks) at each station.
STATIONS = {"westlands": (3, 9), "kilimani": (0, 12), "upper hill": (7, 1), "cbd": (11, 4)}
REFUNDS = []   # what refund_ride would have done, if anyone let it


def station_status(station: str) -> dict:
    """Live count of bikes and free docks at one station.

    Args:
        station: The station name, for example "Westlands".
    """
    key = station.strip().lower()
    if key not in STATIONS:
        return {"error": f"unknown station {station!r}. Known stations: {', '.join(s.title() for s in STATIONS)}"}
    bikes, docks = STATIONS[key]
    return {"station": station, "bikes": bikes, "free_docks": docks}


def ride_price(minutes: int) -> dict:
    """Price in KSh of a single ride of the given length.

    Args:
        minutes: How long the ride lasts, in minutes.
    """
    extra_blocks = -(-max(0, int(minutes) - 30) // 15)     # KSh 20 per started 15 minutes after the first 30
    return {"minutes": int(minutes), "price_ksh": 50 + 20 * extra_blocks}


def refund_ride(ride_id: str, reason: str) -> dict:
    """Refund a rider's ride to their M-Pesa or card.

    Args:
        ride_id: The ride's ID, for example "R-1042".
        reason: Why the ride is being refunded.
    """
    REFUNDS.append(ride_id)
    return {"ride_id": ride_id, "refunded": True}


TOOLS = [station_status, ride_price, refund_ride]
ACTIONS = {"refund_ride"}      # tools that change the world, not just read it


class Chat:
    """Lesson 066's chat model, now told which tools exist."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL).eval()

    def prompt_text(self, messages, tools=TOOLS):
        return self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)

    def __call__(self, messages, tools=TOOLS, max_new_tokens=80):
        ids = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                                 return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


SPEC = "You are the bike-share support assistant. Use the tools for live station data and prices."


def ask(question, spec=SPEC):
    return [{"role": "system", "content": spec}, {"role": "user", "content": question}]


# ---------------------------------------------------------------------------
# 1. A tool is a function plus a description
# ---------------------------------------------------------------------------

def section_schema(chat):
    heading("1. A tool is a function plus a description")
    schema = get_json_schema(ride_price)["function"]
    show("name", schema["name"])
    show("description", schema["description"])
    show("parameters", schema["parameters"]["properties"])
    show("required", schema["parameters"]["required"])
    text = chat.prompt_text(ask("hi"))
    show("tool descriptions in the prompt", text.count('{"type": "function"'))
    show("prompt length with tools (characters)", len(text))
    show("prompt length without tools", len(chat.prompt_text(ask("hi"), tools=None)))
    print("  The model never sees your code, only this description, pasted into")
    print("  its prompt as text. A vague docstring makes a confused tool user.")


# ---------------------------------------------------------------------------
# 2. The model asks, your code runs
# ---------------------------------------------------------------------------

def parse_calls(reply):
    """Pull every <tool_call>{...}</tool_call> out of a reply. Bad JSON is kept as an error."""
    calls = []
    for raw in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", reply, re.S):
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            calls.append({"name": None, "arguments": {}, "error": f"not valid JSON: {raw[:40]}"})
    return calls


def section_one_call(chat):
    heading("2. The model asks, your code runs")
    q = "How much will a 70 minute ride cost?"
    reply = chat(ask(q))
    show("question", q)
    show("raw reply", reply)
    call = parse_calls(reply)[0]
    show("parsed call", call)
    result = {t.__name__: t for t in TOOLS}[call["name"]](**call["arguments"])
    show("your code ran it", result)
    print("  The model wrote a request in a fixed format. It ran nothing. Your")
    print("  code decided to run it, and could just as well have refused.")


# ---------------------------------------------------------------------------
# 3. The agent loop
# ---------------------------------------------------------------------------

def run_tool(call, approve):
    """Run one call safely: unknown tools, bad arguments and actions are all handled here."""
    tools = {t.__name__: t for t in TOOLS}
    name, args = call.get("name"), call.get("arguments", {})
    if "error" in call:
        return {"error": call["error"]}
    if name not in tools:
        return {"error": f"there is no tool called {name!r}"}
    if name in ACTIONS and not approve(name, args):
        return {"error": f"{name} needs a staff member's approval, and it was not given"}
    try:
        return tools[name](**args)
    except Exception as exc:          # a wrong argument name or type
        return {"error": f"{type(exc).__name__}: {exc}"}


def deny(name, args):
    return False


def run_agent(chat, question, spec=SPEC, max_steps=4, approve=deny, verbose=False):
    """Ask, run any tool calls, hand back the results, repeat until a plain answer."""
    messages, trace = ask(question, spec), []
    for step in range(max_steps):
        reply = chat(messages)
        calls = parse_calls(reply)
        if not calls:
            if verbose:
                print(f"    step {step + 1}: answer  {reply!r}")
            return reply, trace
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function", "function": c} for c in calls]})
        for c in calls:
            result = run_tool(c, approve)
            trace.append((c.get("name"), c.get("arguments"), result))
            if verbose:
                print(f"    step {step + 1}: call    {c.get('name')}({c.get('arguments')}) = {result}")
            messages.append({"role": "tool", "name": c.get("name") or "unknown", "content": json.dumps(result)})
    return "(stopped: too many steps)", trace


def section_loop(chat):
    heading("3. The agent loop")
    q = "I want to ride from Westlands. Are there bikes, and what would 50 minutes cost?"
    show("question", q)
    reply, trace = run_agent(chat, q, verbose=True)
    show("tool calls made", len(trace))
    print("  The loop worked: call, read the result, answer. But the rider asked")
    print("  two things and got one. The price was never looked up, and nothing")
    print("  in the loop noticed. 'It stopped' is not the same as 'it finished'.")


# ---------------------------------------------------------------------------
# 4. Where agent loops go wrong
# ---------------------------------------------------------------------------

def section_failures(chat):
    heading("4. Where agent loops go wrong")
    print("  a) It doesn't call the tool when it should")
    q = "How many bikes are at CBD?"
    reply, trace = run_agent(chat, q)
    show("question", q)
    show("tool calls", len(trace))
    show("reply", reply)

    print("  b) It calls a tool with something the tool can't handle")
    q = "Are there free docks at Karen station?"
    reply, trace = run_agent(chat, q, verbose=True)

    print("  c) It reaches for an action")
    q = "Refund ride R-1042, the bike was broken."
    reply, trace = run_agent(chat, q, verbose=True)
    show("refunds actually made", REFUNDS)

    print("  d) It never stops")
    looping = lambda messages, **kw: '<tool_call>\n{"name": "station_status", "arguments": {"station": "CBD"}}\n</tool_call>'
    reply, trace = run_agent(looping, "Any bikes in the CBD?", max_steps=4)
    show("calls before the guard stopped it", len(trace))
    show("reply", reply)
    print("  The guards live in your code, not in the prompt: validate every call,")
    print("  return errors as results, gate actions behind a person, cap the steps.")


# ---------------------------------------------------------------------------
# 5. Measure the agent
# ---------------------------------------------------------------------------

# Question, the tool it should use (None = no tool needed), and a fact the answer
# must contain (a tuple means any one of them will do).
TASKS = [
    ("How much will a 70 minute ride cost?", "ride_price", "110"),
    ("What does a 20 minute ride cost?", "ride_price", "50"),
    ("Are there any bikes at Kilimani right now?", "station_status", ("0 bikes", "no bikes")),
    ("Is there space to dock at Upper Hill?", "station_status", "1"),
    ("How many bikes are at CBD?", "station_status", "11"),
    ("What's the capital of Kenya?", None, "Nairobi"),
]

STRICT_SPEC = (SPEC + " Never ask the rider for station counts or prices: call a tool to get them. "
               "If no tool fits the question, answer it briefly yourself.")


def score(chat, spec):
    rows = []
    for q, tool, fact in TASKS:
        reply, trace = run_agent(chat, q, spec=spec)
        used = [name for name, _, _ in trace]
        right_tool = (tool in used) if tool else not used
        facts = fact if isinstance(fact, tuple) else (fact,)
        rows.append((q, right_tool, any(f.lower() in reply.lower() for f in facts)))
    return rows


def section_measure(chat):
    heading("5. Measure the agent")
    plain, strict = score(chat, SPEC), score(chat, STRICT_SPEC)
    mark = lambda ok: "yes" if ok else "no"
    print(f"    {'question':<40} {'tool':>5} {'ans':>4} | {'tool':>5} {'ans':>4}")
    for (q, t1, a1), (_, t2, a2) in zip(plain, strict):
        print(f"    {q[:40]:<40} {mark(t1):>5} {mark(a1):>4} | {mark(t2):>5} {mark(a2):>4}")
    for name, rows in (("plain spec", plain), ("strict spec", strict)):
        show(f"{name}: right tool / right answer",
             f"{sum(r[1] for r in rows)} / {len(rows)}, {sum(r[2] for r in rows)} / {len(rows)}")
    print("  Score the tool choice and the final answer separately, as with RAG's")
    print("  retrieval and generation: they fail for different reasons.")


def main():
    print("=" * 66)
    print("  Lesson 069: Tool use and agents")
    print("=" * 66)
    chat = Chat()
    section_schema(chat)
    section_one_call(chat)
    section_loop(chat)
    section_failures(chat)
    section_measure(chat)
    print()
    print("The model only writes requests. Your code decides what runs, checks")
    print("every call, keeps a person in the loop for actions, and caps the steps.")
    print()


if __name__ == "__main__":
    main()
