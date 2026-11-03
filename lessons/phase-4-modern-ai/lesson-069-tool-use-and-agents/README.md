# Lesson 069 - Tool use and agents

**Phase 4 - Modern AI: NLP, vision, generative and LLM basics** | Week 9, Day 2 | Tuesday 2026-11-03

> **Goal:** Let a model call Python functions, and learn where agent loops
> go wrong.

Time: about 45 minutes. Same venv as lesson 064, with `torch` and
`transformers` installed.

---

Yesterday's RAG system gave the model notes to read. But some answers
aren't in any document: how many bikes are at Westlands *right now*,
what a 70-minute ride costs *exactly*. Those live in a database or a
pricing function, and the model can't reach either. It can only write
text.

Here's the trick. If the text it writes is a structured request, like
"call `ride_price` with `minutes=70`", your code can run that function
and paste the result back into the conversation. That's **tool use**
(also called function calling). Put it in a loop (the model asks, your
code runs, the model reads the result, maybe asks again) and you have
what people call an **agent**.

Agents are exciting and easy to over-trust. So today we build a small
one for the bike-share's support desk, and then spend most of the
lesson finding the ways it breaks.

## How to follow along

Venv active, `python` in the repo root. Full script:

```bash
python lessons/phase-4-modern-ai/lesson-069-tool-use-and-agents/lesson.py
```

It needs no data file and writes no files. It uses lesson 066's chat
model, `Qwen/Qwen2.5-0.5B-Instruct`, already in your cache if you ran
that lesson. Replies are greedy, so your numbers should match. It takes
about three minutes.

The agent has three tools, all plain Python functions near the top of
the script:

- `station_status(station)` reads made-up live bike and dock counts.
- `ride_price(minutes)` works out a ride's price with the help centre's
  rules from lesson 068 (KSh 50 for 30 minutes, then KSh 20 per started
  15 minutes).
- `refund_ride(ride_id, reason)` *does* something: it sends money back.

## 1. A tool is a function plus a description

```
name                                           -> 'ride_price'
description                                    -> 'Price in KSh of a single ride of the given length.'
parameters                                     -> {'minutes': {'type': 'integer', 'description': 'How long the ride lasts, in minutes.'}}
required                                       -> ['minutes']
tool descriptions in the prompt                -> 3
prompt length with tools (characters)          -> 1597
prompt length without tools                    -> 171
```

`get_json_schema`, from `transformers`, reads a function's type hints
and docstring and turns them into a **tool schema**: a name, a
description, and the arguments with their types. The chat template
pastes all three schemas into the system prompt, which is why "hi"
grows from 171 characters to 1,597.

That's the whole of what the model knows about your tools. It never
sees the code. So the docstring is no longer just for humans: it's an
instruction to the model about *when* to use the tool and *what* to
pass. Vague docstrings make confused tool users.

## 2. The model asks, your code runs

```
question                                       -> 'How much will a 70 minute ride cost?'
raw reply                                      -> '<tool_call>\n{"name": "ride_price", "arguments": {"minutes": 70}}\n</tool_call>'
parsed call                                    -> {'name': 'ride_price', 'arguments': {'minutes': 70}}
your code ran it                               -> {'minutes': 70, 'price_ksh': 110}
```

The model didn't answer the question. It wrote a request, in the exact
JSON format it was trained to use, wrapped in `<tool_call>` tags.
`parse_calls` pulls the JSON out with a regular expression, and then
*your code* looks up the function and runs it.

This is the most important idea in the lesson: **the model never runs
anything**. Every call passes through your code, which can check it,
change it, or refuse it. That's where all the safety in an agent lives.

## 3. The agent loop

```
question                                       -> 'I want to ride from Westlands. Are there bikes, and what would 50 minutes cost?'
  step 1: call    station_status({'station': 'Westlands'}) = {'station': 'Westlands', 'bikes': 3, 'free_docks': 9}
  step 2: answer  'There are currently 3 bikes available at Westlands. The free docks can accommodate up to 9 bikes.'
tool calls made                                -> 1
```

`run_agent` is the loop, and it's short:

1. Send the conversation to the model.
2. If the reply has tool calls, run each one, append the call and its
   result to the conversation (the result goes in as a `tool` message),
   and go back to 1.
3. If the reply is plain text, that's the answer. Stop.

The mechanics worked: one call, a correct read of the result, a plain
answer. But the rider asked *two* things and got one. The price was
never looked up, and nothing in the loop noticed, because the loop only
knows the model stopped, not whether it finished. That gap is the first
lesson about agents: **"it stopped" isn't "it finished"**.

## 4. Where agent loops go wrong

**a) It doesn't call the tool when it should.**

```
question                                       -> 'How many bikes are at CBD?'
tool calls                                     -> 0
reply                                          -> "To provide you with the current status of bikes at the CBD, I will need to check the station data. Could you please specify which station you're interested in?"
```

It even says it needs to check the station data, and then asks the
rider instead. Asked about Kilimani or Westlands, it calls the tool
fine. Something about "CBD" throws it, and you'd never guess that
without trying it.

**b) It calls a tool with something the tool can't handle.**

```
step 1: call    station_status({'station': 'Karen'}) = {'error': "unknown station 'Karen'. Known stations: Westlands, Kilimani, Upper Hill, Cbd"}
step 2: answer  "I'm sorry, but I don't have information about the specific station you're asking about. Could you please provide me with the name of the station?"
```

There's no Karen station. A naive loop would crash here with a
`KeyError`. Ours doesn't, because of two habits worth stealing:

- The tool **checks its input** and returns a helpful error.
- `run_tool` wraps every call: unknown tool names, wrong argument names
  or types, and bad JSON all come back as `{"error": ...}` *results*
  instead of exceptions. The model reads the error and can recover.

It recovered politely here, though it didn't pass on the list of real
stations, which would have helped the rider more.

**c) It reaches for an action.**

```
step 1: call    refund_ride({'ride_id': 'R-1042', 'reason': 'Bike broke'}) = {'error': "refund_ride needs a staff member's approval, and it was not given"}
step 2: answer  "I'm sorry, but I can't proceed without your staff member's approval. Could you please confirm that you have the necessary permissions? If you don't, we might need to escalate this issue further."
refunds actually made                          -> []
```

Reading a bike count is harmless; sending money isn't. `refund_ride`
is listed in `ACTIONS`, so `run_tool` asks an `approve` function first,
and in this lesson it always says no. No refund happened.

Never rely on the prompt for this ("only refund when appropriate"). The
model is easy to talk into things (remember prompt injection from
lesson 066) and a 0.5B model is easy to confuse. **Actions get a gate
in code**, usually a person who clicks "approve".

Then read the reply. The model turned our error message into nonsense
for the *rider* ("confirm that you have the necessary permissions").
Error messages are read by the model and often end up in front of
users, so write them for that: say what happened and what to tell the
person.

**d) It never stops.**

```
calls before the guard stopped it              -> 4
reply                                          -> '(stopped: too many steps)'
```

Here we swapped the model for a fake one that asks for the same tool
forever. Real agents do this too: they retry a failing call, or bounce
between two tools. Each step costs time and money, so `max_steps` caps
the loop. Every agent loop needs one.

All four guards live in **your code**, not in the prompt.

## 5. Measure the agent

```
question                                  tool  ans |  tool  ans
How much will a 70 minute ride cost?       yes  yes |   yes  yes
What does a 20 minute ride cost?           yes  yes |   yes  yes
Are there any bikes at Kilimani right no   yes  yes |   yes  yes
Is there space to dock at Upper Hill?      yes  yes |   yes  yes
How many bikes are at CBD?                  no   no |    no   no
What's the capital of Kenya?               yes   no |   yes   no
plain spec: right tool / right answer          -> '5 / 6, 4 / 6'
strict spec: right tool / right answer         -> '5 / 6, 4 / 6'
```

Six tasks, each with the tool it should use and a fact the answer must
contain, scored two ways, like lesson 068's retrieval and generation:

- **tool**: did it call the right tool (or, for the last question, no
  tool at all)?
- **ans**: did the final answer contain the right fact?

The left columns use the plain system prompt; the right use a stricter
one that says "never ask the rider for station counts or prices: call
a tool", and "if no tool fits, answer it briefly yourself".

Two things to notice:

- **The stricter prompt changed nothing.** Same 5 and 4. Prompt
  advice is cheap to try, and it doesn't always work, which is exactly
  why you measure instead of assuming.
- **The capital of Kenya.** It correctly used no tool, then failed to
  answer anyway: it asked which country we meant. Without tools, this
  model answers "Nairobi" easily. Giving a small model tools can make
  it *worse* at questions that don't need them, because it starts
  treating every question as a tool problem.

One more honest note: the Upper Hill check looks for "1" (one free
dock), which is a very easy string to find by accident, the same kind
of fluke lesson 068 warned about. Read the replies, not just the
score.

## What you can do now

- Describe a Python function as a tool schema, and explain why the
  docstring matters.
- Parse a model's tool calls and run them from your own code.
- Write an agent loop that feeds tool results back to the model.
- Guard the loop: validate calls, return errors as results, gate
  actions behind approval, cap the number of steps.
- Score tool choice and final answers separately on a small task set.

## What to do now

1. Run `lesson.py`. Then add a fourth station to `STATIONS` and ask
   about it. The tool picks it up with no other change.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   tries to make the agent finish the Westlands question.
3. Next weekday, lesson 070: computer vision with pretrained models,
   where we classify, detect and search photos with models someone else
   trained.

The model only writes requests. Your code decides what runs, checks
every call, keeps a person in the loop for actions, and caps the steps.
