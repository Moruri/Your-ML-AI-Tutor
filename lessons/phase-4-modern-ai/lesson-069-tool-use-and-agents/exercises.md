# Lesson 069 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv (with `transformers` installed, as in lesson 064).
For the hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-4-modern-ai/lesson-069-tool-use-and-agents/lesson.py my_lesson_069.py
python my_lesson_069.py
```

`Chat`, `parse_calls`, `run_tool`, `run_agent` and the three tools are
there to reuse.

---

## 1. Who runs the code?

A friend says: "Agents are dangerous because the model can run any code
it wants on your machine." Is that true of the agent in this lesson?

<details>
<summary>Check yourself</summary>

No. The model only writes text. `parse_calls` reads that text, and
`run_tool` decides whether to run anything, and only functions in
`TOOLS` can ever run. The danger is real, but it comes from the tools
*you* hand over: give an agent a tool that runs any shell command, and
then it can effectively run anything. Keep tools narrow (`ride_price`,
not `run_python`), and gate the ones that change the world.

</details>

## 2. Errors as results

`run_tool` catches exceptions and returns `{"error": ...}` instead of
letting the program crash. Give two reasons that's better for an agent.

<details>
<summary>Check yourself</summary>

First, the conversation survives: one bad call (a typo in a station
name, a wrong argument) doesn't end the whole session. Second, the
model reads the error and can fix its own mistake, for example by
calling again with a real station name. A crash teaches the model
nothing. (Do still log the errors for yourself: a tool that errors
often is a sign its description is unclear.)

</details>

## 3. Where should the gate go?

Someone suggests removing the `ACTIONS` check and instead adding "Only
refund a ride if the rider gives a good reason" to the system prompt.
Why is that worse?

<details>
<summary>Check yourself</summary>

A prompt is a request, not a rule. A rider can write a persuasive
"reason", a poisoned document can inject instructions (lesson 068,
exercise 2), and a small model can simply misread the situation. The
check in `run_tool` can't be argued with: no approval, no refund. Use
the prompt to make good behaviour *likely*, and code to make bad
behaviour *impossible*.

</details>

## 4. Hands-on: make it finish the job

In section 3 the agent answered the bike question and forgot the price.
Try a simple fix in code: after the loop ends, if the question mentions
minutes but `ride_price` was never called, run the agent again with a
reminder.

Predict first: does it now give the right price, KSh 90?

<details>
<summary>Expected result</summary>

```
tools called: ['station_status', 'station_status']
reply: 'There are currently 3 bikes available at Westlands. The free docks have 9 of them. If you need to take a ride, it will cost 50 minutes approximately KSh 75.'
```

Worse than before. It looked up the station *again*, never called
`ride_price`, and then made up a price: KSh 75, when the real answer is
KSh 90. The reminder made it *talk* about the price without *fetching*
it.

That's a lesson in itself. An answer can sound complete and still be
invented. The fix that works is a check against the trace: any number
the agent states as a price should appear in a `ride_price` result.
KSh 75 appears in no result, so the reply is flagged instead of sent.
For a step this predictable, you could also skip the model entirely:
if the question contains "N minutes", call `ride_price(N)` in code and
hand the model the result.

</details>

<details>
<summary>One way to write it</summary>

In your copy, replace the `main()` call at the very bottom with:

```python
if __name__ == "__main__":
    chat = Chat()
    q = "I want to ride from Westlands. Are there bikes, and what would 50 minutes cost?"
    reply, trace = run_agent(chat, q)
    if "minute" in q and "ride_price" not in [name for name, _, _ in trace]:
        reply, more = run_agent(chat, q + f" You already found: {trace[-1][2]}. Now look up the price too.")
        trace += more
    print("tools called:", [name for name, _, _ in trace])
    print("reply:", repr(reply))
```

</details>

## 5. (Optional) The CBD mystery

The agent calls `station_status` for Kilimani and Westlands, but not
for "CBD". Suggest two changes you could try, and how you'd know
whether either helped.

<details>
<summary>Check yourself</summary>

Two ideas: list the real station names in the tool's docstring (`One
of: Westlands, Kilimani, Upper Hill, CBD`), so the model knows "CBD"
is a station; or expand the abbreviation before the model sees it
("CBD" to "the CBD station"). Be warned: we tried the docstring change
while writing this lesson and it didn't fix it on its own, and "How
many bikes are at the CBD station?" fails too. Small models have odd
blind spots. The way to know is the same as section 5: add the failing
question to `TASKS`, change one thing, rerun the whole set, and check
nothing else broke.

</details>

---

That's tool use and agents. If one thing sticks: *the model proposes,
your code disposes. Every call, every action and every step count goes
through a check you wrote.*
