# Lesson 079 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere of
your own and edit the copy (it writes an `output/` folder next to
wherever it sits). From the repo root:

```bash
cd lessons/phase-5-projects/lesson-079-serving-a-model-as-an-api
python lesson.py
cp lesson.py ../../../my_service.py
cd ../../..
python my_service.py
```

---

## 1. Which status code?

For each request, say which status code our server should answer with
and why: (a) a valid request for Lakeside; (b) a body that is the text
`hello`, not JSON; (c) `rain_mm` of `-4`; (d) a valid request, but the
history file is missing the day before; (e) a request to
`/predict/market`; (f) `predict` hits a bug and raises `KeyError`.

<details>
<summary>Check yourself</summary>

(a) 200. (b) 400: the body isn't a JSON object, so the caller must fix
it. (c) 400: it's JSON, but the value is impossible. (d) 422: the
request is fine; the server can't answer it with what it knows. (e) 404:
no such route. (f) 500: the caller did nothing wrong. The useful split
is **4xx means "you should change the request"** and **5xx means "we
should change the server"**. Getting it right tells whoever is on call
whose problem a spike of errors is.

</details>

## 2. Why load the model once?

A teammate moves `pickle.loads(...)` inside `predict`, "so it always
uses the latest file". Name two things that get worse, and one better
way to pick up a new model.

<details>
<summary>Check yourself</summary>

Every request now reads and unpickles the file, so each one gets slower
(often much slower for big models). And if a new file is being written
while a request reads it, that request may get a half-written model and
crash. A better way: load at start-up, and to deploy a new model, start
a new server with it, check `/health` shows the new `model_version`,
then send traffic to it and stop the old one. Some services add a
"reload" route that swaps the model in one step, under a lock. Either
way, the version string in every answer tells you which model replied.

</details>

## 3. Same answer, or not?

The service's answer for Market on 2026-10-30 matched the in-process
prediction exactly. Name three changes that would make them disagree
while everything still "works".

<details>
<summary>Check yourself</summary>

Some examples: the server builds features with its own copy of the code
and counts today in `avg_7`; the server orders the feature columns
differently from training (scikit-learn may warn, or may not, depending
on how the data is passed); the client sends Fahrenheit and the range
check is missing; the server loads an older `model.pkl` than the one you
tested; `history.csv` on the server is a day behind. None of these
crash. That's why the "same as predicting in-process" check belongs in
your tests, and why the model and feature versions belong in every
answer.

</details>

## 4. Hands-on: a batch endpoint and a second terminal

Edit `my_service.py`.

1. Add a `POST /predict_batch` route that takes `{"rows": [...]}`, a
   list of the usual request bodies, and returns one result per row in
   the same order. A bad row should give an error *for that row only*,
   not fail the whole batch.
2. Change the server to listen on port `8000` instead of `0`, and
   replace section 7 with `input("Server running. Press Enter to stop.")`
   before the shutdown, so it stays up.
3. While it waits, open a **second terminal** and call it with `curl`
   (see the README), or with a three-line Python script that uses
   `urllib`. Try `/health`, one prediction and a batch of three rows
   with one bad row.

<details>
<summary>Check yourself</summary>

The core of the route can reuse what's there:

```python
results = []
for row in body["rows"]:
    problems = self.service.check(row)
    if problems:
        results.append({"error": problems})
        continue
    pred = self.service.predict(row)
    results.append({"predicted_rides": pred} if pred is not None
                   else {"error": ["no history for the 7 days before that date"]})
return 200, {"results": results, "model_version": self.service.card["model_version"]}, body
```

A 200 with per-row errors is a common, reasonable choice for batches.
Also check `rows` is a list and cap its length (say 1,000), or one huge
request can tie the server up. If your second terminal gets "connection
refused", the server isn't running or is on another port; if port 8000
is "already in use", something else has it, so pick 8001.

</details>

## 5. Optional puzzle: the server that knows the future

Our server reads `history.csv` once, at start-up. It's now running in
production for a month. What goes wrong, and when? How would you fix
it without ever retraining the model?

<details>
<summary>Check yourself</summary>

From the second day on, every request for "today" gets a 422: the
history stops at start-up, so the day before today isn't in it. (A
server without our strict "yesterday must be there" check would be
worse: it would quietly use stale days as if they were yesterday.)
The fix is to make history a **live** input: the server reads recent
rides from wherever they're stored (a database, or a small file updated
every night), or the caller sends the last seven days in the request.
The model itself doesn't need to change for this. Its *inputs* need to
be as fresh as the day you're predicting, which is lesson 078's feature
pipeline again, now on a schedule.

</details>
