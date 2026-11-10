# Lesson 079 - Serving a model as an API

**Phase 5 - Projects and production intuition** | Week 10, Day 2 | Tuesday 2026-11-10

> **Goal:** Wrap a model in a small web service and call it from another
> program.

Time: about 45 minutes. Same venv as lesson 013; nothing new to install.

---

So far every model in this course has lived inside the script that
trained it. That's fine for you. It's no use to the ride-share app, the
operations dashboard or the colleague who writes JavaScript and has
never heard of scikit-learn. They don't want your notebook. They want to
ask a question and get a number back.

The usual way to make that possible is to put the model behind an
**API**: a small web service with an address, that takes a request
written in JSON and answers in JSON. Any program in any language can
talk to it. The model becomes a thing you *call*, like a weather service,
instead of a thing you *run*.

That sounds like a lot of new machinery. It isn't. Python's standard
library has a web server built in, so today we build the whole thing in
about a hundred lines: save the model, load it in a server, answer
requests, turn away bad ones politely, and keep a log of everything.
Real projects usually reach for a framework such as FastAPI or Flask
(more on that below), but they do the same jobs, and you'll understand
them better for having done it by hand once.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-5-projects/lesson-079-serving-a-model-as-an-api/lesson.py
```

It takes a couple of seconds. No downloads and no data file: the ride
data (three stations, 90 days) is simulated, with the same feature
function as lesson 078. The script starts a server on your own machine
(`127.0.0.1`, on whatever port is free), calls it, and shuts it down
before it finishes, so nothing is left running. Files go in an
`output/` folder next to the script, wiped at the start of every run.

Your numbers should match, with three exceptions marked "yours varies"
or explained below: timings, and the **model version** string, which is
a hash of the saved model file and so depends on your scikit-learn
version.

## 1. Train once, save the model and its record

```
  training rows, test rows                       -> (204, 45)
  test MAE (rides per day)                       -> 4.034
  files the server will load                     -> ['history.csv', 'model.pkl', 'model_card.json']
```

Training and serving are two different jobs, and they usually happen in
different programs, often on different machines. So the hand-over is a
set of **files**:

- `model.pkl`, the fitted ridge model, saved with Python's `pickle`;
- `model_card.json`, the record from lesson 078: model version, feature
  version, the feature list, the test score and the last training day;
- `history.csv`, the recent rides. The model uses `lag_1` and `avg_7`,
  so to predict for a day the server must know the seven days before
  it.

One warning that's worth taking seriously: **only unpickle files you
trust**. Loading a pickle can run code. Your own model file is fine; a
`.pkl` from a stranger on the internet is not. Pickles also tend to
break across library versions, which is one more reason the model card
records what was used.

## 2. A tiny prediction service

```
  server running in a background thread          -> True
  endpoints                                      -> ['GET /health', 'POST /predict']
```

The server has three parts, and every web framework has the same three
under different names:

- **Start-up.** `Service` loads the model, the card and the history
  *once*, when the server starts. Loading a model per request would be
  slow and wasteful.
- **Routes.** An address and a verb decide what code runs. `GET /health`
  answers "are you alive, and which model are you?" `POST /predict`
  takes a JSON body and returns a prediction. Anything else gets a 404.
- **Handlers.** `Handler.answer` reads the body, checks it, predicts and
  builds the reply. It returns a **status code** and a JSON payload: 200
  means fine, 400 means "your request is wrong", 404 means "no such
  page", 422 means "I understood you but can't do it", and 500 means
  "the bug is on my side".

Predicting a day works exactly as lesson 078 promised: take the last
seven days of history for that station, add the new row, run the
**same** `build_features`, and use the last row. There is no second,
hand-written copy of the features in the server, so there's no chance
of training-serving skew.

`ThreadingHTTPServer(("127.0.0.1", 0), ...)` listens only on your own
machine, and port `0` asks the system for any free port, so the script
never clashes with something else you're running. It runs in a
background **thread**, which lets the same script play both the server
and its client.

## 3. Calling it from another program

```
  GET /health                                    -> (200, 'ok')
  versions match the model card                  -> True
  POST /predict status                           -> 200
  answer                                         -> {'station': 'Market', 'date': '2026-10-30', 'predicted_rides': 42.7, 'model_version': '6e2e2d03'}
  same as predicting in-process                  -> True
  same day with 20 mm of rain                    -> 26.4
```

The client, `call`, is what any other program would do: turn a
dictionary into JSON, send it to an address, read the status code and
the JSON that comes back. It uses only `urllib` and knows nothing about
pandas, ridge models or features. From a terminal you could do the same
with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"station": "Market", "date": "2026-10-30", "temp_c": 24.0, "rain_mm": 0.0}'
```

(For that you'd run the server on a fixed port such as 8000 and leave it
running; exercise 4 does exactly that.)

The answer for Market on a dry 24-degree Friday is 42.7 rides, and it is
**identical** to predicting directly in the script. Check that whenever
you put a model behind a service: if the two disagree, something in the
plumbing (a column order, a unit, a feature) differs. Rain at 20 mm takes
the prediction down to 26.4, so the model is clearly reading the input.

Notice that the answer carries the **model version**. When somebody
says "the app predicted 43 on Friday", that string tells you which model
said it.

## 4. Bad requests get clear answers, not crashes

```
  missing a field              -> 400 ["missing field 'rain_mm'"]
  Fahrenheit by mistake        -> 400 ['temp_c should be in degrees Celsius, between -10 and 45']
  a station we've never seen   -> 400 ["unknown station 'Airport'"]
  temperature sent as text     -> 400 ["'temp_c' has the wrong type"]
  a date with no history       -> 422 ['no history for the 7 days before that date']
  a page that does not exist   -> 404 'no such page: /predictions'
  server still up afterwards                     -> True
```

Once your model is an API, the inputs come from programs you didn't
write, and they will be wrong in every possible way. The model itself
would happily turn 75.2 "degrees Celsius" into a confident nonsense
number. `Service.check` stands at the door and refuses it, with a
message that tells the caller what to fix. That's the request-sized
version of lesson 078's `compare`.

Two habits worth copying:

- **Return a list of problems, not just the first.** A caller who fixes
  one thing and gets the next error, five times in a row, will not
  thank you.
- **Never let one bad request take the server down.** Every path ends in
  a reply with a status code. Even an unexpected bug inside `predict` is
  caught and returned as a 500, rather than hanging up on the caller
  (which is what actually happened while this lesson was being written,
  and why that `try` is there).

## 5. Many callers at once

```
  requests sent, twice over                      -> 69
  8 callers at once got the same answers         -> True
  median ms inside the server (yours varies)     -> 4.3
```

A real service has many clients at once. We send 69 requests (three
stations, 23 days) one by one, then again from eight threads at the same
time, and get the same 69 answers. `ThreadingHTTPServer` handles each
request in its own thread. The one place threads could collide is the
log file, so writing to it is guarded by a **lock**.

Each prediction takes a few milliseconds inside the server, and most of
that is pandas building one row of features, not the model. Don't expect
eight threads to make it eight times faster: Python threads take turns
on CPU-heavy work, so they help with waiting (slow networks, slow
clients), not with sums. Production servers run several worker
*processes* for that, and lesson 081 is about making things fast enough.

## 6. The request log: the start of monitoring

```
  requests logged                                -> 145
  by status code                                 -> {200: 140, 400: 4, 422: 1}
  one log line                                   -> {'request': {'station': 'Market', 'date': '2026-10-30', 'temp_c': 24.0, 'rain_mm': 0.0}, 'status': 200, 'predicted_rides': 42.7, 'model_version': '6e2e2d03'}
  mean prediction by station                     -> {'Lakeside': 29.0, 'Market': 42.7, 'Uni Square': 57.1}
  share of requests rejected                     -> 0.034
```

Every `/predict` request, good or bad, becomes one line of JSON in
`output/requests.jsonl`: what was asked, the status, the answer, the
model version and how long it took. One JSON object per line ("JSON
Lines") is easy to append to and easy to load into pandas, as the last
two lines show.

This log is the most valuable thing a model service produces after its
predictions. From it you can answer: which inputs are we getting? Are
they changing? How often do we refuse requests, and why? Is it getting
slower? And when the real ride counts arrive, how wrong were we? That's
tomorrow's lesson, monitoring and drift, and it starts from a file like
this one.

## 7. Shutting down cleanly

```
  server thread stopped                          -> True
  anyone still answering?                        -> False
```

`server.shutdown()` stops the loop, `server_close()` releases the port,
and the thread ends. A script that leaves a server running in the
background is a nasty surprise for whoever runs it next.

## From here to a real deployment

What you built is the real shape of a model service; production adds
layers around it rather than changing it:

- **A framework.** FastAPI is the common choice for Python models. The
  `/predict` route becomes a decorated function, and the type checks in
  `check` become a small class that the framework validates for you (and
  documents automatically). Flask is the older, simpler option. Neither
  is in this course's `requirements.txt`, because the standard library
  was enough to learn the idea.
- **A proper server process.** `http.server` is meant for learning and
  local tools. In production you run the app under something like
  uvicorn or gunicorn, with several worker processes.
- **Packaging and hosting.** A container (Docker) or a hosting platform
  runs it somewhere other people can reach, ideally behind HTTPS and with
  an API key, so not just anyone can call it.

The checks, the version in every answer, the log and the clean shutdown
are yours to keep whatever you choose.

## What you can do now

- Save a model with its record, and load it in a separate program.
- Write a small HTTP service with a health check and a predict route.
- Call it from a client and check that it agrees with the model itself.
- Validate requests and answer with honest status codes and messages.
- Explain why serving must reuse the training feature function.
- Log every request so that monitoring is possible.

## What to do now

1. Run `lesson.py`, then open `output/model_card.json` and
   `output/requests.jsonl`.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   adds a batch endpoint and talks to your server from a second
   terminal.
3. Next, lesson 080: monitoring and drift, or how to notice when the
   world moves and your model doesn't.

A model nobody can call is a hobby. A model with an address is a
product.
