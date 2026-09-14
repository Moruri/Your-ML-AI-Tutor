# Lesson 004 - Exercises

Three quick checks, one hands-on task with a fresh dataset, and an optional
"which would you choose?" at the end. Predict first, then run.

For the hands-on task, work in a copy. From the repo root:

```bash
cp lessons/phase-0-python/lesson-004-lists-and-tuples-properly/lesson.py my_lesson_004.py
python my_lesson_004.py
```

(No data file today, so the copy runs from anywhere.)

---

## 1. Predict the slice

With `letters = ["a", "b", "c", "d", "e", "f"]`, what does each give?

```python
letters[1:3]
letters[-3:]
letters[::2]
letters[::-1][0]
letters[10:]
```

<details>
<summary>Check yourself</summary>

```
['b', 'c']            <- stop is not included
['d', 'e', 'f']       <- last three
['a', 'c', 'e']       <- every second, starting at 0
'f'                   <- reverse, then take the first: the original last item
[]                    <- slices never crash; there's just nothing there
```

If you got the last one as an error, that's the one to remember: `letters[10]`
would crash, `letters[10:]` won't.

</details>

## 2. Which list changed?

Read this, don't run it yet. What is `backup` at the end? And what is `x`?

```python
prices = [380, 220, 420]
backup = prices
x = prices.sort()
```

Then change the second line to `backup = sorted(prices)` and answer again.

<details>
<summary>Check yourself</summary>

First version: `backup` is `[220, 380, 420]`, because `backup = prices` made
a second name for the same list, and `.sort()` changed that one list in
place. `x` is `None`, because `.sort()` returns nothing. Two of the day's
traps in three lines.

Second version: `backup` is `[220, 380, 420]` *and* `prices` is still
`[380, 220, 420]` at that point, because `sorted()` builds a new list. (Then
the third line sorts `prices` too, so they end up looking the same, but they
are two separate lists.)

</details>

## 3. Loop or comprehension?

Rewrite (a) as a comprehension and (b) as a loop. Then decide: which of the
two was *better* in its original form, and why?

```python
# (a)
sizes = []
for _, _, size, _, _ in ORDERS:
    sizes.append(size.upper())

# (b)
labels = [f"{d} {s}" if q > 1 else d for _, d, s, _, q in ORDERS if s != "small" and q < 3]
```

<details>
<summary>Check yourself</summary>

```python
# (a) as a comprehension - this one is better short
sizes = [size.upper() for _, _, size, _, _ in ORDERS]

# (b) as a loop - this one is better long
labels = []
for _, drink, size, _, qty in ORDERS:
    if size == "small" or qty >= 3:
        continue
    if qty > 1:
        labels.append(f"{drink} {size}")
    else:
        labels.append(drink)
```

(a) is a single transform over a list: one breath. Comprehension.

(b) has a filter with two conditions *and* an `if`/`else` in the expression.
Nobody can read the one-liner without stopping. The loop is three times the
lines and ten times as clear. Choosing the loop here is not a failure to
know comprehensions; it's knowing when not to use them.

</details>

## 4. Hands-on: the fridge log

The cafe's milk fridge logs its temperature once an hour. Someone left the
door open on Monday afternoon. Here's the day, as `(hour, degrees_c)`
tuples:

```python
readings = [
    (0, 3.9), (1, 3.8), (2, 3.8), (3, 3.7), (4, 3.7), (5, 3.8),
    (6, 4.1), (7, 4.4), (8, 4.6), (9, 4.5), (10, 4.7), (11, 4.9),
    (12, 5.2), (13, 5.0), (14, 6.8), (15, 7.1), (16, 5.9), (17, 5.1),
    (18, 4.8), (19, 4.6), (20, 4.3), (21, 4.1), (22, 4.0), (23, 3.9),
]
```

Paste it into your copy and answer, one at a time:

**a) The three warmest readings.** Which hours, and how warm? (`sorted` with
a `key`, then a slice.)

**b) Over the limit.** Milk should stay at or below 5.0 C. Which hours were
over? Use a comprehension that unpacks each tuple and gives you just the
hours.

**c) Morning vs afternoon.** Split the day into the first 12 readings and
the last 12 with slices. Average temperature of each half, to 2 decimal
places. (Average = total divided by count. `sum(...)` with a generator over
the temperatures is the tidy way.)

**d) Coldest and warmest.** Write a function `coldest_and_warmest(readings)`
that returns *two* readings (the full `(hour, temp)` tuples, not just the
temperatures) and unpack the result into two names. `min` and `max` take
`key=` just like `sorted`.

**e) Biggest jump.** When did the temperature change the most from one hour
to the next? `zip(readings, readings[1:])` pairs each reading with the one
after it. That's slicing and `zip` doing something genuinely useful together;
take a moment to see why it works.

Hints, if you want them:

- For (a), `key=lambda r: r[1]` and `reverse=True`, then `[:3]`.
- For (b), the pattern is `[hour for hour, temp in readings if ...]`.
- For (c), `readings[:12]` and `readings[12:]`.
- For (e), inside the loop you'll have two tuples, say `before` and
  `after`. The jump is `after[1] - before[1]`. Track the biggest by absolute
  size (`abs(...)`), or build a list of `(hour, jump)` tuples and use `max`
  with a key.

<details>
<summary>Expected results</summary>

```
a) warmest three     [(15, 7.1), (14, 6.8), (16, 5.9)]
b) hours over 5.0    [12, 14, 15, 16, 17]   (5 hours)
c) morning avg 4.16, afternoon avg 5.07
d) coldest (3, 3.7), warmest (15, 7.1)
e) biggest jump at hour 14: +1.8 degrees (from 5.0 to 6.8)
```

</details>

<details>
<summary>One way to write it</summary>

```python
# a)
warmest = sorted(readings, key=lambda r: r[1], reverse=True)[:3]
print(f"a) {warmest}")

# b)
over = [hour for hour, temp in readings if temp > 5.0]
print(f"b) {over}   ({len(over)} hours)")

# c)
morning, afternoon = readings[:12], readings[12:]
morning_avg = sum(temp for _, temp in morning) / len(morning)
afternoon_avg = sum(temp for _, temp in afternoon) / len(afternoon)
print(f"c) morning avg {morning_avg:.2f}, afternoon avg {afternoon_avg:.2f}")

# d)
def coldest_and_warmest(readings):
    return min(readings, key=lambda r: r[1]), max(readings, key=lambda r: r[1])

coldest, warmest_one = coldest_and_warmest(readings)
print(f"d) coldest {coldest}, warmest {warmest_one}")

# e)
jumps = []
for before, after in zip(readings, readings[1:]):
    jumps.append((after[0], after[1] - before[1]))
hour, jump = max(jumps, key=lambda j: abs(j[1]))
print(f"e) biggest jump at hour {hour}: {jump:+.1f} degrees")
```

A few things to notice. In (c), `morning, afternoon = readings[:12],
readings[12:]` is unpacking a two-tuple of slices; you'll write that shape a
lot. In (d), the function returns a tuple of tuples, and it reads fine
because unpacking takes it apart on the next line. In (e), `zip(readings,
readings[1:])` works because slicing off the first item shifts the second
list along by one, so each pair is "this hour, next hour". The `:+.1f`
format spec prints the sign, which is a nice touch for a change.

Note the tuple-of-tuples in (e): `jumps` is a list of `(hour, change)`
records. Every time you find yourself wanting to sort or `max` over "a thing
and its score", that's the shape.

</details>

## 5. (Optional) List or tuple?

For each of these, would you reach for a list or a tuple? One-word answer
and a one-line reason.

1. A shopping basket the customer is still adding to.
2. A GPS coordinate `(latitude, longitude)`.
3. One row read from a CSV.
4. The whole `price` column from that CSV.
5. An RGB colour like `(255, 128, 0)`.
6. What a function returns when it computes both a mean and a count.
7. The key for a dict that groups sales by `(month, branch)`.

<details>
<summary>Check yourself</summary>

1. **List.** It grows and shrinks.
2. **Tuple.** Two parts, fixed shape, never changes.
3. **Tuple** (or a dict, as on Friday). A row is one record with fixed fields.
4. **List.** Many of the same kind of thing.
5. **Tuple.** Always exactly three parts.
6. **Tuple.** `return mean, count` and unpack it.
7. **Tuple.** It has to be; lists can't be dict keys.

If you said "list" for 2, 5 or 6 nothing will break. But the tuple tells the
next reader "this shape is fixed", and that's information worth giving them.

</details>

---

That's Week 2, Day 1 done. Lessons 005 and 006 arrive tomorrow (Tuesday). If
today felt like a lot, the best thing you can do tonight is nothing; it'll
have settled by morning.
