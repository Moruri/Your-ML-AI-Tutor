"""
Lesson 001 - Why ML/AI, and how we'll learn

A tiny, honest demonstration of what "learning" means:
search for the rule that makes the smallest error on the examples you have,
then use that rule to guess about something new.

Pure Python. No installs. Run it with:

    python lesson.py

Don't worry if some syntax is new. Lesson 002 covers it. Just follow the story.
"""

# ---------------------------------------------------------------------------
# 1. DATA
#
# Six past pizza deliveries. Each pair is (distance in km, minutes it took).
# This is the whole "training set". Real ones have thousands of rows, but the
# idea is identical.
# ---------------------------------------------------------------------------

deliveries = [
    (1.0, 12),
    (2.0, 17),
    (3.5, 24),
    (5.0, 33),
    (6.0, 36),
    (8.0, 48),
]


def show_data():
    print("1. THE DATA - six past deliveries")
    print("   distance (km)   time (min)")
    for km, minutes in deliveries:
        print(f"   {km:>8.1f}       {minutes:>6}")
    print()


# ---------------------------------------------------------------------------
# 2. A RULE, AND HOW WRONG IT IS
#
# We assume the pattern has this shape:
#
#     minutes = base + per_km * distance
#
# "base" is the fixed cost (boxing, finding the bike). "per_km" is the riding
# cost. We don't know the two numbers yet. Finding them IS the learning.
# ---------------------------------------------------------------------------


def predict(base, per_km, distance):
    """Apply a candidate rule to one distance."""
    return base + per_km * distance


def average_error(base, per_km, data):
    """
    How wrong is this rule, on average, across all our examples?

    For each delivery we ask "how many minutes off were we?", ignore whether
    we were too high or too low (that's what abs does), and average the result.
    Smaller is better. Zero would mean we nailed every single delivery.
    """
    total = 0.0
    for km, actual in data:
        guess = predict(base, per_km, km)
        total += abs(guess - actual)
    return total / len(data)


def show_hand_made_guess():
    print("2. A HAND-MADE GUESS")
    base, per_km = 10, 4
    print(f"   Let's guess: minutes = {base} + {per_km} * km")
    print("   How does it do on our six deliveries?")
    for km, actual in deliveries:
        guess = predict(base, per_km, km)
        print(f"   {km:>5.1f} km -> guessed {guess:>5.1f}, actual {actual:>3}, "
              f"off by {abs(guess - actual):.1f}")
    err = average_error(base, per_km, deliveries)
    print(f"   Average error: {err:.2f} minutes. Not terrible. Can we do better?")
    print()
    return err


# ---------------------------------------------------------------------------
# 3. LEARNING = SEARCHING FOR THE RULE WITH THE SMALLEST ERROR
#
# This is the heart of the lesson. We try lots of candidate rules and keep the
# one with the lowest average error. That's all "training a model" means.
# Later in the course we'll meet much smarter ways to search (gradient descent,
# closed-form solutions). They find the same kind of answer, faster.
# ---------------------------------------------------------------------------


def frange(start, stop, step):
    """Like range(), but for decimals. Yields start, start+step, ... < stop."""
    value = start
    while value < stop:
        yield round(value, 6)
        value += step


def learn(data):
    """Return (best_base, best_per_km, best_error) found by brute-force search."""
    best_error = float("inf")
    best_rule = None
    tried = 0

    for base in frange(0, 20, 0.5):          # fixed cost: 0, 0.5, ..., 19.5
        for per_km in frange(0, 10, 0.1):    # per-km cost: 0, 0.1, ..., 9.9
            tried += 1
            err = average_error(base, per_km, data)
            if err < best_error:
                best_error = err
                best_rule = (base, per_km)

    base, per_km = best_rule
    return base, per_km, best_error, tried


def show_learning(hand_made_error):
    print("3. LEARNING - try many rules, keep the least wrong one")
    base, per_km, err, tried = learn(deliveries)
    print(f"   Tried {tried:,} candidate rules.")
    print(f"   Best one found: minutes = {base:g} + {per_km:g} * km")
    print(f"   Average error:  {err:.2f} minutes "
          f"(hand-made guess was {hand_made_error:.2f})")
    print()
    print("   Notice the error isn't zero. Riders hit red lights. Real data is")
    print("   noisy, and a rule that fit every point perfectly would be")
    print("   memorising, not learning. We'll come back to that in Phase 2.")
    print()
    return base, per_km


# ---------------------------------------------------------------------------
# 4. PREDICTION - use the learned rule on something new
# ---------------------------------------------------------------------------


def show_prediction(base, per_km):
    print("4. PREDICTION - a new order just came in")
    new_distance = 4.5
    guess = predict(base, per_km, new_distance)
    print(f"   Customer is {new_distance} km away.")
    print(f"   Tell them: about {guess:.0f} minutes.")
    print()

    far_away = 40.0
    far_guess = predict(base, per_km, far_away)
    print("   A word of caution. Ask the same rule about a 40 km delivery:")
    print(f"   {far_away} km -> {far_guess:.0f} minutes.")
    print("   The rule will happily answer, but it has never seen anything past")
    print("   8 km. Motorways, fuel stops, the rider quitting... it knows none")
    print("   of that. Predicting far outside your data is called extrapolation,")
    print("   and it's where confident models go to be wrong.")
    print()


# ---------------------------------------------------------------------------
# Run the whole story, top to bottom.
# ---------------------------------------------------------------------------


def main():
    print()
    print("=" * 66)
    print("  Lesson 001: data -> pattern -> prediction")
    print("=" * 66)
    print()
    show_data()
    hand_made_error = show_hand_made_guess()
    base, per_km = show_learning(hand_made_error)
    show_prediction(base, per_km)
    print("That loop in learn() is machine learning. Everything else is detail.")
    print()


if __name__ == "__main__":
    main()
