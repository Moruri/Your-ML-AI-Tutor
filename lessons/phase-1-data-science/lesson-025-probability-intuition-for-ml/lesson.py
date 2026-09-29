"""
Lesson 025 - Probability intuition for ML

Conditional probability, Bayes' rule and expected value — taught with
counts and small simulations before any heavy formula. Uses the September
shop orders for real frequencies, then a seeded Monte Carlo (lesson 016)
so the toy examples land on the same numbers every run.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders_september.csv"

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 14)
RNG = np.random.default_rng(25)   # reproducible toy examples (lesson 016)


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        print(f"  {label}:")
        for line in value.to_string().splitlines():
            print(f"      {line}")
    else:
        print(f"  {label:<46} -> {value!r}")


def load_orders(path):
    orders = pd.read_csv(path, parse_dates=["timestamp"])
    orders["revenue"] = orders["price"] * orders["quantity"]
    orders["is_morning"] = orders["timestamp"].dt.hour < 12
    orders["is_card"] = orders["payment"] == "card"
    return orders


# ---------------------------------------------------------------------------
# 1. Probability is a count with a denominator
# ---------------------------------------------------------------------------

def section_counts(orders):
    heading("1. Probability is a count with a denominator")
    n = len(orders)
    n_latte = int((orders["drink"] == "latte").sum())
    n_card = int(orders["is_card"].sum())
    n_morning = int(orders["is_morning"].sum())
    show("orders, count", n)
    show("P(latte)  = latte / all", round(n_latte / n, 3))
    show("P(card)   = card / all", round(n_card / n, 3))
    show("P(morning)= morning / all", round(n_morning / n, 3))
    print("  A probability is a share: favourable outcomes over possible ones.")
    print("  On this month of tickets, roughly three in ten orders are lattes,")
    print("  about four in five are card, and mornings take a bit over half.")
    return n


# ---------------------------------------------------------------------------
# 2. Conditional probability: 'given that…'
# ---------------------------------------------------------------------------

def section_conditional(orders):
    heading("2. Conditional probability: 'given that…'")
    # P(latte | morning) vs P(latte | afternoon)
    morning = orders[orders["is_morning"]]
    afternoon = orders[~orders["is_morning"]]
    p_latte_morning = (morning["drink"] == "latte").mean()
    p_latte_afternoon = (afternoon["drink"] == "latte").mean()
    p_latte = (orders["drink"] == "latte").mean()
    show("P(latte)", round(float(p_latte), 3))
    show("P(latte | morning)", round(float(p_latte_morning), 3))
    show("P(latte | afternoon)", round(float(p_latte_afternoon), 3))

    # card given drink
    by_drink = (
        orders.groupby("drink")["is_card"]
        .mean()
        .round(3)
        .sort_values(ascending=False)
    )
    show("P(card | drink)", by_drink)
    print("  The '|' means 'given that'. Conditioning on morning (or on drink)")
    print("  changes the share — sometimes a little, sometimes a lot. That is")
    print("  the whole idea: new information updates the denominator.")
    return morning, afternoon


# ---------------------------------------------------------------------------
# 3. Bayes with counts, not formulas
# ---------------------------------------------------------------------------

def section_bayes_counts():
    heading("3. Bayes with counts, not formulas")
    # Tiny medical-style test, told with shop loyalty cards
    # 1000 customers. 5% are "regulars". A "loyalty ping" fires for
    # 90% of regulars and 8% of visitors.
    n = 1000
    n_regular = 50
    n_visitor = n - n_regular
    ping_given_regular = 0.90
    ping_given_visitor = 0.08
    true_pos = int(n_regular * ping_given_regular)     # 45
    false_pos = int(n_visitor * ping_given_visitor)    # 76
    ping_total = true_pos + false_pos
    p_regular_given_ping = true_pos / ping_total

    show("customers in the thought experiment", n)
    show("true regulars (5%)", n_regular)
    show("pings among regulars (90%)", true_pos)
    show("pings among visitors (8%)", false_pos)
    show("total pings", ping_total)
    show("of those pings, how many are regulars?",
         f"{true_pos}/{ping_total} = {p_regular_given_ping:.3f}")
    print("  The ping fires 121 times. Only 45 of those are real regulars.")
    print("  So P(regular | ping) ≈ 0.37 — not 0.90. The test is sensitive,")
    print("  but regulars are rare, so most pings are false alarms. Counts")
    print("  first; the formula just compresses the same arithmetic.")


# ---------------------------------------------------------------------------
# 4. Bayes as an update (same story, formula form)
# ---------------------------------------------------------------------------

def section_bayes_formula():
    heading("4. Bayes as an update (same story, formula form)")
    prior = 0.05
    likelihood = 0.90          # P(ping | regular)
    false_alarm = 0.08         # P(ping | visitor)
    # P(ping) = P(ping|R)P(R) + P(ping|V)P(V)
    p_ping = likelihood * prior + false_alarm * (1 - prior)
    posterior = (likelihood * prior) / p_ping
    show("prior P(regular)", prior)
    show("likelihood P(ping | regular)", likelihood)
    show("P(ping)  (law of total probability)", round(p_ping, 4))
    show("posterior P(regular | ping)", round(posterior, 3))
    print("  Same answer as the count table (~0.37). Bayes' rule is:")
    print("  posterior ∝ likelihood × prior, then renormalise by P(data).")
    print("  In ML you'll meet this as updating beliefs when new evidence")
    print("  arrives — spam filters, medical models, A/B readouts.")


# ---------------------------------------------------------------------------
# 5. Expected value: the long-run average payoff
# ---------------------------------------------------------------------------

def section_expected(orders):
    heading("5. Expected value: the long-run average payoff")
    rev = orders["revenue"]
    # Empirical expected order revenue = mean
    show("E[order revenue]  (= mean)", round(float(rev.mean()), 3))

    # A tip game: with P=0.3 tip is 10% of bill, else 0. Expected tip?
    tip_rate = 0.10
    p_tip = 0.30
    expected_tip = p_tip * tip_rate * float(rev.mean())
    show("P(customer tips)", p_tip)
    show("tip size when they do (10% of bill)", tip_rate)
    show("E[tip per order]", round(expected_tip, 3))

    # Discrete EV from a small drink lottery
    # Suppose a free-drink raffle: 1/100 wins a £4.20 median drink
    p_win = 0.01
    prize = 4.20
    cost = 0.20   # raffle ticket
    ev_raffle = p_win * prize - cost
    show("raffle: E[net] = 0.01×£4.20 − £0.20", round(ev_raffle, 3))
    print("  Expected value is the probability-weighted average outcome.")
    print("  Positive EV games favour you in the long run; negative ones")
    print("  favour the house. One draw can still lose — EV is about many")
    print("  repeats, not a single ticket.")
    return float(rev.mean())


# ---------------------------------------------------------------------------
# 6. A tiny Monte Carlo: simulate, don't just derive
# ---------------------------------------------------------------------------

def section_monte_carlo(mean_rev):
    heading("6. A tiny Monte Carlo: simulate, don't just derive")
    # Simulate 10_000 orders: tip with p=0.3 of 10% of a random bill
    # drawn from a simple model around the shop mean
    n_sim = 10_000
    bills = RNG.choice(
        [2.5, 3.4, 4.2, 5.5, 7.4, 9.0, 12.0],
        size=n_sim,
        p=[0.12, 0.25, 0.28, 0.15, 0.10, 0.07, 0.03],
    )
    tips = np.where(RNG.random(n_sim) < 0.30, 0.10 * bills, 0.0)
    show("simulated orders", n_sim)
    show("mean bill in the sim", round(float(bills.mean()), 3))
    show("mean tip in the sim", round(float(tips.mean()), 3))
    show("theory E[tip] ≈ 0.30 × 0.10 × E[bill]",
         round(0.30 * 0.10 * float(bills.mean()), 3))
    show("share of orders that tipped", round(float((tips > 0).mean()), 3))
    print("  Theory and simulation agree within noise. When a closed-form")
    print("  answer is awkward, draw many random outcomes and average —")
    print("  that's Monte Carlo. Seed the generator (lesson 016) so the")
    print("  demo is reproducible.")

    assert abs(tips.mean() - 0.30 * 0.10 * bills.mean()) < 0.02
    assert mean_rev > 4.0  # shop mean order revenue is around £5.46
    print("  Sanity checks passed.")


def main():
    print("=" * 66)
    print("  Lesson 025: Probability intuition for ML")
    print("=" * 66)
    orders = load_orders(ORDERS_CSV)
    section_counts(orders)
    section_conditional(orders)
    section_bayes_counts()
    section_bayes_formula()
    mean_rev = section_expected(orders)
    section_monte_carlo(mean_rev)
    print()
    print("Counts before formulas, condition before you update, and simulate")
    print("when the algebra gets sticky. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
