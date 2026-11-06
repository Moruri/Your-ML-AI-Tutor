"""
Lesson 076 - Project structure, tests and reproducibility

A notebook that works today and nobody can re-run next month is a liability.
This lesson builds a small, real project on disk (a package, tests, a config
file and pinned requirements), then shows the three habits that make a project
trustworthy: tests that guard the data code, seeds that make runs repeatable,
and a fingerprint that tells you exactly what produced a result.

The project is written by this script into an `output/` folder next to it
(ignored by git), so you can open the files, break them, and re-run.

Run it with (inside the venv from lesson 013):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

No downloads and no data file: the demand data is simulated by the project
itself. It takes about 20 seconds, because it starts Python a few times.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE / "output" / "bike_demand"


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


def run(args, **extra_env):
    """Run a command inside the project folder and return (exit code, text)."""
    env = dict(os.environ, PYTHONPATH=str(PROJECT / "src"), PYTHONDONTWRITEBYTECODE="1", **extra_env)
    done = subprocess.run(args, cwd=PROJECT, env=env, capture_output=True, text=True)
    return done.returncode, (done.stdout + done.stderr).strip()


# ---------------------------------------------------------------------------
# The files of the project. In real life you'd just write these in an editor.

DATA_PY = '''"""Making and checking the raw data."""
import numpy as np
import pandas as pd

REQUIRED = ["date", "station", "temp_c", "rain_mm", "rides"]


def make_rides(seed, days=120, stations=("Uni Square", "Market", "Lakeside")):
    """Simulate daily rides per station. Same seed, same table."""
    rng = np.random.default_rng(seed)
    rows = []
    for s, base in zip(stations, (60, 45, 30)):
        dates = pd.date_range("2026-01-01", periods=days)
        temp = 22 + 5 * np.sin(np.arange(days) / 18) + rng.normal(0, 1.5, days)
        rain = np.where(rng.random(days) < 0.25, rng.exponential(6, days), 0.0)
        weekend = (dates.dayofweek >= 5) * 12
        rides = base + 1.2 * (temp - 22) - 0.9 * rain + weekend + rng.normal(0, 4, days)
        rows.append(pd.DataFrame({"date": dates, "station": s, "temp_c": temp.round(1),
                                  "rain_mm": rain.round(1), "rides": rides.round().clip(0).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def validate(df):
    """Raise ValueError with a clear message if the table can't be trusted."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df[REQUIRED].isna().any().any():
        raise ValueError("missing values in required columns")
    if (df["rides"] < 0).any():
        raise ValueError("negative ride counts")
    if df.duplicated(["date", "station"]).any():
        raise ValueError("duplicate station-days")
    return df
'''

FEATURES_PY = '''"""Turning raw rows into model inputs. Only the past may be used."""
import pandas as pd


def add_features(df):
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    by_station = df.groupby("station")["rides"]
    df["lag_1"] = by_station.shift(1)
    df["avg_7"] = by_station.transform(lambda s: s.shift(1).rolling(7).mean())
    df["weekday"] = df["date"].dt.dayofweek
    return df


FEATURES = ["temp_c", "rain_mm", "lag_1", "avg_7", "weekday"]
'''

TRAIN_PY = '''"""One command: python -m bike_demand_pkg.train config.json"""
import hashlib
import json
import platform
import random
import sys
from importlib.metadata import version

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from bike_demand_pkg.data import make_rides, validate
from bike_demand_pkg.features import FEATURES, add_features


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def run(config):
    set_seeds(config["seed"])
    raw = validate(make_rides(config["seed"]))
    data = add_features(raw).dropna().reset_index(drop=True)
    cut = data["date"].max() - np.timedelta64(config["test_days"], "D")
    train, test = data[data["date"] <= cut], data[data["date"] > cut]
    model = RandomForestRegressor(n_estimators=100, random_state=config["seed"])
    model.fit(train[FEATURES], train["rides"])
    return {
        "mae_model": round(float(mean_absolute_error(test["rides"], model.predict(test[FEATURES]))), 3),
        "mae_yesterday": round(float(mean_absolute_error(test["rides"], test["lag_1"])), 3),
        "train_rows": len(train),
        "test_rows": len(test),
        "fingerprint": {
            "seed": config["seed"],
            "data_hash": hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()[:12],
            "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12],
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "python": platform.python_version(),
        },
    }


if __name__ == "__main__":
    config = json.load(open(sys.argv[1]))
    print(json.dumps(run(config), sort_keys=True))
'''

TEST_DATA = '''import unittest

import pandas as pd

from bike_demand_pkg.data import make_rides, validate


class DataTests(unittest.TestCase):
    def test_same_seed_same_table(self):
        pd.testing.assert_frame_equal(make_rides(1), make_rides(1))

    def test_different_seed_different_table(self):
        self.assertFalse(make_rides(1).equals(make_rides(2)))

    def test_clean_data_passes(self):
        validate(make_rides(1))

    def test_negative_rides_rejected(self):
        df = make_rides(1)
        df.loc[3, "rides"] = -5
        with self.assertRaisesRegex(ValueError, "negative"):
            validate(df)

    def test_duplicates_rejected(self):
        df = make_rides(1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate(pd.concat([df, df.iloc[[0]]]))

    def test_missing_column_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate(make_rides(1).drop(columns="rain_mm"))


if __name__ == "__main__":
    unittest.main()
'''

TEST_FEATURES = '''import unittest

from bike_demand_pkg.data import make_rides
from bike_demand_pkg.features import add_features


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.df = add_features(make_rides(1))

    def test_lag_is_yesterdays_rides_for_the_same_station(self):
        one = self.df[self.df["station"] == "Market"].reset_index(drop=True)
        self.assertEqual(one.loc[10, "lag_1"], one.loc[9, "rides"])

    def test_first_day_of_each_station_has_no_lag(self):
        firsts = self.df.groupby("station").head(1)
        self.assertTrue(firsts["lag_1"].isna().all())

    def test_today_never_leaks_into_todays_features(self):
        # Pretend today's rides were 10,000. Today's features must not move.
        before = self.df.loc[40, ["lag_1", "avg_7"]]
        raw = make_rides(1)
        day, station = self.df.loc[40, "date"], self.df.loc[40, "station"]
        raw.loc[(raw["date"] == day) & (raw["station"] == station), "rides"] = 10_000
        after = add_features(raw).loc[40, ["lag_1", "avg_7"]]
        self.assertEqual(list(before), list(after))


if __name__ == "__main__":
    unittest.main()
'''

CONFIG = {"seed": 76, "test_days": 20}
PKG = "bike_demand_pkg"


def pins():
    return "\n".join(f"{p}=={version(p)}" for p in ["numpy", "pandas", "scikit-learn"]) + "\n"


def write_project():
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    files = {
        "README.md": "# Bike demand\n\nDaily rides per station.\n\n    python -m bike_demand_pkg.train config.json\n"
                     "    python -m unittest discover -s tests\n",
        "requirements.txt": pins(),
        "config.json": json.dumps(CONFIG, indent=2) + "\n",
        f"src/{PKG}/__init__.py": "",
        f"src/{PKG}/data.py": DATA_PY,
        f"src/{PKG}/features.py": FEATURES_PY,
        f"src/{PKG}/train.py": TRAIN_PY,
        "tests/test_data.py": TEST_DATA,
        "tests/test_features.py": TEST_FEATURES,
    }
    for rel, text in files.items():
        path = PROJECT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return sorted(files)


# ---------------------------------------------------------------------------
heading("1. A layout you can find things in")

written = write_project()
for rel in written:
    print("   ", rel)
show("files written", len(written))
show("lines of code in src/", sum(len(p.read_text().splitlines()) for p in (PROJECT / "src").rglob("*.py")))

# ---------------------------------------------------------------------------
heading("2. Tests that guard the data code")

code, text = run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
show("exit code", code)
show("last line", text.splitlines()[-1])
show("tests that ran", int(re.search(r"Ran (\d+) tests", text).group(1)))

# ---------------------------------------------------------------------------
heading("3. A test catches a quiet bug")

# A classic: someone "tidies" the rolling average and forgets shift(1). Now
# today's rides sit inside today's feature, and the model looks brilliant.
fpath = PROJECT / "src" / PKG / "features.py"
good = fpath.read_text()
fpath.write_text(good.replace("s.shift(1).rolling(7).mean()", "s.rolling(7).mean()"))
code, text = run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
failed = [line for line in text.splitlines() if line.startswith("FAIL:")]
show("exit code with the bug", code)
show("failing test", failed[0].split(" ")[1])
show("summary", text.splitlines()[-1])

code, text = run([sys.executable, "-m", "bike_demand_pkg.train", "config.json"])
leaky = json.loads(text)
fpath.write_text(good)
code, text = run([sys.executable, "-m", "bike_demand_pkg.train", "config.json"])
honest = json.loads(text)
show("MAE with the leak (looks great)", leaky["mae_model"])
show("MAE after fixing it (the truth)", honest["mae_model"])
show("MAE of 'same as yesterday'", honest["mae_yesterday"])

# ---------------------------------------------------------------------------
heading("4. Seeds, and what they don't cover")

import numpy as np  # noqa: E402

show("unseeded, two draws equal?", bool(np.array_equal(np.random.rand(3), np.random.rand(3))))
show("seeded, two draws equal?", bool(np.array_equal(np.random.default_rng(76).random(3), np.random.default_rng(76).random(3))))

SNIPPET = "print(list({'uni', 'market', 'lakeside', 'depot', 'airport'}))"
first = {h: run([sys.executable, "-c", SNIPPET], PYTHONHASHSEED=h)[1] for h in ("1", "2")}
again = run([sys.executable, "-c", SNIPPET], PYTHONHASHSEED="1")[1]
show("set order, hash seed 1", first["1"])
show("set order, hash seed 2", first["2"])
show("same hash seed twice gives same order", first["1"] == again)
code, a = run([sys.executable, "-m", "bike_demand_pkg.train", "config.json"])
code, b = run([sys.executable, "-m", "bike_demand_pkg.train", "config.json"])
show("two full runs give identical results", a == b)

# ---------------------------------------------------------------------------
heading("5. Pinning the environment")

lines = (PROJECT / "requirements.txt").read_text().split()
show("pinned packages", [x.split("==")[0] for x in lines])
show("every pin is exact", all("==" in x for x in lines))


def check_env(path):
    """Return packages whose installed version differs from the pin file."""
    problems = []
    for line in Path(path).read_text().split():
        name, want = line.split("==")
        have = version(name)
        if have != want:
            problems.append((name, want, have))
    return problems


show("environment matches pins", check_env(PROJECT / "requirements.txt"))
(PROJECT / "requirements.txt").write_text(pins().replace("numpy==", "numpy==1.0.0+"))
mismatch = check_env(PROJECT / "requirements.txt")
show("after editing a pin", [(m[0], m[1].startswith("1.0.0+")) for m in mismatch])
(PROJECT / "requirements.txt").write_text(pins())

# ---------------------------------------------------------------------------
heading("6. One command, and a fingerprint")

res = json.loads(a)
show("MAE: model", res["mae_model"])
show("MAE: yesterday's rides", res["mae_yesterday"])
show("rows: train, test", (res["train_rows"], res["test_rows"]))
show("fingerprint keys", sorted(res["fingerprint"]))
show("seed in the fingerprint", res["fingerprint"]["seed"])
show("data hash length", len(res["fingerprint"]["data_hash"]))

(PROJECT / "config2.json").write_text(json.dumps({"seed": 77, "test_days": 20}))
code, text = run([sys.executable, "-m", "bike_demand_pkg.train", "config2.json"])
other = json.loads(text)
show("seed 77: MAE model", other["mae_model"])
show("seed 77: data hash changed", other["fingerprint"]["data_hash"] != res["fingerprint"]["data_hash"])
show("seed 77: config hash changed", other["fingerprint"]["config_hash"] != res["fingerprint"]["config_hash"])
