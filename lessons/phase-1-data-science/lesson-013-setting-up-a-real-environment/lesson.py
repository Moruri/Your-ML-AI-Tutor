"""
Lesson 013 - Setting up a real environment

A friendly health check for your Python setup. It tells you which Python is
running this script, whether you're inside a virtual environment, where
installed packages live, which of the Phase 1 packages are installed (and
whether their versions match requirements.txt), and what to do about anything
that's missing.

It uses only the standard library, so it runs before AND after you install
anything. Run it both ways and compare:

    python lesson.py                  # before: probably "not in a venv"
    source .venv/bin/activate         # Windows: .venv\\Scripts\\activate
    python lesson.py                  # after: everything ticked

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files and installs nothing. It only looks.
"""

import importlib
import importlib.metadata
import platform
import re
import sys
import sysconfig
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                       # lessons/phase-1-.../lesson-013-... -> repo root
REQUIREMENTS = REPO_ROOT / "requirements.txt"

# The packages Phase 1 needs, and the name you import each one by.
PHASE_1 = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "scipy": "scipy"}


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<44} {value}")


def tick(ok):
    return "[ok]" if ok else "[!!]"


# ---------------------------------------------------------------------------
# 1. Which Python is this?
# ---------------------------------------------------------------------------


def section_which_python():
    heading("1. Which Python is this?")
    show("Python version", platform.python_version())
    show("running from (sys.executable)", sys.executable)
    show("operating system", f"{platform.system()} {platform.release()}")
    ok = sys.version_info >= (3, 10)
    print(f"  {tick(ok)} Python 3.10 or newer is needed for this course.")
    return ok


# ---------------------------------------------------------------------------
# 2. Am I in a virtual environment?
# ---------------------------------------------------------------------------


def in_virtualenv():
    """True when running inside a venv: sys.prefix points at the venv, base_prefix at the real Python."""
    return sys.prefix != sys.base_prefix


def section_venv():
    heading("2. Am I in a virtual environment?")
    show("sys.prefix       (this environment)", sys.prefix)
    show("sys.base_prefix  (the Python it's built on)", sys.base_prefix)
    active = in_virtualenv()
    if active:
        print(f"  {tick(True)} Yes. Packages you install now go into {Path(sys.prefix).name}/, not your system.")
    else:
        print(f"  {tick(False)} No. You're using the system (or global) Python.")
        print("       Fine for Phase 0. For Phase 1, create and activate a venv (README, section 2).")
    return active


# ---------------------------------------------------------------------------
# 3. Where do packages live?
# ---------------------------------------------------------------------------


def section_site_packages():
    heading("3. Where do packages live?")
    purelib = sysconfig.get_paths()["purelib"]
    show("installed packages go in", purelib)
    inside = Path(purelib).resolve().is_relative_to(Path(sys.prefix).resolve())
    show("is that inside this environment?", "yes" if inside else "no")
    print("  Every environment has its own site-packages folder. That's the whole trick:")
    print("  two projects, two folders, no fights over versions.")


# ---------------------------------------------------------------------------
# 4. What's installed, and does it match requirements.txt?
# ---------------------------------------------------------------------------


def parse_version(text):
    """'2.2.3' -> (2, 2, 3). Stops at the first part that isn't a plain number (e.g. 'rc1')."""
    parts = []
    for piece in text.split("."):
        match = re.match(r"\d+", piece)
        if not match:
            break
        parts.append(int(match.group()))
    return tuple(parts)


def read_requirements(path):
    """requirements.txt -> {name: [(op, version_tuple), ...]} for active (uncommented) lines."""
    wanted = {}
    if not path.exists():
        return wanted
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"([A-Za-z0-9_.\-]+)\s*(.*)", line)
        name, spec = match.group(1).lower(), match.group(2)
        rules = []
        for part in filter(None, (p.strip() for p in spec.split(","))):
            op, version = re.match(r"(>=|<=|==|<|>)\s*(.+)", part).groups()
            rules.append((op, parse_version(version)))
        wanted[name] = rules
    return wanted


def satisfies(version, rules):
    """Does a version tuple meet every (op, bound) rule? Enough for the pins in this repo."""
    checks = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b,
              "<": lambda a, b: a < b, ">": lambda a, b: a > b}
    return all(checks[op](version[:len(bound)], bound) for op, bound in rules)


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def section_installed():
    heading("4. What's installed, and does it match requirements.txt?")
    wanted = read_requirements(REQUIREMENTS)
    show("requirements file", REQUIREMENTS.relative_to(REPO_ROOT) if REQUIREMENTS.exists() else "not found")
    results = {}
    for name in PHASE_1:
        version = installed_version(name)
        rules = wanted.get(name, [])
        pin = ",".join(f"{op}{'.'.join(map(str, b))}" for op, b in rules) or "(any)"
        if version is None:
            status, ok = "not installed", False
        elif satisfies(parse_version(version), rules):
            status, ok = f"{version}", True
        else:
            status, ok = f"{version}  (outside {pin})", False
        print(f"  {tick(ok)} {name:<11} wanted {pin:<14} have {status}")
        results[name] = ok
    return results


# ---------------------------------------------------------------------------
# 5. Can we actually import them?
# ---------------------------------------------------------------------------


def section_imports(installed):
    heading("5. Can we actually import them?")
    for name, module in PHASE_1.items():
        if not installed_version(name):
            print(f"  [--] {name:<11} skipped (not installed)")
            continue
        start = time.perf_counter()
        try:
            imported = importlib.import_module(module)
        except ImportError as err:
            print(f"  [!!] {name:<11} installed but won't import: {err}")
            installed[name] = False
            continue
        took = (time.perf_counter() - start) * 1000
        where = Path(imported.__file__).parent
        print(f"  [ok] {name:<11} imported in {took:5.0f} ms from ...{where.parent.name}/{where.name}")
    print("  'Installed' and 'importable' are nearly always the same, but checking both is how")
    print("  you catch a package installed into a DIFFERENT Python from the one you're running.")


# ---------------------------------------------------------------------------
# 6. Summary: what to do next
# ---------------------------------------------------------------------------


def section_summary(python_ok, venv_ok, packages):
    heading("6. Summary: what to do next")
    missing = [name for name, ok in packages.items() if not ok]
    print(f"  {tick(python_ok)} Python 3.10+")
    print(f"  {tick(venv_ok)} inside a virtual environment")
    print(f"  {tick(not missing)} Phase 1 packages installed at the right versions")
    print()
    if python_ok and venv_ok and not missing:
        print("  All set for Phase 1. Lesson 014 is waiting.")
        return
    print("  To fix, from the repo root:")
    if not venv_ok:
        print("    python -m venv .venv")
        print("    source .venv/bin/activate        # Windows: .venv\\Scripts\\activate")
    if missing:
        print("    python -m pip install -r requirements.txt")
    print("    python " + str(Path(__file__).resolve().relative_to(REPO_ROOT)))
    print("  Then run this script again. It should be all ticks.")


def main():
    print("=" * 66)
    print("  Lesson 013: Setting up a real environment")
    print("=" * 66)
    python_ok = section_which_python()
    venv_ok = section_venv()
    section_site_packages()
    packages = section_installed()
    section_imports(packages)
    section_summary(python_ok, venv_ok, packages)

    # Sanity checks on the little helpers, so the report above can be trusted.
    assert parse_version("2.2.3") == (2, 2, 3)
    assert parse_version("1.26.0rc1") == (1, 26, 0)
    assert satisfies((2, 2, 3), [(">=", (2, 2)), ("<", (3,))])
    assert not satisfies((3, 0, 0), [(">=", (2, 2)), ("<", (3,))])
    print()
    print("Know which Python you're running, keep each project in its own venv. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
