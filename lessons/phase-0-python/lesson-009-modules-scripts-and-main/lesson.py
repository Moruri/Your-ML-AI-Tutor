"""
Lesson 009 - Modules, scripts and `if __name__ == "__main__"`

Move the helpers we keep copying into their own file (coffee_tools.py) and
import them, see the three ways to write an import and where Python looks for
the file, find out what `__name__` is and why the `if __name__ == "__main__":`
line at the bottom of every lesson exists, and turn this script into a small
command-line tool that takes arguments.

Run it with:

    python lesson.py
    python lesson.py --top 2
    python lesson.py --help

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there.

This script writes no files.
"""

import argparse
import contextlib
import io
import subprocess
import sys
from pathlib import Path

import coffee_tools                                   # our own module, sitting next to this file
from coffee_tools import pounds, revenue_by_drink     # two names, borrowed directly
import coffee_tools as ct                             # the same module under a shorter name

HERE = Path(__file__).resolve().parent
ORDERS_CSV = HERE / "orders.csv"                      # the clean week from lessons 005-008


def heading(title):
    print()
    print(title)
    print("-" * len(title))


def show(label, value):
    print(f"  {label:<46} -> {value!r}")


# ---------------------------------------------------------------------------
# 1. A module is just a .py file
# ---------------------------------------------------------------------------


def section_module():
    heading("1. A module is just a .py file")

    print("  coffee_tools.py sits next to this script. `import coffee_tools` ran it once")
    print("  and handed us everything it defined, under the name coffee_tools:")
    show("coffee_tools.pounds(7970)", coffee_tools.pounds(7970))
    show("coffee_tools.MENU['latte']", coffee_tools.MENU["latte"])
    show("coffee_tools.SIZES", coffee_tools.SIZES)
    show("type(coffee_tools)", type(coffee_tools).__name__)
    show("Path(coffee_tools.__file__).name", Path(coffee_tools.__file__).name)

    print()
    print("  No copying, no pasting. Fix a bug in coffee_tools.py once, and every")
    print("  script that imports it gets the fix.")


# ---------------------------------------------------------------------------
# 2. Three ways to import
# ---------------------------------------------------------------------------


def section_import_styles():
    heading("2. Three ways to import")

    show("import coffee_tools; coffee_tools.pounds(380)", coffee_tools.pounds(380))
    show("from coffee_tools import pounds; pounds(380)", pounds(380))
    show("import coffee_tools as ct; ct.pounds(380)", ct.pounds(380))
    show("pounds is coffee_tools.pounds", pounds is coffee_tools.pounds)
    show("ct is coffee_tools", ct is coffee_tools)
    print("  All three reach the very same function. Only the name you type changes.")

    print()
    print("  The standard library works exactly the same way:")
    from collections import Counter
    import datetime as dt
    show("Counter('latte latte tea'.split())", dict(Counter("latte latte tea".split())))
    show("dt.date(2026, 9, 17).strftime('%A')", dt.date(2026, 9, 17).strftime("%A"))

    print()
    print("  The one to avoid: `from coffee_tools import *`. It dumps every name into")
    print("  your file, and a reader can no longer tell where `pounds` came from.")


# ---------------------------------------------------------------------------
# 3. Where Python looks for modules
# ---------------------------------------------------------------------------


def section_search_path():
    heading("3. Where Python looks for modules")

    print("  sys.path is the list of folders Python searches, in order. The first few:")
    for i, folder in enumerate(sys.path[:3]):
        name = Path(folder).name or folder
        print(f"    sys.path[{i}]  ...{name}")
    show("sys.path[0] is this script's folder", Path(sys.path[0]).resolve() == HERE)

    print()
    print("  Folder of the script you RAN comes first. That's why `import coffee_tools`")
    print("  works from anywhere: Python looks next to lesson.py before anywhere else.")
    print("  It's also the trap: call your own file csv.py or random.py and it wins over")
    print("  the standard library one, and nothing you import from it will work.")

    print()
    print("  Modules are only run ONCE per program. After that they come from a cache:")
    show("'coffee_tools' in sys.modules", "coffee_tools" in sys.modules)
    import coffee_tools as again
    show("import coffee_tools again; same object?", again is coffee_tools)


# ---------------------------------------------------------------------------
# 4. __name__, and the line at the bottom of every lesson
# ---------------------------------------------------------------------------


def section_dunder_name():
    heading("4. __name__, and the line at the bottom of every lesson")

    show("__name__ in this file", __name__)
    show("coffee_tools.__name__", coffee_tools.__name__)
    print("  Python sets __name__ in every module. The file you run gets '__main__'.")
    print("  A file that's imported gets its own name.")

    print()
    print("  So `if __name__ == '__main__':` means 'only if I was run directly'.")
    print("  Watch coffee_tools.py run as a script (in a separate Python, so we can see it):")
    result = subprocess.run(
        [sys.executable, str(HERE / "coffee_tools.py")],
        capture_output=True, text=True, check=True,
    )
    for line in result.stdout.splitlines():
        print(f"    {line}")
    print("  When we imported it in section 1, that self-check did NOT run. Same file,")
    print("  two jobs: a library when imported, a program when run.")


# ---------------------------------------------------------------------------
# 5. Command-line arguments: sys.argv, then argparse
# ---------------------------------------------------------------------------


def build_parser():
    """The command-line interface for this script. One place, easy to read."""
    parser = argparse.ArgumentParser(
        description="Weekly revenue report for the coffee shop.",
    )
    parser.add_argument("--csv", type=Path, default=ORDERS_CSV,
                        help="till export to read (default: orders.csv next to this script)")
    parser.add_argument("--top", type=int, default=3,
                        help="how many drinks to show (default: 3)")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the total line")
    return parser


def section_arguments():
    heading("5. Command-line arguments: sys.argv, then argparse")

    show("sys.argv  (what you typed, as strings)", [Path(sys.argv[0]).name, *sys.argv[1:]])
    print("  sys.argv[0] is the script. Everything after it is yours, always as strings.")

    print()
    print("  argparse turns that list into named, typed values, with --help for free.")
    print("  We can feed it a list directly to see what it does with different commands:")
    parser = build_parser()
    for typed in [[], ["--top", "2"], ["--quiet"], ["--top", "5", "--csv", "other.csv"]]:
        args = parser.parse_args(typed)
        pretty = " ".join(typed) or "(nothing)"
        print(f"    python lesson.py {pretty:<27} -> top={args.top!r}, quiet={args.quiet}, csv={args.csv.name}")

    print()
    print("  And a bad one, which argparse refuses with a message (it would normally exit):")
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):     # catch argparse's message so we can print it here
            parser.parse_args(["--top", "three"])
    except SystemExit as err:
        for line in captured.getvalue().splitlines():
            print(f"    {line}")
        print(f"    (and it asks to exit with code {err.code}: the shell's way of saying 'that failed')")


# ---------------------------------------------------------------------------
# 6. Putting it together: a script you can run with options
# ---------------------------------------------------------------------------


def report(csv_path, top, quiet=False):
    """Load, total and print the week. Returns (total_cents, cups) so it can be checked."""
    orders, rejects = ct.load_orders_carefully(csv_path)
    revenue = revenue_by_drink(orders)
    total = sum(revenue.values())
    cups = sum(row["quantity"] for row in orders)
    if not quiet:
        ranked = sorted(revenue.items(), key=lambda item: item[1], reverse=True)
        for drink, cents in ranked[:top]:
            print(f"    {drink:<12} {pounds(cents):>8}")
        if len(ranked) > top:
            print(f"    {'(others)':<12} {pounds(sum(c for _, c in ranked[top:])):>8}")
    print(f"    {'total':<12} {pounds(total):>8}   ({cups} cups, {len(orders)} orders, {len(rejects)} rejected)")
    return total, cups


def main(argv=None):
    args = build_parser().parse_args(argv)

    print("=" * 66)
    print("  Lesson 009: Modules, scripts and `if __name__ == \"__main__\"`")
    print("=" * 66)
    section_module()
    section_import_styles()
    section_search_path()
    section_dunder_name()
    section_arguments()

    heading("6. Putting it together: a script you can run with options")
    print(f"  Running report(csv={args.csv.name}, top={args.top}, quiet={args.quiet}):")
    total, cups = report(args.csv, args.top, args.quiet)
    if args.csv == ORDERS_CSV:
        assert total == 7970, "the clean week should total £79.70"
        assert cups == 23, "the clean week should be 23 cups"
        print("  £79.70 and 23 cups, same as ever, from code that lives in another file.")
    print("  Try: python lesson.py --top 2    or    python lesson.py --help")

    print()
    print("One file per job, import what you need, and let __main__ decide what runs. Now the exercises.")
    print()


if __name__ == "__main__":
    main()
