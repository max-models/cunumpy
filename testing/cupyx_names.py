"""Compare the fake CuPy's ``cupyx.scipy`` names with the installed CuPy.

The fake ``cupyx.scipy`` (``cunumpy._fake_cupyx.NAMES``) lists the names of
the oldest supported CuPy (``NAMES_VERSION``). Run this where the real CuPy
is installed::

    python testing/cupyx_names.py           # report; exit 1 if the fake lists a name CuPy lacks
    python testing/cupyx_names.py --exact   # also exit 1 if CuPy has names the fake lacks
    python testing/cupyx_names.py --print   # print NAMES for the installed CuPy

``--print`` gives the new ``NAMES`` when raising the minimum CuPy version: run
it with that version installed and paste the output into ``_fake_cupyx.py``.
The GPU CI runs the same check in ``tests/unit/test_cupyx_names.py``.
"""

import argparse
import sys
import textwrap


def main() -> int:
    """Run the comparison; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--exact", action="store_true", help="fail on names the fake lacks too"
    )
    parser.add_argument(
        "--print", action="store_true", help="print NAMES for the installed CuPy"
    )
    args = parser.parse_args()

    import cupy

    if getattr(cupy, "__cunumpy_fake__", False):
        sys.exit("the fake CuPy is active: unset CUNUMPY_FAKE_CUPY")

    from cunumpy._fake_cupyx import (
        NAMES_VERSION,
        SUBPACKAGES,
        compare_names,
        installed_names,
    )

    real = installed_names()
    if args.print:
        print(
            f"#: The public names of each ``cupyx.scipy`` subpackage (CuPy {cupy.__version__})."
        )
        print("NAMES = {")
        for path in SUBPACKAGES:
            body = textwrap.fill(
                " ".join(sorted(real[path])),
                width=72,
                initial_indent=" " * 8,
                subsequent_indent=" " * 8,
                break_on_hyphens=False,
                break_long_words=False,
            )
            print(f'    "{path}": _names(\n        """\n{body}\n        """,\n    ),')
        print("}")
        return 0

    missing, added = compare_names(real)
    print(f"installed CuPy {cupy.__version__}; the fake lists CuPy {NAMES_VERSION}")
    for path, names in missing.items():
        print(f"cupyx.scipy.{path} lacks names the fake has: {' '.join(names)}")
    for path, names in added.items():
        print(f"cupyx.scipy.{path} has names the fake lacks: {' '.join(names)}")
    if not missing and not added:
        print("the names agree")
    return 1 if missing or (args.exact and added) else 0


if __name__ == "__main__":
    sys.exit(main())
