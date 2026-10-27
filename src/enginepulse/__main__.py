"""Console-script entry point.

Keeps the wiring in one place so `python -m enginepulse` and the installed
`enginepulse` binary behave identically, including propagating the CLI's
return code to the shell.
"""

from __future__ import annotations

import sys

from enginepulse.cli import main


def run() -> int:
    """Entry point declared in pyproject [project.scripts]."""
    return main()


if __name__ == "__main__":
    sys.exit(run())
