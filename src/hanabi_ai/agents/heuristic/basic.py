from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    SRC_PATH = Path(__file__).resolve().parents[3]
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

from hanabi_ai.agents.heuristic.base import BaseHeuristicAgent


class BasicHeuristicAgent(BaseHeuristicAgent):
    """
    Basic rule-based Hanabi baseline using only partial observations.

    This agent keeps the same local play, hint, and discard priorities as the
    conservative heuristic baseline, but it does not use private hint
    conventions when communicating or interpreting teammate hints.

    In practice, the basic and conservative heuristics differ only in these two
    private communication rules:
    - Color hints are not interpreted or emitted with ascending-rank ordering.
    - Rank hints are not interpreted or emitted with playability-based grouping.
    """

    pass


if __name__ == "__main__":
    raise SystemExit(
        "BasicHeuristicAgent is a library module. "
        "Run 'hanabi-demo-basic' or 'python -m hanabi_ai.tools.demo_basic_trace' "
        "to inspect a full game trace."
    )
