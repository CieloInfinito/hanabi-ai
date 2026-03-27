from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    SRC_PATH = Path(__file__).resolve().parents[3]
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

from hanabi_ai.agents.heuristic.convention_tempo import ConventionTempoHeuristicAgent


class LargeTableHeuristicAgent(ConventionTempoHeuristicAgent):
    """
    Thin compatibility wrapper around ConventionTempo's current 5-player policy.
    """


if __name__ == "__main__":
    raise SystemExit(
        "LargeTableHeuristicAgent is a library module. "
        "Use hanabi-evaluate or import it from hanabi_ai.agents."
    )
