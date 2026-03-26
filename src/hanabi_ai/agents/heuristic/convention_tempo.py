from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    SRC_PATH = Path(__file__).resolve().parents[3]
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

from hanabi_ai.agents.heuristic._convention_mixin import _ConventionHeuristicMixin
from hanabi_ai.agents.heuristic.tempo import TempoHeuristicAgent


class ConventionTempoHeuristicAgent(_ConventionHeuristicMixin, TempoHeuristicAgent):
    """
    Hybrid heuristic that combines convention-aware hint communication with
    Tempo's hint-economy policy.

    In practice this agent:
    - interprets and emits the same private hint conventions as the convention
      heuristic
    - uses Tempo's stricter judgment about when a hint is worth spending
    """


if __name__ == "__main__":
    raise SystemExit(
        "ConventionTempoHeuristicAgent is a library module. "
        "Use hanabi-evaluate or import it from hanabi_ai.agents."
    )
