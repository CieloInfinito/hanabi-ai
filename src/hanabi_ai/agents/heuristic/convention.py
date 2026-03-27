from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    SRC_PATH = Path(__file__).resolve().parents[3]
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

from hanabi_ai.agents.heuristic._convention_mixin import _ConventionHeuristicMixin
from hanabi_ai.agents.heuristic.basic import BasicHeuristicAgent


class ConventionHeuristicAgent(_ConventionHeuristicMixin, BasicHeuristicAgent):
    """
    Basic heuristic plus private communication conventions.

    Relative to `BasicHeuristicAgent`, this variant changes only how hint
    presentations are emitted and interpreted:

    - color hints use ascending-rank pointing order
    - rank hints group playable cards before non-playable cards

    The underlying play, discard, and public-information policy stays shared.
    """


if __name__ == "__main__":
    raise SystemExit(
        "ConventionHeuristicAgent is a library module. "
        "Run 'hanabi-demo-convention' or "
        "'python -m hanabi_ai.tools.demo_convention_trace' "
        "to inspect a full game trace."
    )
