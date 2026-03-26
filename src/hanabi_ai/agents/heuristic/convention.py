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
    Convention-aware Hanabi heuristic using only partial observations.

    Policy summary:
    - Play an own-hand card only when current knowledge guarantees it is playable.
    - Otherwise, choose the most useful legal hint for a teammate.
    - Otherwise, prefer the safest discard candidate.
    - Avoid blind plays whenever a discard is legal.
    - If forced to play, choose the own-hand card with the highest inferred
      probability of being playable.

    Private convention support:
    - This agent can model a color-hint convention where cards are pointed
      in ascending rank order, which also communicates when two or more
      hinted cards share the same rank.
    - This agent can model a rank-hint convention where hinted cards are
      grouped by immediate playability: playable cards are pointed first,
      then non-playable cards.
    - The convention intentionally lives in the agent layer rather than the
      engine so different bots can adopt different hidden signaling schemes.

    Shared public-information inference such as teammate-hand visibility,
    fireworks tracking, and discard-pile reasoning is inherited from the
    heuristic base class and is not unique to this convention-aware variant.

    The agent never accesses hidden information from its own hand directly.
    """


if __name__ == "__main__":
    raise SystemExit(
        "ConventionHeuristicAgent is a library module. "
        "Run 'hanabi-demo-convention' or "
        "'python -m hanabi_ai.tools.demo_convention_trace' "
        "to inspect a full game trace."
    )
