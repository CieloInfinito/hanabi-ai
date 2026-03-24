from __future__ import annotations

from card_game_ai.agents.heuristic.base_agent import BaseHeuristicAgent


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
