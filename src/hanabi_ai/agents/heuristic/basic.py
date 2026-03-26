from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    SRC_PATH = Path(__file__).resolve().parents[3]
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

from hanabi_ai.agents.heuristic.base import BaseHeuristicAgent, _HintPriorityWeights


class BasicHeuristicAgent(BaseHeuristicAgent):
    """
    Basic rule-based Hanabi baseline using only partial observations.

    This agent keeps the same local play, hint, and discard priorities as the
    convention heuristic baseline, but it does not use private hint
    conventions when communicating or interpreting teammate hints.

    In practice, the basic and convention heuristics differ only in these two
    private communication rules:
    - Color hints are not interpreted or emitted with ascending-rank ordering.
    - Rank hints are not interpreted or emitted with playability-based grouping.
    """

    def _base_hint_priority_weights(self, player_count: int) -> _HintPriorityWeights:
        if player_count <= 2:
            return _HintPriorityWeights(
                actionable_hint=1,
                critical_playable=1,
            )
        if player_count == 3:
            return _HintPriorityWeights(
                receiver_needs_help=1,
                immediate_receiver=1,
                near_term_receiver=1,
                actionable_hint=1,
                critical_playable=1,
            )
        if player_count == 4:
            return _HintPriorityWeights(
                follow_on_value=1,
                receiver_needs_help=1,
                immediate_receiver=2,
                near_term_receiver=1,
                actionable_hint=1,
                critical_playable=1,
                turn_distance_penalty=1,
            )
        return _HintPriorityWeights(
            follow_on_value=2,
            receiver_needs_help=1,
            immediate_receiver=1,
            near_term_receiver=2,
            actionable_hint=1,
            critical_playable=1,
            turn_distance_penalty=1,
        )


if __name__ == "__main__":
    raise SystemExit(
        "BasicHeuristicAgent is a library module. "
        "Run 'hanabi-demo-basic' or 'python -m hanabi_ai.tools.demo_basic_trace' "
        "to inspect a full game trace."
    )
