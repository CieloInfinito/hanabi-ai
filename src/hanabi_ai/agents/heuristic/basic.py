from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    SRC_PATH = Path(__file__).resolve().parents[3]
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

from hanabi_ai.agents.heuristic.base import BaseHeuristicAgent, _HintPriorityWeights
from hanabi_ai.agents.heuristic._scoring import HintScore
from hanabi_ai.game.actions import HintColorAction, HintRankAction
from hanabi_ai.game.observation import ObservedHand, PlayerObservation


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

    def _hint_priority_adjustment(
        self,
        *,
        observation: PlayerObservation,
        observed_hand: ObservedHand,
        hint_action: HintColorAction | HintRankAction,
        score: HintScore,
        follow_on_value: int,
        receiver_needs_help: bool,
        immediate_receiver_bonus: int,
        near_term_receiver_bonus: int,
        actionable_hint_bonus: int,
        critical_playable_hits: int,
        turn_distance: int,
    ) -> int:
        player_count = len(observation.other_player_hands) + 1
        if player_count < 4:
            return 0

        guaranteed_play_hits = score[0]
        playable_hits = score[1]
        useful_hits = score[5]
        information_gain = score[6]
        receiver_under_pressure = self._receiver_under_pressure(
            observation,
            observed_hand.player_id,
        )

        if not receiver_under_pressure or turn_distance > 2:
            return 0

        if guaranteed_play_hits >= 1:
            return 3

        if playable_hits >= 1 and information_gain >= 2:
            return 2

        if playable_hits >= 1 and useful_hits >= 1:
            return 1

        return 0


if __name__ == "__main__":
    raise SystemExit(
        "BasicHeuristicAgent is a library module. "
        "Run 'hanabi-demo-basic' or 'python -m hanabi_ai.tools.demo_basic_trace' "
        "to inspect a full game trace."
    )
