from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    SRC_PATH = Path(__file__).resolve().parents[3]
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

from hanabi_ai.agents.heuristic.convention_tempo import ConventionTempoHeuristicAgent
from hanabi_ai.agents.heuristic._scoring import HintScore
from hanabi_ai.game.actions import DiscardAction, HintColorAction, HintRankAction
from hanabi_ai.game.observation import ObservedHand, PlayerObservation


class LargeTableHeuristicAgent(ConventionTempoHeuristicAgent):
    """
    Experimental 5-player branch built on top of ConventionTempo.

    The current experiment isolates one hypothesis:
    some farther hints deserve to survive large-table tempo pressure when they
    preserve meaningful coordination, such as critical cards or visible
    follow-on chains.
    """

    def _should_prefer_discard_over_hint(
        self,
        observation: PlayerObservation,
        discard_action: DiscardAction,
        hint_action: HintColorAction | HintRankAction,
        hint_score: HintScore | None,
    ) -> bool:
        player_count = len(observation.other_player_hands) + 1
        if player_count < 5:
            return super()._should_prefer_discard_over_hint(
                observation,
                discard_action,
                hint_action,
                hint_score,
            )

        if hint_score is None:
            return True

        guaranteed_play_hits = hint_score[0]
        playable_hits = hint_score[1]
        critical_playable_hits = hint_score[4]
        useful_hits = hint_score[5]
        information_gain = hint_score[6]
        turn_distance = self._turn_distance(
            observation.current_player,
            hint_action.target_player,
            player_count,
        )

        if guaranteed_play_hits > 0:
            return False

        if turn_distance <= 2 and playable_hits >= 1 and information_gain >= 2:
            return False

        if turn_distance <= 2 and critical_playable_hits >= 1:
            return False

        if turn_distance == 1 and playable_hits >= 1 and useful_hits >= 1:
            return False

        return super()._should_prefer_discard_over_hint(
            observation,
            discard_action,
            hint_action,
            hint_score,
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
        if player_count < 5:
            return super()._hint_priority_adjustment(
                observation=observation,
                observed_hand=observed_hand,
                hint_action=hint_action,
                score=score,
                follow_on_value=follow_on_value,
                receiver_needs_help=receiver_needs_help,
                immediate_receiver_bonus=immediate_receiver_bonus,
                near_term_receiver_bonus=near_term_receiver_bonus,
                actionable_hint_bonus=actionable_hint_bonus,
                critical_playable_hits=critical_playable_hits,
                turn_distance=turn_distance,
            )

        guaranteed_play_hits = score[0]
        playable_hits = score[1]
        useful_hits = score[5]
        information_gain = score[6]
        bonus = 0

        if turn_distance >= 3 and (
            critical_playable_hits >= 1
            or follow_on_value >= 1
            or guaranteed_play_hits >= 1
        ):
            bonus += 2

        if (
            turn_distance >= 3
            and playable_hits >= 1
            and useful_hits >= 1
            and information_gain >= 4
        ):
            bonus += 1

        return bonus


if __name__ == "__main__":
    raise SystemExit(
        "LargeTableHeuristicAgent is a library module. "
        "Use hanabi-evaluate or import it from hanabi_ai.agents."
    )
