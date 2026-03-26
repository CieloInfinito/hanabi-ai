from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    SRC_PATH = Path(__file__).resolve().parents[3]
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

from hanabi_ai.agents.heuristic.basic import BasicHeuristicAgent
from hanabi_ai.agents.heuristic._scoring import HintScore
from hanabi_ai.game.actions import (
    DiscardAction,
    HintColorAction,
    HintRankAction,
)
from hanabi_ai.game.observation import ObservedHand
from hanabi_ai.game.observation import PlayerObservation


class TempoHeuristicAgent(BasicHeuristicAgent):
    """
    Experimental heuristic variant that protects short-horizon hint economy.

    This agent reuses the same public-information inference as the base
    heuristic, but it becomes less willing to spend the last hint token on a
    purely informational hint when a legal discard can recover tempo instead.
    """

    def _should_prefer_discard_over_hint(
        self,
        observation: PlayerObservation,
        discard_action: DiscardAction,
        hint_action: HintColorAction | HintRankAction,
        hint_score: HintScore | None,
    ) -> bool:
        return not self._should_spend_hint_on_best_hint(
            observation,
            hint_action,
            hint_score,
        )

    def _should_spend_hint_on_best_hint(
        self,
        observation: PlayerObservation,
        hint_action: HintColorAction | HintRankAction,
        best_hint_score: HintScore | None,
    ) -> bool:
        if best_hint_score is None:
            return False

        guaranteed_play_hits = best_hint_score[0]
        playable_hits = best_hint_score[1]
        critical_playable_hits = best_hint_score[4]
        useful_hits = best_hint_score[5]
        information_gain = best_hint_score[6]
        critical_useful_hits = best_hint_score[7]
        player_count = len(observation.other_player_hands) + 1

        if guaranteed_play_hits > 0:
            return True

        if observation.hint_tokens >= 3:
            return True

        turn_distance = self._turn_distance(
            observation.current_player,
            hint_action.target_player,
            player_count,
        )

        if player_count >= 3:
            if observation.hint_tokens >= 2:
                return True

            if turn_distance == 1 and playable_hits >= 1:
                return True

            receiver_needs_help = self._receiver_needs_help(
                observation,
                hint_action.target_player,
            )
            follow_on_value = self._follow_on_play_value(
                observation,
                hint_action.target_player,
                hint_action,
            )

            if (
                player_count >= 4
                and turn_distance <= 2
                and playable_hits >= 1
                and useful_hits + information_gain >= 3
            ):
                return True

            if (
                player_count >= 4
                and receiver_needs_help
                and turn_distance <= 2
                and playable_hits >= 1
            ):
                return True

            if (
                player_count >= 4
                and receiver_needs_help
                and turn_distance <= 2
                and useful_hits >= 2
                and information_gain >= 3
            ):
                return True

            if (
                player_count >= 4
                and turn_distance <= 2
                and follow_on_value >= 1
                and playable_hits >= 1
            ):
                return True

            return (
                playable_hits >= 2
                or critical_playable_hits >= 1
                or (
                    playable_hits >= 1 and useful_hits + information_gain >= 4
                )
                or (
                    critical_useful_hits >= 1
                    and useful_hits + information_gain >= 3
                )
            )

        if observation.hint_tokens == 2:
            return playable_hits >= 1 or useful_hits + information_gain >= 6

        return (
            playable_hits >= 2
            or critical_playable_hits >= 1
            or (
                playable_hits >= 1
                and (
                    critical_useful_hits >= 1 or useful_hits + information_gain >= 6
                )
            )
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

        if player_count <= 2:
            return (
                actionable_hint_bonus
                + (critical_playable_hits * 2)
                + immediate_receiver_bonus
            )
        if player_count == 3:
            return (
                int(receiver_needs_help)
                + immediate_receiver_bonus
                + near_term_receiver_bonus
                + actionable_hint_bonus
                + (critical_playable_hits * 2)
            )
        if player_count == 4:
            return (
                follow_on_value
                + int(receiver_needs_help)
                + immediate_receiver_bonus
                + near_term_receiver_bonus
                + actionable_hint_bonus
                + (critical_playable_hits * 2)
                - turn_distance
            )
        return (
            (follow_on_value * 2)
            + int(receiver_needs_help)
            + immediate_receiver_bonus
            + near_term_receiver_bonus
            + actionable_hint_bonus
            + (critical_playable_hits * 2)
            - turn_distance
        )

    def _tempo_hint_priority(
        self,
        observation: PlayerObservation,
        observed_hand: ObservedHand,
        hint_action: HintColorAction | HintRankAction,
        score: HintScore,
    ) -> tuple[HintScore, int, int, int, int, int, int, int]:
        return self._hint_priority(
            observation,
            observed_hand,
            hint_action,
            score,
        )


if __name__ == "__main__":
    raise SystemExit(
        "TempoHeuristicAgent is a library module. "
        "Use hanabi-evaluate or import it from hanabi_ai.agents."
    )
