from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    SRC_PATH = Path(__file__).resolve().parents[3]
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

from hanabi_ai.agents.heuristic._convention_mixin import _ConventionHeuristicMixin
from hanabi_ai.agents.heuristic._scoring import HintScore
from hanabi_ai.agents.heuristic.tempo import TempoHeuristicAgent
from hanabi_ai.game.actions import DiscardAction, HintColorAction, HintRankAction
from hanabi_ai.game.observation import ObservedHand, PlayerObservation


class ConventionTempoHeuristicAgent(_ConventionHeuristicMixin, TempoHeuristicAgent):
    """
    Hybrid heuristic that combines convention-aware hint communication with
    Tempo's hint-economy policy.

    In practice this agent:
    - interprets and emits the same private hint conventions as the convention
      heuristic
    - uses Tempo's stricter judgment about when a hint is worth spending
    - adds a small 5-player preference for farther hints that preserve
      meaningful coordination, such as critical cards or visible follow-on
      chains
    """

    @staticmethod
    def _is_five_player_game(player_count: int) -> bool:
        return player_count >= 5

    @staticmethod
    def _guarantees_a_play(hint_score: HintScore) -> bool:
        return hint_score[0] > 0

    @staticmethod
    def _is_near_term_actionable_hint(
        hint_score: HintScore,
        turn_distance: int,
    ) -> bool:
        playable_hits = hint_score[1]
        information_gain = hint_score[6]
        return turn_distance <= 2 and playable_hits >= 1 and information_gain >= 2

    @staticmethod
    def _is_near_term_critical_hint(
        hint_score: HintScore,
        turn_distance: int,
    ) -> bool:
        critical_playable_hits = hint_score[4]
        return turn_distance <= 2 and critical_playable_hits >= 1

    @staticmethod
    def _is_immediate_useful_hint(
        hint_score: HintScore,
        turn_distance: int,
    ) -> bool:
        playable_hits = hint_score[1]
        useful_hits = hint_score[5]
        return turn_distance == 1 and playable_hits >= 1 and useful_hits >= 1

    @staticmethod
    def _far_hint_preserves_coordination(
        hint_score: HintScore,
        *,
        follow_on_value: int,
        critical_playable_hits: int,
        turn_distance: int,
    ) -> bool:
        return turn_distance >= 3 and (
            critical_playable_hits >= 1
            or follow_on_value >= 1
            or hint_score[0] >= 1
        )

    @staticmethod
    def _far_hint_is_rich_enough(hint_score: HintScore, turn_distance: int) -> bool:
        playable_hits = hint_score[1]
        useful_hits = hint_score[5]
        information_gain = hint_score[6]
        return (
            turn_distance >= 3
            and playable_hits >= 1
            and useful_hits >= 1
            and information_gain >= 4
        )

    def _should_prefer_discard_over_hint(
        self,
        observation: PlayerObservation,
        discard_action: DiscardAction,
        hint_action: HintColorAction | HintRankAction,
        hint_score: HintScore | None,
    ) -> bool:
        player_count = len(observation.other_player_hands) + 1
        if not self._is_five_player_game(player_count):
            return super()._should_prefer_discard_over_hint(
                observation,
                discard_action,
                hint_action,
                hint_score,
            )

        if hint_score is None:
            return True

        turn_distance = self._turn_distance(
            observation.current_player,
            hint_action.target_player,
            player_count,
        )

        if (
            self._guarantees_a_play(hint_score)
            or self._is_near_term_actionable_hint(hint_score, turn_distance)
            or self._is_near_term_critical_hint(hint_score, turn_distance)
            or self._is_immediate_useful_hint(hint_score, turn_distance)
        ):
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
        base_adjustment = super()._hint_priority_adjustment(
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
        if not self._is_five_player_game(player_count):
            return base_adjustment

        bonus = 0

        if self._far_hint_preserves_coordination(
            score,
            follow_on_value=follow_on_value,
            critical_playable_hits=critical_playable_hits,
            turn_distance=turn_distance,
        ):
            bonus += 2

        if self._far_hint_is_rich_enough(score, turn_distance):
            bonus += 1

        return base_adjustment + bonus


if __name__ == "__main__":
    raise SystemExit(
        "ConventionTempoHeuristicAgent is a library module. "
        "Use hanabi-evaluate or import it from hanabi_ai.agents."
    )
