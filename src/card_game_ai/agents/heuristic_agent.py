from __future__ import annotations

from card_game_ai.game.actions import (
    Action,
    DiscardAction,
    HintColorAction,
    HintRankAction,
    PlayAction,
)
from card_game_ai.game.observation import (
    ObservedHand,
    PlayerObservation,
    get_definitely_playable_card_indices,
)


class HeuristicAgent:
    """
    Simple cooperative baseline using only partial observations.
    """

    def act(self, observation: PlayerObservation) -> Action:
        """
        Choose an action using a small ordered set of Hanabi heuristics.
        """
        if not observation.legal_actions:
            raise ValueError(
                "HeuristicAgent received an observation with no legal actions."
            )

        guaranteed_play = self._choose_definitely_playable_action(observation)
        if guaranteed_play is not None:
            return guaranteed_play

        helpful_hint = self._choose_hint_for_other_players(observation)
        if helpful_hint is not None:
            return helpful_hint

        discard_action = self._choose_discard_action(observation)
        if discard_action is not None:
            return discard_action

        fallback_play = self._choose_any_play_action(observation)
        if fallback_play is not None:
            return fallback_play

        return observation.legal_actions[0]

    def _choose_definitely_playable_action(
        self, observation: PlayerObservation
    ) -> PlayAction | None:
        definitely_playable = get_definitely_playable_card_indices(
            observation.hand_knowledge,
            observation.fireworks,
        )
        legal_plays = {
            action.card_index: action
            for action in observation.legal_actions
            if isinstance(action, PlayAction)
        }

        for index in definitely_playable:
            if index in legal_plays:
                return legal_plays[index]

        return None

    def _choose_hint_for_other_players(
        self, observation: PlayerObservation
    ) -> HintColorAction | HintRankAction | None:
        legal_hints = [
            action
            for action in observation.legal_actions
            if isinstance(action, (HintColorAction, HintRankAction))
        ]
        if not legal_hints:
            return None

        for observed_hand in observation.other_player_hands:
            for card in observed_hand.cards:
                if self._card_is_playable_now(card, observation):
                    color_hint = HintColorAction(
                        target_player=observed_hand.player_id,
                        color=card.color,
                    )
                    if color_hint in legal_hints:
                        return color_hint

                    rank_hint = HintRankAction(
                        target_player=observed_hand.player_id,
                        rank=card.rank,
                    )
                    if rank_hint in legal_hints:
                        return rank_hint

        return None

    def _choose_discard_action(
        self, observation: PlayerObservation
    ) -> DiscardAction | None:
        for action in observation.legal_actions:
            if isinstance(action, DiscardAction):
                return action
        return None

    def _choose_any_play_action(
        self, observation: PlayerObservation
    ) -> PlayAction | None:
        for action in observation.legal_actions:
            if isinstance(action, PlayAction):
                return action
        return None

    def _card_is_playable_now(
        self, card, observation: PlayerObservation
    ) -> bool:
        return observation.fireworks.get(card.color, 0) + 1 == int(card.rank)
