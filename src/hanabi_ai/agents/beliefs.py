from __future__ import annotations

from dataclasses import dataclass

from hanabi_ai.game.actions import HintColorAction, HintRankAction
from hanabi_ai.game.cards import Card
from hanabi_ai.game.observation import (
    CardKnowledge,
    PlayerObservation,
    apply_color_hint_to_public_knowledge,
    apply_rank_hint_to_public_knowledge,
    build_remaining_card_counts,
    estimate_card_distribution,
    get_definitely_playable_card_indices,
    reconstruct_public_hand_knowledge,
)
from hanabi_ai.game.rules import is_card_already_played, is_card_playable


@dataclass(frozen=True, slots=True)
class PublicBeliefState:
    """
    Derived public belief view built from a single player observation.

    This keeps agent-side inference separate from the game engine while making
    public hand knowledge and weighted card beliefs reusable across one turn.
    """

    observation: PlayerObservation
    remaining_card_counts: dict[Card, int]
    public_hand_knowledge_by_player: dict[int, tuple[CardKnowledge, ...]]
    card_distributions_by_player: dict[int, tuple[tuple[tuple[Card, float], ...], ...]]

    @classmethod
    def from_observation(cls, observation: PlayerObservation) -> PublicBeliefState:
        remaining_card_counts = build_remaining_card_counts(observation)
        public_hand_knowledge_by_player: dict[int, tuple[CardKnowledge, ...]] = {
            observation.observing_player: observation.hand_knowledge
        }

        for observed_hand in observation.other_player_hands:
            public_hand_knowledge_by_player[observed_hand.player_id] = (
                reconstruct_public_hand_knowledge(observation, observed_hand.player_id)
            )

        card_distributions_by_player = {
            player_id: tuple(
                estimate_card_distribution(knowledge, observation)
                for knowledge in hand_knowledge
            )
            for player_id, hand_knowledge in public_hand_knowledge_by_player.items()
        }

        return cls(
            observation=observation,
            remaining_card_counts=remaining_card_counts,
            public_hand_knowledge_by_player=public_hand_knowledge_by_player,
            card_distributions_by_player=card_distributions_by_player,
        )

    def knowledge_for_player(self, player_id: int) -> tuple[CardKnowledge, ...]:
        return self.public_hand_knowledge_by_player[player_id]

    def distribution_for_card(
        self, player_id: int, card_index: int
    ) -> tuple[tuple[Card, float], ...]:
        return self.card_distributions_by_player[player_id][card_index]

    def guaranteed_play_indices(self, player_id: int) -> tuple[int, ...]:
        return self.guaranteed_play_indices_for_knowledge(
            self.knowledge_for_player(player_id)
        )

    def guaranteed_play_indices_for_knowledge(
        self, hand_knowledge: tuple[CardKnowledge, ...]
    ) -> tuple[int, ...]:
        return get_definitely_playable_card_indices(
            hand_knowledge,
            self.observation.fireworks,
        )

    def playable_probability(self, player_id: int, card_index: int) -> float:
        return self.probability_for_card(player_id, card_index, self._is_playable_now)

    def critical_probability(
        self,
        player_id: int,
        card_index: int,
        *,
        is_critical,
    ) -> float:
        return self.probability_for_card(player_id, card_index, is_critical)

    def probability_for_card(
        self,
        player_id: int,
        card_index: int,
        predicate,
    ) -> float:
        return sum(
            probability
            for card, probability in self.distribution_for_card(player_id, card_index)
            if predicate(card)
        )

    def distribution_for_knowledge(
        self, knowledge: CardKnowledge
    ) -> tuple[tuple[Card, float], ...]:
        for hand_knowledge, hand_distributions in zip(
            self.public_hand_knowledge_by_player.values(),
            self.card_distributions_by_player.values(),
            strict=True,
        ):
            for known_card, distribution in zip(
                hand_knowledge,
                hand_distributions,
                strict=True,
            ):
                if known_card == knowledge:
                    return distribution
        return ()

    def expected_value_for_knowledge(
        self,
        knowledge: CardKnowledge,
        scorer,
    ) -> float:
        return sum(
            probability * scorer(card)
            for card, probability in self.distribution_for_knowledge(knowledge)
        )

    def possible_cards_for_knowledge(
        self, knowledge: CardKnowledge
    ) -> tuple[Card, ...]:
        distribution = self.distribution_for_knowledge(knowledge)
        if distribution:
            return tuple(card for card, _ in distribution)

        return tuple(
            Card(color=color, rank=rank)
            for color in knowledge.possible_colors
            for rank in knowledge.possible_ranks
        )

    def revealed_indices_for_hint(
        self,
        cards: tuple[Card, ...],
        hint_action: HintColorAction | HintRankAction,
    ) -> tuple[int, ...]:
        if isinstance(hint_action, HintColorAction):
            return tuple(
                index
                for index, card in enumerate(cards)
                if card.color == hint_action.color
            )

        return tuple(
            index for index, card in enumerate(cards) if card.rank == hint_action.rank
        )

    def updated_public_knowledge_after_hint(
        self,
        player_id: int,
        cards: tuple[Card, ...],
        hint_action: HintColorAction | HintRankAction,
    ) -> tuple[tuple[CardKnowledge, ...], tuple[int, ...]]:
        knowledge = list(self.knowledge_for_player(player_id))
        revealed_indices = self.revealed_indices_for_hint(cards, hint_action)

        if isinstance(hint_action, HintColorAction):
            updated_knowledge = apply_color_hint_to_public_knowledge(
                knowledge,
                revealed_indices,
                hint_action.color,
            )
        else:
            updated_knowledge = apply_rank_hint_to_public_knowledge(
                knowledge,
                revealed_indices,
                hint_action.rank,
            )

        return tuple(updated_knowledge), revealed_indices

    def _is_playable_now(self, card: Card) -> bool:
        return is_card_playable(card, self.observation.fireworks)

    def is_already_played(self, card: Card) -> bool:
        return is_card_already_played(card, self.observation.fireworks)
