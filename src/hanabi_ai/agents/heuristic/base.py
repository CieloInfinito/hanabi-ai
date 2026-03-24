from __future__ import annotations

from collections import Counter

from hanabi_ai.game.actions import (
    Action,
    AgentDecision,
    DiscardAction,
    HintColorAction,
    HintRankAction,
    PlayAction,
)
from hanabi_ai.game.cards import CARD_COUNTS_BY_RANK, Card, Rank
from hanabi_ai.game.observation import (
    CardKnowledge,
    ObservedHand,
    PlayerObservation,
    PublicTurnRecord,
)
from hanabi_ai.game.rules import is_card_already_played, is_card_playable


class BaseHeuristicAgent:
    """
    Shared rule-based Hanabi heuristic logic using only partial observations.

    Subclasses can customize how they interpret public hint history or how they
    present hints to teammates, while reusing the same local play, hint, and
    discard policy.
    """

    def act(self, observation: PlayerObservation) -> Action | AgentDecision:
        """
        Choose an action using a small ordered set of Hanabi heuristics.
        """
        if not observation.legal_actions:
            raise ValueError(
                f"{self.__class__.__name__} received an observation with no legal actions."
            )

        observation = self._apply_private_conventions(observation)

        guaranteed_play = self._choose_definitely_playable_action(observation)
        if guaranteed_play is not None:
            return guaranteed_play

        helpful_hint = self._choose_hint_for_other_players(observation)
        if helpful_hint is not None:
            return self._attach_hint_presentation(helpful_hint, observation)

        discard_action = self._choose_discard_action(observation)
        if discard_action is not None:
            return discard_action

        fallback_play = self._choose_any_play_action(observation)
        if fallback_play is not None:
            return fallback_play

        return observation.legal_actions[0]

    def refine_observation_for_display(
        self, observation: PlayerObservation
    ) -> PlayerObservation:
        """
        Return the observation as interpreted by this agent's private conventions.
        """
        return self._apply_private_conventions(observation)

    def describe_public_turn_record(
        self, record: PublicTurnRecord
    ) -> tuple[str, ...]:
        """
        Return optional human-readable notes about how this agent reads a public turn.
        """
        return ()

    def _apply_private_conventions(
        self, observation: PlayerObservation
    ) -> PlayerObservation:
        return observation

    def _attach_hint_presentation(
        self,
        action: HintColorAction | HintRankAction,
        observation: PlayerObservation,
    ) -> HintColorAction | HintRankAction | AgentDecision:
        return action

    def _choose_definitely_playable_action(
        self, observation: PlayerObservation
    ) -> PlayAction | None:
        legal_plays = {
            action.card_index: action
            for action in observation.legal_actions
            if isinstance(action, PlayAction)
        }

        for index, knowledge in enumerate(observation.hand_knowledge):
            if index not in legal_plays:
                continue

            possible_cards = self._possible_cards_for_knowledge(knowledge, observation)
            if possible_cards and all(
                is_card_playable(card, observation.fireworks) for card in possible_cards
            ):
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

        best_hint: HintColorAction | HintRankAction | None = None
        best_score: tuple[int, int, int] | None = None

        for observed_hand in observation.other_player_hands:
            candidate_hints = self._build_candidate_hints(
                observed_hand, legal_hints, observation
            )
            for hint_action, score in candidate_hints:
                if best_score is None or score > best_score:
                    best_hint = hint_action
                    best_score = score

        return best_hint

    def _choose_discard_action(
        self, observation: PlayerObservation
    ) -> DiscardAction | None:
        discard_actions = [
            action
            for action in observation.legal_actions
            if isinstance(action, DiscardAction)
        ]
        if not discard_actions:
            return None

        scored_discards = [
            (
                self._score_discard_knowledge(
                    observation.hand_knowledge[action.card_index],
                    observation,
                ),
                action.card_index,
                action,
            )
            for action in discard_actions
        ]
        scored_discards.sort(reverse=True)
        return scored_discards[0][2]

    def _choose_any_play_action(
        self, observation: PlayerObservation
    ) -> PlayAction | None:
        play_actions = [
            action for action in observation.legal_actions if isinstance(action, PlayAction)
        ]
        if not play_actions:
            return None

        best_action: PlayAction | None = None
        best_score = -1.0

        for action in play_actions:
            knowledge = observation.hand_knowledge[action.card_index]
            score = self._playable_probability(knowledge, observation)
            if score > best_score:
                best_action = action
                best_score = score

        return best_action

    def _card_is_playable_now(
        self, card: Card, observation: PlayerObservation
    ) -> bool:
        return is_card_playable(card, observation.fireworks)

    def _build_candidate_hints(
        self,
        observed_hand: ObservedHand,
        legal_hints: list[HintColorAction | HintRankAction],
        observation: PlayerObservation,
    ) -> list[tuple[HintColorAction | HintRankAction, tuple[int, int, int]]]:
        candidates: list[tuple[HintColorAction | HintRankAction, tuple[int, int, int]]] = []

        seen_colors: set[str] = set()
        seen_ranks: set[int] = set()

        for card in observed_hand.cards:
            if card.color.value not in seen_colors:
                seen_colors.add(card.color.value)
                color_hint = HintColorAction(
                    target_player=observed_hand.player_id,
                    color=card.color,
                )
                if color_hint in legal_hints:
                    candidates.append(
                        (
                            color_hint,
                            self._score_hint_cards(
                                observed_hand.cards,
                                observation,
                                lambda hand_card, hinted_card=card: (
                                    hand_card.color == hinted_card.color
                                ),
                            ),
                        )
                    )

            if int(card.rank) not in seen_ranks:
                seen_ranks.add(int(card.rank))
                rank_hint = HintRankAction(
                    target_player=observed_hand.player_id,
                    rank=card.rank,
                )
                if rank_hint in legal_hints:
                    candidates.append(
                        (
                            rank_hint,
                            self._score_hint_cards(
                                observed_hand.cards,
                                observation,
                                lambda hand_card, hinted_card=card: (
                                    hand_card.rank == hinted_card.rank
                                ),
                            ),
                        )
                    )

        return candidates

    def _score_hint_cards(
        self,
        cards: tuple[Card, ...],
        observation: PlayerObservation,
        matches_hint,
    ) -> tuple[int, int, int]:
        playable_hits = 0
        useful_hits = 0
        touched_cards = 0

        for card in cards:
            if not matches_hint(card):
                continue

            touched_cards += 1
            if self._card_is_playable_now(card, observation):
                playable_hits += 1
            elif not is_card_already_played(card, observation.fireworks):
                useful_hits += 1

        return (playable_hits, useful_hits, touched_cards)

    def _score_discard_knowledge(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> tuple[int, int, float, float, int, int]:
        possible_cards = self._possible_cards(knowledge)
        if not possible_cards:
            return (-1, -1, -1.0, -1.0, -1, -1)

        definitely_discardable = all(
            is_card_already_played(card, observation.fireworks) for card in possible_cards
        )
        definitely_dead = all(
            self._card_is_dead(card, observation) for card in possible_cards
        )
        playable_probability = self._playable_probability(knowledge, observation)
        dead_ratio = sum(
            1 for card in possible_cards if self._card_is_dead(card, observation)
        ) / len(possible_cards)
        already_played_ratio = sum(
            1 for card in possible_cards if is_card_already_played(card, observation.fireworks)
        ) / len(possible_cards)
        hint_count = int(knowledge.hinted_color is not None) + int(
            knowledge.hinted_rank is not None
        )
        highest_possible_rank = max(int(card.rank) for card in possible_cards)

        return (
            int(definitely_discardable),
            int(definitely_dead),
            dead_ratio,
            already_played_ratio,
            -playable_probability,
            -hint_count,
            -highest_possible_rank,
        )

    def _playable_probability(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> float:
        possible_cards = self._possible_cards_for_knowledge(knowledge, observation)
        if not possible_cards:
            return 0.0

        playable_count = sum(
            1 for card in possible_cards if is_card_playable(card, observation.fireworks)
        )
        return playable_count / len(possible_cards)

    def _possible_cards(self, knowledge: CardKnowledge) -> tuple[Card, ...]:
        return tuple(
            Card(color=color, rank=rank)
            for color in knowledge.possible_colors
            for rank in knowledge.possible_ranks
        )

    def _possible_cards_for_knowledge(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> tuple[Card, ...]:
        possible_cards = self._possible_cards(knowledge)
        visible_counts = self._visible_card_counts(observation)
        feasible_cards = tuple(
            card
            for card in possible_cards
            if visible_counts[(card.color, card.rank)] < CARD_COUNTS_BY_RANK[card.rank]
        )

        if feasible_cards:
            return feasible_cards

        return possible_cards

    def _card_is_dead(self, card: Card, observation: PlayerObservation) -> bool:
        if is_card_already_played(card, observation.fireworks):
            return True

        if observation.fireworks.get(card.color, 0) >= int(card.rank):
            return True

        for needed_rank in range(observation.fireworks.get(card.color, 0) + 1, int(card.rank)):
            if self._discarded_count(card.color, Rank(needed_rank), observation) >= CARD_COUNTS_BY_RANK[Rank(needed_rank)]:
                return True

        return False

    def _discarded_count(
        self, color, rank: Rank, observation: PlayerObservation
    ) -> int:
        return sum(
            1
            for discarded_card in observation.discard_pile
            if discarded_card.color == color and discarded_card.rank == rank
        )

    def _visible_card_counts(
        self, observation: PlayerObservation
    ) -> Counter[tuple[object, Rank]]:
        counts: Counter[tuple[object, Rank]] = Counter()

        for discarded_card in observation.discard_pile:
            counts[(discarded_card.color, discarded_card.rank)] += 1

        for observed_hand in observation.other_player_hands:
            for card in observed_hand.cards:
                counts[(card.color, card.rank)] += 1

        for color, highest_rank in observation.fireworks.items():
            for rank_value in range(1, highest_rank + 1):
                counts[(color, Rank(rank_value))] += 1

        return counts
