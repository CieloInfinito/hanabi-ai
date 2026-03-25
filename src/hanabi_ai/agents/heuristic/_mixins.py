from __future__ import annotations

from hanabi_ai.agents.beliefs import PublicBeliefState
from hanabi_ai.agents.heuristic._scoring import (
    DiscardScore,
    HintEffect,
    HintScore,
    PlayScore,
    highest_rank_value,
    knowledge_state_size,
    possible_cards_from_knowledge,
)
from hanabi_ai.game.actions import (
    DiscardAction,
    HintColorAction,
    HintRankAction,
)
from hanabi_ai.game.cards import CARD_COUNTS_BY_RANK, Card, Rank
from hanabi_ai.game.observation import CardKnowledge, ObservedHand, PlayerObservation
from hanabi_ai.game.rules import is_card_already_played, is_card_playable


class _HeuristicBeliefMixin:
    def _cache_belief_state(self, observation: PlayerObservation) -> PublicBeliefState:
        belief_state = PublicBeliefState.from_observation(observation)
        self._cached_belief_observation = observation
        self._cached_belief_state = belief_state
        return belief_state

    def _belief_state(self, observation: PlayerObservation) -> PublicBeliefState:
        cached_observation = getattr(self, "_cached_belief_observation", None)
        cached_belief_state = getattr(self, "_cached_belief_state", None)
        if cached_observation is observation and cached_belief_state is not None:
            return cached_belief_state
        return self._cache_belief_state(observation)

    def _distribution_for_knowledge(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> tuple[tuple[Card, float], ...]:
        return self._belief_state(observation).distribution_for_knowledge(knowledge)

    def _probability_for_knowledge(
        self,
        knowledge: CardKnowledge,
        observation: PlayerObservation,
        predicate,
    ) -> float:
        card_distribution = self._distribution_for_knowledge(knowledge, observation)
        if not card_distribution:
            return 0.0

        return sum(
            probability for card, probability in card_distribution if predicate(card)
        )

    def _possible_cards(self, knowledge: CardKnowledge) -> tuple[Card, ...]:
        return possible_cards_from_knowledge(knowledge)

    def _possible_cards_for_knowledge(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> tuple[Card, ...]:
        return self._belief_state(observation).possible_cards_for_knowledge(knowledge)


class _HeuristicScoringMixin(_HeuristicBeliefMixin):
    def _card_is_playable_now(
        self, card: Card, observation: PlayerObservation
    ) -> bool:
        return is_card_playable(card, observation.fireworks)

    def _build_candidate_hints(
        self,
        observed_hand: ObservedHand,
        legal_hints: list[HintColorAction | HintRankAction],
        observation: PlayerObservation,
    ) -> list[tuple[HintColorAction | HintRankAction, HintScore]]:
        candidates: list[tuple[HintColorAction | HintRankAction, HintScore]] = []
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
                                observed_hand,
                                observed_hand.cards,
                                observation,
                                color_hint,
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
                                observed_hand,
                                observed_hand.cards,
                                observation,
                                rank_hint,
                                lambda hand_card, hinted_card=card: (
                                    hand_card.rank == hinted_card.rank
                                ),
                            ),
                        )
                    )

        return candidates

    def _score_hint_cards(
        self,
        observed_hand: ObservedHand,
        cards: tuple[Card, ...],
        observation: PlayerObservation,
        hint_action: HintColorAction | HintRankAction,
        matches_hint,
    ) -> HintScore:
        hint_effect = self._hint_effect_after_action(
            observed_hand,
            cards,
            observation,
            hint_action,
        )
        playable_hits = 0
        critical_playable_hits = 0
        useful_hits = 0
        critical_useful_hits = 0
        dead_hits = 0
        touched_cards = 0

        for card in cards:
            if not matches_hint(card):
                continue

            touched_cards += 1
            if self._card_is_playable_now(card, observation):
                playable_hits += 1
                if self._card_is_critical(card, observation):
                    critical_playable_hits += 1
            elif self._card_is_dead(card, observation):
                dead_hits += 1
            elif not is_card_already_played(card, observation.fireworks):
                useful_hits += 1
                if self._card_is_critical(card, observation):
                    critical_useful_hits += 1

        non_playable_touched = touched_cards - playable_hits
        broad_future_value = int(useful_hits >= 2)

        return (
            hint_effect.guaranteed_play_hits,
            playable_hits,
            broad_future_value,
            -non_playable_touched,
            critical_playable_hits,
            useful_hits,
            hint_effect.information_gain,
            critical_useful_hits,
            -dead_hits,
        )

    def _hint_effect_after_action(
        self,
        observed_hand: ObservedHand,
        cards: tuple[Card, ...],
        observation: PlayerObservation,
        hint_action: HintColorAction | HintRankAction,
    ) -> HintEffect:
        belief_state = self._belief_state(observation)
        before_knowledge = belief_state.knowledge_for_player(observed_hand.player_id)
        definitely_playable_before = set(
            belief_state.guaranteed_play_indices_for_knowledge(before_knowledge)
        )
        updated_knowledge, revealed_indices = (
            belief_state.updated_public_knowledge_after_hint(
                observed_hand.player_id,
                cards,
                hint_action,
            )
        )
        definitely_playable = set(
            belief_state.guaranteed_play_indices_for_knowledge(updated_knowledge)
        )
        guaranteed_play_hits = sum(
            index in definitely_playable and index not in definitely_playable_before
            for index in revealed_indices
        )
        information_gain = 0
        for index, card in enumerate(cards):
            before_size = knowledge_state_size(before_knowledge[index])
            after_size = knowledge_state_size(updated_knowledge[index])
            delta = max(before_size - after_size, 0)
            if delta == 0:
                continue

            if self._card_is_playable_now(card, observation):
                information_gain += delta * 3
            elif not self._card_is_dead(card, observation):
                information_gain += delta * 2
            else:
                information_gain += delta

        return HintEffect(
            guaranteed_play_hits=guaranteed_play_hits,
            information_gain=information_gain,
        )

    def _score_discard_knowledge(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> DiscardScore:
        card_distribution = self._distribution_for_knowledge(knowledge, observation)
        possible_cards = (
            tuple(card for card, _ in card_distribution)
            if card_distribution
            else possible_cards_from_knowledge(knowledge)
        )
        if not possible_cards:
            return (-1, -1, -1.0, -1.0, -1.0, -1.0, -1.0, -1, -1)

        definitely_discardable = all(
            is_card_already_played(card, observation.fireworks)
            for card in possible_cards
        )
        definitely_dead = all(
            self._card_is_dead(card, observation) for card in possible_cards
        )
        playable_probability = self._playable_probability(knowledge, observation)
        dead_probability = self._dead_probability(knowledge, observation)
        already_played_probability = self._already_played_probability(
            knowledge, observation
        )
        critical_probability = self._critical_probability(knowledge, observation)
        expected_discard_risk = self._expected_discard_risk(knowledge, observation)
        hint_count = int(knowledge.hinted_color is not None) + int(
            knowledge.hinted_rank is not None
        )
        highest_possible_rank = highest_rank_value(possible_cards)

        return (
            int(definitely_discardable),
            int(definitely_dead),
            dead_probability,
            already_played_probability,
            -critical_probability,
            -expected_discard_risk,
            -playable_probability,
            -hint_count,
            -highest_possible_rank,
        )

    def _playable_probability(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> float:
        return self._probability_for_knowledge(
            knowledge,
            observation,
            lambda card: is_card_playable(card, observation.fireworks),
        )

    def _guaranteed_play_score(
        self,
        knowledge: CardKnowledge,
        observation: PlayerObservation,
        *,
        index: int,
    ) -> PlayScore:
        card_distribution = self._distribution_for_knowledge(knowledge, observation)
        if not card_distribution:
            return (0.0, 0.0, 0.0, -index)

        five_probability = sum(
            probability
            for card, probability in card_distribution
            if int(card.rank) == 5
        )
        critical_probability = sum(
            probability
            for card, probability in card_distribution
            if self._card_is_critical(card, observation)
        )
        expected_rank = sum(
            int(card.rank) * probability for card, probability in card_distribution
        )

        return (
            five_probability,
            critical_probability,
            expected_rank,
            -index,
        )

    def _dead_probability(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> float:
        return self._probability_for_knowledge(
            knowledge,
            observation,
            lambda card: self._card_is_dead(card, observation),
        )

    def _already_played_probability(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> float:
        return self._probability_for_knowledge(
            knowledge,
            observation,
            lambda card: is_card_already_played(card, observation.fireworks),
        )

    def _critical_probability(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> float:
        return self._probability_for_knowledge(
            knowledge,
            observation,
            lambda card: self._card_is_critical(card, observation),
        )

    def _expected_discard_risk(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> float:
        return self._belief_state(observation).expected_value_for_knowledge(
            knowledge,
            lambda card: self._discard_risk_for_card(card, observation),
        )

    def _card_is_critical(self, card: Card, observation: PlayerObservation) -> bool:
        if self._card_is_dead(card, observation):
            return False

        remaining_copies = self._belief_state(observation).remaining_card_counts[card]
        return remaining_copies <= 1

    def _discard_risk_for_card(
        self, card: Card, observation: PlayerObservation
    ) -> float:
        if is_card_already_played(card, observation.fireworks):
            return 0.0
        if self._card_is_dead(card, observation):
            return 0.0
        if self._card_is_critical(card, observation):
            return 3.0 + (int(card.rank) / 10.0)
        return 1.0 + (int(card.rank) / 10.0)

    def _expected_play_failure_cost(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> float:
        card_distribution = self._distribution_for_knowledge(knowledge, observation)
        if not card_distribution:
            return 0.0

        return sum(
            probability * self._discard_risk_for_card(card, observation)
            for card, probability in card_distribution
            if not is_card_playable(card, observation.fireworks)
        )

    def _card_is_dead(self, card: Card, observation: PlayerObservation) -> bool:
        if is_card_already_played(card, observation.fireworks):
            return True

        if observation.fireworks.get(card.color, 0) >= int(card.rank):
            return True

        for needed_rank in range(
            observation.fireworks.get(card.color, 0) + 1,
            int(card.rank),
        ):
            rank = Rank(needed_rank)
            if self._discarded_count(card.color, rank, observation) >= CARD_COUNTS_BY_RANK[rank]:
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
