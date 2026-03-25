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
    apply_color_hint_to_knowledge,
    apply_rank_hint_to_knowledge,
    build_visible_card_counts,
    create_initial_hand_knowledge,
    estimate_card_distribution,
    get_definitely_playable_card_indices,
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

        helpful_hint, helpful_hint_score = self._choose_hint_for_other_players(
            observation
        )
        confident_play = self._choose_confident_probabilistic_play(
            observation,
            best_hint_score=helpful_hint_score,
        )
        if confident_play is not None:
            return confident_play

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
    ) -> tuple[
        HintColorAction | HintRankAction | None,
        tuple[int, int, int, int, int, int, int, int] | None,
    ]:
        legal_hints = [
            action
            for action in observation.legal_actions
            if isinstance(action, (HintColorAction, HintRankAction))
        ]
        if not legal_hints:
            return None, None

        best_hint: HintColorAction | HintRankAction | None = None
        best_score: tuple[int, int, int, int, int, int, int, int] | None = None

        for observed_hand in observation.other_player_hands:
            candidate_hints = self._build_candidate_hints(
                observed_hand, legal_hints, observation
            )
            for hint_action, score in candidate_hints:
                if best_score is None or score > best_score:
                    best_hint = hint_action
                    best_score = score

        return best_hint, best_score

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

    def _choose_confident_probabilistic_play(
        self,
        observation: PlayerObservation,
        *,
        best_hint_score: tuple[int, int, int, int, int, int, int, int] | None,
    ) -> PlayAction | None:
        play_actions = [
            action
            for action in observation.legal_actions
            if isinstance(action, PlayAction)
        ]
        if not play_actions:
            return None

        if any(
            isinstance(action, DiscardAction) for action in observation.legal_actions
        ):
            return None

        # If a teammate can be given an immediate playable hint, preserve that
        # stronger cooperative action before taking a personal risk.
        if best_hint_score is not None and best_hint_score[0] > 0:
            return None

        threshold = self._risky_play_probability_threshold(observation)
        best_action: PlayAction | None = None
        best_score: tuple[float, float, float, int] | None = None

        for action in play_actions:
            knowledge = observation.hand_knowledge[action.card_index]
            playable_probability = self._playable_probability(knowledge, observation)
            if playable_probability < threshold:
                continue

            critical_probability = self._critical_probability(knowledge, observation)
            expected_failure_cost = self._expected_play_failure_cost(
                knowledge, observation
            )
            score = (
                playable_probability,
                -critical_probability,
                -expected_failure_cost,
                -action.card_index,
            )
            if best_score is None or score > best_score:
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
    ) -> list[
        tuple[
            HintColorAction | HintRankAction,
            tuple[int, int, int, int, int, int, int, int],
        ]
    ]:
        candidates: list[
            tuple[
                HintColorAction | HintRankAction,
                tuple[int, int, int, int, int, int, int, int],
            ]
        ] = []

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
        cards: tuple[Card, ...],
        observation: PlayerObservation,
        hint_action: HintColorAction | HintRankAction,
        matches_hint,
    ) -> tuple[int, int, int, int, int, int, int, int]:
        guaranteed_play_hits = self._count_guaranteed_plays_after_hint(
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
            guaranteed_play_hits,
            playable_hits,
            broad_future_value,
            -non_playable_touched,
            critical_playable_hits,
            useful_hits,
            critical_useful_hits,
            -dead_hits,
        )

    def _count_guaranteed_plays_after_hint(
        self,
        cards: tuple[Card, ...],
        observation: PlayerObservation,
        hint_action: HintColorAction | HintRankAction,
    ) -> int:
        hand_knowledge = create_initial_hand_knowledge(len(cards))

        if isinstance(hint_action, HintColorAction):
            updated_knowledge = apply_color_hint_to_knowledge(
                hand_knowledge,
                list(cards),
                hint_action.color,
            )
            revealed_indices = tuple(
                index
                for index, card in enumerate(cards)
                if card.color == hint_action.color
            )
        else:
            updated_knowledge = apply_rank_hint_to_knowledge(
                hand_knowledge,
                list(cards),
                hint_action.rank,
            )
            revealed_indices = tuple(
                index
                for index, card in enumerate(cards)
                if card.rank == hint_action.rank
            )

        definitely_playable = set(
            get_definitely_playable_card_indices(
                tuple(updated_knowledge),
                observation.fireworks,
            )
        )
        return sum(index in definitely_playable for index in revealed_indices)

    def _score_discard_knowledge(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> tuple[int, int, float, float, float, float, int, int]:
        card_distribution = estimate_card_distribution(knowledge, observation)
        possible_cards = (
            tuple(card for card, _ in card_distribution)
            if card_distribution
            else self._possible_cards(knowledge)
        )
        if not possible_cards:
            return (-1, -1, -1.0, -1.0, -1.0, -1.0, -1, -1)

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
        highest_possible_rank = max(int(card.rank) for card in possible_cards)

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
        card_distribution = estimate_card_distribution(knowledge, observation)
        if not card_distribution:
            return 0.0

        return sum(
            probability
            for card, probability in card_distribution
            if is_card_playable(card, observation.fireworks)
        )

    def _dead_probability(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> float:
        card_distribution = estimate_card_distribution(knowledge, observation)
        if not card_distribution:
            return 0.0

        return sum(
            probability
            for card, probability in card_distribution
            if self._card_is_dead(card, observation)
        )

    def _already_played_probability(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> float:
        card_distribution = estimate_card_distribution(knowledge, observation)
        if not card_distribution:
            return 0.0

        return sum(
            probability
            for card, probability in card_distribution
            if is_card_already_played(card, observation.fireworks)
        )

    def _critical_probability(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> float:
        card_distribution = estimate_card_distribution(knowledge, observation)
        if not card_distribution:
            return 0.0

        return sum(
            probability
            for card, probability in card_distribution
            if self._card_is_critical(card, observation)
        )

    def _expected_discard_risk(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> float:
        card_distribution = estimate_card_distribution(knowledge, observation)
        if not card_distribution:
            return 0.0

        return sum(
            probability * self._discard_risk_for_card(card, observation)
            for card, probability in card_distribution
        )

    def _possible_cards(self, knowledge: CardKnowledge) -> tuple[Card, ...]:
        return tuple(
            Card(color=color, rank=rank)
            for color in knowledge.possible_colors
            for rank in knowledge.possible_ranks
        )

    def _possible_cards_for_knowledge(
        self, knowledge: CardKnowledge, observation: PlayerObservation
    ) -> tuple[Card, ...]:
        card_distribution = estimate_card_distribution(knowledge, observation)
        if card_distribution:
            return tuple(card for card, _ in card_distribution)
        return self._possible_cards(knowledge)

    def _risky_play_probability_threshold(
        self, observation: PlayerObservation
    ) -> float:
        if observation.strike_tokens >= 2:
            return 1.01
        if observation.strike_tokens == 1:
            return 0.90
        return 0.75

    def _card_is_critical(self, card: Card, observation: PlayerObservation) -> bool:
        if self._card_is_dead(card, observation):
            return False

        visible_count = self._visible_card_counts(observation)[(card.color, card.rank)]
        remaining_copies = CARD_COUNTS_BY_RANK[card.rank] - visible_count
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
        card_distribution = estimate_card_distribution(knowledge, observation)
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
        for card, count in build_visible_card_counts(observation).items():
            counts[(card.color, card.rank)] = count
        return counts
