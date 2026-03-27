from __future__ import annotations

from dataclasses import dataclass

from hanabi_ai.agents.heuristic._mixins import _HeuristicScoringMixin
from hanabi_ai.agents.heuristic._scoring import HintScore
from hanabi_ai.game.actions import (
    Action,
    AgentDecision,
    DiscardAction,
    HintColorAction,
    HintRankAction,
    PlayAction,
)
from hanabi_ai.game.cards import Card, Color
from hanabi_ai.game.observation import (
    CardKnowledge,
    ObservedHand,
    PlayerObservation,
    PublicTurnRecord,
)
from hanabi_ai.game.rules import is_card_playable


@dataclass(frozen=True, slots=True)
class _HintPriorityWeights:
    follow_on_value: int = 0
    receiver_needs_help: int = 0
    immediate_receiver: int = 0
    near_term_receiver: int = 0
    actionable_hint: int = 0
    critical_playable: int = 0
    turn_distance_penalty: int = 0


class BaseHeuristicAgent(_HeuristicScoringMixin):
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
        self._cache_belief_state(observation)

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

        discard_action = self._choose_discard_action(observation)
        if (
            discard_action is not None
            and helpful_hint is not None
            and self._should_prefer_discard_over_hint(
                observation,
                discard_action,
                helpful_hint,
                helpful_hint_score,
            )
        ):
            return discard_action

        if helpful_hint is not None:
            return self._attach_hint_presentation(helpful_hint, observation)

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

    def explain_action_choice(
        self,
        observation: PlayerObservation,
        action: Action,
    ) -> tuple[str, ...]:
        """
        Return optional human-readable notes about why this action was chosen.
        """
        ranked_hints = self._ranked_hint_candidates(observation)
        if not ranked_hints:
            return ()

        lines = ["Hint candidates:"]
        for index, (candidate_action, score, priority) in enumerate(
            ranked_hints[:3],
            start=1,
        ):
            lines.append(
                f"{index}. {candidate_action} | {self._format_hint_debug(score, priority)}"
            )

        if isinstance(action, (HintColorAction, HintRankAction)):
            chosen_entry = next(
                (
                    (score, priority)
                    for candidate_action, score, priority in ranked_hints
                    if candidate_action == action
                ),
                None,
            )
            if chosen_entry is not None:
                score, priority = chosen_entry
                lines.append(
                    "Chosen hint: "
                    f"{action} | {self._format_hint_debug(score, priority)}"
                )
            return tuple(lines)

        best_hint_action, best_hint_score, best_hint_priority = ranked_hints[0]
        lines.append(
            "Best hint alternative: "
            f"{best_hint_action} | "
            f"{self._format_hint_debug(best_hint_score, best_hint_priority)}"
        )
        return tuple(lines)

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

    def _should_prefer_discard_over_hint(
        self,
        observation: PlayerObservation,
        discard_action: DiscardAction,
        hint_action: HintColorAction | HintRankAction,
        hint_score: HintScore | None,
    ) -> bool:
        return False

    def _choose_definitely_playable_action(
        self, observation: PlayerObservation
    ) -> PlayAction | None:
        legal_plays = {
            action.card_index: action
            for action in observation.legal_actions
            if isinstance(action, PlayAction)
        }
        best_action: PlayAction | None = None
        best_score = None

        for index, knowledge in enumerate(observation.hand_knowledge):
            if index not in legal_plays:
                continue

            possible_cards = self._possible_cards_for_knowledge(knowledge, observation)
            if possible_cards and all(
                is_card_playable(card, observation.fireworks) for card in possible_cards
            ):
                score = self._guaranteed_play_score(knowledge, observation, index=index)
                if best_score is None or score > best_score:
                    best_action = legal_plays[index]
                    best_score = score

        return best_action

    def _choose_hint_for_other_players(
        self, observation: PlayerObservation
    ) -> tuple[HintColorAction | HintRankAction | None, HintScore | None]:
        ranked_hints = self._ranked_hint_candidates(observation)
        if not ranked_hints:
            return None, None
        best_hint, best_score, _ = ranked_hints[0]
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
        best_hint_score: HintScore | None,
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
        best_score = None

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


    def _risky_play_probability_threshold(
        self, observation: PlayerObservation
    ) -> float:
        if observation.strike_tokens >= 2:
            return 1.01
        if observation.strike_tokens == 1:
            return 0.90
        return 0.75

    def _hint_priority(
        self,
        observation: PlayerObservation,
        observed_hand: ObservedHand,
        hint_action: HintColorAction | HintRankAction,
        score: HintScore,
    ) -> tuple[HintScore, int, int, int, int, int, int, int]:
        player_count = len(observation.other_player_hands) + 1
        turn_distance = self._turn_distance(
            observation.current_player,
            observed_hand.player_id,
            player_count,
        )
        guaranteed_play_hits = score[0]
        playable_hits = score[1]
        critical_playable_hits = score[4]
        receiver_needs_help = self._receiver_needs_help(
            observation,
            observed_hand.player_id,
        )
        follow_on_value = self._follow_on_play_value(
            observation,
            observed_hand.player_id,
            hint_action,
        )
        immediate_receiver_bonus = int(turn_distance == 1 and playable_hits >= 1)
        near_term_receiver_bonus = int(turn_distance <= 2 and guaranteed_play_hits >= 1)
        actionable_hint_bonus = int(playable_hits >= 1 or guaranteed_play_hits >= 1)
        weighted_bonus = self._base_hint_priority_bonus(
            player_count=player_count,
            follow_on_value=follow_on_value,
            receiver_needs_help=receiver_needs_help,
            immediate_receiver_bonus=immediate_receiver_bonus,
            near_term_receiver_bonus=near_term_receiver_bonus,
            actionable_hint_bonus=actionable_hint_bonus,
            critical_playable_hits=critical_playable_hits,
            turn_distance=turn_distance,
        ) + self._hint_priority_adjustment(
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

        return (
            score,
            weighted_bonus,
            follow_on_value,
            int(receiver_needs_help),
            immediate_receiver_bonus,
            near_term_receiver_bonus,
            actionable_hint_bonus,
            critical_playable_hits,
        )

    def _base_hint_priority_weights(self, player_count: int) -> _HintPriorityWeights:
        return _HintPriorityWeights()

    def _base_hint_priority_bonus(
        self,
        *,
        player_count: int,
        follow_on_value: int,
        receiver_needs_help: bool,
        immediate_receiver_bonus: int,
        near_term_receiver_bonus: int,
        actionable_hint_bonus: int,
        critical_playable_hits: int,
        turn_distance: int,
    ) -> int:
        weights = self._base_hint_priority_weights(player_count)
        return (
            weights.follow_on_value * follow_on_value
            + weights.receiver_needs_help * int(receiver_needs_help)
            + weights.immediate_receiver * immediate_receiver_bonus
            + weights.near_term_receiver * near_term_receiver_bonus
            + weights.actionable_hint * actionable_hint_bonus
            + weights.critical_playable * critical_playable_hits
            - weights.turn_distance_penalty * turn_distance
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
        return 0

    def _turn_distance(
        self,
        current_player: int,
        target_player: int,
        player_count: int,
    ) -> int:
        return (target_player - current_player) % player_count

    def _receiver_needs_help(
        self,
        observation: PlayerObservation,
        player_id: int,
    ) -> bool:
        belief_state = self._belief_state(observation)
        knowledge = belief_state.knowledge_for_player(player_id)
        if belief_state.guaranteed_play_indices_for_knowledge(knowledge):
            return False

        known_attribute_count = sum(
            int(card_knowledge.hinted_color is not None)
            + int(card_knowledge.hinted_rank is not None)
            for card_knowledge in knowledge
        )
        return known_attribute_count <= 1

    def _receiver_under_pressure(
        self,
        observation: PlayerObservation,
        player_id: int,
    ) -> bool:
        belief_state = self._belief_state(observation)
        knowledge = belief_state.knowledge_for_player(player_id)
        if belief_state.guaranteed_play_indices_for_knowledge(knowledge):
            return False

        best_discard_score = max(
            (
                self._score_discard_knowledge(card_knowledge, observation)
                for card_knowledge in knowledge
            ),
            default=None,
        )
        if best_discard_score is None:
            return False

        definitely_safe_discard = bool(best_discard_score[0] or best_discard_score[1])
        expected_discard_risk = -best_discard_score[5]
        return self._receiver_needs_help(observation, player_id) and (
            not definitely_safe_discard and expected_discard_risk >= 1.0
        )

    def _follow_on_play_value(
        self,
        observation: PlayerObservation,
        target_player: int,
        hint_action: HintColorAction | HintRankAction,
    ) -> int:
        target_hand = next(
            (
                hand
                for hand in observation.other_player_hands
                if hand.player_id == target_player
            ),
            None,
        )
        if target_hand is None:
            return 0

        touched_playable_cards = [
            card
            for card in target_hand.cards
            if self._hint_touches_card(card, hint_action)
            and self._card_is_playable_now(card, observation)
        ]
        if not touched_playable_cards:
            return 0

        unlocked_cards: set[tuple[Color, int]] = set()
        for card in touched_playable_cards:
            next_rank = int(card.rank) + 1
            for observed_hand in observation.other_player_hands:
                if observed_hand.player_id == target_player:
                    continue
                for visible_card in observed_hand.cards:
                    if (
                        visible_card.color == card.color
                        and int(visible_card.rank) == next_rank
                    ):
                        unlocked_cards.add((visible_card.color, int(visible_card.rank)))

        return len(unlocked_cards)

    def _hint_touches_card(
        self,
        card: Card,
        hint_action: HintColorAction | HintRankAction,
    ) -> bool:
        if isinstance(hint_action, HintColorAction):
            return card.color == hint_action.color
        return card.rank == hint_action.rank

    def _ranked_hint_candidates(
        self,
        observation: PlayerObservation,
    ) -> list[
        tuple[
            HintColorAction | HintRankAction,
            HintScore,
            tuple[HintScore, int, int, int, int, int, int, int],
        ]
    ]:
        legal_hints = [
            action
            for action in observation.legal_actions
            if isinstance(action, (HintColorAction, HintRankAction))
        ]
        if not legal_hints:
            return []

        ranked_hints: list[
            tuple[
                HintColorAction | HintRankAction,
                HintScore,
                tuple[HintScore, int, int, int, int, int, int, int],
            ]
        ] = []
        insertion_order = 0
        for observed_hand in observation.other_player_hands:
            candidate_hints = self._build_candidate_hints(
                observed_hand,
                legal_hints,
                observation,
            )
            for hint_action, score in candidate_hints:
                priority = self._hint_priority(
                    observation,
                    observed_hand,
                    hint_action,
                    score,
                )
                ranked_hints.append((hint_action, score, priority, insertion_order))
                insertion_order += 1

        ranked_hints.sort(key=lambda item: item[2], reverse=True)
        return [
            (hint_action, score, priority)
            for hint_action, score, priority, _ in ranked_hints
        ]

    def _format_hint_debug(
        self,
        score: HintScore,
        priority: tuple[HintScore, int, int, int, int, int, int, int],
    ) -> str:
        return (
            f"bonus={priority[1]} safe={score[0]} playable={score[1]} "
            f"useful={score[5]} info={score[6]} follow_on={priority[2]} "
            f"needs_help={priority[3]} immediate={priority[4]} "
            f"near_term={priority[5]} critical={priority[7]}"
        )
