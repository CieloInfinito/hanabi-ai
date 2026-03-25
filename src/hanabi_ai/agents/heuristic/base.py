from __future__ import annotations

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
from hanabi_ai.game.observation import (
    CardKnowledge,
    PlayerObservation,
    PublicTurnRecord,
)
from hanabi_ai.game.rules import is_card_playable


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
        legal_hints = [
            action
            for action in observation.legal_actions
            if isinstance(action, (HintColorAction, HintRankAction))
        ]
        if not legal_hints:
            return None, None

        best_hint: HintColorAction | HintRankAction | None = None
        best_score: HintScore | None = None

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
