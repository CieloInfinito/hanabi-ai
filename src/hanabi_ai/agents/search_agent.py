from __future__ import annotations

from dataclasses import dataclass

from hanabi_ai.agents.heuristic.convention_tempo import ConventionTempoHeuristicAgent
from hanabi_ai.game.actions import (
    Action,
    AgentDecision,
    DiscardAction,
    HintColorAction,
    HintRankAction,
    PlayAction,
)
from hanabi_ai.game.observation import PlayerObservation
from hanabi_ai.search.planner import ScoredAction, ShortHorizonPlanner


@dataclass(frozen=True, slots=True)
class SearchDecisionAnalysis:
    baseline_decision: Action | AgentDecision
    baseline_action: Action
    ranked_actions: tuple[ScoredAction, ...]
    planner_selected_action: Action | None
    switched_action: bool


class SearchHeuristicAgent(ConventionTempoHeuristicAgent):
    """
    Hybrid agent that keeps ConventionTempo as a fallback prior and uses
    short compatible-world rollouts to choose among top candidate actions.
    """

    def __init__(
        self,
        *,
        world_samples: int = 4,
        depth: int = 3,
        top_k: int = 4,
        min_improvement: float = 0.0,
    ) -> None:
        super().__init__()
        self._planner = ShortHorizonPlanner(
            world_samples=world_samples,
            depth=depth,
            top_k=top_k,
        )
        self._min_improvement = min_improvement

    def analyze_decision(
        self,
        observation: PlayerObservation,
    ) -> SearchDecisionAnalysis:
        refined_observation = self._apply_private_conventions(observation)
        self._cache_belief_state(refined_observation)

        guaranteed_play = self._choose_definitely_playable_action(refined_observation)
        if guaranteed_play is not None:
            return SearchDecisionAnalysis(
                baseline_decision=guaranteed_play,
                baseline_action=guaranteed_play,
                ranked_actions=(),
                planner_selected_action=guaranteed_play,
                switched_action=False,
            )

        baseline_decision = super().act(observation)
        baseline_action = (
            baseline_decision.action
            if isinstance(baseline_decision, AgentDecision)
            else baseline_decision
        )
        if isinstance(baseline_action, (HintColorAction, HintRankAction, PlayAction)):
            return SearchDecisionAnalysis(
                baseline_decision=baseline_decision,
                baseline_action=baseline_action,
                ranked_actions=(),
                planner_selected_action=baseline_action,
                switched_action=False,
            )

        prioritized_actions = self._candidate_actions(
            refined_observation,
            baseline_action=baseline_action,
        )
        try:
            ranked_actions = tuple(
                self._planner.rank_actions(
                    refined_observation,
                    prioritized_actions=prioritized_actions,
                )
            )
        except ValueError:
            return SearchDecisionAnalysis(
                baseline_decision=baseline_decision,
                baseline_action=baseline_action,
                ranked_actions=(),
                planner_selected_action=baseline_action,
                switched_action=False,
            )
        baseline_entry = next(
            (entry for entry in ranked_actions if entry.action == baseline_action),
            None,
        )
        planner_entry = ranked_actions[0] if ranked_actions else None
        if (
            planner_entry is None
            or baseline_entry is None
            or planner_entry.expected_value
            < baseline_entry.expected_value + self._min_improvement
        ):
            return SearchDecisionAnalysis(
                baseline_decision=baseline_decision,
                baseline_action=baseline_action,
                ranked_actions=ranked_actions,
                planner_selected_action=baseline_action,
                switched_action=False,
            )

        return SearchDecisionAnalysis(
            baseline_decision=baseline_decision,
            baseline_action=baseline_action,
            ranked_actions=ranked_actions,
            planner_selected_action=planner_entry.action,
            switched_action=True,
        )

    def act(self, observation: PlayerObservation):
        analysis = self.analyze_decision(observation)
        if not analysis.switched_action or analysis.planner_selected_action is None:
            return analysis.baseline_decision

        refined_observation = self._apply_private_conventions(observation)
        if isinstance(analysis.planner_selected_action, (HintColorAction, HintRankAction)):
            return self._attach_hint_presentation(
                analysis.planner_selected_action,
                refined_observation,
            )
        return analysis.planner_selected_action

    def _candidate_actions(
        self,
        observation: PlayerObservation,
        *,
        baseline_action: Action,
    ) -> list[Action]:
        candidates: list[Action] = [baseline_action]

        helpful_hint, helpful_hint_score = self._choose_hint_for_other_players(observation)
        confident_play = self._choose_confident_probabilistic_play(
            observation,
            best_hint_score=helpful_hint_score,
        )
        discard_action = self._choose_discard_action(observation)
        fallback_play = self._choose_any_play_action(observation)

        candidate_pool = (
            (helpful_hint, discard_action)
            if isinstance(baseline_action, DiscardAction)
            else (helpful_hint, discard_action, confident_play, fallback_play)
        )
        for action in candidate_pool:
            if action is not None and action not in candidates:
                candidates.append(action)

        if isinstance(baseline_action, DiscardAction):
            for hint_action, _, _ in self._ranked_hint_candidates(observation)[:2]:
                if hint_action not in candidates:
                    candidates.append(hint_action)

        return candidates
