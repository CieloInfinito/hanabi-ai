from __future__ import annotations

from dataclasses import dataclass

from hanabi_ai.game.actions import Action, HintColorAction, HintRankAction
from hanabi_ai.game.observation import PlayerObservation
from hanabi_ai.search.determinization import sample_compatible_worlds
from hanabi_ai.search.hint_value import evaluate_hint_value
from hanabi_ai.search.rollout import evaluate_action_rollout


@dataclass(frozen=True, slots=True)
class ScoredAction:
    action: Action
    expected_value: float
    sample_count: int
    average_strikes_used: float
    terminal_rate: float


class ShortHorizonPlanner:
    """
    Rank legal actions by short rollouts over sampled compatible worlds.
    """

    def __init__(
        self,
        *,
        world_samples: int = 12,
        depth: int = 2,
        top_k: int = 4,
    ) -> None:
        self.world_samples = world_samples
        self.depth = depth
        self.top_k = top_k

    def rank_actions(
        self,
        observation: PlayerObservation,
        *,
        prioritized_actions: list[Action] | None = None,
        seed: int | None = None,
    ) -> list[ScoredAction]:
        candidate_actions = self._candidate_actions(
            observation,
            prioritized_actions=prioritized_actions,
        )
        if not candidate_actions:
            return []

        worlds = sample_compatible_worlds(
            observation,
            self.world_samples,
            seed=seed,
        )
        hint_static_values: dict[Action, float] = {}
        for action in candidate_actions:
            if not isinstance(action, (HintColorAction, HintRankAction)):
                continue
            hint_value = evaluate_hint_value(observation, action)
            hint_static_values[action] = (
                (0.45 * hint_value.guaranteed_play_delta)
                + (0.12 * hint_value.immediate_playable_touches)
                + (0.08 * hint_value.critical_touches)
                + (0.08 * hint_value.receiver_pressure_relief)
                + (0.04 if hint_value.turn_distance == 1 else 0.0)
            )
        scored_actions: list[ScoredAction] = []
        for action in candidate_actions:
            total_value = 0.0
            total_strikes = 0.0
            terminal_count = 0
            for world in worlds:
                summary = evaluate_action_rollout(
                    world.engine,
                    action,
                    depth=self.depth,
                )
                total_value += (
                    summary.score_delta
                    + (0.1 * summary.leaf_value)
                    - (0.75 * summary.strikes_used)
                    + hint_static_values.get(action, 0.0)
                )
                total_strikes += summary.strikes_used
                terminal_count += int(summary.terminated)

            sample_count = len(worlds)
            scored_actions.append(
                ScoredAction(
                    action=action,
                    expected_value=total_value / sample_count,
                    sample_count=sample_count,
                    average_strikes_used=total_strikes / sample_count,
                    terminal_rate=terminal_count / sample_count,
                )
            )

        return sorted(
            scored_actions,
            key=lambda item: (
                -item.expected_value,
                item.average_strikes_used,
                -item.terminal_rate,
                str(item.action),
            ),
        )

    def choose_action(
        self,
        observation: PlayerObservation,
        *,
        prioritized_actions: list[Action] | None = None,
        seed: int | None = None,
    ) -> Action:
        ranked_actions = self.rank_actions(
            observation,
            prioritized_actions=prioritized_actions,
            seed=seed,
        )
        if not ranked_actions:
            raise ValueError("ShortHorizonPlanner received an observation with no legal actions.")
        return ranked_actions[0].action

    def _candidate_actions(
        self,
        observation: PlayerObservation,
        *,
        prioritized_actions: list[Action] | None = None,
    ) -> list[Action]:
        legal_actions = list(observation.legal_actions)
        if not legal_actions:
            return []

        ordered_candidates: list[Action] = []
        seen_actions: set[Action] = set()
        for action in prioritized_actions or ():
            if action in observation.legal_actions and action not in seen_actions:
                ordered_candidates.append(action)
                seen_actions.add(action)

        if ordered_candidates:
            return ordered_candidates[: self.top_k]

        for action in legal_actions:
            if action not in seen_actions:
                ordered_candidates.append(action)
                seen_actions.add(action)

        return ordered_candidates[: self.top_k]
