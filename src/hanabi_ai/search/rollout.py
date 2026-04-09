from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from hanabi_ai.agents.beliefs import PublicBeliefState
from hanabi_ai.agents.heuristic.convention_tempo import ConventionTempoHeuristicAgent
from hanabi_ai.game.actions import Action
from hanabi_ai.game.engine import HanabiGameEngine


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    final_score: int
    score_delta: int
    strikes_used: int
    leaf_value: float
    terminated: bool


def evaluate_action_rollout(
    engine: HanabiGameEngine,
    action: Action,
    *,
    depth: int,
    agent_factory: Callable[[int], object] | None = None,
) -> RolloutSummary:
    """
    Apply one candidate action and roll the game forward for a short horizon.
    """
    rollout_engine = engine.clone()
    starting_score = rollout_engine.get_score()
    starting_strikes = rollout_engine.strike_tokens

    rollout_engine.step(action)
    run_rollout_policy(
        rollout_engine,
        depth=max(depth - 1, 0),
        agent_factory=agent_factory,
    )

    return RolloutSummary(
        final_score=rollout_engine.get_score(),
        score_delta=rollout_engine.get_score() - starting_score,
        strikes_used=rollout_engine.strike_tokens - starting_strikes,
        leaf_value=_evaluate_leaf_state(rollout_engine),
        terminated=rollout_engine.is_terminal(),
    )


def run_rollout_policy(
    engine: HanabiGameEngine,
    *,
    depth: int,
    agent_factory: Callable[[int], object] | None = None,
) -> None:
    """
    Advance a copied engine for a short horizon using heuristic self-play.
    """
    if depth <= 0:
        return

    factory = agent_factory or (lambda _player_id: ConventionTempoHeuristicAgent())
    agents = [factory(player_id) for player_id in range(engine.player_count)]

    for _ in range(depth):
        if engine.is_terminal():
            break
        player_id = engine.current_player
        observation = engine.get_observation(player_id)
        engine.step(agents[player_id].act(observation))


def _evaluate_leaf_state(engine: HanabiGameEngine) -> float:
    """
    Lightweight value estimate for a non-terminal rollout leaf.
    """
    score = float(engine.get_score())
    guaranteed_plays = 0
    known_attributes = 0

    for player_id in range(engine.player_count):
        observation = engine.get_observation(player_id)
        belief_state = PublicBeliefState.from_observation(observation)
        for seat_id in range(engine.player_count):
            guaranteed_plays += len(belief_state.guaranteed_play_indices(seat_id))
        known_attributes += sum(
            int(card_knowledge.hinted_color is not None)
            + int(card_knowledge.hinted_rank is not None)
            for card_knowledge in observation.hand_knowledge
        )

    return (
        (4.0 * score)
        + (0.6 * guaranteed_plays)
        + (0.05 * known_attributes)
        + (0.1 * engine.hint_tokens)
        - (1.5 * engine.strike_tokens)
    )
