from __future__ import annotations

from dataclasses import dataclass

from hanabi_ai.training.behavior_cloning import (
    BehaviorCloningConfig,
    BehaviorCloningStats,
    run_behavior_cloning_iteration,
)
from hanabi_ai.training.reinforce import (
    ReinforceConfig,
    ReinforceIterationStats,
    build_reinforce_policy,
    run_reinforce_iteration,
)


@dataclass(frozen=True, slots=True)
class WarmStartConfig:
    player_count: int
    cloning_episode_count: int
    cloning_epochs: int
    cloning_learning_rate: float
    reinforce_iterations: int
    reinforce_episode_count: int
    reinforce_learning_rate: float = 0.002
    reinforce_discount_factor: float = 0.95
    reinforce_final_score_bonus_weight: float = 0.5
    seed_base: int = 0
    policy_seed: int = 0
    greedy_evaluation: bool = False


@dataclass(frozen=True, slots=True)
class WarmStartStats:
    cloning_stats: BehaviorCloningStats
    reinforce_stats: tuple[ReinforceIterationStats, ...]


def run_warm_started_reinforce(
    config: WarmStartConfig,
    *,
    hidden_size: int | None = None,
) -> WarmStartStats:
    if config.reinforce_iterations <= 0:
        raise ValueError("reinforce_iterations must be positive.")

    encoder, action_indexer, policy = build_reinforce_policy(
        player_count=config.player_count,
        seed=config.policy_seed,
        hidden_size=hidden_size,
    )
    cloning_stats = run_behavior_cloning_iteration(
        policy,
        encoder=encoder,
        action_indexer=action_indexer,
        config=BehaviorCloningConfig(
            player_count=config.player_count,
            episode_count=config.cloning_episode_count,
            learning_rate=config.cloning_learning_rate,
            epochs=config.cloning_epochs,
            seed_base=config.seed_base,
        ),
    )

    reinforce_stats: list[ReinforceIterationStats] = []
    reinforce_seed_base = config.seed_base + config.cloning_episode_count
    for iteration_index in range(config.reinforce_iterations):
        stats = run_reinforce_iteration(
            policy,
            encoder=encoder,
            action_indexer=action_indexer,
            config=ReinforceConfig(
                player_count=config.player_count,
                episode_count=config.reinforce_episode_count,
                learning_rate=config.reinforce_learning_rate,
                discount_factor=config.reinforce_discount_factor,
                final_score_bonus_weight=config.reinforce_final_score_bonus_weight,
                seed_base=(
                    reinforce_seed_base
                    + (iteration_index * config.reinforce_episode_count)
                ),
                policy_seed=config.policy_seed,
                greedy_evaluation=config.greedy_evaluation,
            ),
        )
        reinforce_stats.append(stats)

    return WarmStartStats(
        cloning_stats=cloning_stats,
        reinforce_stats=tuple(reinforce_stats),
    )
