from __future__ import annotations

from dataclasses import dataclass

from hanabi_ai.agents.heuristic.convention_tempo import ConventionTempoHeuristicAgent
from hanabi_ai.agents.rl.encoding import LegalActionIndexer, ObservationVectorEncoder
from hanabi_ai.agents.rl.policy import BehaviorCloningSample, LinearSoftmaxPolicy
from hanabi_ai.game.actions import normalize_agent_decision
from hanabi_ai.game.engine import HanabiGameEngine


@dataclass(frozen=True, slots=True)
class BehaviorCloningConfig:
    player_count: int
    episode_count: int
    learning_rate: float = 0.05
    epochs: int = 4
    seed_base: int = 0
    validation_split: float = 0.2


@dataclass(frozen=True, slots=True)
class BehaviorCloningStats:
    episode_count: int
    sample_count: int
    average_score: float
    min_score: int
    max_score: int
    training_accuracy: float
    validation_sample_count: int
    validation_accuracy: float


def collect_behavior_cloning_samples(
    *,
    player_count: int,
    episode_count: int,
    encoder: ObservationVectorEncoder,
    action_indexer: LegalActionIndexer,
    seed_base: int = 0,
) -> tuple[tuple[BehaviorCloningSample, ...], BehaviorCloningStats]:
    if episode_count <= 0:
        raise ValueError("episode_count must be positive.")

    samples: list[BehaviorCloningSample] = []
    scores: list[int] = []

    for episode_index in range(episode_count):
        engine = HanabiGameEngine(
            player_count=player_count,
            seed=seed_base + episode_index,
        )
        agents = [ConventionTempoHeuristicAgent() for _ in range(player_count)]

        while not engine.is_terminal():
            player_id = engine.current_player
            observation = engine.get_observation(player_id)
            features = encoder.encode(observation)
            legal_action_indices = action_indexer.legal_action_indices(observation)
            expert_decision = normalize_agent_decision(agents[player_id].act(observation))
            target_action_index = action_indexer.action_index_for_action(
                expert_decision.action,
                current_player=observation.current_player,
            )
            samples.append(
                BehaviorCloningSample(
                    features=features,
                    legal_action_indices=legal_action_indices,
                    target_action_index=target_action_index,
                )
            )
            engine.step(expert_decision)

        scores.append(engine.get_score())

    return (
        tuple(samples),
        BehaviorCloningStats(
            episode_count=episode_count,
            sample_count=len(samples),
            average_score=sum(scores) / episode_count,
            min_score=min(scores),
            max_score=max(scores),
            training_accuracy=0.0,
            validation_sample_count=0,
            validation_accuracy=0.0,
        ),
    )


def run_behavior_cloning_iteration(
    policy: LinearSoftmaxPolicy,
    *,
    encoder: ObservationVectorEncoder,
    action_indexer: LegalActionIndexer,
    config: BehaviorCloningConfig,
) -> BehaviorCloningStats:
    samples, stats = collect_behavior_cloning_samples(
        player_count=config.player_count,
        episode_count=config.episode_count,
        encoder=encoder,
        action_indexer=action_indexer,
        seed_base=config.seed_base,
    )
    if not 0.0 <= config.validation_split < 1.0:
        raise ValueError("validation_split must be between 0.0 and 1.0.")

    validation_count = int(len(samples) * config.validation_split)
    if validation_count > 0:
        training_samples = samples[:-validation_count]
        validation_samples = samples[-validation_count:]
    else:
        training_samples = samples
        validation_samples = ()

    policy.apply_behavior_cloning(
        training_samples,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )
    return BehaviorCloningStats(
        episode_count=stats.episode_count,
        sample_count=stats.sample_count,
        average_score=stats.average_score,
        min_score=stats.min_score,
        max_score=stats.max_score,
        training_accuracy=policy.behavior_cloning_accuracy(training_samples),
        validation_sample_count=len(validation_samples),
        validation_accuracy=policy.behavior_cloning_accuracy(validation_samples),
    )
