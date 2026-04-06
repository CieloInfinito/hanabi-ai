from __future__ import annotations

from dataclasses import dataclass
from random import Random

from hanabi_ai.agents.rl.agent import RLPolicyAgent, RLTransitionRecorder
from hanabi_ai.agents.rl.encoding import LegalActionIndexer, ObservationVectorEncoder
from hanabi_ai.agents.rl.policy import (
    LinearSoftmaxPolicy,
    PolicyGradientSample,
    ValueRegressionSample,
)
from hanabi_ai.game.cards import MAX_HINT_TOKENS, Rank
from hanabi_ai.game.engine import EngineStepResult, HanabiGameEngine


@dataclass(frozen=True, slots=True)
class ReinforceConfig:
    player_count: int
    episode_count: int
    learning_rate: float = 0.002
    reward_scale: float = 1.0 / 25.0
    discount_factor: float = 0.95
    strike_penalty: float = 1.0
    hint_recovery_bonus: float = 0.2
    final_score_bonus_weight: float = 0.5
    seed_base: int = 0
    policy_seed: int = 0
    greedy_evaluation: bool = False


@dataclass(frozen=True, slots=True)
class ReinforceIterationStats:
    episode_count: int
    average_score: float
    min_score: int
    max_score: int
    average_return: float
    average_shaped_return: float
    average_value_prediction: float
    average_advantage: float
    total_transitions: int


def build_reinforce_policy(
    *,
    player_count: int,
    seed: int = 0,
    hidden_size: int | None = None,
) -> tuple[ObservationVectorEncoder, LegalActionIndexer, LinearSoftmaxPolicy]:
    encoder = ObservationVectorEncoder(player_count)
    action_indexer = LegalActionIndexer(player_count)
    policy = LinearSoftmaxPolicy(
        input_size=encoder.feature_size,
        action_count=action_indexer.action_count,
        seed=seed,
        hidden_size=hidden_size,
    )
    return encoder, action_indexer, policy


def run_reinforce_iteration(
    policy: LinearSoftmaxPolicy,
    *,
    encoder: ObservationVectorEncoder,
    action_indexer: LegalActionIndexer,
    config: ReinforceConfig,
) -> ReinforceIterationStats:
    if config.episode_count <= 0:
        raise ValueError("episode_count must be positive.")
    if not 0.0 <= config.discount_factor <= 1.0:
        raise ValueError("discount_factor must be between 0.0 and 1.0.")

    recorded_episodes: list[tuple[RLTransitionRecorder, tuple[float, ...], int]] = []
    scores: list[int] = []
    total_transitions = 0
    episode_returns: list[float] = []
    value_predictions: list[float] = []
    advantages: list[float] = []

    for episode_index in range(config.episode_count):
        engine = HanabiGameEngine(
            player_count=config.player_count,
            seed=config.seed_base + episode_index,
        )
        recorder = RLTransitionRecorder(features=[], legal_action_indices=[], chosen_action_indices=[])
        recorder_rewards: list[float] = []
        agents = [
            RLPolicyAgent(
                encoder=encoder,
                action_indexer=action_indexer,
                policy=policy,
                rng=Random(config.policy_seed + (episode_index * config.player_count) + player_id),
                greedy=config.greedy_evaluation,
                recorder=recorder,
            )
            for player_id in range(config.player_count)
        ]

        while not engine.is_terminal():
            player_id = engine.current_player
            observation = engine.get_observation(player_id)
            previous_score = engine.get_score()
            previous_hint_tokens = engine.hint_tokens
            action = agents[player_id].act(observation)
            step_result = engine.step(action)
            transition_reward = _transition_reward(
                step_result=step_result,
                previous_score=previous_score,
                previous_hint_tokens=previous_hint_tokens,
                reward_scale=config.reward_scale,
                strike_penalty=config.strike_penalty,
                hint_recovery_bonus=config.hint_recovery_bonus,
            )
            recorder_rewards.append(transition_reward)

        final_score = engine.get_score()
        if recorder_rewards:
            recorder_rewards[-1] += (
                final_score * config.reward_scale * config.final_score_bonus_weight
            )
        returns = _discounted_returns(
            tuple(recorder_rewards),
            discount_factor=config.discount_factor,
        )
        scores.append(final_score)
        total_transitions += len(recorder.chosen_action_indices)
        episode_returns.append(sum(recorder_rewards))
        recorded_episodes.append((recorder, returns, final_score))

    samples: list[PolicyGradientSample] = []
    value_samples: list[ValueRegressionSample] = []
    for recorder, returns, _final_score in recorded_episodes:
        for features, legal_action_indices, chosen_action_index, target_return in zip(
            recorder.features,
            recorder.legal_action_indices,
            recorder.chosen_action_indices,
            returns,
            strict=True,
        ):
            predicted_value = policy.predict_value(features)
            advantage = target_return - predicted_value
            value_predictions.append(predicted_value)
            advantages.append(advantage)
            samples.append(
                PolicyGradientSample(
                    features=features,
                    legal_action_indices=legal_action_indices,
                    chosen_action_index=chosen_action_index,
                    advantage=advantage,
                )
            )
            value_samples.append(
                ValueRegressionSample(
                    features=features,
                    target_value=target_return,
                )
            )

    if samples:
        policy.apply_policy_gradient(
            tuple(samples),
            learning_rate=config.learning_rate,
        )
        policy.apply_value_regression(
            tuple(value_samples),
            learning_rate=config.learning_rate,
        )

    average_return = sum(scores) / config.episode_count * config.reward_scale

    return ReinforceIterationStats(
        episode_count=config.episode_count,
        average_score=sum(scores) / config.episode_count,
        min_score=min(scores),
        max_score=max(scores),
        average_return=average_return,
        average_shaped_return=(
            sum(episode_returns) / len(episode_returns) if episode_returns else 0.0
        ),
        average_value_prediction=(
            sum(value_predictions) / len(value_predictions) if value_predictions else 0.0
        ),
        average_advantage=(
            sum(advantages) / len(advantages) if advantages else 0.0
        ),
        total_transitions=total_transitions,
    )


def _transition_reward(
    *,
    step_result: EngineStepResult,
    previous_score: int,
    previous_hint_tokens: int,
    reward_scale: float,
    strike_penalty: float,
    hint_recovery_bonus: float,
) -> float:
    score_delta = step_result.score - previous_score
    strike_delta = int(step_result.play_succeeded is False)

    reward = score_delta * reward_scale
    reward -= strike_penalty * strike_delta * reward_scale

    recovered_hint = (
        step_result.play_succeeded is True
        and step_result.removed_card is not None
        and step_result.removed_card.rank == Rank.FIVE
        and previous_hint_tokens < MAX_HINT_TOKENS
    )
    if recovered_hint:
        reward += hint_recovery_bonus * reward_scale

    return reward


def _discounted_returns(
    rewards: tuple[float, ...],
    *,
    discount_factor: float,
) -> tuple[float, ...]:
    running_return = 0.0
    returns_reversed: list[float] = []
    for reward in reversed(rewards):
        running_return = reward + (discount_factor * running_return)
        returns_reversed.append(running_return)
    returns_reversed.reverse()
    return tuple(returns_reversed)
