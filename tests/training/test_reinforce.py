from __future__ import annotations

import unittest
from random import Random

from hanabi_ai.agents.rl.policy import ValueRegressionSample
from hanabi_ai.training.reinforce import (
    ReinforceConfig,
    build_reinforce_policy,
    run_reinforce_iteration,
)


class ReinforceTrainingTests(unittest.TestCase):
    def test_build_reinforce_policy_matches_encoder_and_action_sizes(self) -> None:
        encoder, action_indexer, policy = build_reinforce_policy(player_count=2, seed=3)

        self.assertEqual(policy.input_size, encoder.feature_size)
        self.assertEqual(policy.action_count, action_indexer.action_count)
        self.assertEqual(policy.hidden_size, 48)

    def test_reinforce_config_defaults_match_tuned_values(self) -> None:
        config = ReinforceConfig(player_count=2, episode_count=3)

        self.assertEqual(config.actor_learning_rate, 0.002)
        self.assertEqual(config.critic_learning_rate, 0.002)
        self.assertEqual(config.discount_factor, 0.95)
        self.assertEqual(config.final_score_bonus_weight, 0.5)

    def test_build_reinforce_policy_accepts_hidden_size_override(self) -> None:
        encoder, action_indexer, policy = build_reinforce_policy(
            player_count=2,
            seed=3,
            hidden_size=32,
        )

        self.assertEqual(policy.input_size, encoder.feature_size)
        self.assertEqual(policy.action_count, action_indexer.action_count)
        self.assertEqual(policy.hidden_size, 32)

    def test_reinforce_iteration_returns_non_empty_stats(self) -> None:
        encoder, action_indexer, policy = build_reinforce_policy(player_count=2, seed=1)

        stats = run_reinforce_iteration(
            policy,
            encoder=encoder,
            action_indexer=action_indexer,
            config=ReinforceConfig(
                player_count=2,
                episode_count=3,
                actor_learning_rate=0.02,
                critic_learning_rate=0.01,
                seed_base=20,
                policy_seed=10,
            ),
        )

        self.assertEqual(stats.episode_count, 3)
        self.assertGreaterEqual(stats.average_score, 0.0)
        self.assertGreater(stats.total_transitions, 0)
        self.assertIsInstance(stats.average_shaped_return, float)
        self.assertIsInstance(stats.average_value_prediction, float)
        self.assertIsInstance(stats.average_advantage, float)

    def test_policy_outputs_normalized_legal_probabilities(self) -> None:
        encoder, _, policy = build_reinforce_policy(player_count=2, seed=4)
        features = tuple(0.1 for _ in range(encoder.feature_size))
        legal_action_indices = (0, 1, 8)

        probabilities = policy.legal_action_probabilities(features, legal_action_indices)

        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        sampled = policy.sample_action(features, legal_action_indices, rng=Random(2))
        self.assertIn(sampled.action_index, legal_action_indices)

    def test_policy_value_head_can_learn_simple_target(self) -> None:
        encoder, _, policy = build_reinforce_policy(player_count=2, seed=4)
        features = tuple(0.1 for _ in range(encoder.feature_size))
        before = policy.predict_value(features)

        policy.apply_value_regression(
            (ValueRegressionSample(features=features, target_value=1.0),),
            learning_rate=0.1,
        )
        after = policy.predict_value(features)

        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
