from __future__ import annotations

import unittest

from hanabi_ai.training.behavior_cloning import (
    BehaviorCloningConfig,
    collect_behavior_cloning_samples,
    run_behavior_cloning_iteration,
)
from hanabi_ai.training.reinforce import build_reinforce_policy


class BehaviorCloningTrainingTests(unittest.TestCase):
    def test_collect_behavior_cloning_samples_returns_teacher_dataset(self) -> None:
        encoder, action_indexer, _ = build_reinforce_policy(player_count=2, seed=2)

        samples, stats = collect_behavior_cloning_samples(
            player_count=2,
            episode_count=2,
            encoder=encoder,
            action_indexer=action_indexer,
            seed_base=5,
        )

        self.assertEqual(stats.episode_count, 2)
        self.assertGreater(stats.sample_count, 0)
        self.assertEqual(len(samples), stats.sample_count)
        self.assertGreaterEqual(stats.average_score, 0.0)

    def test_behavior_cloning_improves_accuracy_on_collected_samples(self) -> None:
        encoder, action_indexer, policy = build_reinforce_policy(player_count=2, seed=1)
        samples, _ = collect_behavior_cloning_samples(
            player_count=2,
            episode_count=2,
            encoder=encoder,
            action_indexer=action_indexer,
            seed_base=9,
        )
        accuracy_before = policy.behavior_cloning_accuracy(samples)

        policy.apply_behavior_cloning(samples, learning_rate=0.05, epochs=4)
        accuracy_after = policy.behavior_cloning_accuracy(samples)

        self.assertGreaterEqual(accuracy_after, accuracy_before)
        self.assertGreater(accuracy_after, 0.0)

    def test_behavior_cloning_iteration_reports_training_accuracy(self) -> None:
        encoder, action_indexer, policy = build_reinforce_policy(player_count=2, seed=4)

        stats = run_behavior_cloning_iteration(
            policy,
            encoder=encoder,
            action_indexer=action_indexer,
            config=BehaviorCloningConfig(
                player_count=2,
                episode_count=2,
                learning_rate=0.05,
                epochs=3,
                seed_base=12,
            ),
        )

        self.assertEqual(stats.episode_count, 2)
        self.assertGreater(stats.sample_count, 0)
        self.assertGreaterEqual(stats.training_accuracy, 0.0)
        self.assertGreaterEqual(stats.validation_sample_count, 0)
        self.assertGreaterEqual(stats.validation_accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()
