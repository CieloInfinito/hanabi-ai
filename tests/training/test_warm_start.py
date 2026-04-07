from __future__ import annotations

import unittest

from hanabi_ai.training.warm_start import WarmStartConfig, run_warm_started_reinforce


class WarmStartTrainingTests(unittest.TestCase):
    def test_warm_start_runs_cloning_then_reinforce(self) -> None:
        stats = run_warm_started_reinforce(
            WarmStartConfig(
                player_count=2,
                cloning_episode_count=2,
                cloning_epochs=2,
                cloning_learning_rate=0.05,
                reinforce_iterations=2,
                reinforce_episode_count=2,
                reinforce_actor_learning_rate=0.002,
                reinforce_critic_learning_rate=0.002,
                reinforce_discount_factor=0.95,
                reinforce_final_score_bonus_weight=0.5,
                seed_base=30,
                policy_seed=5,
            )
        )

        self.assertEqual(stats.cloning_stats.episode_count, 2)
        self.assertEqual(len(stats.reinforce_stats), 2)
        self.assertGreater(stats.cloning_stats.sample_count, 0)
        self.assertTrue(
            all(iteration.total_transitions > 0 for iteration in stats.reinforce_stats)
        )


if __name__ == "__main__":
    unittest.main()
