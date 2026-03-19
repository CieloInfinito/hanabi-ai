from __future__ import annotations

import unittest

import _path_setup
from card_game_ai.agents.random_agent import RandomAgent
from card_game_ai.training.self_play import (
    run_self_play_game,
    run_self_play_game_with_trace,
)


class SelfPlayTests(unittest.TestCase):
    def test_run_self_play_game_completes_with_random_agents(self) -> None:
        agents = [RandomAgent(seed=1), RandomAgent(seed=2)]

        result = run_self_play_game(agents, seed=3)

        self.assertEqual(result.player_count, 2)
        self.assertGreater(result.turn_count, 0)
        self.assertGreaterEqual(result.final_score, 0)
        self.assertLessEqual(result.final_score, 25)
        self.assertGreaterEqual(result.hint_tokens, 0)
        self.assertGreaterEqual(result.strike_tokens, 0)

    def test_run_self_play_game_rejects_invalid_agent_count(self) -> None:
        with self.assertRaises(ValueError):
            run_self_play_game([RandomAgent(seed=1)], seed=2)

    def test_run_self_play_game_with_trace_returns_readable_trace(self) -> None:
        agents = [RandomAgent(seed=4), RandomAgent(seed=5)]

        traced_result = run_self_play_game_with_trace(agents, seed=6)

        self.assertGreater(traced_result.summary.turn_count, 0)
        self.assertIn("=== Self-Play Start ===", traced_result.trace)
        self.assertIn("=== Self-Play Turn", traced_result.trace)
        self.assertIn("Chosen action:", traced_result.trace)
        self.assertIn("State after turn:", traced_result.trace)
        self.assertIn("=== Self-Play End ===", traced_result.trace)


if __name__ == "__main__":
    unittest.main()
