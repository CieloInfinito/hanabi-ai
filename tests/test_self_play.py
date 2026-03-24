from __future__ import annotations

import unittest

import _path_setup
from card_game_ai.agents.heuristic.conservative_agent import (
    ConservativeHeuristicAgent,
)
from card_game_ai.agents.random_agent import RandomAgent
from card_game_ai.game.actions import AgentDecision, HintColorAction, HintPresentation
from card_game_ai.training.self_play import (
    evaluate_self_play,
    run_self_play_game,
    run_self_play_game_with_trace,
)


class SelfPlayTests(unittest.TestCase):
    class OrderedHintAgent:
        def act(self, observation):
            # Emits valid hint decisions using a custom reverse-order presentation
            # so self-play exercises the richer agent-decision API end to end.
            for action in observation.legal_actions:
                if isinstance(action, HintColorAction):
                    target_hand = next(
                        hand
                        for hand in observation.other_player_hands
                        if hand.player_id == action.target_player
                    )
                    return AgentDecision(
                        action=action,
                        hint_presentation=HintPresentation(
                            revealed_indices=tuple(
                                reversed(
                                    tuple(
                                        index
                                        for index, card in enumerate(target_hand.cards)
                                        if card.color == action.color
                                    )
                                )
                            ),
                        ),
                    )
            return observation.legal_actions[0]

    def test_run_self_play_game_completes_with_random_agents(self) -> None:
        # Verifies that a full self-play game runs to completion and returns sane summary values.
        agents = [RandomAgent(seed=1), RandomAgent(seed=2)]

        result = run_self_play_game(agents, seed=3)

        self.assertEqual(result.player_count, 2)
        self.assertGreater(result.turn_count, 0)
        self.assertGreaterEqual(result.final_score, 0)
        self.assertLessEqual(result.final_score, 25)
        self.assertGreaterEqual(result.hint_tokens, 0)
        self.assertGreaterEqual(result.strike_tokens, 0)

    def test_run_self_play_game_rejects_invalid_agent_count(self) -> None:
        # Verifies that self-play enforces the supported minimum player count.
        with self.assertRaises(ValueError):
            run_self_play_game([RandomAgent(seed=1)], seed=2)

    def test_run_self_play_game_with_trace_returns_readable_trace(self) -> None:
        # Verifies that traced self-play returns a human-readable trace with key sections.
        agents = [RandomAgent(seed=4), RandomAgent(seed=5)]

        traced_result = run_self_play_game_with_trace(agents, seed=6)

        self.assertGreater(traced_result.summary.turn_count, 0)
        self.assertIn("=== Self-Play Start ===", traced_result.trace)
        self.assertIn("=== Self-Play Turn", traced_result.trace)
        self.assertIn("Chosen action:", traced_result.trace)
        self.assertIn("State after turn:", traced_result.trace)
        self.assertIn("=== Self-Play End ===", traced_result.trace)

    def test_run_self_play_game_accepts_agent_decisions(self) -> None:
        # Verifies that self-play accepts agents returning richer decisions.
        agents = [self.OrderedHintAgent(), RandomAgent(seed=9)]

        result = run_self_play_game(agents, seed=7)

        self.assertEqual(result.player_count, 2)
        self.assertGreater(result.turn_count, 0)

    def test_evaluate_self_play_returns_aggregate_metrics(self) -> None:
        # Verifies that batched evaluation returns sensible aggregate metrics.
        evaluation = evaluate_self_play(
            lambda player_id, game_index: RandomAgent(seed=100 + (game_index * 2) + player_id),
            player_count=2,
            game_count=5,
            seed_base=10,
        )

        self.assertEqual(evaluation.game_count, 5)
        self.assertEqual(evaluation.player_count, 2)
        self.assertGreaterEqual(evaluation.average_score, 0.0)
        self.assertGreaterEqual(evaluation.median_score, 0.0)
        self.assertLessEqual(evaluation.average_score, 25.0)
        self.assertGreaterEqual(evaluation.min_score, 0)
        self.assertLessEqual(evaluation.max_score, 25)
        self.assertGreaterEqual(evaluation.average_turn_count, 1.0)
        self.assertGreaterEqual(evaluation.average_hint_tokens, 0.0)
        self.assertGreaterEqual(evaluation.average_strike_tokens, 0.0)
        self.assertGreaterEqual(evaluation.average_completed_stacks, 0.0)
        self.assertGreaterEqual(evaluation.win_rate, 0.0)
        self.assertLessEqual(evaluation.win_rate, 1.0)
        self.assertGreaterEqual(evaluation.loss_rate, 0.0)
        self.assertLessEqual(evaluation.loss_rate, 1.0)
        self.assertGreaterEqual(evaluation.score_at_least_10_rate, 0.0)
        self.assertLessEqual(evaluation.score_at_least_10_rate, 1.0)
        self.assertGreaterEqual(evaluation.score_at_least_15_rate, 0.0)
        self.assertLessEqual(evaluation.score_at_least_15_rate, 1.0)
        self.assertEqual(sum(count for _, count in evaluation.score_distribution), 5)

    def test_evaluate_self_play_shows_heuristic_beating_random_baseline(self) -> None:
        # Verifies that the heuristic baseline outperforms the random baseline
        # on average score under a fixed deterministic evaluation setup.
        heuristic_evaluation = evaluate_self_play(
            lambda player_id, game_index: ConservativeHeuristicAgent(),
            player_count=2,
            game_count=30,
            seed_base=0,
        )
        random_evaluation = evaluate_self_play(
            lambda player_id, game_index: RandomAgent(seed=1000 + (game_index * 2) + player_id),
            player_count=2,
            game_count=30,
            seed_base=0,
        )

        self.assertGreater(
            heuristic_evaluation.average_score,
            random_evaluation.average_score,
        )


if __name__ == "__main__":
    unittest.main()
