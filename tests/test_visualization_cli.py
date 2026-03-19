from __future__ import annotations

import unittest

import _path_setup
from card_game_ai.game.actions import PlayAction
from card_game_ai.game.cards import Card, Color, Rank
from card_game_ai.game.engine import HanabiGameEngine
from card_game_ai.visualization.cli import (
    render_game_state,
    render_player_observation,
    render_step_result,
)


class VisualizationCliTests(unittest.TestCase):
    def test_render_game_state_includes_public_summary_and_real_hands(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=21)
        engine.hands[0][0] = Card(Color.RED, Rank.ONE)

        rendered = render_game_state(engine)

        self.assertIn("=== Hanabi Game State ===", rendered)
        self.assertIn("Current player: 0", rendered)
        self.assertIn("Fireworks:", rendered)
        self.assertIn("Player 0:", rendered)
        self.assertIn("R1", rendered)

    def test_render_player_observation_hides_own_real_cards(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=22)
        engine.hands[0][0] = Card(Color.RED, Rank.ONE)
        observation = engine.get_observation(0)

        rendered = render_player_observation(observation)

        self.assertIn("=== Hanabi Player Observation ===", rendered)
        self.assertIn("Own hand knowledge:", rendered)
        self.assertIn("Visible other hands:", rendered)
        self.assertIn("Definitely playable own indices:", rendered)
        self.assertIn("colors:RYGBW", rendered)
        self.assertNotIn("Player 0: R1", rendered)

    def test_render_player_observation_lists_legal_actions(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=23)
        observation = engine.get_observation(0)

        rendered = render_player_observation(observation)

        self.assertIn("Legal actions:", rendered)
        self.assertIn(str(PlayAction(card_index=0)), rendered)

    def test_render_step_result_includes_action_and_score(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=24)
        result = engine.step(PlayAction(card_index=0))

        rendered = render_step_result(result)

        self.assertIn("Turn result:", rendered)
        self.assertIn("Action:", rendered)
        self.assertIn("Score:", rendered)


if __name__ == "__main__":
    unittest.main()
