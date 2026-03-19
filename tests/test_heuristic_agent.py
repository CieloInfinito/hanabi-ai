from __future__ import annotations

import unittest

import _path_setup
from card_game_ai.agents.heuristic_agent import HeuristicAgent
from card_game_ai.game.actions import HintColorAction, PlayAction
from card_game_ai.game.cards import Card, Color, Rank
from card_game_ai.game.engine import HanabiGameEngine


class HeuristicAgentTests(unittest.TestCase):
    def test_heuristic_agent_plays_definitely_playable_card(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=31)
        engine.fireworks[Color.RED] = 1
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED}),
            possible_ranks=frozenset({Rank.TWO}),
            hinted_color=Color.RED,
            hinted_rank=Rank.TWO,
        )
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(action, PlayAction(card_index=0))

    def test_heuristic_agent_gives_hint_for_other_players_playable_card(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=32)
        engine.hands[1][0] = Card(Color.BLUE, Rank.ONE)
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(action, HintColorAction(target_player=1, color=Color.BLUE))

    def test_heuristic_agent_returns_legal_action(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=33)
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertIn(action, observation.legal_actions)


if __name__ == "__main__":
    unittest.main()
