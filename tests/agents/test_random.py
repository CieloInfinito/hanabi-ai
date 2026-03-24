from __future__ import annotations

import unittest

from hanabi_ai.agents.random import RandomAgent
from hanabi_ai.game.engine import HanabiGameEngine


class RandomAgentTests(unittest.TestCase):
    def test_random_agent_returns_legal_action(self) -> None:
        # Verifies that the random baseline never returns an illegal move.
        engine = HanabiGameEngine(player_count=2, seed=7)
        observation = engine.get_observation(0)
        agent = RandomAgent(seed=11)

        action = agent.act(observation)

        self.assertIn(action, observation.legal_actions)


if __name__ == "__main__":
    unittest.main()
