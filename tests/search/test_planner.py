from __future__ import annotations

import unittest

from hanabi_ai.agents.search_agent import SearchHeuristicAgent
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.search.planner import ShortHorizonPlanner


class PlannerTests(unittest.TestCase):
    def test_planner_returns_only_legal_actions(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=80)
        observation = engine.get_observation(0)
        planner = ShortHorizonPlanner(world_samples=4, depth=1, top_k=3)

        ranked_actions = planner.rank_actions(observation, seed=3)

        self.assertTrue(ranked_actions)
        self.assertTrue(all(entry.action in observation.legal_actions for entry in ranked_actions))

    def test_search_agent_returns_legal_decision(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=81)
        observation = engine.get_observation(0)
        agent = SearchHeuristicAgent(world_samples=4, depth=1, top_k=3)

        decision = agent.act(observation)
        action = getattr(decision, "action", decision)

        self.assertIn(action, observation.legal_actions)


if __name__ == "__main__":
    unittest.main()
