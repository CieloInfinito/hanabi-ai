from __future__ import annotations

import unittest

from hanabi_ai.agents.heuristic.basic import BasicHeuristicAgent
from hanabi_ai.agents.heuristic.base import _HintPriorityWeights
from ._shared import SharedBasicHeuristicVariantTests, SharedHeuristicAgentTests


# Basic heuristic tests verify that this variant does not emit or interpret
# any private hint-ordering conventions shared with teammates.
class BasicHeuristicAgentTests(
    SharedBasicHeuristicVariantTests,
    SharedHeuristicAgentTests,
    unittest.TestCase,
):
    def make_agent(self):
        return BasicHeuristicAgent()

    def test_basic_agent_exposes_player_count_hint_priority_weights(self) -> None:
        agent = BasicHeuristicAgent()

        self.assertEqual(
            agent._base_hint_priority_weights(2),
            _HintPriorityWeights(
                actionable_hint=1,
                critical_playable=1,
            ),
        )
        self.assertEqual(
            agent._base_hint_priority_weights(3),
            _HintPriorityWeights(
                receiver_needs_help=1,
                immediate_receiver=1,
                near_term_receiver=1,
                actionable_hint=1,
                critical_playable=1,
            ),
        )
        self.assertEqual(
            agent._base_hint_priority_weights(4),
            _HintPriorityWeights(
                follow_on_value=1,
                receiver_needs_help=1,
                immediate_receiver=2,
                near_term_receiver=1,
                actionable_hint=1,
                critical_playable=1,
                turn_distance_penalty=1,
            ),
        )
        self.assertEqual(
            agent._base_hint_priority_weights(5),
            _HintPriorityWeights(
                follow_on_value=2,
                receiver_needs_help=1,
                immediate_receiver=1,
                near_term_receiver=2,
                actionable_hint=1,
                critical_playable=1,
                turn_distance_penalty=1,
            ),
        )


if __name__ == "__main__":
    unittest.main()
