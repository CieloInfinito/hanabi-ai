from __future__ import annotations

import unittest

from hanabi_ai.agents.heuristic.basic import BasicHeuristicAgent
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


if __name__ == "__main__":
    unittest.main()
