from __future__ import annotations

import unittest

from hanabi_ai.tools.compare_agent_decisions import compare_agents


class CompareAgentDecisionsToolTests(unittest.TestCase):
    def test_compare_agents_reports_output_shape(self) -> None:
        rendered = compare_agents(
            player_count=5,
            seed=0,
            left_agent_name="convention-tempo",
            right_agent_name="large-table",
            show_all=False,
        )

        self.assertIn("=== Agent Decision Comparison ===", rendered)
        self.assertIn("Final scores | Left:", rendered)


if __name__ == "__main__":
    unittest.main()
