from __future__ import annotations

import unittest

from hanabi_ai.tools.analyze_search_opportunities import (
    find_search_opportunities,
    format_search_opportunities,
)


class AnalyzeSearchOpportunitiesToolTests(unittest.TestCase):
    def test_format_search_opportunities_handles_empty_scan(self) -> None:
        rendered = format_search_opportunities([], limit=5)

        self.assertIn("=== Search Opportunities ===", rendered)
        self.assertIn("No opportunities found", rendered)

    def test_find_search_opportunities_returns_list(self) -> None:
        opportunities = find_search_opportunities(
            player_count=2,
            seed_start=0,
            seed_count=3,
            world_samples=4,
            depth=2,
            top_k=4,
            min_gap=0.0,
        )

        self.assertIsInstance(opportunities, list)


if __name__ == "__main__":
    unittest.main()
