from __future__ import annotations

import json
import unittest
from pathlib import Path

from hanabi_ai.tools.evaluate_agents import (
    build_benchmark_report,
    build_report_delta,
    format_benchmark_report,
    format_report_delta,
    load_json_report,
    parse_args,
    validate_args,
    write_json_report,
)


class EvaluateAgentsToolTests(unittest.TestCase):
    def test_parse_args_accepts_multiple_player_counts(self) -> None:
        args = parse_args(["--players", "2", "3", "5", "--games", "7"])

        self.assertEqual(args.players, [2, 3, 5])
        self.assertEqual(args.games, 7)

    def test_parse_args_accepts_compare_json(self) -> None:
        args = parse_args(["--compare-json", "reports\\previous.json"])

        self.assertEqual(args.compare_json, Path("reports\\previous.json"))

    def test_validate_args_rejects_out_of_range_player_counts(self) -> None:
        args = parse_args(["--players", "1", "2"])

        with self.assertRaises(ValueError):
            validate_args(args)

    def test_build_benchmark_report_returns_expected_sections(self) -> None:
        report = build_benchmark_report(
            player_counts=[2, 3],
            game_count=2,
            seed_base=11,
            agent_seed_base=101,
        )

        self.assertEqual(report["config"]["player_counts"], [2, 3])
        self.assertEqual(len(report["result_sets"]), 2)
        self.assertEqual(
            set(report["aggregate_average_score_by_agent"]),
            {
                "BasicHeuristicAgent",
                "ConventionHeuristicAgent",
                "ConventionTempoHeuristicAgent",
                "LargeTableHeuristicAgent",
                "TempoHeuristicAgent",
                "RandomAgent",
            },
        )
        for result_set in report["result_sets"]:
            self.assertIn("ranking", result_set)
            self.assertIn("comparisons", result_set)
            self.assertIn("Tempo vs Random", result_set["comparisons"])
            self.assertIn("ConventionTempo vs Tempo", result_set["comparisons"])
            self.assertIn("LargeTable vs ConventionTempo", result_set["comparisons"])
            self.assertEqual(
                set(result_set["evaluations"]),
                {
                    "BasicHeuristicAgent",
                    "ConventionHeuristicAgent",
                    "ConventionTempoHeuristicAgent",
                    "LargeTableHeuristicAgent",
                    "TempoHeuristicAgent",
                    "RandomAgent",
                },
            )

    def test_format_benchmark_report_includes_table_headers(self) -> None:
        report = build_benchmark_report(player_counts=[2], game_count=1)

        rendered = format_benchmark_report(report)

        self.assertIn("Benchmark configuration", rendered)
        self.assertIn("=== 2-Player Table ===", rendered)
        self.assertIn("Ranking", rendered)
        self.assertIn("Comparisons", rendered)
        self.assertIn("=== Aggregate Ranking Across Player Counts ===", rendered)

    def test_write_json_report_writes_serialized_report(self) -> None:
        report = build_benchmark_report(player_counts=[2], game_count=1)
        output_dir = Path("tests") / "_artifacts"
        output_path = output_dir / "benchmark.json"
        self.addCleanup(lambda: output_path.unlink(missing_ok=True))
        self.addCleanup(
            lambda: output_dir.rmdir() if output_dir.exists() and not any(output_dir.iterdir()) else None
        )
        write_json_report(output_path, report)
        loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["config"]["player_counts"], [2])
        self.assertEqual(len(loaded["result_sets"]), 1)

    def test_load_json_report_reads_serialized_report(self) -> None:
        report = build_benchmark_report(player_counts=[2], game_count=1)
        output_dir = Path("tests") / "_artifacts"
        output_path = output_dir / "benchmark_load.json"
        self.addCleanup(lambda: output_path.unlink(missing_ok=True))
        self.addCleanup(
            lambda: output_dir.rmdir() if output_dir.exists() and not any(output_dir.iterdir()) else None
        )
        write_json_report(output_path, report)

        loaded = load_json_report(output_path)

        self.assertEqual(loaded["config"]["player_counts"], [2])

    def test_build_report_delta_returns_shared_player_counts(self) -> None:
        previous_report = build_benchmark_report(
            player_counts=[2, 3],
            game_count=1,
            seed_base=0,
            agent_seed_base=100,
        )
        current_report = build_benchmark_report(
            player_counts=[3, 4],
            game_count=1,
            seed_base=1,
            agent_seed_base=101,
        )

        delta = build_report_delta(current_report, previous_report)

        self.assertEqual(delta["shared_player_counts"], [3])
        self.assertEqual(len(delta["per_table_deltas"]), 1)
        self.assertIn(
            "BasicHeuristicAgent",
            delta["per_table_deltas"][0]["agent_deltas"],
        )

    def test_format_report_delta_handles_no_shared_player_counts(self) -> None:
        previous_report = build_benchmark_report(player_counts=[2], game_count=1)
        current_report = build_benchmark_report(player_counts=[5], game_count=1)

        rendered = format_report_delta(
            current_report,
            previous_report,
            Path("reports\\previous.json"),
        )

        self.assertIn("No shared player counts", rendered)

    def test_format_report_delta_includes_delta_sections(self) -> None:
        previous_report = build_benchmark_report(
            player_counts=[2],
            game_count=1,
            seed_base=0,
            agent_seed_base=100,
        )
        current_report = build_benchmark_report(
            player_counts=[2],
            game_count=1,
            seed_base=1,
            agent_seed_base=101,
        )

        rendered = format_report_delta(
            current_report,
            previous_report,
            Path("reports\\previous.json"),
        )

        self.assertIn("=== Delta vs reports\\previous.json ===", rendered)
        self.assertIn("2-Player Delta", rendered)
        self.assertIn("Aggregate average score delta by agent", rendered)


if __name__ == "__main__":
    unittest.main()
