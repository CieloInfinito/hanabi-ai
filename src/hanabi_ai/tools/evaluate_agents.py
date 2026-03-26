from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from hanabi_ai.agents.heuristic.basic import BasicHeuristicAgent
from hanabi_ai.agents.heuristic.convention import ConventionHeuristicAgent
from hanabi_ai.agents.heuristic.convention_tempo import ConventionTempoHeuristicAgent
from hanabi_ai.agents.heuristic.tempo import TempoHeuristicAgent
from hanabi_ai.agents.random import RandomAgent
from hanabi_ai.training.self_play import SelfPlayEvaluation, evaluate_self_play

AgentFactory = Callable[[int, int, int], object]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Hanabi agents across many self-play games."
    )
    parser.add_argument(
        "--players",
        type=int,
        nargs="+",
        default=[2],
        help="One or more player counts. Each value must be between 2 and 5.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Number of games to evaluate for each agent setup.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base seed used for the Hanabi engine.",
    )
    parser.add_argument(
        "--agent-seed-base",
        type=int,
        default=1000,
        help="Base seed used when creating random agents.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path where a JSON benchmark report will be written.",
    )
    parser.add_argument(
        "--compare-json",
        type=Path,
        default=None,
        help="Optional path to a previous JSON benchmark report used for delta comparison.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)

    report = build_benchmark_report(
        player_counts=args.players,
        game_count=args.games,
        seed_base=args.seed_base,
        agent_seed_base=args.agent_seed_base,
    )

    print(format_benchmark_report(report))
    if args.compare_json is not None:
        previous_report = load_json_report(args.compare_json)
        print()
        print(format_report_delta(report, previous_report, args.compare_json))
    if args.json_output is not None:
        write_json_report(args.json_output, report)


def validate_args(args: argparse.Namespace) -> None:
    invalid_player_counts = [player_count for player_count in args.players if not 2 <= player_count <= 5]
    if invalid_player_counts:
        raise ValueError(
            "--players values must be between 2 and 5. "
            f"Received {invalid_player_counts}."
        )
    if args.games <= 0:
        raise ValueError(f"--games must be positive. Received {args.games}.")


def build_benchmark_report(
    *,
    player_counts: Sequence[int],
    game_count: int,
    seed_base: int = 0,
    agent_seed_base: int = 1000,
) -> dict[str, Any]:
    unique_player_counts = tuple(dict.fromkeys(player_counts))
    agent_factories: dict[str, AgentFactory] = {
        "BasicHeuristicAgent": lambda player_id, game_index, player_count: BasicHeuristicAgent(),
        "ConventionHeuristicAgent": (
            lambda player_id, game_index, player_count: ConventionHeuristicAgent()
        ),
        "ConventionTempoHeuristicAgent": (
            lambda player_id, game_index, player_count: ConventionTempoHeuristicAgent()
        ),
        "TempoHeuristicAgent": lambda player_id, game_index, player_count: TempoHeuristicAgent(),
        "RandomAgent": (
            lambda player_id, game_index, player_count: RandomAgent(
                seed=agent_seed_base + (game_index * player_count) + player_id
            )
        ),
    }

    result_sets: list[dict[str, Any]] = []
    aggregate_scores: dict[str, list[float]] = {name: [] for name in agent_factories}

    for player_count in unique_player_counts:
        evaluations = {
            name: evaluate_self_play(
                lambda player_id, game_index, factory=factory, current_player_count=player_count: factory(
                    player_id,
                    game_index,
                    current_player_count,
                ),
                player_count=player_count,
                game_count=game_count,
                seed_base=seed_base,
            )
            for name, factory in agent_factories.items()
        }
        for name, evaluation in evaluations.items():
            aggregate_scores[name].append(evaluation.average_score)

        result_sets.append(
            {
                "player_count": player_count,
                "evaluations": {
                    name: serialize_evaluation(evaluation)
                    for name, evaluation in evaluations.items()
                },
                "comparisons": {
                    "Basic vs Random": build_comparison_dict(
                        evaluations["BasicHeuristicAgent"],
                        evaluations["RandomAgent"],
                    ),
                    "Convention vs Random": build_comparison_dict(
                        evaluations["ConventionHeuristicAgent"],
                        evaluations["RandomAgent"],
                    ),
                    "Convention vs Basic": build_comparison_dict(
                        evaluations["ConventionHeuristicAgent"],
                        evaluations["BasicHeuristicAgent"],
                    ),
                    "ConventionTempo vs Convention": build_comparison_dict(
                        evaluations["ConventionTempoHeuristicAgent"],
                        evaluations["ConventionHeuristicAgent"],
                    ),
                    "ConventionTempo vs Tempo": build_comparison_dict(
                        evaluations["ConventionTempoHeuristicAgent"],
                        evaluations["TempoHeuristicAgent"],
                    ),
                    "ConventionTempo vs Basic": build_comparison_dict(
                        evaluations["ConventionTempoHeuristicAgent"],
                        evaluations["BasicHeuristicAgent"],
                    ),
                    "ConventionTempo vs Random": build_comparison_dict(
                        evaluations["ConventionTempoHeuristicAgent"],
                        evaluations["RandomAgent"],
                    ),
                    "Tempo vs Basic": build_comparison_dict(
                        evaluations["TempoHeuristicAgent"],
                        evaluations["BasicHeuristicAgent"],
                    ),
                    "Tempo vs Random": build_comparison_dict(
                        evaluations["TempoHeuristicAgent"],
                        evaluations["RandomAgent"],
                    ),
                },
                "ranking": build_agent_ranking(evaluations),
            }
        )

    aggregate_summary = {
        name: sum(scores) / len(scores) if scores else 0.0
        for name, scores in aggregate_scores.items()
    }

    return {
        "config": {
            "player_counts": list(unique_player_counts),
            "game_count": game_count,
            "seed_base": seed_base,
            "agent_seed_base": agent_seed_base,
        },
        "result_sets": result_sets,
        "aggregate_average_score_by_agent": aggregate_summary,
        "aggregate_ranking": [
            {"agent": name, "average_score": score}
            for name, score in sorted(
                aggregate_summary.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
    }


def serialize_evaluation(evaluation: SelfPlayEvaluation) -> dict[str, Any]:
    serialized = asdict(evaluation)
    serialized["score_distribution"] = [list(item) for item in evaluation.score_distribution]
    return serialized


def build_agent_ranking(
    evaluations: dict[str, SelfPlayEvaluation]
) -> list[dict[str, float | str]]:
    return [
        {
            "agent": name,
            "average_score": evaluation.average_score,
            "score_at_least_15_rate": evaluation.score_at_least_15_rate,
            "loss_rate": evaluation.loss_rate,
        }
        for name, evaluation in sorted(
            evaluations.items(),
            key=lambda item: (
                -item[1].average_score,
                -item[1].score_at_least_15_rate,
                item[1].loss_rate,
                item[0],
            ),
        )
    ]


def build_comparison_dict(
    left_evaluation: SelfPlayEvaluation,
    right_evaluation: SelfPlayEvaluation,
) -> dict[str, float]:
    return {
        "average_score_delta": (
            left_evaluation.average_score - right_evaluation.average_score
        ),
        "average_turn_count_delta": (
            left_evaluation.average_turn_count - right_evaluation.average_turn_count
        ),
        "average_completed_stacks_delta": (
            left_evaluation.average_completed_stacks
            - right_evaluation.average_completed_stacks
        ),
        "win_rate_delta": left_evaluation.win_rate - right_evaluation.win_rate,
        "loss_rate_delta": left_evaluation.loss_rate - right_evaluation.loss_rate,
        "score_at_least_10_rate_delta": (
            left_evaluation.score_at_least_10_rate
            - right_evaluation.score_at_least_10_rate
        ),
        "score_at_least_15_rate_delta": (
            left_evaluation.score_at_least_15_rate
            - right_evaluation.score_at_least_15_rate
        ),
    }


def format_benchmark_report(report: dict[str, Any]) -> str:
    config = report["config"]
    blocks = [
        (
            "Benchmark configuration\n"
            f"  player_counts: {', '.join(str(value) for value in config['player_counts'])}\n"
            f"  games_per_agent: {config['game_count']}\n"
            f"  seed_base: {config['seed_base']}\n"
            f"  agent_seed_base: {config['agent_seed_base']}"
        )
    ]

    for result_set in report["result_sets"]:
        player_count = result_set["player_count"]
        blocks.append(f"=== {player_count}-Player Table ===")

        for agent_name, evaluation in result_set["evaluations"].items():
            blocks.append(_format_evaluation(agent_name, evaluation))

        ranking_lines = ["Ranking"]
        ranking_lines.extend(
            (
                f"  {index}. {entry['agent']} | average_score={entry['average_score']:.3f} "
                f"| score_at_least_15_rate={entry['score_at_least_15_rate']:.3%} "
                f"| loss_rate={entry['loss_rate']:.3%}"
            )
            for index, entry in enumerate(result_set["ranking"], start=1)
        )
        blocks.append("\n".join(ranking_lines))

        comparison_lines = ["Comparisons"]
        comparison_lines.extend(
            _format_comparison(label, comparison)
            for label, comparison in result_set["comparisons"].items()
        )
        blocks.append("\n".join(comparison_lines))

    aggregate_lines = ["=== Aggregate Ranking Across Player Counts ==="]
    aggregate_lines.extend(
        f"  {index}. {entry['agent']} | average_score={entry['average_score']:.3f}"
        for index, entry in enumerate(report["aggregate_ranking"], start=1)
    )
    blocks.append("\n".join(aggregate_lines))

    return "\n\n".join(blocks)


def load_json_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report_delta(
    current_report: dict[str, Any],
    previous_report: dict[str, Any],
) -> dict[str, Any]:
    current_sets = {
        result_set["player_count"]: result_set
        for result_set in current_report["result_sets"]
    }
    previous_sets = {
        result_set["player_count"]: result_set
        for result_set in previous_report["result_sets"]
    }
    shared_player_counts = sorted(set(current_sets) & set(previous_sets))

    per_table_deltas = []
    for player_count in shared_player_counts:
        current_set = current_sets[player_count]
        previous_set = previous_sets[player_count]
        shared_agents = sorted(
            set(current_set["evaluations"]) & set(previous_set["evaluations"])
        )
        agent_deltas = {
            agent_name: build_evaluation_delta(
                current_set["evaluations"][agent_name],
                previous_set["evaluations"][agent_name],
            )
            for agent_name in shared_agents
        }
        per_table_deltas.append(
            {
                "player_count": player_count,
                "agent_deltas": agent_deltas,
                "ranking_by_average_score_delta": [
                    {
                        "agent": agent_name,
                        "average_score_delta": delta["average_score_delta"],
                        "score_at_least_15_rate_delta": delta[
                            "score_at_least_15_rate_delta"
                        ],
                        "loss_rate_delta": delta["loss_rate_delta"],
                    }
                    for agent_name, delta in sorted(
                        agent_deltas.items(),
                        key=lambda item: (
                            -item[1]["average_score_delta"],
                            -item[1]["score_at_least_15_rate_delta"],
                            item[1]["loss_rate_delta"],
                            item[0],
                        ),
                    )
                ],
            }
        )

    current_aggregate = current_report.get("aggregate_average_score_by_agent", {})
    previous_aggregate = previous_report.get("aggregate_average_score_by_agent", {})
    shared_aggregate_agents = sorted(set(current_aggregate) & set(previous_aggregate))

    return {
        "shared_player_counts": shared_player_counts,
        "per_table_deltas": per_table_deltas,
        "aggregate_average_score_delta_by_agent": {
            agent_name: current_aggregate[agent_name] - previous_aggregate[agent_name]
            for agent_name in shared_aggregate_agents
        },
    }


def build_evaluation_delta(
    current_evaluation: dict[str, Any],
    previous_evaluation: dict[str, Any],
) -> dict[str, float]:
    return {
        "average_score_delta": (
            current_evaluation["average_score"] - previous_evaluation["average_score"]
        ),
        "median_score_delta": (
            current_evaluation["median_score"] - previous_evaluation["median_score"]
        ),
        "average_turn_count_delta": (
            current_evaluation["average_turn_count"]
            - previous_evaluation["average_turn_count"]
        ),
        "average_completed_stacks_delta": (
            current_evaluation["average_completed_stacks"]
            - previous_evaluation["average_completed_stacks"]
        ),
        "successful_play_rate_delta": (
            current_evaluation["successful_play_rate"]
            - previous_evaluation["successful_play_rate"]
        ),
        "loss_rate_delta": (
            current_evaluation["loss_rate"] - previous_evaluation["loss_rate"]
        ),
        "score_at_least_10_rate_delta": (
            current_evaluation["score_at_least_10_rate"]
            - previous_evaluation["score_at_least_10_rate"]
        ),
        "score_at_least_15_rate_delta": (
            current_evaluation["score_at_least_15_rate"]
            - previous_evaluation["score_at_least_15_rate"]
        ),
    }


def format_report_delta(
    current_report: dict[str, Any],
    previous_report: dict[str, Any],
    previous_path: Path,
) -> str:
    delta_report = build_report_delta(current_report, previous_report)
    blocks = [f"=== Delta vs {previous_path} ==="]
    if not delta_report["shared_player_counts"]:
        blocks.append("No shared player counts between the current report and the previous report.")
        return "\n".join(blocks)

    blocks.append(
        "Shared player counts: "
        + ", ".join(str(value) for value in delta_report["shared_player_counts"])
    )

    for table_delta in delta_report["per_table_deltas"]:
        lines = [f"{table_delta['player_count']}-Player Delta"]
        for entry in table_delta["ranking_by_average_score_delta"]:
            lines.append(
                (
                    f"  {entry['agent']} | average_score_delta={entry['average_score_delta']:+.3f} "
                    f"| score_at_least_15_rate_delta={entry['score_at_least_15_rate_delta']:+.3%} "
                    f"| loss_rate_delta={entry['loss_rate_delta']:+.3%}"
                )
            )
        blocks.append("\n".join(lines))

    aggregate_lines = ["Aggregate average score delta by agent"]
    aggregate_lines.extend(
        f"  {agent_name}: {delta:+.3f}"
        for agent_name, delta in sorted(
            delta_report["aggregate_average_score_delta_by_agent"].items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    blocks.append("\n".join(aggregate_lines))

    return "\n\n".join(blocks)


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _format_evaluation(name: str, evaluation: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"{name} over {evaluation['game_count']} games ({evaluation['player_count']} players)",
            f"  average_score: {evaluation['average_score']:.3f}",
            f"  median_score: {evaluation['median_score']:.3f}",
            f"  min_score: {evaluation['min_score']}",
            f"  max_score: {evaluation['max_score']}",
            f"  average_turn_count: {evaluation['average_turn_count']:.3f}",
            f"  average_hint_tokens: {evaluation['average_hint_tokens']:.3f}",
            f"  average_strike_tokens: {evaluation['average_strike_tokens']:.3f}",
            f"  average_completed_stacks: {evaluation['average_completed_stacks']:.3f}",
            f"  average_play_actions: {evaluation['average_play_actions']:.3f}",
            f"  average_successful_plays: {evaluation['average_successful_plays']:.3f}",
            f"  average_failed_plays: {evaluation['average_failed_plays']:.3f}",
            f"  average_discards: {evaluation['average_discards']:.3f}",
            f"  average_hints_given: {evaluation['average_hints_given']:.3f}",
            f"  successful_play_rate: {evaluation['successful_play_rate']:.3%}",
            f"  win_rate: {evaluation['win_rate']:.3%}",
            f"  loss_rate: {evaluation['loss_rate']:.3%}",
            f"  score_at_least_10_rate: {evaluation['score_at_least_10_rate']:.3%}",
            f"  score_at_least_15_rate: {evaluation['score_at_least_15_rate']:.3%}",
            "  score_distribution: "
            f"{_format_distribution(evaluation['score_distribution'])}",
        ]
    )


def _format_comparison(label: str, comparison: dict[str, float]) -> str:
    return "\n".join(
        [
            label,
            f"  average_score_delta: {comparison['average_score_delta']:.3f}",
            f"  average_turn_count_delta: {comparison['average_turn_count_delta']:.3f}",
            (
                "  average_completed_stacks_delta: "
                f"{comparison['average_completed_stacks_delta']:.3f}"
            ),
            f"  win_rate_delta: {comparison['win_rate_delta']:.3%}",
            f"  loss_rate_delta: {comparison['loss_rate_delta']:.3%}",
            (
                "  score_at_least_10_rate_delta: "
                f"{comparison['score_at_least_10_rate_delta']:.3%}"
            ),
            (
                "  score_at_least_15_rate_delta: "
                f"{comparison['score_at_least_15_rate_delta']:.3%}"
            ),
        ]
    )


def _format_distribution(distribution: Sequence[Sequence[int]]) -> str:
    return ", ".join(f"{score}:{count}" for score, count in distribution)


if __name__ == "__main__":
    main()
