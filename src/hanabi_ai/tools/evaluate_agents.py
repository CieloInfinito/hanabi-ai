from __future__ import annotations

import argparse

from hanabi_ai.agents.heuristic.basic import BasicHeuristicAgent
from hanabi_ai.agents.heuristic.convention import ConventionHeuristicAgent
from hanabi_ai.agents.heuristic.tempo import TempoHeuristicAgent
from hanabi_ai.agents.random import RandomAgent
from hanabi_ai.training.self_play import SelfPlayEvaluation, evaluate_self_play


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Hanabi agents across many self-play games."
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        help="Number of players. Must be between 2 and 5.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 2 <= args.players <= 5:
        raise ValueError(f"--players must be between 2 and 5. Received {args.players}.")
    if args.games <= 0:
        raise ValueError(f"--games must be positive. Received {args.games}.")

    basic_evaluation = evaluate_self_play(
        lambda player_id, game_index: BasicHeuristicAgent(),
        player_count=args.players,
        game_count=args.games,
        seed_base=args.seed_base,
    )
    convention_evaluation = evaluate_self_play(
        lambda player_id, game_index: ConventionHeuristicAgent(),
        player_count=args.players,
        game_count=args.games,
        seed_base=args.seed_base,
    )
    tempo_evaluation = evaluate_self_play(
        lambda player_id, game_index: TempoHeuristicAgent(),
        player_count=args.players,
        game_count=args.games,
        seed_base=args.seed_base,
    )
    random_evaluation = evaluate_self_play(
        lambda player_id, game_index: RandomAgent(
            seed=args.agent_seed_base + (game_index * args.players) + player_id
        ),
        player_count=args.players,
        game_count=args.games,
        seed_base=args.seed_base,
    )

    print(_format_evaluation("BasicHeuristicAgent", basic_evaluation))
    print()
    print(_format_evaluation("ConventionHeuristicAgent", convention_evaluation))
    print()
    print(_format_evaluation("TempoHeuristicAgent", tempo_evaluation))
    print()
    print(_format_evaluation("RandomAgent", random_evaluation))
    print()
    print(_format_comparison("Basic vs Random", basic_evaluation, random_evaluation))
    print()
    print(
        _format_comparison(
            "Convention vs Random",
            convention_evaluation,
            random_evaluation,
        )
    )
    print()
    print(
        _format_comparison(
            "Convention vs Basic",
            convention_evaluation,
            basic_evaluation,
        )
    )
    print()
    print(
        _format_comparison(
            "Tempo vs Basic",
            tempo_evaluation,
            basic_evaluation,
        )
    )


def _format_evaluation(name: str, evaluation: SelfPlayEvaluation) -> str:
    return "\n".join(
        [
            f"{name} over {evaluation.game_count} games ({evaluation.player_count} players)",
            f"  average_score: {evaluation.average_score:.3f}",
            f"  median_score: {evaluation.median_score:.3f}",
            f"  min_score: {evaluation.min_score}",
            f"  max_score: {evaluation.max_score}",
            f"  average_turn_count: {evaluation.average_turn_count:.3f}",
            f"  average_hint_tokens: {evaluation.average_hint_tokens:.3f}",
            f"  average_strike_tokens: {evaluation.average_strike_tokens:.3f}",
            f"  average_completed_stacks: {evaluation.average_completed_stacks:.3f}",
            f"  average_play_actions: {evaluation.average_play_actions:.3f}",
            f"  average_successful_plays: {evaluation.average_successful_plays:.3f}",
            f"  average_failed_plays: {evaluation.average_failed_plays:.3f}",
            f"  average_discards: {evaluation.average_discards:.3f}",
            f"  average_hints_given: {evaluation.average_hints_given:.3f}",
            f"  successful_play_rate: {evaluation.successful_play_rate:.3%}",
            f"  win_rate: {evaluation.win_rate:.3%}",
            f"  loss_rate: {evaluation.loss_rate:.3%}",
            f"  score_at_least_10_rate: {evaluation.score_at_least_10_rate:.3%}",
            f"  score_at_least_15_rate: {evaluation.score_at_least_15_rate:.3%}",
            f"  score_distribution: {_format_distribution(evaluation.score_distribution)}",
        ]
    )


def _format_comparison(
    label: str,
    left_evaluation: SelfPlayEvaluation,
    right_evaluation: SelfPlayEvaluation,
) -> str:
    return "\n".join(
        [
            label,
            (
                "  average_score_delta: "
                f"{left_evaluation.average_score - right_evaluation.average_score:.3f}"
            ),
            (
                "  average_turn_count_delta: "
                f"{left_evaluation.average_turn_count - right_evaluation.average_turn_count:.3f}"
            ),
            (
                "  average_completed_stacks_delta: "
                f"{left_evaluation.average_completed_stacks - right_evaluation.average_completed_stacks:.3f}"
            ),
            (
                "  win_rate_delta: "
                f"{left_evaluation.win_rate - right_evaluation.win_rate:.3%}"
            ),
            (
                "  loss_rate_delta: "
                f"{left_evaluation.loss_rate - right_evaluation.loss_rate:.3%}"
            ),
            (
                "  score_at_least_10_rate_delta: "
                f"{left_evaluation.score_at_least_10_rate - right_evaluation.score_at_least_10_rate:.3%}"
            ),
            (
                "  score_at_least_15_rate_delta: "
                f"{left_evaluation.score_at_least_15_rate - right_evaluation.score_at_least_15_rate:.3%}"
            ),
        ]
    )


def _format_distribution(distribution: tuple[tuple[int, int], ...]) -> str:
    return ", ".join(f"{score}:{count}" for score, count in distribution)


if __name__ == "__main__":
    main()
