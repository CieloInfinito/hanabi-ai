from __future__ import annotations

import argparse

from hanabi_ai.agents.heuristic.convention import ConventionHeuristicAgent
from hanabi_ai.training.self_play import run_self_play_game_with_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 2-player Hanabi trace with two convention heuristic agents."
    )
    parser.add_argument(
        "--game-seed",
        type=int,
        default=3,
        help="Seed used by the Hanabi engine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agents = [ConventionHeuristicAgent(), ConventionHeuristicAgent()]
    traced_game = run_self_play_game_with_trace(agents, seed=args.game_seed)
    print(traced_game.trace)


if __name__ == "__main__":
    main()
