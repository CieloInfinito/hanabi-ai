from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from card_game_ai.agents.heuristic.conservative_agent import (
    ConservativeHeuristicAgent,
)
from card_game_ai.agents.random_agent import RandomAgent
from card_game_ai.training.self_play import run_self_play_game_with_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Hanabi self-play game and print a full text trace."
    )
    parser.add_argument(
        "--agent",
        choices=("random", "conservative", "heuristic"),
        default="random",
        help="Agent type to use for all players.",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        help="Number of players. Must be between 2 and 5.",
    )
    parser.add_argument(
        "--game-seed",
        type=int,
        default=3,
        help="Seed used by the Hanabi engine.",
    )
    parser.add_argument(
        "--agent-seed-base",
        type=int,
        default=1,
        help="Base seed used to initialize the random agents.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 2 <= args.players <= 5:
        raise ValueError(f"--players must be between 2 and 5. Received {args.players}.")

    agents = [_build_agent(args.agent, args.agent_seed_base + player_id) for player_id in range(args.players)]
    traced_game = run_self_play_game_with_trace(agents, seed=args.game_seed)

    print(traced_game.trace)


def _build_agent(agent_name: str, seed: int):
    if agent_name == "random":
        return RandomAgent(seed=seed)
    if agent_name in ("conservative", "heuristic"):
        return ConservativeHeuristicAgent()
    raise ValueError(f"Unsupported agent type: {agent_name}.")


if __name__ == "__main__":
    main()
