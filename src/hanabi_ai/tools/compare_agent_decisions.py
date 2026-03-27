from __future__ import annotations

import argparse

from hanabi_ai.agents.heuristic.basic import BasicHeuristicAgent
from hanabi_ai.agents.heuristic.convention import ConventionHeuristicAgent
from hanabi_ai.agents.heuristic.convention_tempo import ConventionTempoHeuristicAgent
from hanabi_ai.agents.heuristic.large_table import LargeTableHeuristicAgent
from hanabi_ai.agents.heuristic.tempo import TempoHeuristicAgent
from hanabi_ai.game.actions import normalize_agent_decision
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.visualization.cli import render_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two heuristic agents turn by turn on the same game seed."
    )
    parser.add_argument("--players", type=int, default=5, help="Number of players.")
    parser.add_argument("--game-seed", type=int, default=0, help="Seed used by the Hanabi engine.")
    parser.add_argument(
        "--left-agent",
        default="convention-tempo",
        help="Left agent name: basic, convention, tempo, convention-tempo, large-table.",
    )
    parser.add_argument(
        "--right-agent",
        default="large-table",
        help="Right agent name: basic, convention, tempo, convention-tempo, large-table.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show every turn, not only turns where the chosen action differs.",
    )
    return parser.parse_args()


def _build_agent(name: str):
    normalized = name.strip().lower()
    mapping = {
        "basic": BasicHeuristicAgent,
        "convention": ConventionHeuristicAgent,
        "tempo": TempoHeuristicAgent,
        "convention-tempo": ConventionTempoHeuristicAgent,
        "large-table": LargeTableHeuristicAgent,
    }
    if normalized not in mapping:
        raise ValueError(f"Unknown agent '{name}'.")
    return mapping[normalized]()


def compare_agents(
    *,
    player_count: int,
    seed: int,
    left_agent_name: str,
    right_agent_name: str,
    show_all: bool = False,
) -> str:
    left_engine = HanabiGameEngine(player_count=player_count, seed=seed)
    right_engine = HanabiGameEngine(player_count=player_count, seed=seed)
    left_agents = [_build_agent(left_agent_name) for _ in range(player_count)]
    right_agents = [_build_agent(right_agent_name) for _ in range(player_count)]

    lines = [
        "=== Agent Decision Comparison ===",
        f"Players: {player_count} | Seed: {seed}",
        f"Left agent: {left_agent_name}",
        f"Right agent: {right_agent_name}",
    ]

    while not left_engine.is_terminal() and not right_engine.is_terminal():
        player_id = left_engine.current_player
        left_observation = left_engine.get_observation(player_id)
        right_observation = right_engine.get_observation(player_id)
        left_agent = left_agents[player_id]
        right_agent = right_agents[player_id]
        left_decision = normalize_agent_decision(left_agent.act(left_observation))
        right_decision = normalize_agent_decision(right_agent.act(right_observation))
        differs = left_decision.action != right_decision.action

        if show_all or differs:
            lines.append("")
            lines.append(
                f"Turn {left_engine.turn_number} | Player {player_id} | Differs: {differs}"
            )
            lines.append(f"  Left:  {render_action(left_decision.action)}")
            for note in left_agent.explain_action_choice(
                left_observation,
                left_decision.action,
            ):
                lines.append(f"    {note}")
            lines.append(f"  Right: {render_action(right_decision.action)}")
            for note in right_agent.explain_action_choice(
                right_observation,
                right_decision.action,
            ):
                lines.append(f"    {note}")

        left_engine.step(left_decision)
        right_engine.step(right_decision)

    lines.append("")
    lines.append(
        f"Final scores | Left: {left_engine.get_score()} | Right: {right_engine.get_score()}"
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    print(
        compare_agents(
            player_count=args.players,
            seed=args.game_seed,
            left_agent_name=args.left_agent,
            right_agent_name=args.right_agent,
            show_all=args.show_all,
        )
    )


if __name__ == "__main__":
    main()
