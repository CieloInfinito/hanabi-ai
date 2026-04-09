from __future__ import annotations

import argparse
from dataclasses import dataclass

from hanabi_ai.agents.search_agent import SearchHeuristicAgent
from hanabi_ai.game.actions import DiscardAction
from hanabi_ai.game.engine import HanabiGameEngine


@dataclass(frozen=True, slots=True)
class SearchOpportunity:
    seed: int
    turn_number: int
    player_id: int
    baseline_action: str
    best_action: str
    baseline_value: float
    best_value: float

    @property
    def value_gap(self) -> float:
        return self.best_value - self.baseline_value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan baseline self-play seeds for discard turns where the search "
            "planner sees a competitive alternative."
        )
    )
    parser.add_argument("--players", type=int, default=2, help="Number of players.")
    parser.add_argument("--seed-start", type=int, default=0, help="First game seed.")
    parser.add_argument(
        "--seed-count",
        type=int,
        default=20,
        help="Number of consecutive seeds to scan.",
    )
    parser.add_argument(
        "--world-samples",
        type=int,
        default=8,
        help="Compatible-world samples used by the planner.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Short rollout depth used by the planner.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Maximum candidate actions considered by the planner.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of opportunities to print.",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.0,
        help="Minimum planner value gap over the baseline discard to report.",
    )
    return parser.parse_args(argv)


def find_search_opportunities(
    *,
    player_count: int,
    seed_start: int,
    seed_count: int,
    world_samples: int,
    depth: int,
    top_k: int,
    min_gap: float = 0.0,
) -> list[SearchOpportunity]:
    agent = SearchHeuristicAgent(
        world_samples=world_samples,
        depth=depth,
        top_k=top_k,
        min_improvement=0.0,
    )
    opportunities: list[SearchOpportunity] = []

    for seed in range(seed_start, seed_start + seed_count):
        engine = HanabiGameEngine(player_count=player_count, seed=seed)
        while not engine.is_terminal():
            player_id = engine.current_player
            observation = engine.get_observation(player_id)
            analysis = agent.analyze_decision(observation)
            if isinstance(analysis.baseline_action, DiscardAction) and analysis.ranked_actions:
                baseline_entry = next(
                    (
                        entry
                        for entry in analysis.ranked_actions
                        if entry.action == analysis.baseline_action
                    ),
                    None,
                )
                best_entry = analysis.ranked_actions[0]
                if (
                    baseline_entry is not None
                    and best_entry.action != analysis.baseline_action
                    and (best_entry.expected_value - baseline_entry.expected_value) >= min_gap
                ):
                    opportunities.append(
                        SearchOpportunity(
                            seed=seed,
                            turn_number=engine.turn_number,
                            player_id=player_id,
                            baseline_action=str(analysis.baseline_action),
                            best_action=str(best_entry.action),
                            baseline_value=baseline_entry.expected_value,
                            best_value=best_entry.expected_value,
                        )
                    )

            engine.step(analysis.baseline_decision)

    return sorted(
        opportunities,
        key=lambda item: (
            -item.value_gap,
            item.seed,
            item.turn_number,
            item.player_id,
        ),
    )


def format_search_opportunities(
    opportunities: list[SearchOpportunity],
    *,
    limit: int,
) -> str:
    lines = ["=== Search Opportunities ==="]
    if not opportunities:
        lines.append("No opportunities found under the current scan settings.")
        return "\n".join(lines)

    for index, item in enumerate(opportunities[:limit], start=1):
        lines.append(
            (
                f"{index}. seed={item.seed} turn={item.turn_number} player={item.player_id} "
                f"| baseline={item.baseline_action} ({item.baseline_value:.3f}) "
                f"| best={item.best_action} ({item.best_value:.3f}) "
                f"| gap={item.value_gap:+.3f}"
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    opportunities = find_search_opportunities(
        player_count=args.players,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        world_samples=args.world_samples,
        depth=args.depth,
        top_k=args.top_k,
        min_gap=args.min_gap,
    )
    print(format_search_opportunities(opportunities, limit=args.limit))


if __name__ == "__main__":
    main()
