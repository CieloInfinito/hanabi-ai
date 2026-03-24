from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Protocol

from card_game_ai.game.actions import Action, ActionLike, normalize_agent_decision
from card_game_ai.game.cards import Color, Rank
from card_game_ai.game.engine import HanabiGameEngine
from card_game_ai.game.observation import PlayerObservation
from card_game_ai.visualization.cli import render_game_state, render_self_play_turn


class Agent(Protocol):
    def act(self, observation: PlayerObservation) -> ActionLike:
        """Return one legal action or a richer action selection."""


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    player_count: int
    final_score: int
    turn_count: int
    hint_tokens: int
    strike_tokens: int
    deck_size: int
    completed_stacks: int
    game_won: bool
    game_lost: bool


@dataclass(frozen=True, slots=True)
class SelfPlayTraceResult:
    summary: SelfPlayResult
    trace: str


@dataclass(frozen=True, slots=True)
class SelfPlayEvaluation:
    game_count: int
    player_count: int
    average_score: float
    median_score: float
    min_score: int
    max_score: int
    average_turn_count: float
    average_hint_tokens: float
    average_strike_tokens: float
    average_completed_stacks: float
    win_rate: float
    loss_rate: float
    score_at_least_10_rate: float
    score_at_least_15_rate: float
    score_distribution: tuple[tuple[int, int], ...]


def run_self_play_game(
    agents: list[Agent],
    *,
    seed: int | None = None,
) -> SelfPlayResult:
    """
    Run one full Hanabi game using the provided agents.

    Args:
        agents:
            One agent per player seat. Each agent only receives its own partial
            observation on its turn.
        seed:
            Optional random seed forwarded to the engine.

    Returns:
        A compact summary of the completed game.
    """
    if not 2 <= len(agents) <= 5:
        raise ValueError(
            f"Hanabi self-play requires 2 to 5 agents. Received {len(agents)}."
        )

    engine = HanabiGameEngine(player_count=len(agents), seed=seed)

    while not engine.is_terminal():
        player_id = engine.current_player
        observation = engine.get_observation(player_id)
        decision = normalize_agent_decision(agents[player_id].act(observation))
        engine.step(decision)

    return SelfPlayResult(
        player_count=engine.player_count,
        final_score=engine.get_score(),
        turn_count=engine.turn_number,
        hint_tokens=engine.hint_tokens,
        strike_tokens=engine.strike_tokens,
        deck_size=len(engine.deck),
        completed_stacks=sum(
            1
            for color in Color
            if engine.fireworks.get(color, 0) == int(Rank.FIVE)
        ),
        game_won=engine.get_score() == 25,
        game_lost=engine.strike_tokens >= 3,
    )


def evaluate_self_play(
    agent_factory: Callable[[int, int], Agent],
    *,
    player_count: int,
    game_count: int,
    seed_base: int = 0,
) -> SelfPlayEvaluation:
    """
    Run multiple self-play games and aggregate outcome metrics.

    Args:
        agent_factory:
            Callable that builds one agent for a given seat and game index.
            Signature: ``agent_factory(player_id, game_index)``.
        player_count:
            Number of players in each game. Must be between 2 and 5.
        game_count:
            Number of games to evaluate. Must be positive.
        seed_base:
            Seed offset used to build deterministic engine seeds.

    Returns:
        Aggregate metrics across the completed games.
    """
    if not 2 <= player_count <= 5:
        raise ValueError(
            f"Hanabi self-play requires 2 to 5 agents. Received {player_count}."
        )
    if game_count <= 0:
        raise ValueError(f"game_count must be positive. Received {game_count}.")

    scores: list[int] = []
    turn_counts: list[int] = []
    hint_tokens: list[int] = []
    strike_tokens: list[int] = []
    completed_stacks: list[int] = []
    wins = 0
    losses = 0

    for game_index in range(game_count):
        agents = [
            agent_factory(player_id, game_index)
            for player_id in range(player_count)
        ]
        result = run_self_play_game(agents, seed=seed_base + game_index)
        scores.append(result.final_score)
        turn_counts.append(result.turn_count)
        hint_tokens.append(result.hint_tokens)
        strike_tokens.append(result.strike_tokens)
        completed_stacks.append(result.completed_stacks)
        wins += int(result.game_won)
        losses += int(result.game_lost)

    sorted_scores = sorted(scores)
    middle = game_count // 2
    median_score = (
        float(sorted_scores[middle])
        if game_count % 2 == 1
        else (sorted_scores[middle - 1] + sorted_scores[middle]) / 2
    )
    score_distribution = tuple(sorted(Counter(scores).items()))

    return SelfPlayEvaluation(
        game_count=game_count,
        player_count=player_count,
        average_score=sum(scores) / game_count,
        median_score=median_score,
        min_score=min(scores),
        max_score=max(scores),
        average_turn_count=sum(turn_counts) / game_count,
        average_hint_tokens=sum(hint_tokens) / game_count,
        average_strike_tokens=sum(strike_tokens) / game_count,
        average_completed_stacks=sum(completed_stacks) / game_count,
        win_rate=wins / game_count,
        loss_rate=losses / game_count,
        score_at_least_10_rate=sum(score >= 10 for score in scores) / game_count,
        score_at_least_15_rate=sum(score >= 15 for score in scores) / game_count,
        score_distribution=score_distribution,
    )


def run_self_play_game_with_trace(
    agents: list[Agent],
    *,
    seed: int | None = None,
) -> SelfPlayTraceResult:
    """
    Run one full Hanabi game and return a human-readable turn trace.
    """
    if not 2 <= len(agents) <= 5:
        raise ValueError(
            f"Hanabi self-play requires 2 to 5 agents. Received {len(agents)}."
        )

    engine = HanabiGameEngine(player_count=len(agents), seed=seed)
    trace_blocks = ["=== Self-Play Start ===", render_game_state(engine)]

    while not engine.is_terminal():
        player_id = engine.current_player
        observation = engine.get_observation(player_id)
        decision = normalize_agent_decision(agents[player_id].act(observation))
        step_result = engine.step(decision)
        trace_blocks.append(
            render_self_play_turn(
                turn_index=engine.turn_number,
                player_id=player_id,
                observation=observation,
                action=decision.action,
                step_result=step_result,
                engine=engine,
            )
        )

    summary = SelfPlayResult(
        player_count=engine.player_count,
        final_score=engine.get_score(),
        turn_count=engine.turn_number,
        hint_tokens=engine.hint_tokens,
        strike_tokens=engine.strike_tokens,
        deck_size=len(engine.deck),
        completed_stacks=sum(
            1
            for color in Color
            if engine.fireworks.get(color, 0) == int(Rank.FIVE)
        ),
        game_won=engine.get_score() == 25,
        game_lost=engine.strike_tokens >= 3,
    )
    trace_blocks.append("=== Self-Play End ===")
    trace_blocks.append(
        (
            f"Final score: {summary.final_score} | Turns: {summary.turn_count} | "
            f"Hint tokens: {summary.hint_tokens} | Strike tokens: {summary.strike_tokens}"
        )
    )

    return SelfPlayTraceResult(summary=summary, trace="\n\n".join(trace_blocks))
