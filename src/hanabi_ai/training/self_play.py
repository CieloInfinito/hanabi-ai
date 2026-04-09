from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Protocol

from hanabi_ai.agents.heuristic.base import BaseHeuristicAgent
from hanabi_ai.game.actions import (
    Action,
    ActionLike,
    DiscardAction,
    HintColorAction,
    HintRankAction,
    PlayAction,
    normalize_agent_decision,
)
from hanabi_ai.game.cards import Color, Rank
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.game.observation import PlayerObservation
from hanabi_ai.visualization.cli import render_game_state, render_self_play_turn


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
    play_action_count: int
    successful_play_count: int
    failed_play_count: int
    discard_action_count: int
    hint_action_count: int
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
    average_play_actions: float
    average_successful_plays: float
    average_failed_plays: float
    average_discards: float
    average_hints_given: float
    successful_play_rate: float
    win_rate: float
    loss_rate: float
    score_at_least_10_rate: float
    score_at_least_15_rate: float
    score_at_least_20_rate: float
    score_at_least_24_rate: float
    average_gap_to_25: float
    perfect_game_rate: float
    score_distribution: tuple[tuple[int, int], ...]


@dataclass(slots=True)
class _SelfPlayCounters:
    play_action_count: int = 0
    successful_play_count: int = 0
    failed_play_count: int = 0
    discard_action_count: int = 0
    hint_action_count: int = 0


def _validate_player_count(player_count: int) -> None:
    if not 2 <= player_count <= 5:
        raise ValueError(
            f"Hanabi self-play requires 2 to 5 agents. Received {player_count}."
        )


def _update_action_counters(counters: _SelfPlayCounters, action: Action, *, play_succeeded: bool | None) -> None:
    if isinstance(action, PlayAction):
        counters.play_action_count += 1
        if play_succeeded:
            counters.successful_play_count += 1
        else:
            counters.failed_play_count += 1
        return

    if isinstance(action, DiscardAction):
        counters.discard_action_count += 1
        return

    if isinstance(action, (HintColorAction, HintRankAction)):
        counters.hint_action_count += 1


def _build_self_play_result(
    engine: HanabiGameEngine,
    counters: _SelfPlayCounters,
) -> SelfPlayResult:
    final_score = engine.get_score()
    return SelfPlayResult(
        player_count=engine.player_count,
        final_score=final_score,
        turn_count=engine.turn_number,
        hint_tokens=engine.hint_tokens,
        strike_tokens=engine.strike_tokens,
        deck_size=len(engine.deck),
        completed_stacks=sum(
            1
            for color in Color
            if engine.fireworks.get(color, 0) == int(Rank.FIVE)
        ),
        play_action_count=counters.play_action_count,
        successful_play_count=counters.successful_play_count,
        failed_play_count=counters.failed_play_count,
        discard_action_count=counters.discard_action_count,
        hint_action_count=counters.hint_action_count,
        game_won=final_score == 25,
        game_lost=engine.strike_tokens >= 3,
    )


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
    _validate_player_count(len(agents))

    engine = HanabiGameEngine(player_count=len(agents), seed=seed)
    counters = _SelfPlayCounters()

    while not engine.is_terminal():
        player_id = engine.current_player
        observation = engine.get_observation(player_id)
        decision = normalize_agent_decision(agents[player_id].act(observation))
        step_result = engine.step(decision)
        _update_action_counters(
            counters,
            step_result.action,
            play_succeeded=step_result.play_succeeded,
        )

    return _build_self_play_result(engine, counters)


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
    _validate_player_count(player_count)
    if game_count <= 0:
        raise ValueError(f"game_count must be positive. Received {game_count}.")

    scores: list[int] = []
    turn_counts: list[int] = []
    hint_tokens: list[int] = []
    strike_tokens: list[int] = []
    completed_stacks: list[int] = []
    play_action_counts: list[int] = []
    successful_play_counts: list[int] = []
    failed_play_counts: list[int] = []
    discard_action_counts: list[int] = []
    hint_action_counts: list[int] = []
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
        play_action_counts.append(result.play_action_count)
        successful_play_counts.append(result.successful_play_count)
        failed_play_counts.append(result.failed_play_count)
        discard_action_counts.append(result.discard_action_count)
        hint_action_counts.append(result.hint_action_count)
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
    total_play_actions = sum(play_action_counts)
    total_successful_plays = sum(successful_play_counts)

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
        average_play_actions=sum(play_action_counts) / game_count,
        average_successful_plays=sum(successful_play_counts) / game_count,
        average_failed_plays=sum(failed_play_counts) / game_count,
        average_discards=sum(discard_action_counts) / game_count,
        average_hints_given=sum(hint_action_counts) / game_count,
        successful_play_rate=(
            total_successful_plays / total_play_actions if total_play_actions else 0.0
        ),
        win_rate=wins / game_count,
        loss_rate=losses / game_count,
        score_at_least_10_rate=sum(score >= 10 for score in scores) / game_count,
        score_at_least_15_rate=sum(score >= 15 for score in scores) / game_count,
        score_at_least_20_rate=sum(score >= 20 for score in scores) / game_count,
        score_at_least_24_rate=sum(score >= 24 for score in scores) / game_count,
        average_gap_to_25=sum(25 - score for score in scores) / game_count,
        perfect_game_rate=sum(score == 25 for score in scores) / game_count,
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
    _validate_player_count(len(agents))

    engine = HanabiGameEngine(player_count=len(agents), seed=seed)
    trace_blocks = ["=== Self-Play Start ===", render_game_state(engine)]
    counters = _SelfPlayCounters()

    while not engine.is_terminal():
        player_id = engine.current_player
        observation = engine.get_observation(player_id)
        decision = normalize_agent_decision(agents[player_id].act(observation))
        step_result = engine.step(decision)
        _update_action_counters(
            counters,
            step_result.action,
            play_succeeded=step_result.play_succeeded,
        )
        trace_blocks.append(
            render_self_play_turn(
                turn_index=engine.turn_number,
                player_id=player_id,
                observation=observation,
                action=decision.action,
                step_result=step_result,
                engine=engine,
                acting_agent=(
                    agents[player_id]
                    if isinstance(agents[player_id], BaseHeuristicAgent)
                    else None
                ),
            )
        )

    summary = _build_self_play_result(engine, counters)
    trace_blocks.append("=== Self-Play End ===")
    trace_blocks.append(
        (
            f"Final score: {summary.final_score} | Turns: {summary.turn_count} | "
            f"Hint tokens: {summary.hint_tokens} | Strike tokens: {summary.strike_tokens}"
        )
    )

    return SelfPlayTraceResult(summary=summary, trace="\n\n".join(trace_blocks))
