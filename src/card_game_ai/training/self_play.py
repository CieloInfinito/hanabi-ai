from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from card_game_ai.game.actions import Action
from card_game_ai.game.engine import HanabiGameEngine
from card_game_ai.game.observation import PlayerObservation


class Agent(Protocol):
    def act(self, observation: PlayerObservation) -> Action:
        """Return one legal action for the current player."""


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    player_count: int
    final_score: int
    turn_count: int
    hint_tokens: int
    strike_tokens: int
    deck_size: int
    game_won: bool
    game_lost: bool


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
        action = agents[player_id].act(observation)
        engine.step(action)

    return SelfPlayResult(
        player_count=engine.player_count,
        final_score=engine.get_score(),
        turn_count=engine.turn_number,
        hint_tokens=engine.hint_tokens,
        strike_tokens=engine.strike_tokens,
        deck_size=len(engine.deck),
        game_won=engine.get_score() == 25,
        game_lost=engine.strike_tokens >= 3,
    )
