from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .cards import Color, Rank


@dataclass(frozen=True, slots=True)
class PlayAction:
    card_index: int

    def __post_init__(self) -> None:
        if self.card_index < 0:
            raise ValueError(
                f"card_index must be >= 0 for PlayAction. Received {self.card_index}."
            )

    def __str__(self) -> str:
        return f"Play(card_index={self.card_index})"


@dataclass(frozen=True, slots=True)
class DiscardAction:
    card_index: int

    def __post_init__(self) -> None:
        if self.card_index < 0:
            raise ValueError(
                "card_index must be >= 0 for DiscardAction. "
                f"Received {self.card_index}."
            )

    def __str__(self) -> str:
        return f"Discard(card_index={self.card_index})"


@dataclass(frozen=True, slots=True)
class HintColorAction:
    target_player: int
    color: Color

    def __post_init__(self) -> None:
        if self.target_player < 0:
            raise ValueError(
                "target_player must be >= 0 for HintColorAction. "
                f"Received {self.target_player}."
            )

    def __str__(self) -> str:
        return (
            f"HintColor(target_player={self.target_player}, color={self.color.value})"
        )


@dataclass(frozen=True, slots=True)
class HintRankAction:
    target_player: int
    rank: Rank

    def __post_init__(self) -> None:
        if self.target_player < 0:
            raise ValueError(
                "target_player must be >= 0 for HintRankAction. "
                f"Received {self.target_player}."
            )

    def __str__(self) -> str:
        return f"HintRank(target_player={self.target_player}, rank={int(self.rank)})"


Action: TypeAlias = PlayAction | DiscardAction | HintColorAction | HintRankAction


def is_hint_action(action: Action) -> bool:
    return isinstance(action, (HintColorAction, HintRankAction))


def is_play_action(action: Action) -> bool:
    return isinstance(action, PlayAction)


def is_discard_action(action: Action) -> bool:
    return isinstance(action, DiscardAction)
