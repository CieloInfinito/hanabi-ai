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


@dataclass(frozen=True, slots=True)
class HintPresentation:
    revealed_indices: tuple[int, ...] | None = None
    revealed_groups: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.revealed_indices is None and self.revealed_groups is None:
            raise ValueError(
                "HintPresentation requires revealed_indices, revealed_groups, or both."
            )

        flattened_groups = (
            tuple(index for group in self.revealed_groups for index in group)
            if self.revealed_groups is not None
            else None
        )

        if self.revealed_indices is None and flattened_groups is not None:
            object.__setattr__(self, "revealed_indices", flattened_groups)
        elif (
            self.revealed_indices is not None
            and flattened_groups is not None
            and self.revealed_indices != flattened_groups
        ):
            raise ValueError(
                "HintPresentation.revealed_indices must match the flattened revealed_groups."
            )


@dataclass(frozen=True, slots=True)
class AgentDecision:
    action: Action
    hint_presentation: HintPresentation | None = None


Action: TypeAlias = PlayAction | DiscardAction | HintColorAction | HintRankAction
ActionLike: TypeAlias = Action | AgentDecision


def normalize_agent_decision(action_like: ActionLike) -> AgentDecision:
    if isinstance(action_like, AgentDecision):
        return action_like
    return AgentDecision(action=action_like)


def is_hint_action(action: Action) -> bool:
    return isinstance(action, (HintColorAction, HintRankAction))


def is_play_action(action: Action) -> bool:
    return isinstance(action, PlayAction)


def is_discard_action(action: Action) -> bool:
    return isinstance(action, DiscardAction)
