from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from random import Random
from typing import Final


class Color(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    WHITE = "white"


class Rank(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


@dataclass(frozen=True, slots=True)
class Card:
    color: Color
    rank: Rank

    def __str__(self) -> str:
        return f"{self.color.value}-{int(self.rank)}"

    def __repr__(self) -> str:
        return f"Card(color={self.color.value!r}, rank={int(self.rank)})"


HANABI_COLORS: Final[tuple[Color, ...]] = (
    Color.RED,
    Color.YELLOW,
    Color.GREEN,
    Color.BLUE,
    Color.WHITE,
)

HANABI_RANKS: Final[tuple[Rank, ...]] = (
    Rank.ONE,
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
)

# Standard Hanabi deck distribution by rank:
# 1 -> 3 copies
# 2 -> 2 copies
# 3 -> 2 copies
# 4 -> 2 copies
# 5 -> 1 copy
CARD_COUNTS_BY_RANK: Final[dict[Rank, int]] = {
    Rank.ONE: 3,
    Rank.TWO: 2,
    Rank.THREE: 2,
    Rank.FOUR: 2,
    Rank.FIVE: 1,
}

MAX_HINT_TOKENS: Final[int] = 8
MAX_STRIKE_TOKENS: Final[int] = 3


def build_standard_deck() -> list[Card]:
    """
    Build the standard 50-card Hanabi deck.

    Returns:
        A list with the full standard deck, unshuffled.
    """
    deck: list[Card] = []

    for color in HANABI_COLORS:
        for rank in HANABI_RANKS:
            count = CARD_COUNTS_BY_RANK[rank]
            for _ in range(count):
                deck.append(Card(color=color, rank=rank))

    return deck


def shuffled_standard_deck(rng: Random | None = None) -> list[Card]:
    """
    Build and shuffle the standard Hanabi deck.

    Args:
        rng:
            Optional random.Random instance for reproducibility.

    Returns:
        A shuffled list of cards.
    """
    deck = build_standard_deck()
    random_generator = rng if rng is not None else Random()
    random_generator.shuffle(deck)
    return deck


def hand_size_for_player_count(player_count: int) -> int:
    """
    Return the standard Hanabi hand size for the given number of players.

    Standard rules:
    - 2 or 3 players -> 5 cards each
    - 4 or 5 players -> 4 cards each
    """
    if player_count in (2, 3):
        return 5
    if player_count in (4, 5):
        return 4
    raise ValueError(
        f"Hanabi supports 2 to 5 players. Received player_count={player_count}."
    )
