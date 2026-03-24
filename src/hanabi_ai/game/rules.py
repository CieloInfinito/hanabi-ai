from __future__ import annotations

from typing import Mapping, TypeAlias

from .cards import (
    HANABI_COLORS,
    MAX_HINT_TOKENS,
    MAX_STRIKE_TOKENS,
    Card,
    Color,
    Rank,
)

Fireworks: TypeAlias = Mapping[Color, int]


def build_initial_fireworks() -> dict[Color, int]:
    """
    Build the initial empty fireworks state.

    Returns:
        A dictionary mapping each color to the highest successfully played rank.
        Empty stacks are represented by 0.
    """
    return {color: 0 for color in HANABI_COLORS}


def get_highest_played_rank(fireworks: Fireworks, color: Color) -> int:
    """
    Return the highest played rank for a color.

    Missing colors are treated as empty stacks.
    """
    return fireworks.get(color, 0)


def get_next_required_rank(fireworks: Fireworks, color: Color) -> int | None:
    """
    Return the next rank needed to advance a firework stack.

    Returns:
        The next required rank as an integer from 1 to 5, or None if the stack
        is already complete.
    """
    highest_rank = get_highest_played_rank(fireworks, color)
    if highest_rank >= int(Rank.FIVE):
        return None
    return highest_rank + 1


def is_card_playable(card: Card, fireworks: Fireworks) -> bool:
    """
    Return whether a card can be legally played onto the current fireworks.
    """
    next_required_rank = get_next_required_rank(fireworks, card.color)
    return next_required_rank == int(card.rank)


def is_card_already_played(card: Card, fireworks: Fireworks) -> bool:
    """
    Return whether a card's value is already covered by its firework stack.
    """
    return get_highest_played_rank(fireworks, card.color) >= int(card.rank)


def is_card_discardable(card: Card, fireworks: Fireworks) -> bool:
    """
    Return whether discarding a card is trivially safe from current fireworks only.

    This helper is intentionally conservative: it returns True only when the card
    has already been played and is therefore no longer needed to finish the game.
    """
    return is_card_already_played(card, fireworks)


def score_fireworks(fireworks: Fireworks) -> int:
    """
    Compute the current Hanabi score from all firework stacks.
    """
    return sum(get_highest_played_rank(fireworks, color) for color in HANABI_COLORS)


def game_is_won(fireworks: Fireworks) -> bool:
    """
    Return whether all five stacks have reached rank five.
    """
    return score_fireworks(fireworks) == len(HANABI_COLORS) * int(Rank.FIVE)


def game_is_lost(strike_tokens: int, max_strike_tokens: int = MAX_STRIKE_TOKENS) -> bool:
    """
    Return whether the game is lost because the strike limit was reached.
    """
    return strike_tokens >= max_strike_tokens


def can_give_hint(hint_tokens: int) -> bool:
    """
    Return whether the current state allows spending a hint token.
    """
    return hint_tokens > 0


def can_recover_hint_token(
    hint_tokens: int, max_hint_tokens: int = MAX_HINT_TOKENS
) -> bool:
    """
    Return whether a hint token can be recovered without exceeding the cap.
    """
    return hint_tokens < max_hint_tokens


def can_discard(
    hint_tokens: int, max_hint_tokens: int = MAX_HINT_TOKENS
) -> bool:
    """
    Return whether discarding is legal under the current hint-token count.
    """
    return hint_tokens < max_hint_tokens


def playing_card_recovers_hint(card: Card) -> bool:
    """
    Return whether successfully playing the card would recover a hint token.
    """
    return card.rank == Rank.FIVE
