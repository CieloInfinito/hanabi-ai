from __future__ import annotations

from dataclasses import dataclass

from .actions import Action
from .cards import HANABI_COLORS, HANABI_RANKS, Card, Color, Rank
from .rules import Fireworks


@dataclass(frozen=True, slots=True)
class CardKnowledge:
    possible_colors: frozenset[Color]
    possible_ranks: frozenset[Rank]
    hinted_color: Color | None = None
    hinted_rank: Rank | None = None


@dataclass(frozen=True, slots=True)
class ObservedHand:
    player_id: int
    cards: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class PublicTurnRecord:
    player_id: int
    action: Action
    revealed_indices: tuple[int, ...] = ()
    revealed_groups: tuple[tuple[int, ...], ...] = ()
    fireworks_before: dict[Color, int] | None = None


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    observing_player: int
    current_player: int
    hand_knowledge: tuple[CardKnowledge, ...]
    other_player_hands: tuple[ObservedHand, ...]
    fireworks: dict[Color, int]
    discard_pile: tuple[Card, ...]
    hint_tokens: int
    strike_tokens: int
    deck_size: int
    public_history: tuple[PublicTurnRecord, ...]
    legal_actions: tuple[Action, ...]


def get_definitely_playable_card_indices(
    hand_knowledge: tuple[CardKnowledge, ...], fireworks: Fireworks
) -> tuple[int, ...]:
    """
    Return own-hand indices that are guaranteed playable from current knowledge.

    A card is considered definitely playable only if every remaining
    color/rank combination compatible with the player's knowledge would be
    playable on the current fireworks.
    """
    playable_indices: list[int] = []

    for index, knowledge in enumerate(hand_knowledge):
        if _knowledge_is_definitely_playable(knowledge, fireworks):
            playable_indices.append(index)

    return tuple(playable_indices)


def create_initial_card_knowledge() -> CardKnowledge:
    """
    Build the default knowledge state for a newly dealt hidden card.
    """
    return CardKnowledge(
        possible_colors=frozenset(HANABI_COLORS),
        possible_ranks=frozenset(HANABI_RANKS),
    )


def create_initial_hand_knowledge(card_count: int) -> list[CardKnowledge]:
    """
    Build initial knowledge entries for a hidden hand of the given size.
    """
    return [create_initial_card_knowledge() for _ in range(card_count)]


def apply_color_hint_to_knowledge(
    knowledge: list[CardKnowledge], hand: list[Card], color: Color
) -> list[CardKnowledge]:
    """
    Apply a color hint to one player's hidden hand knowledge.
    """
    updated_knowledge: list[CardKnowledge] = []

    for knowledge_item, card in zip(knowledge, hand, strict=True):
        if card.color == color:
            updated_knowledge.append(
                CardKnowledge(
                    possible_colors=frozenset({color}),
                    possible_ranks=knowledge_item.possible_ranks,
                    hinted_color=color,
                    hinted_rank=knowledge_item.hinted_rank,
                )
            )
        else:
            updated_knowledge.append(
                CardKnowledge(
                    possible_colors=knowledge_item.possible_colors.difference({color}),
                    possible_ranks=knowledge_item.possible_ranks,
                    hinted_color=knowledge_item.hinted_color,
                    hinted_rank=knowledge_item.hinted_rank,
                )
            )

    return updated_knowledge


def apply_rank_hint_to_knowledge(
    knowledge: list[CardKnowledge], hand: list[Card], rank: Rank
) -> list[CardKnowledge]:
    """
    Apply a rank hint to one player's hidden hand knowledge.
    """
    updated_knowledge: list[CardKnowledge] = []

    for knowledge_item, card in zip(knowledge, hand, strict=True):
        if card.rank == rank:
            updated_knowledge.append(
                CardKnowledge(
                    possible_colors=knowledge_item.possible_colors,
                    possible_ranks=frozenset({rank}),
                    hinted_color=knowledge_item.hinted_color,
                    hinted_rank=rank,
                )
            )
        else:
            updated_knowledge.append(
                CardKnowledge(
                    possible_colors=knowledge_item.possible_colors,
                    possible_ranks=knowledge_item.possible_ranks.difference({rank}),
                    hinted_color=knowledge_item.hinted_color,
                    hinted_rank=knowledge_item.hinted_rank,
                )
            )

    return updated_knowledge


def build_player_observation(
    *,
    observing_player: int,
    current_player: int,
    hands: list[list[Card]],
    knowledge_by_player: list[list[CardKnowledge]],
    fireworks: Fireworks,
    discard_pile: list[Card],
    hint_tokens: int,
    strike_tokens: int,
    deck_size: int,
    public_history: tuple[PublicTurnRecord, ...],
    legal_actions: list[Action],
) -> PlayerObservation:
    """
    Build the partial observation exposed to one player.
    """
    other_player_hands = tuple(
        ObservedHand(player_id=player_id, cards=tuple(hand))
        for player_id, hand in enumerate(hands)
        if player_id != observing_player
    )

    return PlayerObservation(
        observing_player=observing_player,
        current_player=current_player,
        hand_knowledge=tuple(knowledge_by_player[observing_player]),
        other_player_hands=other_player_hands,
        fireworks=dict(fireworks),
        discard_pile=tuple(discard_pile),
        hint_tokens=hint_tokens,
        strike_tokens=strike_tokens,
        deck_size=deck_size,
        public_history=public_history,
        legal_actions=tuple(legal_actions),
    )


def _knowledge_is_definitely_playable(
    knowledge: CardKnowledge, fireworks: Fireworks
) -> bool:
    possible_cards = [
        Card(color=color, rank=rank)
        for color in knowledge.possible_colors
        for rank in knowledge.possible_ranks
    ]

    if not possible_cards:
        return False

    for card in possible_cards:
        next_required_rank = fireworks.get(card.color, 0) + 1
        if int(card.rank) != next_required_rank:
            return False

    return True
