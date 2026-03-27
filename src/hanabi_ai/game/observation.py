from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .actions import (
    Action,
    DiscardAction,
    HintColorAction,
    HintRankAction,
    PlayAction,
)
from .cards import (
    CARD_COUNTS_BY_RANK,
    HANABI_COLORS,
    HANABI_RANKS,
    Card,
    Color,
    Rank,
    hand_size_for_player_count,
)
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
    drew_replacement: bool = False


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


def possible_cards_from_knowledge(knowledge: CardKnowledge) -> tuple[Card, ...]:
    """
    Enumerate every card identity still compatible with one knowledge state.
    """
    return tuple(
        Card(color=color, rank=rank)
        for color in knowledge.possible_colors
        for rank in knowledge.possible_ranks
    )


def build_visible_card_counts(observation: PlayerObservation) -> Counter[Card]:
    """
    Count public cards that are no longer candidates for the observer's hidden hand.

    This includes discarded cards, all visible teammate hands, and cards already
    committed to the fireworks stacks.
    """
    counts: Counter[Card] = Counter()

    for discarded_card in observation.discard_pile:
        counts[discarded_card] += 1

    for observed_hand in observation.other_player_hands:
        for card in observed_hand.cards:
            counts[card] += 1

    for color, highest_rank in observation.fireworks.items():
        for rank_value in range(1, highest_rank + 1):
            counts[Card(color=color, rank=Rank(rank_value))] += 1

    return counts


def build_remaining_card_counts(observation: PlayerObservation) -> dict[Card, int]:
    """
    Return the number of unseen copies still compatible with the observer's hand.
    """
    visible_counts = build_visible_card_counts(observation)
    remaining_counts: dict[Card, int] = {}

    for color in HANABI_COLORS:
        for rank in HANABI_RANKS:
            card = Card(color=color, rank=rank)
            remaining_counts[card] = (
                CARD_COUNTS_BY_RANK[rank] - visible_counts[card]
            )

    return remaining_counts


def estimate_card_distribution(
    knowledge: CardKnowledge,
    observation: PlayerObservation,
) -> tuple[tuple[Card, float], ...]:
    """
    Estimate a hidden card distribution using remaining public copy counts.

    Candidate cards with more unseen copies receive proportionally more weight.
    If all compatible cards are exhausted by public information, the function
    falls back to a uniform distribution over the raw knowledge state so callers
    still receive a usable conservative estimate.
    """
    possible_cards = possible_cards_from_knowledge(knowledge)
    if not possible_cards:
        return ()

    remaining_counts = build_remaining_card_counts(observation)
    weighted_candidates = [
        (card, remaining_counts[card])
        for card in possible_cards
        if remaining_counts[card] > 0
    ]

    if weighted_candidates:
        total_weight = sum(weight for _, weight in weighted_candidates)
        return tuple(
            (card, weight / total_weight) for card, weight in weighted_candidates
        )

    fallback_probability = 1.0 / len(possible_cards)
    return tuple((card, fallback_probability) for card in possible_cards)


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
    return [
        _knowledge_after_color_hint(knowledge_item, color, matched=(card.color == color))
        for knowledge_item, card in zip(knowledge, hand, strict=True)
    ]


def apply_rank_hint_to_knowledge(
    knowledge: list[CardKnowledge], hand: list[Card], rank: Rank
) -> list[CardKnowledge]:
    """
    Apply a rank hint to one player's hidden hand knowledge.
    """
    return [
        _knowledge_after_rank_hint(knowledge_item, rank, matched=(card.rank == rank))
        for knowledge_item, card in zip(knowledge, hand, strict=True)
    ]


def apply_color_hint_to_public_knowledge(
    knowledge: list[CardKnowledge],
    revealed_indices: tuple[int, ...],
    color: Color,
) -> list[CardKnowledge]:
    """
    Apply a public color hint update using only revealed indices.
    """
    revealed_index_set = set(revealed_indices)
    return [
        _knowledge_after_color_hint(
            knowledge_item,
            color,
            matched=(index in revealed_index_set),
        )
        for index, knowledge_item in enumerate(knowledge)
    ]


def apply_rank_hint_to_public_knowledge(
    knowledge: list[CardKnowledge],
    revealed_indices: tuple[int, ...],
    rank: Rank,
) -> list[CardKnowledge]:
    """
    Apply a public rank hint update using only revealed indices.
    """
    revealed_index_set = set(revealed_indices)
    return [
        _knowledge_after_rank_hint(
            knowledge_item,
            rank,
            matched=(index in revealed_index_set),
        )
        for index, knowledge_item in enumerate(knowledge)
    ]


def reconstruct_public_hand_knowledge(
    observation: PlayerObservation,
    target_player: int,
) -> tuple[CardKnowledge, ...]:
    """
    Reconstruct one player's public hand knowledge from the visible turn history.
    """
    knowledge = create_initial_hand_knowledge(
        hand_size_for_player_count(len(observation.other_player_hands) + 1)
    )

    for record in observation.public_history:
        action = record.action
        if isinstance(action, HintColorAction) and action.target_player == target_player:
            knowledge = apply_color_hint_to_public_knowledge(
                knowledge,
                record.revealed_indices,
                action.color,
            )
            continue

        if isinstance(action, HintRankAction) and action.target_player == target_player:
            knowledge = apply_rank_hint_to_public_knowledge(
                knowledge,
                record.revealed_indices,
                action.rank,
            )
            continue

        if record.player_id != target_player:
            continue

        if isinstance(action, (PlayAction, DiscardAction)):
            knowledge.pop(action.card_index)
            if record.drew_replacement:
                knowledge.append(create_initial_card_knowledge())

    target_hand_size = _observed_hand_size(observation, target_player)

    if len(knowledge) != target_hand_size:
        knowledge = knowledge[:target_hand_size]

    return tuple(knowledge)


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
    possible_cards = possible_cards_from_knowledge(knowledge)

    if not possible_cards:
        return False

    for card in possible_cards:
        next_required_rank = fireworks.get(card.color, 0) + 1
        if int(card.rank) != next_required_rank:
            return False

    return True


def _knowledge_after_color_hint(
    knowledge: CardKnowledge,
    color: Color,
    *,
    matched: bool,
) -> CardKnowledge:
    return CardKnowledge(
        possible_colors=(
            frozenset({color})
            if matched
            else knowledge.possible_colors.difference({color})
        ),
        possible_ranks=knowledge.possible_ranks,
        hinted_color=color if matched else knowledge.hinted_color,
        hinted_rank=knowledge.hinted_rank,
    )


def _knowledge_after_rank_hint(
    knowledge: CardKnowledge,
    rank: Rank,
    *,
    matched: bool,
) -> CardKnowledge:
    return CardKnowledge(
        possible_colors=knowledge.possible_colors,
        possible_ranks=(
            frozenset({rank})
            if matched
            else knowledge.possible_ranks.difference({rank})
        ),
        hinted_color=knowledge.hinted_color,
        hinted_rank=rank if matched else knowledge.hinted_rank,
    )


def _observed_hand_size(
    observation: PlayerObservation,
    target_player: int,
) -> int:
    if target_player == observation.observing_player:
        return len(observation.hand_knowledge)

    return len(
        next(
            hand.cards
            for hand in observation.other_player_hands
            if hand.player_id == target_player
        )
    )
