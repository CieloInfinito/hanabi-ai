from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from hanabi_ai.game.cards import Card, Rank
from hanabi_ai.game.observation import CardKnowledge

HintScore: TypeAlias = tuple[int, int, int, int, int, int, int, int, int]
DiscardScore: TypeAlias = tuple[int, int, float, float, float, float, float, int, int]
PlayScore: TypeAlias = tuple[float, float, float, int]


@dataclass(frozen=True, slots=True)
class HintEffect:
    guaranteed_play_hits: int
    information_gain: int


def knowledge_state_size(knowledge: CardKnowledge) -> int:
    return len(knowledge.possible_colors) * len(knowledge.possible_ranks)


def possible_cards_from_knowledge(knowledge: CardKnowledge) -> tuple[Card, ...]:
    return tuple(
        Card(color=color, rank=rank)
        for color in knowledge.possible_colors
        for rank in knowledge.possible_ranks
    )


def highest_rank_value(cards: tuple[Card, ...]) -> int:
    return max(int(card.rank) for card in cards) if cards else int(Rank.ONE)
