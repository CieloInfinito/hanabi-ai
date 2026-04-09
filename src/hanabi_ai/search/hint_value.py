from __future__ import annotations

from dataclasses import dataclass

from hanabi_ai.game.actions import HintColorAction, HintRankAction
from hanabi_ai.game.cards import Card
from hanabi_ai.game.observation import (
    PlayerObservation,
    apply_color_hint_to_public_knowledge,
    apply_rank_hint_to_public_knowledge,
    build_remaining_card_counts,
    get_definitely_playable_card_indices,
    reconstruct_public_hand_knowledge,
)
from hanabi_ai.game.rules import is_card_already_played, is_card_playable


@dataclass(frozen=True, slots=True)
class HintValueBreakdown:
    total_value: float
    guaranteed_play_delta: int
    immediate_playable_touches: int
    critical_touches: int
    information_gain: int
    known_attribute_delta: int
    receiver_pressure_relief: int
    turn_distance: int


def evaluate_hint_value(
    observation: PlayerObservation,
    hint_action: HintColorAction | HintRankAction,
) -> HintValueBreakdown:
    """
    Approximate the immediate strategic value of a hint at the root state.
    """
    player_count = len(observation.other_player_hands) + 1
    target_hand = next(
        hand for hand in observation.other_player_hands if hand.player_id == hint_action.target_player
    )
    before_knowledge = reconstruct_public_hand_knowledge(
        observation,
        hint_action.target_player,
    )
    revealed_indices = _revealed_indices_for_hint(target_hand.cards, hint_action)
    if isinstance(hint_action, HintColorAction):
        updated_knowledge = tuple(
            apply_color_hint_to_public_knowledge(
                list(before_knowledge),
                revealed_indices,
                hint_action.color,
            )
        )
    else:
        updated_knowledge = tuple(
            apply_rank_hint_to_public_knowledge(
                list(before_knowledge),
                revealed_indices,
                hint_action.rank,
            )
        )
    remaining_card_counts = build_remaining_card_counts(observation)

    before_guaranteed = set(
        get_definitely_playable_card_indices(before_knowledge, observation.fireworks)
    )
    after_guaranteed = set(
        get_definitely_playable_card_indices(updated_knowledge, observation.fireworks)
    )
    guaranteed_play_delta = len(after_guaranteed.difference(before_guaranteed))

    immediate_playable_touches = sum(
        1
        for index in revealed_indices
        if is_card_playable(target_hand.cards[index], observation.fireworks)
    )
    critical_touches = sum(
        1
        for index in revealed_indices
        if _is_critical_card(target_hand.cards[index], remaining_card_counts, observation)
    )
    information_gain = sum(
        _knowledge_size(before_knowledge[index]) - _knowledge_size(updated_knowledge[index])
        for index in range(len(before_knowledge))
    )
    known_attribute_delta = sum(
        _known_attribute_count(updated_knowledge[index])
        - _known_attribute_count(before_knowledge[index])
        for index in range(len(before_knowledge))
    )
    receiver_pressure_relief = int(
        not before_guaranteed
        and bool(after_guaranteed)
    )
    turn_distance = _turn_distance(
        observation.current_player,
        hint_action.target_player,
        player_count,
    )

    total_value = (
        (2.5 * guaranteed_play_delta)
        + (1.2 * immediate_playable_touches)
        + (0.8 * critical_touches)
        + (0.08 * information_gain)
        + (0.15 * known_attribute_delta)
        + (0.6 * receiver_pressure_relief)
        + (0.3 if turn_distance == 1 and immediate_playable_touches > 0 else 0.0)
    )

    return HintValueBreakdown(
        total_value=total_value,
        guaranteed_play_delta=guaranteed_play_delta,
        immediate_playable_touches=immediate_playable_touches,
        critical_touches=critical_touches,
        information_gain=information_gain,
        known_attribute_delta=known_attribute_delta,
        receiver_pressure_relief=receiver_pressure_relief,
        turn_distance=turn_distance,
    )


def _knowledge_size(knowledge) -> int:
    return len(knowledge.possible_colors) * len(knowledge.possible_ranks)


def _known_attribute_count(knowledge) -> int:
    return int(knowledge.hinted_color is not None) + int(knowledge.hinted_rank is not None)


def _turn_distance(current_player: int, target_player: int, player_count: int) -> int:
    return (target_player - current_player) % player_count


def _is_critical_card(
    card: Card,
    remaining_card_counts: dict[Card, int],
    observation: PlayerObservation,
) -> bool:
    if is_card_already_played(card, observation.fireworks):
        return False
    return remaining_card_counts.get(card, 0) <= 1


def _revealed_indices_for_hint(
    cards: tuple[Card, ...],
    hint_action: HintColorAction | HintRankAction,
) -> tuple[int, ...]:
    if isinstance(hint_action, HintColorAction):
        return tuple(
            index for index, card in enumerate(cards) if card.color == hint_action.color
        )
    return tuple(
        index for index, card in enumerate(cards) if card.rank == hint_action.rank
    )
