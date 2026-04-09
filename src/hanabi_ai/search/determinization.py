from __future__ import annotations

from dataclasses import dataclass
from random import Random

from hanabi_ai.game.cards import Card, build_standard_deck, hand_size_for_player_count
from hanabi_ai.game.engine import EngineState, HanabiGameEngine, TurnRecord
from hanabi_ai.game.observation import (
    CardKnowledge,
    PlayerObservation,
    build_remaining_card_counts,
    reconstruct_public_hand_knowledge,
)


@dataclass(frozen=True, slots=True)
class CompatibleWorld:
    engine: HanabiGameEngine
    sampled_own_hand: tuple[Card, ...]


def sample_hidden_hand(
    hand_knowledge: tuple[CardKnowledge, ...],
    observation: PlayerObservation,
    *,
    rng: Random | None = None,
) -> tuple[Card, ...]:
    """
    Sample one hidden hand compatible with the observer's current knowledge.
    """
    random_generator = rng if rng is not None else Random()
    remaining_counts = build_remaining_card_counts(observation)
    sampled_cards: list[Card] = []

    for knowledge in hand_knowledge:
        candidates = [
            (card, count)
            for card, count in remaining_counts.items()
            if count > 0
            and card.color in knowledge.possible_colors
            and card.rank in knowledge.possible_ranks
        ]
        if not candidates:
            raise ValueError(
                "Cannot sample a hidden hand compatible with the provided knowledge."
            )

        cards, weights = zip(*candidates, strict=True)
        selected_card = random_generator.choices(cards, weights=weights, k=1)[0]
        remaining_counts[selected_card] -= 1
        sampled_cards.append(selected_card)

    return tuple(sampled_cards)


def build_determinized_engine_state(
    observation: PlayerObservation,
    sampled_own_hand: tuple[Card, ...],
    *,
    rng: Random | None = None,
) -> EngineState:
    """
    Build a simulation-ready engine state from one observation plus a sampled own hand.
    """
    player_count = len(observation.other_player_hands) + 1
    expected_hand_size = hand_size_for_player_count(player_count)
    if len(sampled_own_hand) != len(observation.hand_knowledge):
        raise ValueError("sampled_own_hand length must match observation.hand_knowledge.")
    if len(sampled_own_hand) > expected_hand_size:
        raise ValueError("sampled_own_hand is larger than the supported hand size.")

    remaining_counts = build_remaining_card_counts(observation)
    for card in sampled_own_hand:
        if remaining_counts.get(card, 0) <= 0:
            raise ValueError("sampled_own_hand contains cards incompatible with observation.")
        remaining_counts[card] -= 1

    return EngineState(
        player_count=player_count,
        hands=_build_hands(observation, sampled_own_hand),
        knowledge_by_player=_build_knowledge_by_player(observation),
        deck=_build_unknown_deck(remaining_counts, observation.deck_size, rng=rng),
        discard_pile=tuple(observation.discard_pile),
        fireworks=dict(observation.fireworks),
        hint_tokens=observation.hint_tokens,
        strike_tokens=observation.strike_tokens,
        current_player=observation.current_player,
        turn_number=len(observation.public_history),
        final_turns_remaining=_reconstruct_final_turns_remaining(observation),
        history=_rebuild_turn_history(observation),
    )


def sample_compatible_world(
    observation: PlayerObservation,
    *,
    seed: int | None = None,
) -> CompatibleWorld:
    """
    Sample one full hidden world compatible with the current observation.
    """
    random_generator = Random(seed)
    sampled_own_hand = sample_hidden_hand(
        observation.hand_knowledge,
        observation,
        rng=random_generator,
    )
    engine_state = build_determinized_engine_state(
        observation,
        sampled_own_hand,
        rng=random_generator,
    )
    return CompatibleWorld(
        engine=HanabiGameEngine.from_state(engine_state, rng=Random(seed)),
        sampled_own_hand=sampled_own_hand,
    )


def sample_compatible_worlds(
    observation: PlayerObservation,
    count: int,
    *,
    seed: int | None = None,
) -> list[CompatibleWorld]:
    """
    Sample several hidden worlds compatible with the same observation.
    """
    if count <= 0:
        raise ValueError(f"count must be positive. Received {count}.")

    random_generator = Random(seed)
    worlds: list[CompatibleWorld] = []
    for _ in range(count):
        sampled_own_hand = sample_hidden_hand(
            observation.hand_knowledge,
            observation,
            rng=random_generator,
        )
        engine_state = build_determinized_engine_state(
            observation,
            sampled_own_hand,
            rng=random_generator,
        )
        worlds.append(
            CompatibleWorld(
                engine=HanabiGameEngine.from_state(
                    engine_state,
                    rng=Random(random_generator.random()),
                ),
                sampled_own_hand=sampled_own_hand,
            )
        )
    return worlds


def _build_hands(
    observation: PlayerObservation,
    sampled_own_hand: tuple[Card, ...],
) -> tuple[tuple[Card, ...], ...]:
    player_count = len(observation.other_player_hands) + 1
    hands_by_player: list[tuple[Card, ...] | None] = [None] * player_count
    hands_by_player[observation.observing_player] = sampled_own_hand

    for observed_hand in observation.other_player_hands:
        hands_by_player[observed_hand.player_id] = tuple(observed_hand.cards)

    return tuple(hand if hand is not None else () for hand in hands_by_player)


def _build_knowledge_by_player(
    observation: PlayerObservation,
) -> tuple[tuple[CardKnowledge, ...], ...]:
    player_count = len(observation.other_player_hands) + 1
    knowledge_by_player: list[tuple[CardKnowledge, ...] | None] = [None] * player_count
    knowledge_by_player[observation.observing_player] = observation.hand_knowledge

    for observed_hand in observation.other_player_hands:
        knowledge_by_player[observed_hand.player_id] = reconstruct_public_hand_knowledge(
            observation,
            observed_hand.player_id,
        )

    return tuple(
        hand_knowledge if hand_knowledge is not None else ()
        for hand_knowledge in knowledge_by_player
    )


def _build_unknown_deck(
    remaining_counts: dict[Card, int],
    expected_deck_size: int,
    *,
    rng: Random | None = None,
) -> tuple[Card, ...]:
    deck = [card for card, count in remaining_counts.items() for _ in range(count)]
    if len(deck) != expected_deck_size:
        raise ValueError(
            "Remaining compatible cards do not match the observation deck size."
        )

    random_generator = rng if rng is not None else Random()
    random_generator.shuffle(deck)
    return tuple(deck)


def _rebuild_turn_history(
    observation: PlayerObservation,
) -> tuple[TurnRecord, ...]:
    return tuple(
        TurnRecord(
            player_id=record.player_id,
            action=record.action,
            revealed_indices=record.revealed_indices,
            revealed_groups=record.revealed_groups,
            fireworks_before=(
                dict(record.fireworks_before)
                if record.fireworks_before is not None
                else None
            ),
            drew_replacement=record.drew_replacement,
        )
        for record in observation.public_history
    )


def _reconstruct_final_turns_remaining(
    observation: PlayerObservation,
) -> int | None:
    if observation.deck_size > 0:
        return None

    player_count = len(observation.other_player_hands) + 1
    initial_deck_size = len(build_standard_deck()) - (
        player_count * hand_size_for_player_count(player_count)
    )
    draws_seen = 0
    turn_of_last_draw: int | None = None
    for turn_index, record in enumerate(observation.public_history):
        if record.drew_replacement:
            draws_seen += 1
            if draws_seen == initial_deck_size:
                turn_of_last_draw = turn_index
                break

    if turn_of_last_draw is None:
        return 0 if observation.public_history else player_count - 1

    turns_after_last_draw = len(observation.public_history) - turn_of_last_draw - 1
    return max(player_count - 1 - turns_after_last_draw, 0)
