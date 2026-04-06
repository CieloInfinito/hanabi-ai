from __future__ import annotations

from dataclasses import dataclass

from hanabi_ai.game.actions import (
    Action,
    DiscardAction,
    HintColorAction,
    HintRankAction,
    PlayAction,
)
from hanabi_ai.game.cards import (
    CARD_COUNTS_BY_RANK,
    HANABI_COLORS,
    HANABI_RANKS,
    MAX_HINT_TOKENS,
    MAX_STRIKE_TOKENS,
    Card,
    Color,
    Rank,
    hand_size_for_player_count,
)
from hanabi_ai.game.observation import PlayerObservation
from hanabi_ai.game.rules import is_card_playable

_MAX_DECK_SIZE = 50


@dataclass(frozen=True, slots=True)
class ActionTemplate:
    kind: str
    card_index: int | None = None
    target_offset: int | None = None
    color: Color | None = None
    rank: Rank | None = None


class LegalActionIndexer:
    """
    Encode legal Hanabi actions into a fixed policy head for one table size.

    Hint targets are represented by seat offset relative to the current player,
    which keeps the action semantics stable across seats during self-play.
    """

    def __init__(self, player_count: int) -> None:
        self.player_count = player_count
        self.hand_size = hand_size_for_player_count(player_count)
        self._templates = self._build_templates()

    @property
    def action_count(self) -> int:
        return len(self._templates)

    def legal_action_indices(self, observation: PlayerObservation) -> tuple[int, ...]:
        return tuple(
            self.action_index_for_action(
                action,
                current_player=observation.current_player,
            )
            for action in observation.legal_actions
        )

    def action_index_for_action(self, action: Action, *, current_player: int) -> int:
        if isinstance(action, PlayAction):
            return action.card_index
        if isinstance(action, DiscardAction):
            return self.hand_size + action.card_index

        hint_block_start = self.hand_size * 2
        target_offset = (action.target_player - current_player) % self.player_count
        if target_offset <= 0:
            raise ValueError(
                "Hint actions must target another player when encoded for RL."
            )

        target_block_start = hint_block_start + ((target_offset - 1) * 10)
        if isinstance(action, HintColorAction):
            return target_block_start + HANABI_COLORS.index(action.color)
        if isinstance(action, HintRankAction):
            return target_block_start + 5 + HANABI_RANKS.index(action.rank)

        raise TypeError(f"Unsupported action type: {type(action)!r}.")

    def action_for_index(
        self,
        action_index: int,
        *,
        current_player: int,
    ) -> Action:
        template = self._templates[action_index]
        if template.kind == "play":
            return PlayAction(card_index=template.card_index or 0)
        if template.kind == "discard":
            return DiscardAction(card_index=template.card_index or 0)

        if template.target_offset is None:
            raise RuntimeError("Hint template is missing target_offset.")
        target_player = (current_player + template.target_offset) % self.player_count
        if template.kind == "hint-color":
            if template.color is None:
                raise RuntimeError("Color hint template is missing color.")
            return HintColorAction(target_player=target_player, color=template.color)
        if template.kind == "hint-rank":
            if template.rank is None:
                raise RuntimeError("Rank hint template is missing rank.")
            return HintRankAction(target_player=target_player, rank=template.rank)

        raise RuntimeError(f"Unsupported action template kind: {template.kind!r}.")

    def _build_templates(self) -> tuple[ActionTemplate, ...]:
        templates: list[ActionTemplate] = []
        for card_index in range(self.hand_size):
            templates.append(ActionTemplate(kind="play", card_index=card_index))
        for card_index in range(self.hand_size):
            templates.append(ActionTemplate(kind="discard", card_index=card_index))
        for target_offset in range(1, self.player_count):
            for color in HANABI_COLORS:
                templates.append(
                    ActionTemplate(
                        kind="hint-color",
                        target_offset=target_offset,
                        color=color,
                    )
                )
            for rank in HANABI_RANKS:
                templates.append(
                    ActionTemplate(
                        kind="hint-rank",
                        target_offset=target_offset,
                        rank=rank,
                    )
                )
        return tuple(templates)


class ObservationVectorEncoder:
    """
    Encode a partial Hanabi observation into a fixed numeric feature vector.

    The vector is intentionally simple and framework-agnostic so it can be used
    by a pure Python baseline policy today and a tensor-based learner later.
    """

    def __init__(self, player_count: int) -> None:
        self.player_count = player_count
        self.hand_size = hand_size_for_player_count(player_count)
        self.visible_player_count = player_count - 1
        self.feature_size = (
            4  # score + tokens + deck
            + len(HANABI_COLORS)  # fireworks
            + (len(HANABI_COLORS) * len(HANABI_RANKS))  # discard counts
            + (self.hand_size * 5)  # own hand knowledge summary
            + (self.visible_player_count * self.hand_size * 13)  # visible hands
        )

    def encode(self, observation: PlayerObservation) -> tuple[float, ...]:
        features: list[float] = []

        score = sum(observation.fireworks.values())
        features.extend(
            (
                score / 25.0,
                observation.hint_tokens / MAX_HINT_TOKENS,
                observation.strike_tokens / MAX_STRIKE_TOKENS,
                observation.deck_size / _MAX_DECK_SIZE,
            )
        )

        for color in HANABI_COLORS:
            features.append(observation.fireworks.get(color, 0) / 5.0)

        discard_counts = self._discard_count_vector(observation.discard_pile)
        features.extend(discard_counts)

        for card_index in range(self.hand_size):
            if card_index < len(observation.hand_knowledge):
                knowledge = observation.hand_knowledge[card_index]
                features.extend(
                    (
                        1.0,
                        len(knowledge.possible_colors) / len(HANABI_COLORS),
                        len(knowledge.possible_ranks) / len(HANABI_RANKS),
                        float(knowledge.hinted_color is not None),
                        float(knowledge.hinted_rank is not None),
                    )
                )
            else:
                features.extend((0.0, 0.0, 0.0, 0.0, 0.0))

        ordered_hands = tuple(
            sorted(
                observation.other_player_hands,
                key=lambda hand: (hand.player_id - observation.current_player) % self.player_count,
            )
        )
        for hand_offset in range(self.visible_player_count):
            if hand_offset < len(ordered_hands):
                observed_hand = ordered_hands[hand_offset]
                features.extend(self._encode_visible_hand(observed_hand.cards, observation))
            else:
                features.extend((0.0,) * (self.hand_size * 13))

        return tuple(features)

    def _discard_count_vector(self, discard_pile: tuple[Card, ...]) -> tuple[float, ...]:
        discard_counts: dict[tuple[Color, Rank], int] = {}
        for card in discard_pile:
            key = (card.color, card.rank)
            discard_counts[key] = discard_counts.get(key, 0) + 1

        values: list[float] = []
        for color in HANABI_COLORS:
            for rank in HANABI_RANKS:
                count = discard_counts.get((color, rank), 0)
                values.append(count / CARD_COUNTS_BY_RANK[rank])
        return tuple(values)

    def _encode_visible_hand(
        self,
        cards: tuple[Card, ...],
        observation: PlayerObservation,
    ) -> tuple[float, ...]:
        values: list[float] = []
        for card_index in range(self.hand_size):
            if card_index < len(cards):
                card = cards[card_index]
                values.append(1.0)
                values.extend(
                    1.0 if card.color == candidate else 0.0 for candidate in HANABI_COLORS
                )
                values.extend(
                    1.0 if card.rank == candidate else 0.0 for candidate in HANABI_RANKS
                )
                values.append(float(is_card_playable(card, observation.fireworks)))
                values.append(float(card.rank == Rank.FIVE))
            else:
                values.extend((0.0,) * 13)
        return tuple(values)
