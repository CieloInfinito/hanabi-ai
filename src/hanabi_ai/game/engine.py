from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .actions import (
    Action,
    ActionLike,
    DiscardAction,
    HintPresentation,
    HintColorAction,
    HintRankAction,
    PlayAction,
    is_hint_action,
    normalize_agent_decision,
)
from .cards import (
    MAX_HINT_TOKENS,
    Card,
    Color,
    Rank,
    hand_size_for_player_count,
    shuffled_standard_deck,
)
from .observation import (
    CardKnowledge,
    PlayerObservation,
    PublicTurnRecord,
    apply_color_hint_to_knowledge,
    apply_rank_hint_to_knowledge,
    build_player_observation,
    create_initial_card_knowledge,
)
from .rules import (
    build_initial_fireworks,
    can_discard,
    can_give_hint,
    can_recover_hint_token,
    game_is_lost,
    game_is_won,
    is_card_playable,
    playing_card_recovers_hint,
    score_fireworks,
)


@dataclass(frozen=True, slots=True)
class TurnRecord:
    player_id: int
    action: Action
    removed_card: Card | None = None
    revealed_indices: tuple[int, ...] = ()
    revealed_groups: tuple[tuple[int, ...], ...] = ()
    fireworks_before: dict[Color, int] | None = None
    play_succeeded: bool | None = None
    drew_replacement: bool = False


@dataclass(frozen=True, slots=True)
class EngineStepResult:
    acting_player: int
    action: Action
    removed_card: Card | None
    revealed_indices: tuple[int, ...]
    revealed_groups: tuple[tuple[int, ...], ...]
    fireworks_before: dict[Color, int] | None
    play_succeeded: bool | None
    drew_replacement: bool
    score: int
    game_over: bool


class HanabiGameEngine:
    """
    Omniscient Hanabi game state and transition logic.
    """

    def __init__(
        self,
        player_count: int,
        *,
        seed: int | None = None,
        rng: Random | None = None,
    ) -> None:
        if rng is not None and seed is not None:
            raise ValueError("Provide either seed or rng, but not both.")

        hand_size_for_player_count(player_count)

        self.player_count = player_count
        self._rng = rng if rng is not None else Random(seed)

        self.hands: list[list[Card]] = []
        self.knowledge_by_player: list[list[CardKnowledge]] = []
        self.deck: list[Card] = []
        self.discard_pile: list[Card] = []
        self.fireworks: dict[Color, int] = {}
        self.hint_tokens = MAX_HINT_TOKENS
        self.strike_tokens = 0
        self.current_player = 0
        self.turn_number = 0
        self.final_turns_remaining: int | None = None
        self.history: list[TurnRecord] = []

        self.reset()

    def reset(self) -> None:
        """
        Reset the engine to a fresh shuffled game.
        """
        self.deck = shuffled_standard_deck(self._rng)
        self.discard_pile = []
        self.fireworks = build_initial_fireworks()
        self.hint_tokens = MAX_HINT_TOKENS
        self.strike_tokens = 0
        self.current_player = 0
        self.turn_number = 0
        self.final_turns_remaining = None
        self.history = []

        hand_size = hand_size_for_player_count(self.player_count)
        self.hands = [[] for _ in range(self.player_count)]
        self.knowledge_by_player = [[] for _ in range(self.player_count)]

        for _ in range(hand_size):
            for player_id in range(self.player_count):
                self._draw_card_to_player(player_id)

    def step(self, action: ActionLike) -> EngineStepResult:
        """
        Apply one legal action for the current player.
        """
        if self.is_terminal():
            raise RuntimeError("Cannot call step() on a terminal game.")

        decision = normalize_agent_decision(action)
        action = decision.action

        legal_actions = self.get_legal_actions(self.current_player)
        if action not in legal_actions:
            raise ValueError(
                f"Illegal action for player {self.current_player}: {action!s}."
            )
        self._validate_hint_presentation(action, decision.hint_presentation)

        acting_player = self.current_player
        removed_card: Card | None = None
        revealed_indices: tuple[int, ...] = ()
        revealed_groups: tuple[tuple[int, ...], ...] = ()
        fireworks_before = dict(self.fireworks)
        play_succeeded: bool | None = None
        drew_replacement = False

        if isinstance(action, PlayAction):
            removed_card, drew_replacement, play_succeeded = self._handle_play_action(
                acting_player, action
            )
        elif isinstance(action, DiscardAction):
            removed_card, drew_replacement = self._handle_discard_action(
                acting_player, action
            )
        elif isinstance(action, HintColorAction):
            revealed_indices, revealed_groups = self._handle_hint_color_action(
                acting_player,
                action,
                decision.hint_presentation,
            )
        elif isinstance(action, HintRankAction):
            revealed_indices, revealed_groups = self._handle_hint_rank_action(
                acting_player,
                action,
                decision.hint_presentation,
            )
        else:
            raise TypeError(f"Unsupported action type: {type(action)!r}.")

        record = TurnRecord(
            player_id=acting_player,
            action=action,
            removed_card=removed_card,
            revealed_indices=revealed_indices,
            revealed_groups=revealed_groups,
            fireworks_before=fireworks_before if is_hint_action(action) else None,
            play_succeeded=play_succeeded,
            drew_replacement=drew_replacement,
        )
        self.history.append(record)

        self.turn_number += 1
        self._advance_final_round_counter()

        game_over = self.is_terminal()
        if not game_over:
            self.current_player = (self.current_player + 1) % self.player_count

        return EngineStepResult(
            acting_player=acting_player,
            action=action,
            removed_card=removed_card,
            revealed_indices=revealed_indices,
            revealed_groups=revealed_groups,
            fireworks_before=fireworks_before if is_hint_action(action) else None,
            play_succeeded=play_succeeded,
            drew_replacement=drew_replacement,
            score=self.get_score(),
            game_over=game_over,
        )

    def get_legal_actions(self, player_id: int) -> list[Action]:
        """
        Return all legal actions for the specified player on the current turn.
        """
        self._validate_player_id(player_id)

        if self.is_terminal() or player_id != self.current_player:
            return []

        hand = self.hands[player_id]
        actions: list[Action] = [
            PlayAction(card_index=index) for index in range(len(hand))
        ]
        if can_discard(self.hint_tokens):
            actions.extend(
                DiscardAction(card_index=index) for index in range(len(hand))
            )

        if can_give_hint(self.hint_tokens):
            for target_player in range(self.player_count):
                if target_player == player_id:
                    continue

                target_hand = self.hands[target_player]
                present_colors = {card.color for card in target_hand}
                present_ranks = {card.rank for card in target_hand}

                actions.extend(
                    HintColorAction(target_player=target_player, color=color)
                    for color in sorted(present_colors, key=lambda value: value.value)
                )
                actions.extend(
                    HintRankAction(target_player=target_player, rank=rank)
                    for rank in sorted(present_ranks, key=int)
                )

        return actions

    def get_observation(self, player_id: int) -> PlayerObservation:
        """
        Build the partial observation for one player.
        """
        self._validate_player_id(player_id)

        legal_actions = (
            self.get_legal_actions(player_id) if player_id == self.current_player else []
        )
        return build_player_observation(
            observing_player=player_id,
            current_player=self.current_player,
            hands=self.hands,
            knowledge_by_player=self.knowledge_by_player,
            fireworks=self.fireworks,
            discard_pile=self.discard_pile,
            hint_tokens=self.hint_tokens,
            strike_tokens=self.strike_tokens,
            deck_size=len(self.deck),
            public_history=tuple(
                PublicTurnRecord(
                    player_id=record.player_id,
                    action=record.action,
                    revealed_indices=record.revealed_indices,
                    revealed_groups=record.revealed_groups,
                    fireworks_before=dict(record.fireworks_before)
                    if record.fireworks_before is not None
                    else None,
                    drew_replacement=record.drew_replacement,
                )
                for record in self.history
            ),
            legal_actions=legal_actions,
        )

    def is_terminal(self) -> bool:
        """
        Return whether the current game has ended.
        """
        return (
            game_is_won(self.fireworks)
            or game_is_lost(self.strike_tokens)
            or self.final_turns_remaining == 0
        )

    def get_score(self) -> int:
        """
        Return the current Hanabi score.
        """
        return score_fireworks(self.fireworks)

    def _handle_play_action(
        self, player_id: int, action: PlayAction
    ) -> tuple[Card, bool, bool]:
        card = self._remove_card_from_hand(player_id, action.card_index)
        play_succeeded = is_card_playable(card, self.fireworks)

        if play_succeeded:
            self.fireworks[card.color] = int(card.rank)
            if playing_card_recovers_hint(card) and can_recover_hint_token(
                self.hint_tokens
            ):
                self.hint_tokens += 1
        else:
            self.discard_pile.append(card)
            self.strike_tokens += 1

        drew_replacement = self._draw_card_to_player(player_id)
        return card, drew_replacement, play_succeeded

    def _handle_discard_action(
        self, player_id: int, action: DiscardAction
    ) -> tuple[Card, bool]:
        card = self._remove_card_from_hand(player_id, action.card_index)
        self.discard_pile.append(card)

        if can_recover_hint_token(self.hint_tokens):
            self.hint_tokens += 1

        drew_replacement = self._draw_card_to_player(player_id)
        return card, drew_replacement

    def _handle_hint_color_action(
        self,
        player_id: int,
        action: HintColorAction,
        hint_presentation: HintPresentation | None,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        self._spend_hint_token()
        target_hand = self.hands[action.target_player]
        revealed_indices, revealed_groups = self._resolve_revealed_indices(
            tuple(
                index for index, card in enumerate(target_hand) if card.color == action.color
            ),
            hint_presentation,
        )
        self.knowledge_by_player[action.target_player] = apply_color_hint_to_knowledge(
            self.knowledge_by_player[action.target_player],
            target_hand,
            action.color,
        )
        return revealed_indices, revealed_groups

    def _handle_hint_rank_action(
        self,
        player_id: int,
        action: HintRankAction,
        hint_presentation: HintPresentation | None,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        self._spend_hint_token()
        target_hand = self.hands[action.target_player]
        revealed_indices, revealed_groups = self._resolve_revealed_indices(
            tuple(
                index for index, card in enumerate(target_hand) if card.rank == action.rank
            ),
            hint_presentation,
        )
        self.knowledge_by_player[action.target_player] = apply_rank_hint_to_knowledge(
            self.knowledge_by_player[action.target_player],
            target_hand,
            action.rank,
        )
        return revealed_indices, revealed_groups

    def _validate_hint_presentation(
        self,
        action: Action,
        hint_presentation: HintPresentation | None,
    ) -> None:
        if hint_presentation is None:
            return
        if not is_hint_action(action):
            raise ValueError(
                "hint_presentation can only be provided for hint actions."
            )

    def _resolve_revealed_indices(
        self,
        default_indices: tuple[int, ...],
        hint_presentation: HintPresentation | None,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        if hint_presentation is None:
            return default_indices, (default_indices,) if default_indices else ()

        custom_indices = hint_presentation.revealed_indices
        custom_groups = hint_presentation.revealed_groups
        if custom_indices is None:
            raise ValueError("hint_presentation must resolve to revealed_indices.")
        if tuple(sorted(custom_indices)) != tuple(sorted(default_indices)):
            raise ValueError(
                "hint_presentation.revealed_indices must match the cards revealed by the hint."
            )
        if custom_groups is None:
            custom_groups = (custom_indices,) if custom_indices else ()
        return custom_indices, custom_groups

    def _spend_hint_token(self) -> None:
        if not can_give_hint(self.hint_tokens):
            raise RuntimeError("Cannot spend a hint token when none are available.")
        self.hint_tokens -= 1

    def _remove_card_from_hand(self, player_id: int, card_index: int) -> Card:
        hand = self.hands[player_id]
        knowledge = self.knowledge_by_player[player_id]

        if card_index >= len(hand):
            raise IndexError(
                f"card_index {card_index} is out of range for player {player_id}."
            )

        knowledge.pop(card_index)
        return hand.pop(card_index)

    def _draw_card_to_player(self, player_id: int) -> bool:
        if not self.deck:
            return False

        card = self.deck.pop()
        self.hands[player_id].append(card)
        self.knowledge_by_player[player_id].append(create_initial_card_knowledge())

        if not self.deck and self.final_turns_remaining is None:
            self.final_turns_remaining = self.player_count

        return True

    def _advance_final_round_counter(self) -> None:
        if self.final_turns_remaining is not None:
            self.final_turns_remaining -= 1

    def _validate_player_id(self, player_id: int) -> None:
        if not 0 <= player_id < self.player_count:
            raise ValueError(
                f"player_id must be in [0, {self.player_count - 1}]. "
                f"Received {player_id}."
            )
