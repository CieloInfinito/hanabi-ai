from __future__ import annotations

from hanabi_ai.game.actions import (
    AgentDecision,
    DiscardAction,
    HintPresentation,
    HintColorAction,
    HintRankAction,
    PlayAction,
)
from hanabi_ai.game.cards import Card, Rank
from hanabi_ai.game.observation import (
    CardKnowledge,
    PlayerObservation,
    PublicTurnRecord,
)
from hanabi_ai.game.rules import is_card_playable


class _ConventionHeuristicMixin:
    def _apply_private_conventions(
        self, observation: PlayerObservation
    ) -> PlayerObservation:
        hand_knowledge = observation.hand_knowledge

        for record in self._relevant_public_history(observation):
            hand_knowledge = self._apply_convention_record(
                hand_knowledge,
                observation,
                record,
            )

        return PlayerObservation(
            observing_player=observation.observing_player,
            current_player=observation.current_player,
            hand_knowledge=hand_knowledge,
            other_player_hands=observation.other_player_hands,
            fireworks=observation.fireworks,
            discard_pile=observation.discard_pile,
            hint_tokens=observation.hint_tokens,
            strike_tokens=observation.strike_tokens,
            deck_size=observation.deck_size,
            public_history=observation.public_history,
            legal_actions=observation.legal_actions,
        )

    def describe_public_turn_record(
        self, record: PublicTurnRecord
    ) -> tuple[str, ...]:
        notes: list[str] = []

        if isinstance(record.action, HintColorAction) and record.revealed_indices:
            notes.append(
                "Convention heuristic: color hints point matching cards in ascending rank order, including equal-rank ties."
            )

        if isinstance(record.action, HintRankAction) and record.revealed_groups:
            notes.append(
                "Convention heuristic: rank hints group playable cards first, then non-playable cards."
            )

        return tuple(notes)

    def _apply_convention_record(
        self,
        hand_knowledge: tuple[CardKnowledge, ...],
        observation: PlayerObservation,
        record: PublicTurnRecord,
    ) -> tuple[CardKnowledge, ...]:
        action = record.action
        if isinstance(action, HintColorAction):
            if action.target_player != observation.observing_player:
                return hand_knowledge
            return self._apply_ascending_color_hint_convention(
                hand_knowledge,
                record.revealed_indices,
            )

        if not isinstance(action, HintRankAction):
            return hand_knowledge
        if action.target_player != observation.observing_player:
            return hand_knowledge
        if record.fireworks_before is None:
            return hand_knowledge

        return self._apply_rank_playability_hint_convention(
            hand_knowledge,
            record.revealed_groups,
            record.fireworks_before,
        )

    def _relevant_public_history(
        self, observation: PlayerObservation
    ) -> tuple[PublicTurnRecord, ...]:
        start_index = 0

        for index, record in enumerate(observation.public_history):
            if record.player_id != observation.observing_player:
                continue
            if isinstance(record.action, (PlayAction, DiscardAction)):
                start_index = index + 1

        return observation.public_history[start_index:]

    def _apply_ascending_color_hint_convention(
        self,
        hand_knowledge: tuple[CardKnowledge, ...],
        revealed_indices: tuple[int, ...],
    ) -> tuple[CardKnowledge, ...]:
        if not revealed_indices:
            return hand_knowledge

        refined_knowledge = list(hand_knowledge)
        ordered_knowledge = [hand_knowledge[index] for index in revealed_indices]
        valid_rank_sequences = self._build_valid_ascending_rank_sequences(
            ordered_knowledge
        )
        if not valid_rank_sequences:
            return hand_knowledge

        for position, card_index in enumerate(revealed_indices):
            allowed_ranks = frozenset(
                sequence[position] for sequence in valid_rank_sequences
            )
            refined_knowledge[card_index] = self._replace_knowledge_ranks(
                refined_knowledge[card_index],
                allowed_ranks,
            )

        return tuple(refined_knowledge)

    def _build_valid_ascending_rank_sequences(
        self, ordered_knowledge: list[CardKnowledge]
    ) -> tuple[tuple[Rank, ...], ...]:
        valid_sequences: list[tuple[Rank, ...]] = []

        def search(
            position: int,
            minimum_rank: int,
            current_sequence: list[Rank],
        ) -> None:
            if position == len(ordered_knowledge):
                valid_sequences.append(tuple(current_sequence))
                return

            knowledge_item = ordered_knowledge[position]
            candidate_ranks = sorted(
                rank
                for rank in knowledge_item.possible_ranks
                if int(rank) >= minimum_rank
            )

            for rank in candidate_ranks:
                current_sequence.append(rank)
                search(position + 1, int(rank), current_sequence)
                current_sequence.pop()

        search(position=0, minimum_rank=1, current_sequence=[])
        return tuple(valid_sequences)

    def _attach_hint_presentation(
        self,
        action: HintColorAction | HintRankAction,
        observation: PlayerObservation,
    ) -> HintColorAction | HintRankAction | AgentDecision:
        target_hand = next(
            hand
            for hand in observation.other_player_hands
            if hand.player_id == action.target_player
        )

        if isinstance(action, HintColorAction):
            revealed_indices = tuple(
                index
                for index, card in sorted(
                    enumerate(target_hand.cards),
                    key=lambda item: (int(item[1].rank), item[0]),
                )
                if card.color == action.color
            )
            revealed_groups = tuple((index,) for index in revealed_indices)
        else:
            revealed_groups = self._build_rank_hint_revealed_groups(
                target_hand.cards,
                action.rank,
                observation,
            )
            revealed_indices = self._flatten_revealed_groups(revealed_groups)

        return AgentDecision(
            action=action,
            hint_presentation=HintPresentation(
                revealed_indices=revealed_indices,
                revealed_groups=revealed_groups,
            ),
        )

    def _build_rank_hint_revealed_groups(
        self,
        cards: tuple[Card, ...],
        rank: Rank,
        observation: PlayerObservation,
    ) -> tuple[tuple[int, ...], ...]:
        playable_indices = tuple(
            index
            for index, card in enumerate(cards)
            if card.rank == rank and self._card_is_playable_now(card, observation)
        )
        non_playable_indices = tuple(
            index
            for index, card in enumerate(cards)
            if card.rank == rank and not self._card_is_playable_now(card, observation)
        )

        if playable_indices and non_playable_indices:
            return (playable_indices, non_playable_indices)
        if playable_indices:
            return (playable_indices,)
        if non_playable_indices:
            return (non_playable_indices,)
        return ()

    def _apply_rank_playability_hint_convention(
        self,
        hand_knowledge: tuple[CardKnowledge, ...],
        revealed_groups: tuple[tuple[int, ...], ...],
        fireworks_before: dict,
    ) -> tuple[CardKnowledge, ...]:
        if not revealed_groups:
            return hand_knowledge

        flattened_indices = self._flatten_revealed_groups(revealed_groups)
        hinted_knowledge = [hand_knowledge[index] for index in flattened_indices]
        valid_assignments = self._build_valid_rank_hint_assignments(
            hinted_knowledge,
            revealed_groups,
            fireworks_before,
        )
        if not valid_assignments:
            return hand_knowledge

        refined_knowledge = list(hand_knowledge)
        for position, card_index in enumerate(flattened_indices):
            allowed_cards = {assignment[position] for assignment in valid_assignments}
            knowledge = refined_knowledge[card_index]
            refined_knowledge[card_index] = CardKnowledge(
                possible_colors=frozenset(card.color for card in allowed_cards),
                possible_ranks=frozenset(card.rank for card in allowed_cards),
                hinted_color=knowledge.hinted_color,
                hinted_rank=knowledge.hinted_rank,
            )

        return tuple(refined_knowledge)

    def _build_valid_rank_hint_assignments(
        self,
        hinted_knowledge: list[CardKnowledge],
        revealed_groups: tuple[tuple[int, ...], ...],
        fireworks_before: dict,
    ) -> tuple[tuple[Card, ...], ...]:
        possible_cards_per_position = [
            self._possible_cards(knowledge) for knowledge in hinted_knowledge
        ]
        valid_assignments: list[tuple[Card, ...]] = []

        def search(position: int, current_assignment: list[Card]) -> None:
            if position == len(possible_cards_per_position):
                if self._assignment_matches_rank_hint_groups(
                    tuple(current_assignment),
                    revealed_groups,
                    fireworks_before,
                ):
                    valid_assignments.append(tuple(current_assignment))
                return

            for card in possible_cards_per_position[position]:
                current_assignment.append(card)
                search(position + 1, current_assignment)
                current_assignment.pop()

        search(position=0, current_assignment=[])
        return tuple(valid_assignments)

    def _assignment_matches_rank_hint_groups(
        self,
        assignment: tuple[Card, ...],
        revealed_groups: tuple[tuple[int, ...], ...],
        fireworks_before: dict,
    ) -> bool:
        assignment_groups = self._partition_assignment_by_groups(
            assignment,
            revealed_groups,
        )

        if len(assignment_groups) == 1:
            group = assignment_groups[0]
            return all(
                is_card_playable(card, fireworks_before) for card in group
            ) or all(
                not is_card_playable(card, fireworks_before) for card in group
            )

        if not all(
            is_card_playable(card, fireworks_before) for card in assignment_groups[0]
        ):
            return False

        for group in assignment_groups[1:]:
            if not all(not is_card_playable(card, fireworks_before) for card in group):
                return False

        return True

    def _partition_assignment_by_groups(
        self,
        assignment: tuple[Card, ...],
        revealed_groups: tuple[tuple[int, ...], ...],
    ) -> list[tuple[Card, ...]]:
        assignment_groups: list[tuple[Card, ...]] = []
        position = 0

        for group in revealed_groups:
            group_length = len(group)
            assignment_groups.append(assignment[position : position + group_length])
            position += group_length

        return assignment_groups

    def _flatten_revealed_groups(
        self, revealed_groups: tuple[tuple[int, ...], ...]
    ) -> tuple[int, ...]:
        return tuple(index for group in revealed_groups for index in group)

    def _replace_knowledge_ranks(
        self,
        knowledge: CardKnowledge,
        ranks: frozenset[Rank],
    ) -> CardKnowledge:
        return CardKnowledge(
            possible_colors=knowledge.possible_colors,
            possible_ranks=ranks,
            hinted_color=knowledge.hinted_color,
            hinted_rank=knowledge.hinted_rank,
        )
