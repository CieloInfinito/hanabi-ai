from __future__ import annotations

import unittest

from hanabi_ai.agents.heuristic.convention import ConventionHeuristicAgent
from hanabi_ai.game.actions import (
    AgentDecision,
    HintColorAction,
    HintPresentation,
    HintRankAction,
    PlayAction,
)
from hanabi_ai.game.cards import Card, Color, Rank
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.game.observation import CardKnowledge, ObservedHand, PlayerObservation, PublicTurnRecord
from ._shared import SharedHeuristicAgentTests


# Convention heuristic tests verify that this variant emits and
# interprets its private color-order and rank-playability conventions.
class ConventionHeuristicAgentTests(SharedHeuristicAgentTests, unittest.TestCase):
    def make_agent(self):
        return ConventionHeuristicAgent()

    def test_convention_heuristic_agent_gives_hint_for_other_players_playable_card_with_private_presentation(self) -> None:
        # Verifies that the convention heuristic augments an otherwise shared
        # rank hint with its private presentation convention.
        engine = HanabiGameEngine(player_count=2, seed=32)
        engine.hands[1] = [
            Card(Color.BLUE, Rank.ONE),
            Card(Color.RED, Rank.TWO),
            Card(Color.GREEN, Rank.THREE),
            Card(Color.YELLOW, Rank.FOUR),
            Card(Color.WHITE, Rank.FIVE),
        ]
        observation = engine.get_observation(0)
        agent = ConventionHeuristicAgent()

        decision = agent.act(observation)

        self.assertIsInstance(decision, AgentDecision)
        self.assertEqual(decision.action, HintRankAction(target_player=1, rank=Rank.ONE))
        self.assertEqual(
            decision.hint_presentation,
            HintPresentation(revealed_indices=(0,), revealed_groups=((0,),)),
        )

    def test_convention_heuristic_agent_keeps_ascending_color_hint_convention_private(self) -> None:
        # Verifies that the agent can refine rank knowledge from an
        # ascending-order color-hint convention without requiring engine support,
        # including the fact that two hinted cards can be inferred to share a rank.
        agent = ConventionHeuristicAgent()
        hand_knowledge = (
            CardKnowledge(
                possible_colors=frozenset({Color.YELLOW}),
                possible_ranks=frozenset({Rank.ONE, Rank.TWO}),
                hinted_color=Color.YELLOW,
            ),
            CardKnowledge(
                possible_colors=frozenset({Color.YELLOW}),
                possible_ranks=frozenset({Rank.TWO, Rank.THREE}),
                hinted_color=Color.YELLOW,
            ),
            CardKnowledge(
                possible_colors=frozenset({Color.WHITE, Color.GREEN}),
                possible_ranks=frozenset({Rank.FOUR, Rank.FIVE}),
            ),
            CardKnowledge(
                possible_colors=frozenset({Color.YELLOW}),
                possible_ranks=frozenset({Rank.TWO, Rank.FOUR}),
                hinted_color=Color.YELLOW,
            ),
            CardKnowledge(
                possible_colors=frozenset({Color.YELLOW}),
                possible_ranks=frozenset({Rank.ONE, Rank.TWO}),
                hinted_color=Color.YELLOW,
            ),
        )

        refined_knowledge = agent._apply_ascending_color_hint_convention(
            hand_knowledge,
            revealed_indices=(0, 1, 4, 3),
        )

        self.assertEqual(refined_knowledge[1].possible_ranks, frozenset({Rank.TWO}))
        self.assertEqual(refined_knowledge[4].possible_ranks, frozenset({Rank.TWO}))
        self.assertEqual(
            refined_knowledge[3].possible_ranks,
            frozenset({Rank.TWO, Rank.FOUR}),
        )
        self.assertEqual(
            refined_knowledge[2].possible_ranks,
            frozenset({Rank.FOUR, Rank.FIVE}),
        )

    def test_convention_heuristic_agent_uses_public_hint_history_for_private_convention(self) -> None:
        # Verifies that the agent consumes public hint history from the
        # observation and applies its private ascending-order convention.
        agent = ConventionHeuristicAgent()
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(
                CardKnowledge(
                    possible_colors=frozenset({Color.YELLOW}),
                    possible_ranks=frozenset({Rank.ONE, Rank.TWO}),
                    hinted_color=Color.YELLOW,
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.YELLOW}),
                    possible_ranks=frozenset({Rank.TWO, Rank.THREE}),
                    hinted_color=Color.YELLOW,
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.WHITE, Color.GREEN}),
                    possible_ranks=frozenset({Rank.FOUR, Rank.FIVE}),
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.YELLOW}),
                    possible_ranks=frozenset({Rank.TWO, Rank.FOUR}),
                    hinted_color=Color.YELLOW,
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.YELLOW}),
                    possible_ranks=frozenset({Rank.ONE, Rank.TWO}),
                    hinted_color=Color.YELLOW,
                ),
            ),
            other_player_hands=(
                ObservedHand(
                    player_id=1,
                    cards=(
                        Card(Color.RED, Rank.ONE),
                        Card(Color.BLUE, Rank.ONE),
                        Card(Color.GREEN, Rank.THREE),
                        Card(Color.YELLOW, Rank.FOUR),
                        Card(Color.WHITE, Rank.FIVE),
                    ),
                ),
            ),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=10,
            public_history=(
                PublicTurnRecord(
                    player_id=1,
                    action=HintColorAction(target_player=0, color=Color.YELLOW),
                    revealed_indices=(0, 1, 4, 3),
                    revealed_groups=((0,), (1,), (4,), (3,)),
                ),
            ),
            legal_actions=(PlayAction(card_index=0), PlayAction(card_index=1)),
        )

        refined_observation = agent._apply_private_conventions(observation)

        self.assertEqual(
            refined_observation.hand_knowledge[1].possible_ranks,
            frozenset({Rank.TWO}),
        )
        self.assertEqual(
            refined_observation.hand_knowledge[4].possible_ranks,
            frozenset({Rank.TWO}),
        )

    def test_convention_heuristic_agent_prefers_rank_hint_when_it_creates_safe_play(self) -> None:
        # Verifies that the convention heuristic may choose a rank hint when
        # that more directly creates a safe play than a broader color hint.
        engine = HanabiGameEngine(player_count=2, seed=41)
        engine.hands[1] = [
            Card(Color.YELLOW, Rank.TWO),
            Card(Color.YELLOW, Rank.ONE),
            Card(Color.WHITE, Rank.FIVE),
            Card(Color.YELLOW, Rank.FOUR),
            Card(Color.YELLOW, Rank.TWO),
        ]
        observation = engine.get_observation(0)
        agent = ConventionHeuristicAgent()

        decision = agent.act(observation)

        self.assertIsInstance(decision, AgentDecision)
        self.assertEqual(
            decision.action,
            HintRankAction(target_player=1, rank=Rank.ONE),
        )
        self.assertEqual(
            decision.hint_presentation,
            HintPresentation(
                revealed_indices=(1,),
                revealed_groups=((1,),),
            ),
        )

    def test_convention_heuristic_agent_emits_rank_hint_groups_by_playability(self) -> None:
        # Verifies that rank hints are grouped by current playability:
        # playable cards first, then non-playable cards.
        engine = HanabiGameEngine(player_count=2, seed=42)
        engine.fireworks[Color.RED] = 1
        engine.hands[1] = [
            Card(Color.RED, Rank.TWO),
            Card(Color.BLUE, Rank.TWO),
            Card(Color.GREEN, Rank.FIVE),
            Card(Color.YELLOW, Rank.TWO),
            Card(Color.WHITE, Rank.FIVE),
        ]
        observation = engine.get_observation(0)
        agent = ConventionHeuristicAgent()

        decision = agent._attach_hint_presentation(
            HintRankAction(target_player=1, rank=Rank.TWO),
            observation,
        )

        self.assertIsInstance(decision, AgentDecision)
        self.assertEqual(
            decision.hint_presentation,
            HintPresentation(
                revealed_indices=(0, 1, 3),
                revealed_groups=((0,), (1, 3)),
            ),
        )

    def test_convention_heuristic_agent_uses_rank_hint_groups_from_public_history(self) -> None:
        # Verifies that the agent interprets grouped rank hints using the
        # playability state from when the hint was given.
        agent = ConventionHeuristicAgent()
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(
                CardKnowledge(
                    possible_colors=frozenset({Color.RED, Color.BLUE}),
                    possible_ranks=frozenset({Rank.TWO}),
                    hinted_rank=Rank.TWO,
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.BLUE, Color.YELLOW}),
                    possible_ranks=frozenset({Rank.TWO}),
                    hinted_rank=Rank.TWO,
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.WHITE}),
                    possible_ranks=frozenset({Rank.FIVE}),
                ),
            ),
            other_player_hands=(
                ObservedHand(
                    player_id=1,
                    cards=(
                        Card(Color.RED, Rank.ONE),
                        Card(Color.BLUE, Rank.ONE),
                        Card(Color.GREEN, Rank.THREE),
                    ),
                ),
            ),
            fireworks={
                Color.RED: 1,
                Color.YELLOW: 0,
                Color.GREEN: 0,
                Color.BLUE: 0,
                Color.WHITE: 0,
            },
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=10,
            public_history=(
                PublicTurnRecord(
                    player_id=1,
                    action=HintRankAction(target_player=0, rank=Rank.TWO),
                    revealed_indices=(0, 1),
                    revealed_groups=((0,), (1,)),
                    fireworks_before={
                        Color.RED: 1,
                        Color.YELLOW: 0,
                        Color.GREEN: 0,
                        Color.BLUE: 0,
                        Color.WHITE: 0,
                    },
                ),
            ),
            legal_actions=(PlayAction(card_index=0), PlayAction(card_index=1)),
        )

        refined_observation = agent._apply_private_conventions(observation)

        self.assertEqual(
            refined_observation.hand_knowledge[0].possible_colors,
            frozenset({Color.RED}),
        )
        self.assertEqual(
            refined_observation.hand_knowledge[1].possible_colors,
            frozenset({Color.BLUE, Color.YELLOW}),
        )


if __name__ == "__main__":
    unittest.main()
