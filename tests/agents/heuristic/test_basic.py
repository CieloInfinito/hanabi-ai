from __future__ import annotations

import unittest

from hanabi_ai.agents.heuristic.basic import BasicHeuristicAgent
from hanabi_ai.game.actions import HintColorAction, HintRankAction
from hanabi_ai.game.cards import Card, Color, Rank
from hanabi_ai.game.observation import CardKnowledge, ObservedHand, PlayerObservation, PublicTurnRecord
from ._shared import SharedHeuristicAgentTests


# Basic heuristic tests verify that this variant does not emit or interpret
# any private hint-ordering conventions shared with teammates.
class BasicHeuristicAgentTests(SharedHeuristicAgentTests, unittest.TestCase):
    def make_agent(self):
        return BasicHeuristicAgent()

    def test_basic_heuristic_agent_returns_plain_color_hint_without_private_presentation(self) -> None:
        # Verifies that the basic heuristic keeps the same hint choice as the
        # conservative variant but does not emit a private color-hint presentation.
        from hanabi_ai.game.engine import HanabiGameEngine

        engine = HanabiGameEngine(player_count=2, seed=51)
        engine.hands[1] = [
            Card(Color.YELLOW, Rank.TWO),
            Card(Color.YELLOW, Rank.ONE),
            Card(Color.WHITE, Rank.FIVE),
            Card(Color.YELLOW, Rank.FOUR),
            Card(Color.YELLOW, Rank.TWO),
        ]
        observation = engine.get_observation(0)
        agent = BasicHeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(
            action,
            HintColorAction(target_player=1, color=Color.YELLOW),
        )

    def test_basic_heuristic_agent_returns_plain_rank_hint_without_private_presentation(self) -> None:
        # Verifies that the basic heuristic emits rank hints as plain actions
        # instead of grouping hinted cards by immediate playability.
        from hanabi_ai.game.engine import HanabiGameEngine

        engine = HanabiGameEngine(player_count=2, seed=52)
        engine.fireworks[Color.RED] = 1
        engine.hands[1] = [
            Card(Color.RED, Rank.TWO),
            Card(Color.BLUE, Rank.TWO),
            Card(Color.GREEN, Rank.FIVE),
            Card(Color.YELLOW, Rank.TWO),
            Card(Color.WHITE, Rank.FIVE),
        ]
        observation = engine.get_observation(0)
        agent = BasicHeuristicAgent()

        action = agent._attach_hint_presentation(
            HintRankAction(target_player=1, rank=Rank.TWO),
            observation,
        )

        self.assertEqual(
            action,
            HintRankAction(target_player=1, rank=Rank.TWO),
        )

    def test_basic_heuristic_agent_does_not_apply_color_hint_private_inference(self) -> None:
        # Verifies that the basic heuristic ignores the conservative
        # color-order convention when interpreting public hint history.
        agent = BasicHeuristicAgent()
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
            legal_actions=(),
        )

        refined_observation = agent._apply_private_conventions(observation)

        self.assertEqual(
            refined_observation.hand_knowledge[1].possible_ranks,
            frozenset({Rank.TWO, Rank.THREE}),
        )
        self.assertEqual(
            refined_observation.hand_knowledge[4].possible_ranks,
            frozenset({Rank.ONE, Rank.TWO}),
        )


if __name__ == "__main__":
    unittest.main()
