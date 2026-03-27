from __future__ import annotations

import unittest

from hanabi_ai.agents.heuristic.basic import BasicHeuristicAgent
from hanabi_ai.agents.heuristic.base import _HintPriorityWeights
from hanabi_ai.game.actions import HintColorAction, HintRankAction
from hanabi_ai.game.cards import Card, Color, Rank
from hanabi_ai.game.observation import CardKnowledge, ObservedHand, PlayerObservation, PublicTurnRecord
from ._shared import SharedBasicHeuristicVariantTests, SharedHeuristicAgentTests


# Basic heuristic tests verify that this variant does not emit or interpret
# any private hint-ordering conventions shared with teammates.
class BasicHeuristicAgentTests(
    SharedBasicHeuristicVariantTests,
    SharedHeuristicAgentTests,
    unittest.TestCase,
):
    def make_agent(self):
        return BasicHeuristicAgent()

    def test_basic_agent_exposes_player_count_hint_priority_weights(self) -> None:
        agent = BasicHeuristicAgent()

        self.assertEqual(
            agent._base_hint_priority_weights(2),
            _HintPriorityWeights(
                actionable_hint=1,
                critical_playable=1,
            ),
        )
        self.assertEqual(
            agent._base_hint_priority_weights(3),
            _HintPriorityWeights(
                receiver_needs_help=1,
                immediate_receiver=1,
                near_term_receiver=1,
                actionable_hint=1,
                critical_playable=1,
            ),
        )
        self.assertEqual(
            agent._base_hint_priority_weights(4),
            _HintPriorityWeights(
                follow_on_value=1,
                receiver_needs_help=1,
                immediate_receiver=2,
                near_term_receiver=1,
                actionable_hint=1,
                critical_playable=1,
                turn_distance_penalty=1,
            ),
        )
        self.assertEqual(
            agent._base_hint_priority_weights(5),
            _HintPriorityWeights(
                follow_on_value=2,
                receiver_needs_help=1,
                immediate_receiver=1,
                near_term_receiver=2,
                actionable_hint=1,
                critical_playable=1,
                turn_distance_penalty=1,
            ),
        )

    def test_basic_agent_prioritizes_hint_that_relieves_near_term_pressure(self) -> None:
        agent = BasicHeuristicAgent()
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(CardKnowledge(
                possible_colors=frozenset({Color.BLUE}),
                possible_ranks=frozenset({Rank.ONE}),
            ),),
            other_player_hands=(
                ObservedHand(
                    player_id=1,
                    cards=(
                        Card(Color.RED, Rank.THREE),
                        Card(Color.RED, Rank.FOUR),
                        Card(Color.GREEN, Rank.FIVE),
                        Card(Color.WHITE, Rank.THREE),
                    ),
                ),
                ObservedHand(
                    player_id=2,
                    cards=(
                        Card(Color.BLUE, Rank.ONE),
                        Card(Color.GREEN, Rank.THREE),
                        Card(Color.YELLOW, Rank.FOUR),
                        Card(Color.WHITE, Rank.FIVE),
                    ),
                ),
                ObservedHand(
                    player_id=3,
                    cards=(
                        Card(Color.RED, Rank.THREE),
                        Card(Color.GREEN, Rank.FOUR),
                        Card(Color.YELLOW, Rank.FIVE),
                        Card(Color.WHITE, Rank.THREE),
                    ),
                ),
            ),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=40,
            public_history=(
                PublicTurnRecord(
                    player_id=0,
                    action=HintRankAction(target_player=2, rank=Rank.ONE),
                    revealed_indices=(0,),
                    revealed_groups=((0,),),
                ),
            ),
            legal_actions=(),
        )

        pressured_hint = agent._hint_priority(
            observation,
            observation.other_player_hands[0],
            HintColorAction(target_player=1, color=Color.RED),
            (0, 1, 0, 0, 0, 2, 3, 0, 0),
        )
        stable_hint = agent._hint_priority(
            observation,
            observation.other_player_hands[1],
            HintColorAction(target_player=2, color=Color.BLUE),
            (0, 1, 0, 0, 0, 2, 3, 0, 0),
        )

        self.assertGreater(pressured_hint[1], stable_hint[1])


if __name__ == "__main__":
    unittest.main()
