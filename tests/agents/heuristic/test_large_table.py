from __future__ import annotations

import unittest

from hanabi_ai.agents.heuristic.convention_tempo import ConventionTempoHeuristicAgent
from hanabi_ai.agents.heuristic.large_table import LargeTableHeuristicAgent
from hanabi_ai.game.actions import AgentDecision, DiscardAction, HintPresentation, HintRankAction
from hanabi_ai.game.cards import Card, Color, Rank
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.game.observation import CardKnowledge, ObservedHand, PlayerObservation
from ._shared import SharedHeuristicAgentTests


class LargeTableHeuristicAgentTests(SharedHeuristicAgentTests, unittest.TestCase):
    def make_agent(self):
        return LargeTableHeuristicAgent()

    def test_large_table_agent_keeps_convention_private_hint_presentation(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=32)
        engine.hands[1] = [
            Card(Color.BLUE, Rank.ONE),
            Card(Color.RED, Rank.TWO),
            Card(Color.GREEN, Rank.THREE),
            Card(Color.YELLOW, Rank.FOUR),
            Card(Color.WHITE, Rank.FIVE),
        ]
        observation = engine.get_observation(0)
        agent = LargeTableHeuristicAgent()

        decision = agent.act(observation)

        self.assertIsInstance(decision, AgentDecision)
        self.assertEqual(decision.action, HintRankAction(target_player=1, rank=Rank.ONE))
        self.assertEqual(
            decision.hint_presentation,
            HintPresentation(revealed_indices=(0,), revealed_groups=((0,),)),
        )

    def test_large_table_agent_keeps_hint_over_discard_for_near_term_actionable_hint(self) -> None:
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(CardKnowledge(
                possible_colors=frozenset({Color.BLUE}),
                possible_ranks=frozenset({Rank.ONE}),
            ),),
            other_player_hands=(
                ObservedHand(player_id=1, cards=(Card(Color.RED, Rank.ONE),)),
                ObservedHand(player_id=2, cards=(Card(Color.GREEN, Rank.THREE),)),
                ObservedHand(player_id=3, cards=(Card(Color.YELLOW, Rank.FOUR),)),
                ObservedHand(player_id=4, cards=(Card(Color.WHITE, Rank.FIVE),)),
            ),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=40,
            public_history=(),
            legal_actions=(DiscardAction(card_index=0),),
        )
        agent = LargeTableHeuristicAgent()

        prefer_discard = agent._should_prefer_discard_over_hint(
            observation,
            DiscardAction(card_index=0),
            HintRankAction(target_player=1, rank=Rank.ONE),
            (0, 1, 0, 0, 0, 1, 2, 0, 0),
        )

        self.assertFalse(prefer_discard)

    def test_large_table_agent_matches_convention_tempo_five_player_hint_priority(self) -> None:
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(CardKnowledge(
                possible_colors=frozenset({Color.BLUE}),
                possible_ranks=frozenset({Rank.ONE}),
            ),),
            other_player_hands=(
                ObservedHand(player_id=1, cards=(Card(Color.RED, Rank.ONE),)),
                ObservedHand(player_id=2, cards=(Card(Color.GREEN, Rank.ONE),)),
                ObservedHand(player_id=3, cards=(Card(Color.YELLOW, Rank.THREE),)),
                ObservedHand(player_id=4, cards=(Card(Color.WHITE, Rank.FOUR),)),
            ),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=40,
            public_history=(),
            legal_actions=(DiscardAction(card_index=0),),
        )
        agent = LargeTableHeuristicAgent()
        baseline = ConventionTempoHeuristicAgent()

        large_table_priority = agent._hint_priority(
            observation,
            observation.other_player_hands[3],
            HintRankAction(target_player=4, rank=Rank.ONE),
            (1, 1, 0, 0, 1, 1, 6, 0, 0),
        )
        convention_tempo_priority = baseline._hint_priority(
            observation,
            observation.other_player_hands[3],
            HintRankAction(target_player=4, rank=Rank.ONE),
            (1, 1, 0, 0, 1, 1, 6, 0, 0),
        )

        self.assertEqual(large_table_priority, convention_tempo_priority)


if __name__ == "__main__":
    unittest.main()
