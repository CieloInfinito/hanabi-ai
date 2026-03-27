from __future__ import annotations

import unittest

from hanabi_ai.agents.heuristic.convention_tempo import ConventionTempoHeuristicAgent
from hanabi_ai.game.actions import AgentDecision, DiscardAction, HintColorAction, HintPresentation, HintRankAction
from hanabi_ai.game.cards import Card, Color, Rank
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.game.observation import CardKnowledge, ObservedHand, PlayerObservation, PublicTurnRecord
from ._shared import SharedHeuristicAgentTests


class ConventionTempoHeuristicAgentTests(SharedHeuristicAgentTests, unittest.TestCase):
    def make_agent(self):
        return ConventionTempoHeuristicAgent()

    def test_convention_tempo_agent_uses_convention_private_hint_presentation(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=32)
        engine.hands[1] = [
            Card(Color.BLUE, Rank.ONE),
            Card(Color.RED, Rank.TWO),
            Card(Color.GREEN, Rank.THREE),
            Card(Color.YELLOW, Rank.FOUR),
            Card(Color.WHITE, Rank.FIVE),
        ]
        observation = engine.get_observation(0)
        agent = ConventionTempoHeuristicAgent()

        decision = agent.act(observation)

        self.assertIsInstance(decision, AgentDecision)
        self.assertEqual(decision.action, HintRankAction(target_player=1, rank=Rank.ONE))
        self.assertEqual(
            decision.hint_presentation,
            HintPresentation(revealed_indices=(0,), revealed_groups=((0,),)),
        )

    def test_convention_tempo_agent_keeps_tempo_threshold_for_last_hint_in_two_player_games(self) -> None:
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(
                CardKnowledge(
                    possible_colors=frozenset({Color.BLUE}),
                    possible_ranks=frozenset({Rank.ONE}),
                    hinted_color=Color.BLUE,
                    hinted_rank=Rank.ONE,
                ),
            ),
            other_player_hands=(
                ObservedHand(
                    player_id=1,
                    cards=(Card(Color.RED, Rank.THREE),),
                ),
            ),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=40,
            public_history=(),
            legal_actions=(DiscardAction(card_index=0),),
        )
        agent = ConventionTempoHeuristicAgent()

        should_spend = agent._should_spend_hint_on_best_hint(
            observation,
            HintRankAction(target_player=1, rank=Rank.ONE),
            (
                0,
                1,
                0,
                0,
                0,
                1,
                2,
                0,
                0,
            ),
        )

        self.assertFalse(should_spend)

    def test_convention_tempo_agent_inherits_pressure_relief_priority(self) -> None:
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
                ObservedHand(
                    player_id=4,
                    cards=(
                        Card(Color.BLUE, Rank.THREE),
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
            legal_actions=(DiscardAction(card_index=0),),
        )
        agent = ConventionTempoHeuristicAgent()

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

        self.assertGreater(pressured_hint, stable_hint)

    def test_convention_tempo_agent_prefers_far_hint_that_preserves_coordination_in_five_player_games(self) -> None:
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
                ObservedHand(player_id=4, cards=(Card(Color.WHITE, Rank.ONE),)),
            ),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=40,
            public_history=(),
            legal_actions=(DiscardAction(card_index=0),),
        )
        agent = ConventionTempoHeuristicAgent()

        near_plain_hint = agent._hint_priority(
            observation,
            observation.other_player_hands[0],
            HintRankAction(target_player=1, rank=Rank.ONE),
            (0, 1, 0, 0, 0, 0, 2, 0, 0),
        )
        far_coordination_hint = agent._hint_priority(
            observation,
            observation.other_player_hands[3],
            HintRankAction(target_player=4, rank=Rank.ONE),
            (1, 1, 0, 0, 1, 1, 6, 0, 0),
        )

        self.assertGreater(far_coordination_hint, near_plain_hint)


if __name__ == "__main__":
    unittest.main()
