from __future__ import annotations

import unittest

from hanabi_ai.agents.heuristic.tempo import TempoHeuristicAgent
from hanabi_ai.game.actions import DiscardAction, HintColorAction, HintRankAction
from hanabi_ai.game.cards import Card, Color, Rank
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.game.observation import CardKnowledge, ObservedHand, PlayerObservation
from ._shared import SharedHeuristicAgentTests


class TempoHeuristicAgentTests(SharedHeuristicAgentTests, unittest.TestCase):
    def make_agent(self):
        return TempoHeuristicAgent()

    def test_tempo_agent_discards_instead_of_spending_last_hint_on_pure_information(self) -> None:
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
                    cards=(
                        Card(Color.RED, Rank.TWO),
                        Card(Color.RED, Rank.FIVE),
                        Card(Color.BLUE, Rank.THREE),
                        Card(Color.GREEN, Rank.FOUR),
                        Card(Color.YELLOW, Rank.FIVE),
                    ),
                ),
            ),
            fireworks={
                color: (1 if color == Color.BLUE else 0) for color in Color
            },
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=40,
            public_history=(),
            legal_actions=(
                DiscardAction(card_index=0),
                HintColorAction(target_player=1, color=Color.RED),
                HintRankAction(target_player=1, rank=Rank.FIVE),
            ),
        )
        agent = TempoHeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(action, DiscardAction(card_index=0))

    def test_tempo_agent_still_spends_last_hint_when_it_creates_safe_play(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=81)
        engine.hint_tokens = 1
        engine.hands[1] = [
            Card(Color.BLUE, Rank.ONE),
            Card(Color.RED, Rank.TWO),
            Card(Color.GREEN, Rank.THREE),
            Card(Color.YELLOW, Rank.FOUR),
            Card(Color.WHITE, Rank.FIVE),
        ]
        observation = engine.get_observation(0)
        agent = TempoHeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(action, HintRankAction(target_player=1, rank=Rank.ONE))


if __name__ == "__main__":
    unittest.main()
