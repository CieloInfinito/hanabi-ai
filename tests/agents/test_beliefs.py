from __future__ import annotations

import unittest

from hanabi_ai.agents.beliefs import PublicBeliefState
from hanabi_ai.game.actions import DiscardAction, HintColorAction, HintRankAction
from hanabi_ai.game.cards import Card, Color, Rank
from hanabi_ai.game.engine import HanabiGameEngine


class PublicBeliefStateTests(unittest.TestCase):
    def test_builds_public_knowledge_for_other_player_from_history(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=71)
        engine.hands[1] = [
            Card(Color.RED, Rank.ONE),
            Card(Color.BLUE, Rank.THREE),
            Card(Color.GREEN, Rank.TWO),
            Card(Color.YELLOW, Rank.FOUR),
            Card(Color.WHITE, Rank.FIVE),
        ]

        engine.step(HintColorAction(target_player=1, color=Color.RED))
        engine.step(DiscardAction(card_index=4))
        engine.step(HintRankAction(target_player=1, rank=Rank.THREE))
        observation = engine.get_observation(0)

        beliefs = PublicBeliefState.from_observation(observation)
        knowledge = beliefs.knowledge_for_player(1)

        self.assertEqual(knowledge[0].possible_colors, frozenset({Color.RED}))
        self.assertEqual(knowledge[1].possible_ranks, frozenset({Rank.THREE}))
        self.assertNotIn(Color.RED, knowledge[2].possible_colors)
        self.assertNotIn(Rank.THREE, knowledge[2].possible_ranks)

    def test_tracks_hand_shift_after_public_discard(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=72)
        engine.hands[1] = [
            Card(Color.RED, Rank.ONE),
            Card(Color.BLUE, Rank.THREE),
            Card(Color.GREEN, Rank.TWO),
            Card(Color.YELLOW, Rank.FOUR),
            Card(Color.WHITE, Rank.FIVE),
        ]

        engine.step(HintColorAction(target_player=1, color=Color.RED))
        engine.step(DiscardAction(card_index=0))
        observation = engine.get_observation(0)

        beliefs = PublicBeliefState.from_observation(observation)
        knowledge = beliefs.knowledge_for_player(1)

        self.assertNotEqual(knowledge[0].possible_colors, frozenset({Color.RED}))
        self.assertEqual(len(knowledge), len(observation.other_player_hands[0].cards))

    def test_reuses_weighted_distributions_for_observer_hand(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=73)
        engine.discard_pile.append(Card(Color.RED, Rank.TWO))
        engine.hands[1] = [
            Card(Color.WHITE, Rank.FIVE),
            Card(Color.GREEN, Rank.ONE),
            Card(Color.YELLOW, Rank.THREE),
            Card(Color.BLUE, Rank.ONE),
            Card(Color.GREEN, Rank.FOUR),
        ]
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED, Color.BLUE}),
            possible_ranks=frozenset({Rank.TWO}),
            hinted_rank=Rank.TWO,
        )

        observation = engine.get_observation(0)
        beliefs = PublicBeliefState.from_observation(observation)
        distribution = dict(beliefs.distribution_for_card(0, 0))

        self.assertAlmostEqual(distribution[Card(Color.RED, Rank.TWO)], 1 / 3)
        self.assertAlmostEqual(distribution[Card(Color.BLUE, Rank.TWO)], 2 / 3)

    def test_updates_public_knowledge_after_hint(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=74)
        engine.hands[1] = [
            Card(Color.RED, Rank.ONE),
            Card(Color.BLUE, Rank.THREE),
            Card(Color.GREEN, Rank.TWO),
            Card(Color.YELLOW, Rank.THREE),
            Card(Color.WHITE, Rank.FIVE),
        ]

        observation = engine.get_observation(0)
        beliefs = PublicBeliefState.from_observation(observation)
        updated_knowledge, revealed_indices = beliefs.updated_public_knowledge_after_hint(
            1,
            observation.other_player_hands[0].cards,
            HintRankAction(target_player=1, rank=Rank.THREE),
        )

        self.assertEqual(revealed_indices, (1, 3))
        self.assertEqual(updated_knowledge[1].possible_ranks, frozenset({Rank.THREE}))
        self.assertEqual(updated_knowledge[3].possible_ranks, frozenset({Rank.THREE}))
        self.assertNotIn(Rank.THREE, updated_knowledge[0].possible_ranks)


if __name__ == "__main__":
    unittest.main()
