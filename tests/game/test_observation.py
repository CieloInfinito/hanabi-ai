from __future__ import annotations

import unittest

from hanabi_ai.game.cards import Card, Color, Rank
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.game.observation import (
    build_remaining_card_counts,
    build_visible_card_counts,
    estimate_card_distribution,
)


class ObservationInferenceTests(unittest.TestCase):
    def test_build_visible_card_counts_includes_discards_hands_and_fireworks(self) -> None:
        # Verifies that public visibility counts cover all public zones that
        # should reduce the observer's hidden-card possibilities.
        engine = HanabiGameEngine(player_count=2, seed=61)
        engine.fireworks[Color.RED] = 2
        engine.discard_pile.extend(
            [Card(Color.BLUE, Rank.THREE), Card(Color.BLUE, Rank.THREE)]
        )
        engine.hands[1] = [
            Card(Color.WHITE, Rank.FIVE),
            Card(Color.GREEN, Rank.ONE),
            Card(Color.YELLOW, Rank.TWO),
            Card(Color.BLUE, Rank.ONE),
            Card(Color.GREEN, Rank.THREE),
        ]

        observation = engine.get_observation(0)
        visible_counts = build_visible_card_counts(observation)

        self.assertEqual(visible_counts[Card(Color.RED, Rank.ONE)], 1)
        self.assertEqual(visible_counts[Card(Color.RED, Rank.TWO)], 1)
        self.assertEqual(visible_counts[Card(Color.BLUE, Rank.THREE)], 2)
        self.assertEqual(visible_counts[Card(Color.WHITE, Rank.FIVE)], 1)

    def test_build_remaining_card_counts_removes_exhausted_public_copies(self) -> None:
        # Verifies that remaining-copy counts drop to zero once every public
        # copy of a compatible card is already visible elsewhere.
        engine = HanabiGameEngine(player_count=2, seed=62)
        engine.hands[1] = [
            Card(Color.RED, Rank.ONE),
            Card(Color.RED, Rank.ONE),
            Card(Color.RED, Rank.ONE),
            Card(Color.GREEN, Rank.THREE),
            Card(Color.WHITE, Rank.FIVE),
        ]

        observation = engine.get_observation(0)
        remaining_counts = build_remaining_card_counts(observation)

        self.assertEqual(remaining_counts[Card(Color.RED, Rank.ONE)], 0)
        self.assertEqual(remaining_counts[Card(Color.BLUE, Rank.ONE)], 3)

    def test_estimate_card_distribution_filters_exhausted_candidates(self) -> None:
        # Verifies that exhausted possibilities disappear from the estimated
        # hidden-card distribution instead of receiving non-zero weight.
        engine = HanabiGameEngine(player_count=2, seed=63)
        engine.hands[1] = [
            Card(Color.RED, Rank.ONE),
            Card(Color.RED, Rank.ONE),
            Card(Color.RED, Rank.ONE),
            Card(Color.GREEN, Rank.THREE),
            Card(Color.WHITE, Rank.FIVE),
        ]
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED, Color.BLUE}),
            possible_ranks=frozenset({Rank.ONE}),
            hinted_rank=Rank.ONE,
        )

        observation = engine.get_observation(0)
        distribution = estimate_card_distribution(
            observation.hand_knowledge[0],
            observation,
        )

        self.assertEqual(distribution, ((Card(Color.BLUE, Rank.ONE), 1.0),))

    def test_estimate_card_distribution_weights_by_remaining_copies(self) -> None:
        # Verifies that the estimator gives proportionally more weight to card
        # identities with more unseen copies still available.
        engine = HanabiGameEngine(player_count=2, seed=64)
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
        distribution = dict(
            estimate_card_distribution(observation.hand_knowledge[0], observation)
        )

        self.assertAlmostEqual(distribution[Card(Color.RED, Rank.TWO)], 1 / 3)
        self.assertAlmostEqual(distribution[Card(Color.BLUE, Rank.TWO)], 2 / 3)


if __name__ == "__main__":
    unittest.main()
