from __future__ import annotations

import unittest
from collections import Counter

from hanabi_ai.game.actions import HintColorAction, PlayAction
from hanabi_ai.game.cards import Card, Color, Rank, build_standard_deck
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.search.determinization import (
    build_determinized_engine_state,
    sample_compatible_world,
    sample_compatible_worlds,
)


class DeterminizationTests(unittest.TestCase):
    def test_sample_compatible_world_respects_visible_state_and_knowledge(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=70)
        engine.step(HintColorAction(target_player=1, color=engine.hands[1][0].color))
        engine.step(PlayAction(card_index=0))
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED}),
            possible_ranks=engine.knowledge_by_player[0][0].possible_ranks,
            hinted_color=Color.RED,
            hinted_rank=engine.knowledge_by_player[0][0].hinted_rank,
        )

        observation = engine.get_observation(0)
        world = sample_compatible_world(observation, seed=5)

        self.assertEqual(world.engine.current_player, observation.current_player)
        self.assertEqual(tuple(world.engine.hands[1]), observation.other_player_hands[0].cards)
        self.assertEqual(world.engine.get_score(), sum(observation.fireworks.values()))
        self.assertEqual(len(world.engine.deck), observation.deck_size)
        self.assertEqual(world.sampled_own_hand[0].color, Color.RED)
        self.assertEqual(len(world.engine.history), len(observation.public_history))

    def test_build_determinized_engine_state_preserves_full_card_multiset(self) -> None:
        engine = HanabiGameEngine(player_count=3, seed=71)
        observation = engine.get_observation(0)
        sampled_own_hand = tuple(engine.hands[0])

        state = build_determinized_engine_state(observation, sampled_own_hand)
        all_cards = [card for hand in state.hands for card in hand]
        all_cards.extend(state.deck)
        all_cards.extend(state.discard_pile)
        for color, highest_rank in state.fireworks.items():
            for rank_value in range(1, highest_rank + 1):
                all_cards.append(Card(color=color, rank=Rank(rank_value)))

        self.assertEqual(Counter(all_cards), Counter(build_standard_deck()))

    def test_sample_compatible_worlds_returns_requested_number_of_worlds(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=72)
        observation = engine.get_observation(0)

        worlds = sample_compatible_worlds(observation, 3, seed=9)

        self.assertEqual(len(worlds), 3)
        self.assertTrue(all(len(world.engine.deck) == observation.deck_size for world in worlds))


if __name__ == "__main__":
    unittest.main()
