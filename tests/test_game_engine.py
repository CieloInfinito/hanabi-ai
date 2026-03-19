from __future__ import annotations

import unittest

import _path_setup
from card_game_ai.game.actions import DiscardAction, HintColorAction, PlayAction
from card_game_ai.game.cards import Card, Color, MAX_HINT_TOKENS, Rank
from card_game_ai.game.engine import HanabiGameEngine


class HanabiGameEngineTests(unittest.TestCase):
    def test_reset_deals_correct_initial_hands(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=1)

        self.assertEqual(engine.current_player, 0)
        self.assertEqual(len(engine.hands), 2)
        self.assertEqual(len(engine.hands[0]), 5)
        self.assertEqual(len(engine.hands[1]), 5)
        self.assertEqual(len(engine.deck), 40)
        self.assertEqual(engine.hint_tokens, MAX_HINT_TOKENS)
        self.assertEqual(engine.strike_tokens, 0)

    def test_hand_size_depends_on_player_count(self) -> None:
        two_player_engine = HanabiGameEngine(player_count=2, seed=10)
        three_player_engine = HanabiGameEngine(player_count=3, seed=11)
        four_player_engine = HanabiGameEngine(player_count=4, seed=12)
        five_player_engine = HanabiGameEngine(player_count=5, seed=13)

        self.assertTrue(all(len(hand) == 5 for hand in two_player_engine.hands))
        self.assertTrue(all(len(hand) == 5 for hand in three_player_engine.hands))
        self.assertTrue(all(len(hand) == 4 for hand in four_player_engine.hands))
        self.assertTrue(all(len(hand) == 4 for hand in five_player_engine.hands))

    def test_cannot_give_hint_without_hint_tokens(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=2)
        engine.hint_tokens = 0

        legal_actions = engine.get_legal_actions(0)

        self.assertFalse(any(isinstance(action, HintColorAction) for action in legal_actions))

    def test_cannot_discard_when_hint_tokens_are_full(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=3)

        legal_actions = engine.get_legal_actions(0)

        self.assertFalse(any(isinstance(action, DiscardAction) for action in legal_actions))

    def test_playing_incorrect_card_adds_strike_and_discards_card(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=4)
        engine.hands[0][0] = Card(Color.BLUE, Rank.THREE)

        result = engine.step(PlayAction(card_index=0))

        self.assertFalse(result.play_succeeded)
        self.assertEqual(engine.strike_tokens, 1)
        self.assertIn(Card(Color.BLUE, Rank.THREE), engine.discard_pile)
        self.assertEqual(engine.fireworks[Color.BLUE], 0)

    def test_playing_correct_card_advances_fireworks(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=5)
        engine.hands[0][0] = Card(Color.RED, Rank.ONE)

        result = engine.step(PlayAction(card_index=0))

        self.assertTrue(result.play_succeeded)
        self.assertEqual(engine.fireworks[Color.RED], 1)
        self.assertEqual(engine.get_score(), 1)

    def test_observation_hides_real_cards_from_observing_player(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=6)

        observation = engine.get_observation(0)

        self.assertEqual(len(observation.hand_knowledge), 5)
        self.assertEqual(len(observation.other_player_hands), 1)
        self.assertEqual(observation.other_player_hands[0].player_id, 1)
        self.assertFalse(hasattr(observation.hand_knowledge[0], "color"))
        self.assertFalse(hasattr(observation.hand_knowledge[0], "rank"))


if __name__ == "__main__":
    unittest.main()
