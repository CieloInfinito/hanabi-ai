from __future__ import annotations

import unittest

import _path_setup
from card_game_ai.game.actions import DiscardAction, HintColorAction, PlayAction
from card_game_ai.game.cards import Card, Color, MAX_HINT_TOKENS, Rank
from card_game_ai.game.engine import HanabiGameEngine
from card_game_ai.game.observation import get_definitely_playable_card_indices


class HanabiGameEngineTests(unittest.TestCase):
    def test_reset_deals_correct_initial_hands(self) -> None:
        # Verifies that a fresh reset creates the standard two-player opening state.
        engine = HanabiGameEngine(player_count=2, seed=1)

        self.assertEqual(engine.current_player, 0)
        self.assertEqual(len(engine.hands), 2)
        self.assertEqual(len(engine.hands[0]), 5)
        self.assertEqual(len(engine.hands[1]), 5)
        self.assertEqual(len(engine.deck), 40)
        self.assertEqual(engine.hint_tokens, MAX_HINT_TOKENS)
        self.assertEqual(engine.strike_tokens, 0)

    def test_hand_size_depends_on_player_count(self) -> None:
        # Verifies that hand size follows standard Hanabi rules for 2 to 5 players.
        two_player_engine = HanabiGameEngine(player_count=2, seed=10)
        three_player_engine = HanabiGameEngine(player_count=3, seed=11)
        four_player_engine = HanabiGameEngine(player_count=4, seed=12)
        five_player_engine = HanabiGameEngine(player_count=5, seed=13)

        self.assertTrue(all(len(hand) == 5 for hand in two_player_engine.hands))
        self.assertTrue(all(len(hand) == 5 for hand in three_player_engine.hands))
        self.assertTrue(all(len(hand) == 4 for hand in four_player_engine.hands))
        self.assertTrue(all(len(hand) == 4 for hand in five_player_engine.hands))

    def test_cannot_give_hint_without_hint_tokens(self) -> None:
        # Verifies that hint actions disappear when no hint tokens remain.
        engine = HanabiGameEngine(player_count=2, seed=2)
        engine.hint_tokens = 0

        legal_actions = engine.get_legal_actions(0)

        self.assertFalse(any(isinstance(action, HintColorAction) for action in legal_actions))

    def test_cannot_discard_when_hint_tokens_are_full(self) -> None:
        # Verifies that discard actions are illegal while hint tokens are already full.
        engine = HanabiGameEngine(player_count=2, seed=3)

        legal_actions = engine.get_legal_actions(0)

        self.assertFalse(any(isinstance(action, DiscardAction) for action in legal_actions))

    def test_playing_incorrect_card_adds_strike_and_discards_card(self) -> None:
        # Verifies that an incorrect play costs a strike and sends the card to discards.
        engine = HanabiGameEngine(player_count=2, seed=4)
        engine.hands[0][0] = Card(Color.BLUE, Rank.THREE)

        result = engine.step(PlayAction(card_index=0))

        self.assertFalse(result.play_succeeded)
        self.assertEqual(engine.strike_tokens, 1)
        self.assertIn(Card(Color.BLUE, Rank.THREE), engine.discard_pile)
        self.assertEqual(engine.fireworks[Color.BLUE], 0)

    def test_game_is_lost_when_shared_lives_reach_zero(self) -> None:
        # Verifies that the game ends immediately once the shared strike limit is reached.
        engine = HanabiGameEngine(player_count=2, seed=14)
        engine.strike_tokens = 2
        engine.hands[0][0] = Card(Color.GREEN, Rank.THREE)

        result = engine.step(PlayAction(card_index=0))

        self.assertFalse(result.play_succeeded)
        self.assertEqual(engine.strike_tokens, 3)
        self.assertTrue(engine.is_terminal())
        self.assertTrue(result.game_over)

    def test_playing_correct_card_advances_fireworks(self) -> None:
        # Verifies that a correct play advances the matching firework stack and score.
        engine = HanabiGameEngine(player_count=2, seed=5)
        engine.hands[0][0] = Card(Color.RED, Rank.ONE)

        result = engine.step(PlayAction(card_index=0))

        self.assertTrue(result.play_succeeded)
        self.assertEqual(engine.fireworks[Color.RED], 1)
        self.assertEqual(engine.get_score(), 1)

    def test_observation_hides_real_cards_from_observing_player(self) -> None:
        # Verifies that player observations expose knowledge, not the observer's real cards.
        engine = HanabiGameEngine(player_count=2, seed=6)

        observation = engine.get_observation(0)

        self.assertEqual(len(observation.hand_knowledge), 5)
        self.assertEqual(len(observation.other_player_hands), 1)
        self.assertEqual(observation.other_player_hands[0].player_id, 1)
        self.assertFalse(hasattr(observation.hand_knowledge[0], "color"))
        self.assertFalse(hasattr(observation.hand_knowledge[0], "rank"))

    def test_definitely_playable_indices_use_partial_knowledge_only(self) -> None:
        # Verifies that definitely-playable detection depends only on public knowledge.
        engine = HanabiGameEngine(player_count=2, seed=15)
        engine.fireworks[Color.RED] = 1
        engine.fireworks[Color.BLUE] = 2

        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED}),
            possible_ranks=frozenset({Rank.TWO}),
            hinted_color=Color.RED,
            hinted_rank=Rank.TWO,
        )
        engine.knowledge_by_player[0][1] = engine.knowledge_by_player[0][1].__class__(
            possible_colors=frozenset({Color.RED, Color.BLUE}),
            possible_ranks=frozenset({Rank.TWO, Rank.THREE}),
            hinted_color=None,
            hinted_rank=None,
        )

        observation = engine.get_observation(0)
        definitely_playable = get_definitely_playable_card_indices(
            observation.hand_knowledge, observation.fireworks
        )

        self.assertEqual(definitely_playable, (0,))


if __name__ == "__main__":
    unittest.main()
