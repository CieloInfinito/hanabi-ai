from __future__ import annotations

import unittest

from hanabi_ai.agents.rl.encoding import LegalActionIndexer, ObservationVectorEncoder
from hanabi_ai.game.actions import HintColorAction, PlayAction
from hanabi_ai.game.engine import HanabiGameEngine


class ObservationVectorEncoderTests(unittest.TestCase):
    def test_encoder_returns_fixed_feature_size(self) -> None:
        engine = HanabiGameEngine(player_count=3, seed=7)
        observation = engine.get_observation(0)
        encoder = ObservationVectorEncoder(player_count=3)

        features = encoder.encode(observation)

        self.assertEqual(len(features), encoder.feature_size)
        self.assertTrue(any(value != 0.0 for value in features))

    def test_action_indexer_round_trips_relative_hint_targets(self) -> None:
        engine = HanabiGameEngine(player_count=3, seed=11)
        observation = engine.get_observation(0)
        indexer = LegalActionIndexer(player_count=3)
        hint_action = next(
            action
            for action in observation.legal_actions
            if isinstance(action, HintColorAction)
        )

        action_index = indexer.action_index_for_action(
            hint_action,
            current_player=observation.current_player,
        )
        decoded_action = indexer.action_for_index(
            action_index,
            current_player=observation.current_player,
        )

        self.assertEqual(decoded_action, hint_action)

    def test_action_indexer_includes_play_actions_in_legal_indices(self) -> None:
        engine = HanabiGameEngine(player_count=2, seed=5)
        observation = engine.get_observation(0)
        indexer = LegalActionIndexer(player_count=2)

        legal_action_indices = indexer.legal_action_indices(observation)
        play_index = indexer.action_index_for_action(
            PlayAction(card_index=0),
            current_player=observation.current_player,
        )

        self.assertIn(play_index, legal_action_indices)


if __name__ == "__main__":
    unittest.main()
