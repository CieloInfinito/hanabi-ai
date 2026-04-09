from __future__ import annotations

import unittest
from unittest.mock import patch

from hanabi_ai.game.actions import DiscardAction, HintColorAction, HintRankAction
from hanabi_ai.game.cards import Card, Color, Rank
from hanabi_ai.game.engine import HanabiGameEngine, TurnRecord
from hanabi_ai.search.hint_value import evaluate_hint_value
from hanabi_ai.search.planner import ShortHorizonPlanner
from hanabi_ai.search.rollout import RolloutSummary


def _build_two_player_observation_with_public_red_hint():
    engine = HanabiGameEngine(player_count=2, seed=70)
    engine.current_player = 0
    engine.hint_tokens = 1
    engine.hands[1] = [
        Card(Color.RED, Rank.ONE),
        Card(Color.RED, Rank.FIVE),
        Card(Color.GREEN, Rank.THREE),
        Card(Color.YELLOW, Rank.FOUR),
        Card(Color.WHITE, Rank.FIVE),
    ]
    engine.history = [
        TurnRecord(
            player_id=0,
            action=HintRankAction(target_player=1, rank=Rank.FIVE),
            revealed_indices=(1, 4),
            revealed_groups=((1,), (4,)),
            fireworks_before=dict(engine.fireworks),
            drew_replacement=False,
        ),
        TurnRecord(
            player_id=1,
            action=HintRankAction(target_player=0, rank=Rank.ONE),
            revealed_indices=(),
            revealed_groups=(),
            fireworks_before=dict(engine.fireworks),
            drew_replacement=False,
        ),
        TurnRecord(
            player_id=0,
            action=HintColorAction(target_player=1, color=Color.RED),
            revealed_indices=(0, 1),
            revealed_groups=((0,), (1,)),
            fireworks_before=dict(engine.fireworks),
            drew_replacement=False,
        ),
    ]
    engine.turn_number = len(engine.history)
    return engine.get_observation(0)


class HintValueTests(unittest.TestCase):
    def test_evaluate_hint_value_rewards_hint_that_creates_guaranteed_play(self) -> None:
        observation = _build_two_player_observation_with_public_red_hint()

        playable_hint = evaluate_hint_value(
            observation,
            HintRankAction(target_player=1, rank=Rank.ONE),
        )
        noisier_hint = evaluate_hint_value(
            observation,
            HintRankAction(target_player=1, rank=Rank.FIVE),
        )

        self.assertGreater(playable_hint.guaranteed_play_delta, 0)
        self.assertGreater(playable_hint.total_value, noisier_hint.total_value)

    def test_planner_can_prefer_hint_over_discard_when_static_hint_value_is_strong(self) -> None:
        observation = _build_two_player_observation_with_public_red_hint()
        planner = ShortHorizonPlanner(world_samples=2, depth=1, top_k=2)
        flat_rollout = RolloutSummary(
            final_score=0,
            score_delta=0,
            strikes_used=0,
            leaf_value=0.0,
            terminated=False,
        )

        with patch(
            "hanabi_ai.search.planner.evaluate_action_rollout",
            return_value=flat_rollout,
        ):
            ranked_actions = planner.rank_actions(
                observation,
                prioritized_actions=[
                    DiscardAction(card_index=0),
                    HintRankAction(target_player=1, rank=Rank.ONE),
                ],
                seed=1,
            )

        self.assertEqual(
            ranked_actions[0].action,
            HintRankAction(target_player=1, rank=Rank.ONE),
        )


if __name__ == "__main__":
    unittest.main()
