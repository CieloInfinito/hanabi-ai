from __future__ import annotations

import unittest

from hanabi_ai.agents.heuristic.basic import BasicHeuristicAgent
from hanabi_ai.agents.heuristic.tempo import TempoHeuristicAgent
from hanabi_ai.game.actions import DiscardAction, HintColorAction, HintRankAction
from hanabi_ai.game.cards import Card, Color, Rank
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.game.observation import (
    CardKnowledge,
    ObservedHand,
    PlayerObservation,
    PublicTurnRecord,
)
from ._shared import SharedBasicHeuristicVariantTests, SharedHeuristicAgentTests


class TempoHeuristicAgentTests(
    SharedBasicHeuristicVariantTests,
    SharedHeuristicAgentTests,
    unittest.TestCase,
):
    def make_agent(self):
        return TempoHeuristicAgent()

    def test_tempo_agent_inherits_basic_player_count_hint_weights(self) -> None:
        basic_agent = BasicHeuristicAgent()
        tempo_agent = TempoHeuristicAgent()

        self.assertEqual(
            tempo_agent._base_hint_priority_weights(2),
            basic_agent._base_hint_priority_weights(2),
        )
        self.assertEqual(
            tempo_agent._base_hint_priority_weights(4),
            basic_agent._base_hint_priority_weights(4),
        )
        self.assertEqual(
            tempo_agent._base_hint_priority_weights(5),
            basic_agent._base_hint_priority_weights(5),
        )

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

    def test_tempo_agent_keeps_last_hint_for_stronger_value_thresholds(self) -> None:
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
            other_player_hands=(),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=40,
            public_history=(),
            legal_actions=(DiscardAction(card_index=0),),
        )
        agent = TempoHeuristicAgent()

        should_spend = agent._should_spend_hint_on_best_hint(
            observation,
            HintRankAction(target_player=1, rank=Rank.ONE),
            (
                0,  # guaranteed_play_hits
                1,  # playable_hits
                0,
                0,
                0,  # critical_playable_hits
                1,  # useful_hits
                2,  # information_gain
                0,  # critical_useful_hits
                0,
            ),
        )

        self.assertFalse(should_spend)

    def test_tempo_agent_is_less_conservative_with_last_hint_in_three_player_games(self) -> None:
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
                ObservedHand(
                    player_id=2,
                    cards=(Card(Color.GREEN, Rank.FOUR),),
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
        agent = TempoHeuristicAgent()

        should_spend = agent._should_spend_hint_on_best_hint(
            observation,
            HintRankAction(target_player=1, rank=Rank.ONE),
            (
                0,  # guaranteed_play_hits
                1,  # playable_hits
                0,
                0,
                0,  # critical_playable_hits
                1,  # useful_hits
                3,  # information_gain
                0,  # critical_useful_hits
                0,
            ),
        )

        self.assertTrue(should_spend)

    def test_tempo_agent_prefers_next_players_actionable_hint_in_four_player_games(self) -> None:
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
                ObservedHand(player_id=1, cards=(Card(Color.RED, Rank.ONE),)),
                ObservedHand(player_id=2, cards=(Card(Color.GREEN, Rank.ONE),)),
                ObservedHand(player_id=3, cards=(Card(Color.YELLOW, Rank.THREE),)),
            ),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=40,
            public_history=(),
            legal_actions=(DiscardAction(card_index=0),),
        )
        agent = TempoHeuristicAgent()

        next_player_priority = agent._tempo_hint_priority(
            observation,
            observation.other_player_hands[0],
            HintRankAction(target_player=1, rank=Rank.ONE),
            (0, 1, 0, 0, 0, 1, 3, 0, 0),
        )
        farther_player_priority = agent._tempo_hint_priority(
            observation,
            observation.other_player_hands[1],
            HintRankAction(target_player=2, rank=Rank.ONE),
            (0, 1, 0, 0, 0, 1, 3, 0, 0),
        )

        self.assertGreater(next_player_priority, farther_player_priority)

    def test_tempo_agent_spends_last_hint_for_next_players_playable_card_in_four_player_games(self) -> None:
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
                ObservedHand(player_id=1, cards=(Card(Color.RED, Rank.ONE),)),
                ObservedHand(player_id=2, cards=(Card(Color.GREEN, Rank.ONE),)),
                ObservedHand(player_id=3, cards=(Card(Color.YELLOW, Rank.THREE),)),
            ),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=40,
            public_history=(),
            legal_actions=(DiscardAction(card_index=0),),
        )
        agent = TempoHeuristicAgent()

        should_spend = agent._should_spend_hint_on_best_hint(
            observation,
            HintRankAction(target_player=1, rank=Rank.ONE),
            (0, 1, 0, 0, 0, 1, 1, 0, 0),
        )

        self.assertTrue(should_spend)

    def test_tempo_agent_spends_last_hint_for_near_term_chain_value_in_five_player_games(self) -> None:
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
                ObservedHand(player_id=1, cards=(Card(Color.RED, Rank.THREE),)),
                ObservedHand(player_id=2, cards=(Card(Color.GREEN, Rank.FOUR),)),
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
        agent = TempoHeuristicAgent()

        should_spend = agent._should_spend_hint_on_best_hint(
            observation,
            HintColorAction(target_player=2, color=Color.GREEN),
            (0, 1, 0, 0, 0, 2, 4, 0, 0),
        )

        self.assertTrue(should_spend)

    def test_tempo_agent_detects_receiver_needing_help_when_no_safe_play_and_little_information(self) -> None:
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(
                CardKnowledge(
                    possible_colors=frozenset({Color.BLUE}),
                    possible_ranks=frozenset({Rank.ONE}),
                ),
            ),
            other_player_hands=(
                ObservedHand(
                    player_id=1,
                    cards=(
                        Card(Color.RED, Rank.THREE),
                        Card(Color.GREEN, Rank.FOUR),
                        Card(Color.YELLOW, Rank.FIVE),
                        Card(Color.WHITE, Rank.THREE),
                    ),
                ),
                ObservedHand(
                    player_id=2,
                    cards=(
                        Card(Color.RED, Rank.ONE),
                        Card(Color.GREEN, Rank.THREE),
                        Card(Color.YELLOW, Rank.FOUR),
                        Card(Color.WHITE, Rank.FIVE),
                    ),
                ),
                ObservedHand(
                    player_id=3,
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
        agent = TempoHeuristicAgent()

        needs_help = agent._receiver_needs_help(observation, 1)
        still_needs_help = agent._receiver_needs_help(observation, 2)

        self.assertTrue(needs_help)
        self.assertFalse(still_needs_help)

    def test_tempo_agent_prioritizes_useful_hint_for_stuck_near_term_receiver(self) -> None:
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(
                CardKnowledge(
                    possible_colors=frozenset({Color.BLUE}),
                    possible_ranks=frozenset({Rank.ONE}),
                ),
            ),
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
                        Card(Color.BLUE, Rank.THREE),
                        Card(Color.GREEN, Rank.FOUR),
                        Card(Color.YELLOW, Rank.FIVE),
                        Card(Color.WHITE, Rank.THREE),
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
            ),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=40,
            public_history=(),
            legal_actions=(DiscardAction(card_index=0),),
        )
        agent = TempoHeuristicAgent()

        stuck_receiver_priority = agent._tempo_hint_priority(
            observation,
            observation.other_player_hands[0],
            HintColorAction(target_player=1, color=Color.RED),
            (0, 1, 0, 0, 0, 2, 3, 0, 0),
        )
        non_stuck_receiver_priority = agent._tempo_hint_priority(
            observation,
            observation.other_player_hands[1],
            HintColorAction(target_player=2, color=Color.BLUE),
            (0, 1, 0, 0, 0, 2, 3, 0, 0),
        )

        self.assertGreater(stuck_receiver_priority, non_stuck_receiver_priority)

    def test_tempo_agent_detects_follow_on_play_value_from_visible_chain(self) -> None:
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
                        Card(Color.RED, Rank.ONE),
                        Card(Color.GREEN, Rank.FOUR),
                        Card(Color.YELLOW, Rank.FIVE),
                        Card(Color.WHITE, Rank.THREE),
                    ),
                ),
                ObservedHand(
                    player_id=2,
                    cards=(
                        Card(Color.RED, Rank.TWO),
                        Card(Color.BLUE, Rank.THREE),
                        Card(Color.YELLOW, Rank.FOUR),
                        Card(Color.WHITE, Rank.FIVE),
                    ),
                ),
                ObservedHand(
                    player_id=3,
                    cards=(
                        Card(Color.RED, Rank.TWO),
                        Card(Color.GREEN, Rank.THREE),
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
            public_history=(),
            legal_actions=(DiscardAction(card_index=0),),
        )
        agent = TempoHeuristicAgent()

        follow_on_value = agent._follow_on_play_value(
            observation,
            1,
            HintRankAction(target_player=1, rank=Rank.ONE),
        )

        self.assertEqual(follow_on_value, 1)

    def test_tempo_agent_prioritizes_hint_with_follow_on_chain_value(self) -> None:
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
                        Card(Color.RED, Rank.ONE),
                        Card(Color.GREEN, Rank.FOUR),
                        Card(Color.YELLOW, Rank.FIVE),
                        Card(Color.WHITE, Rank.THREE),
                    ),
                ),
                ObservedHand(
                    player_id=2,
                    cards=(
                        Card(Color.RED, Rank.TWO),
                        Card(Color.BLUE, Rank.THREE),
                        Card(Color.YELLOW, Rank.FOUR),
                        Card(Color.WHITE, Rank.FIVE),
                    ),
                ),
                ObservedHand(
                    player_id=3,
                    cards=(
                        Card(Color.BLUE, Rank.ONE),
                        Card(Color.GREEN, Rank.THREE),
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
            public_history=(),
            legal_actions=(DiscardAction(card_index=0),),
        )
        agent = TempoHeuristicAgent()

        chain_priority = agent._tempo_hint_priority(
            observation,
            observation.other_player_hands[0],
            HintRankAction(target_player=1, rank=Rank.ONE),
            (0, 1, 0, 0, 0, 1, 2, 0, 0),
        )
        no_chain_priority = agent._tempo_hint_priority(
            observation,
            observation.other_player_hands[2],
            HintRankAction(target_player=3, rank=Rank.ONE),
            (0, 1, 0, 0, 0, 1, 2, 0, 0),
        )

        self.assertGreater(chain_priority, no_chain_priority)

    def test_tempo_agent_gives_extra_weight_to_follow_on_value_in_five_player_games(self) -> None:
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
                        Card(Color.RED, Rank.ONE),
                        Card(Color.GREEN, Rank.FOUR),
                        Card(Color.YELLOW, Rank.FIVE),
                        Card(Color.WHITE, Rank.THREE),
                    ),
                ),
                ObservedHand(
                    player_id=2,
                    cards=(
                        Card(Color.RED, Rank.TWO),
                        Card(Color.BLUE, Rank.THREE),
                        Card(Color.YELLOW, Rank.FOUR),
                        Card(Color.WHITE, Rank.FIVE),
                    ),
                ),
                ObservedHand(
                    player_id=3,
                    cards=(
                        Card(Color.BLUE, Rank.ONE),
                        Card(Color.GREEN, Rank.THREE),
                        Card(Color.YELLOW, Rank.FIVE),
                        Card(Color.WHITE, Rank.THREE),
                    ),
                ),
                ObservedHand(
                    player_id=4,
                    cards=(
                        Card(Color.GREEN, Rank.TWO),
                        Card(Color.YELLOW, Rank.THREE),
                        Card(Color.WHITE, Rank.FIVE),
                        Card(Color.BLUE, Rank.FOUR),
                    ),
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
        agent = TempoHeuristicAgent()

        higher_chain_priority = agent._tempo_hint_priority(
            observation,
            observation.other_player_hands[0],
            HintRankAction(target_player=1, rank=Rank.ONE),
            (0, 1, 0, 0, 0, 1, 2, 0, 0),
        )
        lower_chain_priority = agent._tempo_hint_priority(
            observation,
            observation.other_player_hands[2],
            HintRankAction(target_player=3, rank=Rank.ONE),
            (0, 1, 0, 0, 0, 1, 2, 0, 0),
        )

        self.assertGreater(higher_chain_priority, lower_chain_priority)


if __name__ == "__main__":
    unittest.main()
