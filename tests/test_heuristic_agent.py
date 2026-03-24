from __future__ import annotations

import unittest

import _path_setup
from card_game_ai.agents.heuristic_agent import HeuristicAgent
from card_game_ai.game.actions import DiscardAction, HintColorAction, HintRankAction, PlayAction
from card_game_ai.game.cards import Card, Color, Rank
from card_game_ai.game.engine import HanabiGameEngine


class HeuristicAgentTests(unittest.TestCase):
    def test_heuristic_agent_plays_definitely_playable_card(self) -> None:
        # Verifies that the agent plays one of its own cards when partial
        # knowledge guarantees that the card is currently playable.
        engine = HanabiGameEngine(player_count=2, seed=31)
        engine.fireworks[Color.RED] = 1
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED}),
            possible_ranks=frozenset({Rank.TWO}),
            hinted_color=Color.RED,
            hinted_rank=Rank.TWO,
        )
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(action, PlayAction(card_index=0))

    def test_heuristic_agent_gives_hint_for_other_players_playable_card(self) -> None:
        # Verifies that, when it has no safe self-play, the agent prioritizes
        # a hint that enables a teammate to play a visible card immediately.
        engine = HanabiGameEngine(player_count=2, seed=32)
        engine.hands[1] = [
            Card(Color.BLUE, Rank.ONE),
            Card(Color.RED, Rank.TWO),
            Card(Color.GREEN, Rank.THREE),
            Card(Color.YELLOW, Rank.FOUR),
            Card(Color.WHITE, Rank.FIVE),
        ]
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(action, HintColorAction(target_player=1, color=Color.BLUE))

    def test_heuristic_agent_prefers_hint_touching_more_playable_cards(self) -> None:
        # Verifies that, among multiple legal hints, the agent chooses the one
        # that makes more immediately useful cards stand out, here two playable ones.
        engine = HanabiGameEngine(player_count=2, seed=34)
        engine.hands[1] = [
            Card(Color.RED, Rank.ONE),
            Card(Color.BLUE, Rank.ONE),
            Card(Color.GREEN, Rank.THREE),
            Card(Color.YELLOW, Rank.FOUR),
            Card(Color.WHITE, Rank.FIVE),
        ]
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(action, HintRankAction(target_player=1, rank=Rank.ONE))

    def test_heuristic_agent_discards_definitely_useless_card(self) -> None:
        # Verifies that the agent discards one of its own cards when it knows
        # that the card is already fully obsolete on the fireworks stacks.
        engine = HanabiGameEngine(player_count=2, seed=35)
        engine.hint_tokens = 0
        engine.fireworks[Color.RED] = 2
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED}),
            possible_ranks=frozenset({Rank.ONE}),
            hinted_color=Color.RED,
            hinted_rank=Rank.ONE,
        )
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(action, DiscardAction(card_index=0))

    def test_heuristic_agent_uses_other_hands_to_identify_safe_play(self) -> None:
        # Verifies that the agent uses visible teammate hands to eliminate
        # incompatible possibilities and upgrade a candidate play to a safe play.
        engine = HanabiGameEngine(player_count=2, seed=39)
        engine.hands[1] = [
            Card(Color.RED, Rank.ONE),
            Card(Color.RED, Rank.ONE),
            Card(Color.RED, Rank.ONE),
            Card(Color.GREEN, Rank.TWO),
            Card(Color.YELLOW, Rank.THREE),
        ]
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED, Color.BLUE}),
            possible_ranks=frozenset({Rank.ONE}),
            hinted_rank=Rank.ONE,
        )
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(action, PlayAction(card_index=0))

    def test_heuristic_agent_uses_other_hands_to_identify_safe_discard(self) -> None:
        # Verifies that the agent also uses visible teammate hands to infer
        # that one of its own cards can only be a safely discardable card.
        engine = HanabiGameEngine(player_count=2, seed=40)
        engine.hint_tokens = 0
        engine.fireworks[Color.RED] = 1
        engine.hands[1] = [
            Card(Color.BLUE, Rank.ONE),
            Card(Color.BLUE, Rank.ONE),
            Card(Color.BLUE, Rank.ONE),
            Card(Color.GREEN, Rank.TWO),
            Card(Color.YELLOW, Rank.THREE),
        ]
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED, Color.BLUE}),
            possible_ranks=frozenset({Rank.ONE}),
            hinted_rank=Rank.ONE,
        )
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(action, DiscardAction(card_index=0))

    def test_heuristic_agent_discards_instead_of_blind_play_when_possible(self) -> None:
        # Verifies that, when it lacks enough information for a safe play,
        # the agent prefers discarding over making a blind play.
        engine = HanabiGameEngine(player_count=2, seed=36)
        engine.hint_tokens = 0
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertIsInstance(action, DiscardAction)

    def test_heuristic_agent_discards_dead_card_when_prerequisite_is_exhausted(self) -> None:
        # Verifies that the agent detects dead cards when a required lower rank
        # can no longer be completed because all copies are in the discard pile.
        engine = HanabiGameEngine(player_count=2, seed=37)
        engine.hint_tokens = 0
        engine.discard_pile.extend(
            [Card(Color.RED, Rank.TWO), Card(Color.RED, Rank.TWO)]
        )
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED}),
            possible_ranks=frozenset({Rank.THREE}),
            hinted_color=Color.RED,
            hinted_rank=Rank.THREE,
        )
        engine.knowledge_by_player[0][1] = engine.knowledge_by_player[0][1].__class__(
            possible_colors=frozenset({Color.BLUE}),
            possible_ranks=frozenset({Rank.FIVE}),
            hinted_color=Color.BLUE,
            hinted_rank=Rank.FIVE,
        )
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(action, DiscardAction(card_index=0))

    def test_heuristic_agent_avoids_discarding_high_rank_card_when_lower_risk_exists(self) -> None:
        # Verifies that, when choosing a discard, the agent avoids throwing away
        # a high-value card if there is another clearly lower-risk option.
        engine = HanabiGameEngine(player_count=2, seed=38)
        engine.hint_tokens = 0
        engine.fireworks[Color.RED] = 1
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.BLUE}),
            possible_ranks=frozenset({Rank.FIVE}),
            hinted_color=Color.BLUE,
            hinted_rank=Rank.FIVE,
        )
        engine.knowledge_by_player[0][1] = engine.knowledge_by_player[0][1].__class__(
            possible_colors=frozenset({Color.RED}),
            possible_ranks=frozenset({Rank.ONE}),
            hinted_color=Color.RED,
            hinted_rank=Rank.ONE,
        )
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertEqual(action, DiscardAction(card_index=1))

    def test_heuristic_agent_returns_legal_action(self) -> None:
        # Verifies the agent's most basic property: every returned decision
        # must belong to the legal action set in the observation.
        engine = HanabiGameEngine(player_count=2, seed=33)
        observation = engine.get_observation(0)
        agent = HeuristicAgent()

        action = agent.act(observation)

        self.assertIn(action, observation.legal_actions)


if __name__ == "__main__":
    unittest.main()
