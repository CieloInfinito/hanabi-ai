from __future__ import annotations

import unittest

from hanabi_ai.agents.heuristic.basic import BasicHeuristicAgent
from hanabi_ai.agents.heuristic.conservative import ConservativeHeuristicAgent
from hanabi_ai.game.actions import HintColorAction, HintRankAction
from hanabi_ai.game.actions import PlayAction
from hanabi_ai.game.cards import Card, Color, Rank
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.game.observation import CardKnowledge, PlayerObservation, PublicTurnRecord
from hanabi_ai.visualization.cli import (
    render_game_state,
    render_player_observation,
    render_step_result,
)


class VisualizationCliTests(unittest.TestCase):
    def test_render_game_state_includes_public_summary_and_real_hands(self) -> None:
        # Verifies that full-state rendering includes public status and omniscient hands.
        engine = HanabiGameEngine(player_count=2, seed=21)
        engine.hands[0][0] = Card(Color.RED, Rank.ONE)

        rendered = render_game_state(engine)

        self.assertIn("=== Hanabi Game State ===", rendered)
        self.assertIn("Current player: 0", rendered)
        self.assertIn("Fireworks:", rendered)
        self.assertIn("Player 0:", rendered)
        self.assertIn("R1", rendered)

    def test_render_player_observation_hides_own_real_cards(self) -> None:
        # Verifies that observation rendering shows knowledge-only data for the viewer's hand.
        engine = HanabiGameEngine(player_count=2, seed=22)
        engine.hands[0][0] = Card(Color.RED, Rank.ONE)
        observation = engine.get_observation(0)

        rendered = render_player_observation(observation)

        self.assertIn("=== Hanabi Player Observation ===", rendered)
        self.assertIn("Own hand knowledge:", rendered)
        self.assertIn("Visible other hands:", rendered)
        self.assertIn("Definitely playable own indices:", rendered)
        self.assertIn("colors:RYGBW", rendered)
        self.assertNotIn("Player 0: R1", rendered)

    def test_render_player_observation_lists_legal_actions(self) -> None:
        # Verifies that observation rendering lists the current legal actions.
        engine = HanabiGameEngine(player_count=2, seed=23)
        observation = engine.get_observation(0)

        rendered = render_player_observation(observation)

        self.assertIn("Legal actions:", rendered)
        self.assertIn(str(PlayAction(card_index=0)), rendered)

    def test_render_step_result_includes_action_and_score(self) -> None:
        # Verifies that step-result rendering includes the chosen action and updated score.
        engine = HanabiGameEngine(player_count=2, seed=24)
        result = engine.step(PlayAction(card_index=0))

        rendered = render_step_result(result)

        self.assertIn("Turn result:", rendered)
        self.assertIn("Action:", rendered)
        self.assertIn("Score:", rendered)

    def test_render_player_observation_shows_conservative_private_interpretation(self) -> None:
        # Verifies that CLI observation rendering can display the conservative
        # agent's private reading of public hint history.
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(
                CardKnowledge(
                    possible_colors=frozenset({Color.YELLOW}),
                    possible_ranks=frozenset({Rank.ONE, Rank.TWO}),
                    hinted_color=Color.YELLOW,
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.YELLOW}),
                    possible_ranks=frozenset({Rank.TWO, Rank.THREE}),
                    hinted_color=Color.YELLOW,
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.WHITE, Color.GREEN}),
                    possible_ranks=frozenset({Rank.FOUR, Rank.FIVE}),
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.YELLOW}),
                    possible_ranks=frozenset({Rank.TWO, Rank.FOUR}),
                    hinted_color=Color.YELLOW,
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.YELLOW}),
                    possible_ranks=frozenset({Rank.ONE, Rank.TWO}),
                    hinted_color=Color.YELLOW,
                ),
            ),
            other_player_hands=(),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=10,
            public_history=(
                PublicTurnRecord(
                    player_id=1,
                    action=HintColorAction(target_player=0, color=Color.YELLOW),
                    revealed_indices=(0, 1, 4, 3),
                    revealed_groups=((0,), (1,), (4,), (3,)),
                ),
            ),
            legal_actions=(PlayAction(card_index=0),),
        )

        rendered = render_player_observation(
            observation,
            viewer_agent=ConservativeHeuristicAgent(),
        )

        self.assertIn(
            "Own hand knowledge with ConservativeHeuristicAgent conventions:",
            rendered,
        )
        self.assertIn("Last revealed groups: [0] then [1] then [4] then [3]", rendered)
        self.assertIn(
            "Private interpretation: Conservative convention: color hints point matching cards in ascending rank order, including equal-rank ties.",
            rendered,
        )

    def test_render_player_observation_keeps_basic_agent_free_of_private_notes(self) -> None:
        # Verifies that the basic heuristic renderer does not invent any
        # private-convention interpretation for the same public hint history.
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(
                CardKnowledge(
                    possible_colors=frozenset({Color.YELLOW}),
                    possible_ranks=frozenset({Rank.ONE, Rank.TWO}),
                    hinted_color=Color.YELLOW,
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.YELLOW}),
                    possible_ranks=frozenset({Rank.TWO, Rank.THREE}),
                    hinted_color=Color.YELLOW,
                ),
            ),
            other_player_hands=(),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=10,
            public_history=(
                PublicTurnRecord(
                    player_id=1,
                    action=HintColorAction(target_player=0, color=Color.YELLOW),
                    revealed_indices=(0, 1),
                    revealed_groups=((0,), (1,)),
                ),
            ),
            legal_actions=(PlayAction(card_index=0),),
        )

        rendered = render_player_observation(
            observation,
            viewer_agent=BasicHeuristicAgent(),
        )

        self.assertNotIn("Own hand knowledge with BasicHeuristicAgent conventions:", rendered)
        self.assertNotIn("Private interpretation:", rendered)

    def test_render_step_result_shows_rank_hint_groups_for_conservative_agent(self) -> None:
        # Verifies that step-result rendering exposes grouped rank hints and
        # their conservative private meaning.
        engine = HanabiGameEngine(player_count=2, seed=42)
        engine.fireworks[Color.RED] = 1
        engine.hands[1] = [
            Card(Color.RED, Rank.TWO),
            Card(Color.BLUE, Rank.TWO),
            Card(Color.GREEN, Rank.FIVE),
            Card(Color.YELLOW, Rank.TWO),
            Card(Color.WHITE, Rank.FIVE),
        ]

        result = engine.step(
            ConservativeHeuristicAgent()._attach_hint_presentation(
                HintRankAction(target_player=1, rank=Rank.TWO),
                engine.get_observation(0),
            )
        )

        rendered = render_step_result(
            result,
            acting_agent=ConservativeHeuristicAgent(),
        )

        self.assertIn("Revealed groups: [0] then [1, 3]", rendered)
        self.assertIn(
            "Private interpretation: Conservative convention: rank hints group playable cards first, then non-playable cards.",
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
