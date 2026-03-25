from __future__ import annotations

from abc import ABC, abstractmethod

from hanabi_ai.game.actions import (
    DiscardAction,
    HintColorAction,
    HintRankAction,
    PlayAction,
    normalize_agent_decision,
)
from hanabi_ai.game.cards import Card, Color, Rank
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.game.observation import (
    CardKnowledge,
    ObservedHand,
    PlayerObservation,
    PublicTurnRecord,
)


# Shared heuristic tests verify the baseline decision logic that every
# heuristic agent in this family should satisfy.
class SharedHeuristicAgentTests(ABC):
    @abstractmethod
    def make_agent(self):
        raise NotImplementedError

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
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, PlayAction(card_index=0))

    def test_heuristic_agent_prefers_safe_five_to_recover_hint(self) -> None:
        # Verifies that, among multiple safe plays, the agent prefers a known
        # five because it recovers a hint token on success.
        engine = HanabiGameEngine(player_count=2, seed=72)
        engine.hint_tokens = 0
        engine.fireworks[Color.RED] = 1
        engine.fireworks[Color.BLUE] = 4
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED}),
            possible_ranks=frozenset({Rank.TWO}),
            hinted_color=Color.RED,
            hinted_rank=Rank.TWO,
        )
        engine.knowledge_by_player[0][1] = engine.knowledge_by_player[0][1].__class__(
            possible_colors=frozenset({Color.BLUE}),
            possible_ranks=frozenset({Rank.FIVE}),
            hinted_color=Color.BLUE,
            hinted_rank=Rank.FIVE,
        )
        observation = engine.get_observation(0)
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, PlayAction(card_index=1))

    def test_heuristic_agent_prefers_safe_critical_play_over_lower_value_safe_play(self) -> None:
        # Verifies that, among multiple safe plays, the agent prefers the one
        # that is publicly the last remaining live copy.
        engine = HanabiGameEngine(player_count=2, seed=73)
        engine.discard_pile.extend(
            [Card(Color.RED, Rank.THREE), Card(Color.RED, Rank.THREE)]
        )
        engine.fireworks[Color.RED] = 2
        engine.fireworks[Color.GREEN] = 0
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.GREEN}),
            possible_ranks=frozenset({Rank.ONE}),
            hinted_color=Color.GREEN,
            hinted_rank=Rank.ONE,
        )
        engine.knowledge_by_player[0][1] = engine.knowledge_by_player[0][1].__class__(
            possible_colors=frozenset({Color.RED}),
            possible_ranks=frozenset({Rank.THREE}),
            hinted_color=Color.RED,
            hinted_rank=Rank.THREE,
        )
        observation = engine.get_observation(0)
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, PlayAction(card_index=1))

    def test_heuristic_agent_gives_hint_for_other_players_playable_card(self) -> None:
        # Verifies that, when it has no safe self-play, the agent prioritizes
        # a hint that enables a teammate to identify a visible playable card immediately.
        engine = HanabiGameEngine(player_count=2, seed=32)
        engine.hands[1] = [
            Card(Color.BLUE, Rank.ONE),
            Card(Color.RED, Rank.TWO),
            Card(Color.GREEN, Rank.THREE),
            Card(Color.YELLOW, Rank.FOUR),
            Card(Color.WHITE, Rank.FIVE),
        ]
        observation = engine.get_observation(0)
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, HintRankAction(target_player=1, rank=Rank.ONE))

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
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, HintRankAction(target_player=1, rank=Rank.ONE))

    def test_heuristic_agent_prefers_hint_that_creates_guaranteed_play(self) -> None:
        # Verifies that the agent prefers a hint that should let the receiver
        # identify a safe play, not just notice a generally useful color.
        engine = HanabiGameEngine(player_count=2, seed=70)
        engine.hands[1] = [
            Card(Color.RED, Rank.ONE),
            Card(Color.RED, Rank.FIVE),
            Card(Color.GREEN, Rank.THREE),
            Card(Color.YELLOW, Rank.FOUR),
            Card(Color.WHITE, Rank.FIVE),
        ]
        observation = engine.get_observation(0)
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, HintRankAction(target_player=1, rank=Rank.ONE))

    def test_heuristic_agent_avoids_redundant_hint_when_playable_card_is_already_known(self) -> None:
        # Verifies that the agent avoids repeating a hint that would not add
        # new certainty about an already-known playable card.
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(
                CardKnowledge(
                    possible_colors=frozenset({Color.RED, Color.YELLOW}),
                    possible_ranks=frozenset({Rank.TWO, Rank.THREE}),
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.BLUE, Color.GREEN}),
                    possible_ranks=frozenset({Rank.ONE, Rank.FIVE}),
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.WHITE}),
                    possible_ranks=frozenset({Rank.FOUR}),
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.RED, Color.GREEN}),
                    possible_ranks=frozenset({Rank.ONE, Rank.TWO}),
                ),
                CardKnowledge(
                    possible_colors=frozenset({Color.YELLOW, Color.WHITE}),
                    possible_ranks=frozenset({Rank.THREE, Rank.FIVE}),
                ),
            ),
            other_player_hands=(
                ObservedHand(
                    player_id=1,
                    cards=(
                        Card(Color.BLUE, Rank.ONE),
                        Card(Color.RED, Rank.TWO),
                        Card(Color.GREEN, Rank.THREE),
                        Card(Color.YELLOW, Rank.FOUR),
                        Card(Color.WHITE, Rank.FIVE),
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
                    action=HintRankAction(target_player=1, rank=Rank.ONE),
                    revealed_indices=(0,),
                    revealed_groups=((0,),),
                ),
            ),
            legal_actions=(
                HintColorAction(target_player=1, color=Color.BLUE),
                HintColorAction(target_player=1, color=Color.RED),
                HintRankAction(target_player=1, rank=Rank.ONE),
                HintRankAction(target_player=1, rank=Rank.TWO),
            ),
        )
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertNotEqual(action, HintRankAction(target_player=1, rank=Rank.ONE))

    def test_heuristic_agent_prefers_cleaner_hint_when_playable_value_is_tied(self) -> None:
        # Verifies that, when two hints expose the same immediate playable value,
        # the agent prefers the less noisy hint that touches fewer extra cards.
        engine = HanabiGameEngine(player_count=2, seed=67)
        engine.hands[1] = [
            Card(Color.RED, Rank.ONE),
            Card(Color.RED, Rank.FIVE),
            Card(Color.BLUE, Rank.THREE),
            Card(Color.GREEN, Rank.FOUR),
            Card(Color.WHITE, Rank.FOUR),
        ]
        observation = engine.get_observation(0)
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, HintRankAction(target_player=1, rank=Rank.ONE))

    def test_heuristic_agent_prefers_hint_with_higher_useful_information_gain(self) -> None:
        # Verifies that, when no hint creates an immediate safe play, the agent
        # prefers the hint that reduces more useful uncertainty for the receiver.
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(
                CardKnowledge(
                    possible_colors=frozenset({Color.RED, Color.YELLOW}),
                    possible_ranks=frozenset({Rank.TWO, Rank.THREE}),
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
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=40,
            public_history=( ),
            legal_actions=(
                HintColorAction(target_player=1, color=Color.RED),
                HintRankAction(target_player=1, rank=Rank.FIVE),
            ),
        )
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, HintColorAction(target_player=1, color=Color.RED))

    def test_heuristic_agent_values_negative_information_on_unhinted_cards(self) -> None:
        # Verifies that the agent values a hint that rules out possibilities on
        # several other cards, not only the cards directly pointed at.
        observation = PlayerObservation(
            observing_player=0,
            current_player=0,
            hand_knowledge=(
                CardKnowledge(
                    possible_colors=frozenset({Color.RED, Color.BLUE}),
                    possible_ranks=frozenset({Rank.ONE, Rank.TWO}),
                ),
            ),
            other_player_hands=(
                ObservedHand(
                    player_id=1,
                    cards=(
                        Card(Color.RED, Rank.THREE),
                        Card(Color.BLUE, Rank.TWO),
                        Card(Color.GREEN, Rank.THREE),
                        Card(Color.YELLOW, Rank.FOUR),
                        Card(Color.WHITE, Rank.FIVE),
                    ),
                ),
            ),
            fireworks={color: 0 for color in Color},
            discard_pile=(),
            hint_tokens=1,
            strike_tokens=0,
            deck_size=40,
            public_history=(),
            legal_actions=(
                HintColorAction(target_player=1, color=Color.RED),
                HintRankAction(target_player=1, rank=Rank.TWO),
            ),
        )
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, HintColorAction(target_player=1, color=Color.RED))

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
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

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
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

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
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, DiscardAction(card_index=0))

    def test_heuristic_agent_discards_instead_of_blind_play_when_possible(self) -> None:
        # Verifies that, when it lacks enough information for a safe play,
        # the agent prefers discarding over making a blind play.
        engine = HanabiGameEngine(player_count=2, seed=36)
        engine.hint_tokens = 0
        observation = engine.get_observation(0)
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertIsInstance(action, DiscardAction)

    def test_heuristic_agent_takes_high_confidence_play_when_hint_is_weak(self) -> None:
        # Verifies that the agent may play a high-confidence own card instead
        # of spending tempo on a hint that does not unlock any immediate play.
        engine = HanabiGameEngine(player_count=2, seed=68)
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED}),
            possible_ranks=frozenset({Rank.ONE, Rank.FIVE}),
            hinted_color=Color.RED,
        )
        engine.hands[1] = [
            Card(Color.BLUE, Rank.THREE),
            Card(Color.YELLOW, Rank.THREE),
            Card(Color.GREEN, Rank.FOUR),
            Card(Color.WHITE, Rank.FOUR),
            Card(Color.BLUE, Rank.FIVE),
        ]
        observation = engine.get_observation(0)
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, PlayAction(card_index=0))

    def test_heuristic_agent_avoids_risky_play_with_two_strikes(self) -> None:
        # Verifies that the agent stops taking probabilistic plays once the
        # game is one mistake away from ending.
        engine = HanabiGameEngine(player_count=2, seed=69)
        engine.strike_tokens = 2
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED}),
            possible_ranks=frozenset({Rank.ONE, Rank.FIVE}),
            hinted_color=Color.RED,
        )
        engine.hands[1] = [
            Card(Color.BLUE, Rank.THREE),
            Card(Color.YELLOW, Rank.THREE),
            Card(Color.GREEN, Rank.FOUR),
            Card(Color.WHITE, Rank.FOUR),
            Card(Color.BLUE, Rank.FIVE),
        ]
        observation = engine.get_observation(0)
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertNotEqual(action, PlayAction(card_index=0))
        self.assertIsInstance(action, (HintColorAction, HintRankAction))

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
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

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
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, DiscardAction(card_index=1))

    def test_heuristic_agent_protects_last_remaining_copy_when_other_discard_exists(self) -> None:
        # Verifies that the agent avoids discarding a card that is the last
        # remaining live copy when another lower-risk discard is available.
        engine = HanabiGameEngine(player_count=2, seed=65)
        engine.hint_tokens = 0
        engine.discard_pile.extend(
            [Card(Color.BLUE, Rank.FOUR), Card(Color.BLUE, Rank.FOUR)]
        )
        engine.fireworks[Color.RED] = 1
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.BLUE}),
            possible_ranks=frozenset({Rank.FOUR}),
            hinted_color=Color.BLUE,
            hinted_rank=Rank.FOUR,
        )
        engine.knowledge_by_player[0][1] = engine.knowledge_by_player[0][1].__class__(
            possible_colors=frozenset({Color.RED}),
            possible_ranks=frozenset({Rank.ONE}),
            hinted_color=Color.RED,
            hinted_rank=Rank.ONE,
        )
        observation = engine.get_observation(0)
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, DiscardAction(card_index=1))

    def test_heuristic_agent_prefers_lower_expected_discard_risk_under_uncertainty(self) -> None:
        # Verifies that, under uncertainty, the agent discards the card whose
        # probability-weighted downside is lower.
        engine = HanabiGameEngine(player_count=2, seed=66)
        engine.hint_tokens = 0
        engine.fireworks[Color.RED] = 1
        engine.hands[1] = [
            Card(Color.BLUE, Rank.FOUR),
            Card(Color.BLUE, Rank.FOUR),
            Card(Color.RED, Rank.ONE),
            Card(Color.GREEN, Rank.THREE),
            Card(Color.WHITE, Rank.FIVE),
        ]
        engine.knowledge_by_player[0][0] = engine.knowledge_by_player[0][0].__class__(
            possible_colors=frozenset({Color.RED, Color.BLUE}),
            possible_ranks=frozenset({Rank.ONE}),
            hinted_rank=Rank.ONE,
        )
        engine.knowledge_by_player[0][1] = engine.knowledge_by_player[0][1].__class__(
            possible_colors=frozenset({Color.BLUE}),
            possible_ranks=frozenset({Rank.FIVE}),
            hinted_color=Color.BLUE,
            hinted_rank=Rank.FIVE,
        )
        observation = engine.get_observation(0)
        agent = self.make_agent()

        action = normalize_agent_decision(agent.act(observation)).action

        self.assertEqual(action, DiscardAction(card_index=0))

    def test_heuristic_agent_returns_legal_action(self) -> None:
        # Verifies the agent's most basic property: every returned decision
        # must belong to the legal action set in the observation.
        engine = HanabiGameEngine(player_count=2, seed=33)
        observation = engine.get_observation(0)
        agent = self.make_agent()

        decision = normalize_agent_decision(agent.act(observation))

        self.assertIn(decision.action, observation.legal_actions)
