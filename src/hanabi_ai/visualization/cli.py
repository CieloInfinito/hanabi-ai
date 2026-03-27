from __future__ import annotations

from collections.abc import Iterable, Mapping

from hanabi_ai.agents.heuristic.base import BaseHeuristicAgent
from hanabi_ai.game.actions import Action
from hanabi_ai.game.cards import HANABI_COLORS, Card, Color, Rank
from hanabi_ai.game.engine import EngineStepResult, HanabiGameEngine
from hanabi_ai.game.observation import (
    CardKnowledge,
    ObservedHand,
    PlayerObservation,
    PublicTurnRecord,
    get_definitely_playable_card_indices,
)


COLOR_SHORT_NAMES: dict[Color, str] = {
    Color.RED: "R",
    Color.YELLOW: "Y",
    Color.GREEN: "G",
    Color.BLUE: "B",
    Color.WHITE: "W",
}


def _append_public_history(
    lines: list[str],
    observation: PlayerObservation,
    *,
    viewer_agent: BaseHeuristicAgent | None = None,
) -> None:
    if not observation.public_history:
        return

    last_record = observation.public_history[-1]
    lines.append(
        "Last public action: "
        f"Player {last_record.player_id} -> {render_action(last_record.action)}"
    )
    if last_record.revealed_indices:
        lines.append(f"Last revealed indices: {last_record.revealed_indices}")
    if last_record.revealed_groups:
        lines.append(
            "Last revealed groups: "
            f"{render_revealed_groups(last_record.revealed_groups)}"
        )
    if viewer_agent is not None:
        for note in viewer_agent.describe_public_turn_record(last_record):
            lines.append(f"Private interpretation: {note}")


def _append_legal_actions(lines: list[str], observation: PlayerObservation) -> None:
    if not observation.legal_actions:
        return

    lines.append("Legal actions:")
    for index, action in enumerate(observation.legal_actions):
        lines.append(f"  [{index}] {render_action(action)}")


def _append_step_interpretation(
    lines: list[str],
    step_result: EngineStepResult,
    *,
    acting_agent: BaseHeuristicAgent | None = None,
) -> None:
    if acting_agent is None:
        return

    for note in acting_agent.describe_public_turn_record(
        PublicTurnRecord(
            player_id=step_result.acting_player,
            action=step_result.action,
            revealed_indices=step_result.revealed_indices,
            revealed_groups=step_result.revealed_groups,
            fireworks_before=step_result.fireworks_before,
        )
    ):
        lines.append(f"  Private interpretation: {note}")


def render_card(card: Card) -> str:
    """
    Render a real card in compact form.
    """
    return f"{COLOR_SHORT_NAMES[card.color]}{int(card.rank)}"


def render_cards(cards: Iterable[Card]) -> str:
    """
    Render a sequence of real cards.
    """
    rendered = [render_card(card) for card in cards]
    return " ".join(rendered) if rendered else "-"


def render_card_knowledge(knowledge: CardKnowledge) -> str:
    """
    Render the hidden knowledge associated with one own-hand card.
    """
    colors = "".join(
        COLOR_SHORT_NAMES[color]
        for color in HANABI_COLORS
        if color in knowledge.possible_colors
    )
    ranks = "".join(
        str(int(rank))
        for rank in Rank
        if rank in knowledge.possible_ranks
    )

    hinted_parts: list[str] = []
    if knowledge.hinted_color is not None:
        hinted_parts.append(f"color={COLOR_SHORT_NAMES[knowledge.hinted_color]}")
    if knowledge.hinted_rank is not None:
        hinted_parts.append(f"rank={int(knowledge.hinted_rank)}")

    hint_text = f" hint[{', '.join(hinted_parts)}]" if hinted_parts else ""
    return f"[colors:{colors or '-'} ranks:{ranks or '-'}{hint_text}]"


def render_fireworks(fireworks: Mapping[Color, int]) -> str:
    """
    Render the public fireworks piles.
    """
    return " ".join(
        f"{COLOR_SHORT_NAMES[color]}:{fireworks.get(color, 0)}" for color in HANABI_COLORS
    )


def render_action(action: Action) -> str:
    """
    Render an action using its string representation.
    """
    return str(action)


def render_revealed_groups(revealed_groups: tuple[tuple[int, ...], ...]) -> str:
    """
    Render grouped hint presentation in a readable order-of-pointing format.
    """
    if not revealed_groups:
        return "-"

    rendered_groups = []
    for group in revealed_groups:
        rendered_groups.append("[" + ", ".join(str(index) for index in group) + "]")

    return " then ".join(rendered_groups)


def render_observed_hand(observed_hand: ObservedHand) -> str:
    """
    Render another player's visible hand.
    """
    return f"Player {observed_hand.player_id}: {render_cards(observed_hand.cards)}"


def render_game_state(engine: HanabiGameEngine) -> str:
    """
    Render the omniscient game state for debugging.
    """
    lines = [
        "=== Hanabi Game State ===",
        (
            f"Turn: {engine.turn_number} | Current player: {engine.current_player} | "
            f"Score: {engine.get_score()}"
        ),
        (
            f"Hint tokens: {engine.hint_tokens} | Strike tokens: {engine.strike_tokens} | "
            f"Deck size: {len(engine.deck)}"
        ),
        f"Fireworks: {render_fireworks(engine.fireworks)}",
        f"Discard pile: {render_cards(engine.discard_pile)}",
        "Hands:",
    ]

    for player_id, hand in enumerate(engine.hands):
        current_marker = " <- current" if player_id == engine.current_player else ""
        lines.append(f"  Player {player_id}: {render_cards(hand)}{current_marker}")

    if engine.history:
        last_record = engine.history[-1]
        lines.append(f"Last action: Player {last_record.player_id} -> {render_action(last_record.action)}")

    return "\n".join(lines)


def render_player_observation(
    observation: PlayerObservation,
    *,
    viewer_agent: BaseHeuristicAgent | None = None,
) -> str:
    """
    Render the partial observation available to one player.
    """
    interpreted_observation = (
        viewer_agent.refine_observation_for_display(observation)
        if viewer_agent is not None
        else observation
    )
    lines = [
        "=== Hanabi Player Observation ===",
        (
            f"Observing player: {observation.observing_player} | "
            f"Current player: {observation.current_player}"
        ),
        (
            f"Hint tokens: {observation.hint_tokens} | "
            f"Strike tokens: {observation.strike_tokens} | "
            f"Deck size: {observation.deck_size}"
        ),
        f"Fireworks: {render_fireworks(observation.fireworks)}",
        f"Discard pile: {render_cards(observation.discard_pile)}",
        "Own hand knowledge:",
    ]

    for index, knowledge in enumerate(observation.hand_knowledge):
        lines.append(f"  [{index}] {render_card_knowledge(knowledge)}")

    if (
        viewer_agent is not None
        and interpreted_observation.hand_knowledge != observation.hand_knowledge
    ):
        lines.append(
            f"Own hand knowledge with {viewer_agent.__class__.__name__} conventions:"
        )
        for index, knowledge in enumerate(interpreted_observation.hand_knowledge):
            lines.append(f"  [{index}] {render_card_knowledge(knowledge)}")

    lines.append("Visible other hands:")
    for observed_hand in observation.other_player_hands:
        lines.append(f"  {render_observed_hand(observed_hand)}")

    definitely_playable = get_definitely_playable_card_indices(
        observation.hand_knowledge, observation.fireworks
    )
    lines.append(
        "Definitely playable own indices: "
        + (str(definitely_playable) if definitely_playable else "-")
    )
    _append_public_history(lines, observation, viewer_agent=viewer_agent)
    _append_legal_actions(lines, observation)

    return "\n".join(lines)


def render_step_result(
    step_result: EngineStepResult,
    *,
    acting_agent: BaseHeuristicAgent | None = None,
) -> str:
    """
    Render the outcome of a single engine step.
    """
    lines = [
        "Turn result:",
        f"  Acting player: {step_result.acting_player}",
        f"  Action: {render_action(step_result.action)}",
        f"  Score: {step_result.score}",
    ]

    if step_result.removed_card is not None:
        lines.append(f"  Removed card: {render_card(step_result.removed_card)}")
    if step_result.play_succeeded is not None:
        lines.append(f"  Play succeeded: {step_result.play_succeeded}")
    if step_result.revealed_indices:
        lines.append(f"  Revealed indices: {step_result.revealed_indices}")
    if step_result.revealed_groups:
        lines.append(
            f"  Revealed groups: {render_revealed_groups(step_result.revealed_groups)}"
        )
    _append_step_interpretation(lines, step_result, acting_agent=acting_agent)
    lines.append(f"  Drew replacement: {step_result.drew_replacement}")
    lines.append(f"  Game over: {step_result.game_over}")
    return "\n".join(lines)


def render_self_play_turn(
    *,
    turn_index: int,
    player_id: int,
    observation: PlayerObservation,
    action: Action,
    step_result: EngineStepResult,
    engine: HanabiGameEngine,
    acting_agent: BaseHeuristicAgent | None = None,
) -> str:
    """
    Render a readable trace block for one self-play turn.
    """
    lines = [
        f"=== Self-Play Turn {turn_index} ===",
        f"Active player: {player_id}",
        render_player_observation(observation, viewer_agent=acting_agent),
        "",
        "Chosen action:",
        f"  {render_action(action)}",
    ]
    if acting_agent is not None:
        for note in acting_agent.explain_action_choice(observation, action):
            lines.append(f"  {note}")
    lines.extend(
        [
            "",
            render_step_result(step_result, acting_agent=acting_agent),
            "",
            "State after turn:",
            render_game_state(engine),
        ]
    )
    return "\n".join(lines)
