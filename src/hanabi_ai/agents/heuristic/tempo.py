from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    SRC_PATH = Path(__file__).resolve().parents[3]
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

from hanabi_ai.agents.heuristic.base import BaseHeuristicAgent
from hanabi_ai.game.actions import (
    Action,
    AgentDecision,
    DiscardAction,
    HintColorAction,
    HintRankAction,
)
from hanabi_ai.game.observation import PlayerObservation


class TempoHeuristicAgent(BaseHeuristicAgent):
    """
    Experimental heuristic variant that protects short-horizon hint economy.

    This agent reuses the same public-information inference as the base
    heuristic, but it becomes less willing to spend the last hint token on a
    purely informational hint when a legal discard can recover tempo instead.
    """

    def act(self, observation: PlayerObservation) -> Action | AgentDecision:
        if not observation.legal_actions:
            raise ValueError(
                f"{self.__class__.__name__} received an observation with no legal actions."
            )

        observation = self._apply_private_conventions(observation)
        self._cache_belief_state(observation)

        guaranteed_play = self._choose_definitely_playable_action(observation)
        if guaranteed_play is not None:
            return guaranteed_play

        helpful_hint, helpful_hint_score = self._choose_hint_for_other_players(
            observation
        )
        confident_play = self._choose_confident_probabilistic_play(
            observation,
            best_hint_score=helpful_hint_score,
        )
        if confident_play is not None:
            return confident_play

        discard_action = self._choose_discard_action(observation)
        if (
            discard_action is not None
            and helpful_hint is not None
            and not self._should_spend_hint_on_best_hint(observation, helpful_hint_score)
        ):
            return discard_action

        if helpful_hint is not None:
            return self._attach_hint_presentation(helpful_hint, observation)

        if discard_action is not None:
            return discard_action

        fallback_play = self._choose_any_play_action(observation)
        if fallback_play is not None:
            return fallback_play

        return observation.legal_actions[0]

    def _should_spend_hint_on_best_hint(
        self,
        observation: PlayerObservation,
        best_hint_score: tuple[int, int, int, int, int, int, int, int, int] | None,
    ) -> bool:
        if best_hint_score is None:
            return False

        guaranteed_play_hits = best_hint_score[0]
        playable_hits = best_hint_score[1]
        useful_hits = best_hint_score[5]
        information_gain = best_hint_score[6]

        if guaranteed_play_hits > 0:
            return True

        if observation.hint_tokens >= 2:
            return True

        return playable_hits >= 2 or (playable_hits >= 1 and useful_hits + information_gain >= 4)


if __name__ == "__main__":
    raise SystemExit(
        "TempoHeuristicAgent is a library module. "
        "Use hanabi-evaluate or import it from hanabi_ai.agents."
    )
