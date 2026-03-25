from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    SRC_PATH = Path(__file__).resolve().parents[3]
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

from hanabi_ai.agents.heuristic.basic import BasicHeuristicAgent
from hanabi_ai.agents.heuristic._scoring import HintScore
from hanabi_ai.game.actions import (
    DiscardAction,
    HintColorAction,
    HintRankAction,
)
from hanabi_ai.game.observation import PlayerObservation


class TempoHeuristicAgent(BasicHeuristicAgent):
    """
    Experimental heuristic variant that protects short-horizon hint economy.

    This agent reuses the same public-information inference as the base
    heuristic, but it becomes less willing to spend the last hint token on a
    purely informational hint when a legal discard can recover tempo instead.
    """

    def _should_prefer_discard_over_hint(
        self,
        observation: PlayerObservation,
        discard_action: DiscardAction,
        hint_action: HintColorAction | HintRankAction,
        hint_score: HintScore | None,
    ) -> bool:
        return not self._should_spend_hint_on_best_hint(observation, hint_score)

    def _should_spend_hint_on_best_hint(
        self,
        observation: PlayerObservation,
        best_hint_score: HintScore | None,
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

        return playable_hits >= 2 or (
            playable_hits >= 1 and useful_hits + information_gain >= 4
        )


if __name__ == "__main__":
    raise SystemExit(
        "TempoHeuristicAgent is a library module. "
        "Use hanabi-evaluate or import it from hanabi_ai.agents."
    )
