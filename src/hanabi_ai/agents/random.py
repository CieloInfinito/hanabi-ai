from __future__ import annotations

from random import Random

from hanabi_ai.game.actions import Action
from hanabi_ai.game.observation import PlayerObservation


class RandomAgent:
    """
    Baseline agent that samples uniformly from legal actions.
    """

    def __init__(self, *, seed: int | None = None, rng: Random | None = None) -> None:
        if rng is not None and seed is not None:
            raise ValueError("Provide either seed or rng, but not both.")
        self._rng = rng if rng is not None else Random(seed)

    def act(self, observation: PlayerObservation) -> Action:
        """
        Sample one legal action from the player's current observation.
        """
        if not observation.legal_actions:
            raise ValueError("RandomAgent received an observation with no legal actions.")
        return self._rng.choice(observation.legal_actions)
