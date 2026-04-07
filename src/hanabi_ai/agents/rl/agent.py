from __future__ import annotations

from dataclasses import dataclass
from random import Random

from hanabi_ai.agents.rl.encoding import LegalActionIndexer, ObservationVectorEncoder
from hanabi_ai.agents.rl.policy import LinearSoftmaxPolicy
from hanabi_ai.game.actions import Action
from hanabi_ai.game.observation import PlayerObservation


@dataclass(slots=True)
class RLTransitionRecorder:
    features: list[tuple[float, ...]]
    legal_action_indices: list[tuple[int, ...]]
    chosen_action_indices: list[int]
    chosen_action_probabilities: list[float]


class RLPolicyAgent:
    """
    Agent wrapper that turns a trainable policy into a legal Hanabi actor.
    """

    def __init__(
        self,
        *,
        encoder: ObservationVectorEncoder,
        action_indexer: LegalActionIndexer,
        policy: LinearSoftmaxPolicy,
        rng: Random | None = None,
        greedy: bool = False,
        recorder: RLTransitionRecorder | None = None,
    ) -> None:
        self._encoder = encoder
        self._action_indexer = action_indexer
        self._policy = policy
        self._rng = rng if rng is not None else Random()
        self._greedy = greedy
        self._recorder = recorder

    def act(self, observation: PlayerObservation) -> Action:
        features = self._encoder.encode(observation)
        legal_action_indices = self._action_indexer.legal_action_indices(observation)
        decision = self._policy.sample_action(
            features,
            legal_action_indices,
            rng=self._rng,
            greedy=self._greedy,
        )
        if self._recorder is not None:
            self._recorder.features.append(features)
            self._recorder.legal_action_indices.append(legal_action_indices)
            self._recorder.chosen_action_indices.append(decision.action_index)
            self._recorder.chosen_action_probabilities.append(decision.probability)
        return self._action_indexer.action_for_index(
            decision.action_index,
            current_player=observation.current_player,
        )
