from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action_index: int
    probability: float


@dataclass(frozen=True, slots=True)
class PolicyGradientSample:
    features: tuple[float, ...]
    legal_action_indices: tuple[int, ...]
    chosen_action_index: int
    advantage: float
    action_probability: float


@dataclass(frozen=True, slots=True)
class BehaviorCloningSample:
    features: tuple[float, ...]
    legal_action_indices: tuple[int, ...]
    target_action_index: int


@dataclass(frozen=True, slots=True)
class ValueRegressionSample:
    features: tuple[float, ...]
    target_value: float


class LinearSoftmaxPolicy:
    """
    Dependency-free softmax policy with a small tanh hidden layer.

    The class keeps its original name for compatibility with the rest of the
    repository, but it now behaves like a tiny MLP policy/value model rather
    than a purely linear head.
    """

    def __init__(
        self,
        *,
        input_size: int,
        action_count: int,
        seed: int = 0,
        hidden_size: int | None = None,
    ) -> None:
        self.input_size = input_size
        self.action_count = action_count
        self.hidden_size = hidden_size if hidden_size is not None else 48

        rng = Random(seed)
        self._hidden_weights = [
            [rng.uniform(-0.05, 0.05) for _ in range(input_size)]
            for _ in range(self.hidden_size)
        ]
        self._hidden_biases = [0.0 for _ in range(self.hidden_size)]

        self._policy_weights = [
            [rng.uniform(-0.05, 0.05) for _ in range(self.hidden_size)]
            for _ in range(action_count)
        ]
        self._policy_biases = [0.0 for _ in range(action_count)]

        self._value_weights = [rng.uniform(-0.05, 0.05) for _ in range(self.hidden_size)]
        self._value_bias = 0.0

    def sample_action(
        self,
        features: tuple[float, ...],
        legal_action_indices: tuple[int, ...],
        *,
        rng: Random,
        greedy: bool = False,
    ) -> PolicyDecision:
        probabilities = self.legal_action_probabilities(
            features,
            legal_action_indices,
        )

        if greedy:
            action_index, probability = max(
                probabilities.items(),
                key=lambda item: (item[1], -item[0]),
            )
            return PolicyDecision(action_index=action_index, probability=probability)

        threshold = rng.random()
        cumulative = 0.0
        last_action_index = legal_action_indices[-1]
        for action_index in legal_action_indices:
            probability = probabilities[action_index]
            cumulative += probability
            if threshold <= cumulative:
                return PolicyDecision(
                    action_index=action_index,
                    probability=probability,
                )

        return PolicyDecision(
            action_index=last_action_index,
            probability=probabilities[last_action_index],
        )

    def legal_action_probabilities(
        self,
        features: tuple[float, ...],
        legal_action_indices: tuple[int, ...],
    ) -> dict[int, float]:
        if not legal_action_indices:
            raise ValueError("legal_action_indices must not be empty.")

        _hidden_pre, hidden = self._hidden_forward(features)
        logits = {
            action_index: self._policy_logit_from_hidden(action_index, hidden)
            for action_index in legal_action_indices
        }
        max_logit = max(logits.values())
        exp_values = {
            action_index: math.exp(logit - max_logit)
            for action_index, logit in logits.items()
        }
        partition = sum(exp_values.values())
        return {
            action_index: value / partition for action_index, value in exp_values.items()
        }

    def apply_policy_gradient(
        self,
        samples: tuple[PolicyGradientSample, ...],
        *,
        learning_rate: float,
        entropy_coefficient: float = 0.0,
        gradient_clip: float | None = None,
    ) -> None:
        for sample in samples:
            hidden_pre, hidden = self._hidden_forward(sample.features)
            probabilities = self._probabilities_from_hidden(
                hidden,
                sample.legal_action_indices,
            )
            policy_weight_snapshot = tuple(
                tuple(weights) for weights in self._policy_weights
            )
            output_signals = {
                action_index: sample.advantage * (
                    (1.0 if action_index == sample.chosen_action_index else 0.0)
                    - probabilities[action_index]
                )
                for action_index in sample.legal_action_indices
            }
            if entropy_coefficient > 0.0:
                entropy_signals = self._entropy_output_signals(
                    probabilities=probabilities,
                    legal_action_indices=sample.legal_action_indices,
                )
                output_signals = {
                    action_index: output_signals[action_index]
                    + (entropy_coefficient * entropy_signals[action_index])
                    for action_index in sample.legal_action_indices
                }
            if gradient_clip is not None:
                output_signals = {
                    action_index: max(
                        -gradient_clip,
                        min(gradient_clip, signal),
                    )
                    for action_index, signal in output_signals.items()
                }
            self._apply_policy_output_update(
                hidden,
                output_signals=output_signals,
                learning_rate=learning_rate,
            )
            hidden_signal = self._hidden_signal_from_policy(
                hidden_pre,
                policy_weight_snapshot=policy_weight_snapshot,
                output_signals=output_signals,
            )
            self._apply_hidden_update(
                sample.features,
                hidden_signal=hidden_signal,
                learning_rate=learning_rate,
            )

    def apply_behavior_cloning(
        self,
        samples: tuple[BehaviorCloningSample, ...],
        *,
        learning_rate: float,
        epochs: int = 1,
    ) -> None:
        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        for _ in range(epochs):
            for sample in samples:
                hidden_pre, hidden = self._hidden_forward(sample.features)
                probabilities = self._probabilities_from_hidden(
                    hidden,
                    sample.legal_action_indices,
                )
                policy_weight_snapshot = tuple(
                    tuple(weights) for weights in self._policy_weights
                )
                output_signals = {
                    action_index: (
                        (1.0 if action_index == sample.target_action_index else 0.0)
                        - probabilities[action_index]
                    )
                    for action_index in sample.legal_action_indices
                }
                self._apply_policy_output_update(
                    hidden,
                    output_signals=output_signals,
                    learning_rate=learning_rate,
                )
                hidden_signal = self._hidden_signal_from_policy(
                    hidden_pre,
                    policy_weight_snapshot=policy_weight_snapshot,
                    output_signals=output_signals,
                )
                self._apply_hidden_update(
                    sample.features,
                    hidden_signal=hidden_signal,
                    learning_rate=learning_rate,
                )

    def behavior_cloning_accuracy(
        self,
        samples: tuple[BehaviorCloningSample, ...],
    ) -> float:
        if not samples:
            return 0.0

        correct_predictions = 0
        for sample in samples:
            decision = self.sample_action(
                sample.features,
                sample.legal_action_indices,
                rng=Random(0),
                greedy=True,
            )
            correct_predictions += int(
                decision.action_index == sample.target_action_index
            )
        return correct_predictions / len(samples)

    def predict_value(self, features: tuple[float, ...]) -> float:
        _hidden_pre, hidden = self._hidden_forward(features)
        return self._value_bias + sum(
            weight * hidden_value
            for weight, hidden_value in zip(self._value_weights, hidden, strict=True)
        )

    def apply_value_regression(
        self,
        samples: tuple[ValueRegressionSample, ...],
        *,
        learning_rate: float,
    ) -> None:
        for sample in samples:
            hidden_pre, hidden = self._hidden_forward(sample.features)
            prediction = self._value_bias + sum(
                weight * hidden_value
                for weight, hidden_value in zip(self._value_weights, hidden, strict=True)
            )
            error = sample.target_value - prediction

            old_value_weights = tuple(self._value_weights)
            self._value_bias += learning_rate * error
            for hidden_index, hidden_value in enumerate(hidden):
                self._value_weights[hidden_index] += learning_rate * error * hidden_value

            hidden_signal = tuple(
                (1.0 - (hidden_value * hidden_value)) * error * old_value_weights[hidden_index]
                for hidden_index, hidden_value in enumerate(hidden)
            )
            self._apply_hidden_update(
                sample.features,
                hidden_signal=hidden_signal,
                learning_rate=learning_rate,
            )

    def _hidden_forward(
        self,
        features: tuple[float, ...],
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        hidden_pre: list[float] = []
        hidden: list[float] = []
        for hidden_index in range(self.hidden_size):
            pre_activation = self._hidden_biases[hidden_index] + sum(
                weight * feature_value
                for weight, feature_value in zip(
                    self._hidden_weights[hidden_index],
                    features,
                    strict=True,
                )
            )
            hidden_pre.append(pre_activation)
            hidden.append(math.tanh(pre_activation))
        return tuple(hidden_pre), tuple(hidden)

    def _policy_logit_from_hidden(
        self,
        action_index: int,
        hidden: tuple[float, ...],
    ) -> float:
        return self._policy_biases[action_index] + sum(
            weight * hidden_value
            for weight, hidden_value in zip(
                self._policy_weights[action_index],
                hidden,
                strict=True,
            )
        )

    def _probabilities_from_hidden(
        self,
        hidden: tuple[float, ...],
        legal_action_indices: tuple[int, ...],
    ) -> dict[int, float]:
        logits = {
            action_index: self._policy_logit_from_hidden(action_index, hidden)
            for action_index in legal_action_indices
        }
        max_logit = max(logits.values())
        exp_values = {
            action_index: math.exp(logit - max_logit)
            for action_index, logit in logits.items()
        }
        partition = sum(exp_values.values())
        return {
            action_index: value / partition
            for action_index, value in exp_values.items()
        }

    def _apply_policy_output_update(
        self,
        hidden: tuple[float, ...],
        *,
        output_signals: dict[int, float],
        learning_rate: float,
    ) -> None:
        for action_index, signal in output_signals.items():
            if signal == 0.0:
                continue
            self._policy_biases[action_index] += learning_rate * signal
            for hidden_index, hidden_value in enumerate(hidden):
                self._policy_weights[action_index][hidden_index] += (
                    learning_rate * signal * hidden_value
                )

    def _hidden_signal_from_policy(
        self,
        hidden_pre: tuple[float, ...],
        *,
        policy_weight_snapshot: tuple[tuple[float, ...], ...],
        output_signals: dict[int, float],
    ) -> tuple[float, ...]:
        hidden_signal: list[float] = []
        for hidden_index in range(self.hidden_size):
            downstream = sum(
                signal * policy_weight_snapshot[action_index][hidden_index]
                for action_index, signal in output_signals.items()
            )
            activation_derivative = 1.0 - (math.tanh(hidden_pre[hidden_index]) ** 2)
            hidden_signal.append(activation_derivative * downstream)
        return tuple(hidden_signal)

    def _apply_hidden_update(
        self,
        features: tuple[float, ...],
        *,
        hidden_signal: tuple[float, ...],
        learning_rate: float,
    ) -> None:
        for hidden_index, signal in enumerate(hidden_signal):
            if signal == 0.0:
                continue
            self._hidden_biases[hidden_index] += learning_rate * signal
            for feature_index, feature_value in enumerate(features):
                self._hidden_weights[hidden_index][feature_index] += (
                    learning_rate * signal * feature_value
                )

    def _entropy_output_signals(
        self,
        *,
        probabilities: dict[int, float],
        legal_action_indices: tuple[int, ...],
    ) -> dict[int, float]:
        entropy = -sum(
            probability * math.log(max(probability, 1e-12))
            for probability in probabilities.values()
        )
        expected_neg_log = {
            action_index: -math.log(max(probabilities[action_index], 1e-12)) - entropy
            for action_index in legal_action_indices
        }
        return {
            action_index: probabilities[action_index] * expected_neg_log[action_index]
            for action_index in legal_action_indices
        }
