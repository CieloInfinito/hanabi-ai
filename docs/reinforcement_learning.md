## Reinforcement Learning

The repository now includes a first reinforcement-learning scaffold built on
top of the existing imperfect-information environment.

It also now includes a first behavior-cloning path that uses
`ConventionTempoHeuristicAgent` as a teacher policy.

### What Exists Today

- `hanabi_ai.agents.rl.encoding.ObservationVectorEncoder`
  Encodes a `PlayerObservation` into a fixed numeric vector for one table size.
- `hanabi_ai.agents.rl.encoding.LegalActionIndexer`
  Maps legal Hanabi actions into a fixed action head using seat-relative hint
  targets.
- `hanabi_ai.agents.rl.policy.LinearSoftmaxPolicy`
  Pure Python masked softmax policy with a small tanh hidden layer and a simple
  value baseline. The current default hidden size is `48`, which performed
  better than the earlier larger default in short behavior-cloning sweeps.
- `hanabi_ai.agents.rl.agent.RLPolicyAgent`
  Wraps the encoder, action indexer, and policy into a legal Hanabi actor.
- `hanabi_ai.training.reinforce`
  Runs a minimal self-play REINFORCE iteration and returns compact stats.
- `hanabi_ai.training.behavior_cloning`
  Collects expert demonstrations from `ConventionTempo` and trains the current
  policy with supervised updates.

### Why Start This Small

This scaffold is intentionally dependency-free.

The goal of the first RL step is not to beat the heuristic baseline
immediately. The goal is to stabilize:

- the observation interface
- the legal-action policy head
- the self-play collection path
- the training loop contract

Once these interfaces are stable, the project can replace the pure Python
policy with a tensor-backed implementation without redesigning the rest of the
pipeline.

### Current Limitations

- the encoder is hand-designed and intentionally simple
- the policy is still a very small pure Python MLP
- the reward is still lightweight shaped reward, not a richer task-specific design
- the value baseline is linear and lightweight, not a richer critic
- the current warm-started RL agent is still far below the strongest heuristic baseline

So this is a research scaffold, not a competitive RL stack.

### Smoke Run

```powershell
hanabi-train-behavior-cloning --players 2 --episodes 8 --epochs 4
```

```powershell
hanabi-train-reinforce --players 2 --episodes 10 --iterations 3
```

```powershell
hanabi-train-warm-start --players 2 --bc-episodes 8 --bc-epochs 4 --rl-iterations 3 --rl-episodes 10
```

All three training commands also accept `--hidden-size` if you want to sweep
model capacity explicitly.

In short local sweeps with the current pure Python MLP, `hidden_size=48`,
`learning_rate=0.05`, and a behavior-cloning setup around `8` episodes and `4`
epochs gave the strongest imitation accuracy among the configurations tested.

For the current warm-start REINFORCE path, short sweeps favored a more
conservative setup than the earlier defaults: `rl_learning_rate=0.002`,
`discount_factor=0.95`, and `final_score_bonus_weight=0.5` produced slightly
lower variance and a positive mean shaped return in the small comparison run.

### Current Comparison

On a short 2-player comparison run over 50 games with shared seeds:

- `ConventionTempoHeuristicAgent`: `16.44` average score
- RL policy after behavior cloning: `2.70`
- RL policy after behavior cloning plus warm-start REINFORCE: `2.76`

So the current RL path should be understood as infrastructure and early
experimentation, not as a policy that is yet competitive with the best
heuristic agent in this repository.

### Likely Next Steps

1. Save and reload warm-started policies as checkpoints.
2. Add richer reward shaping or a learned value baseline.
3. Swap the linear policy for a neural implementation.
4. Save checkpoints and benchmark the learned policy against the heuristic JSON
   baselines in `reports/`.
