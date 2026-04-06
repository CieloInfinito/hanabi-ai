from __future__ import annotations

import argparse

from hanabi_ai.training.behavior_cloning import (
    BehaviorCloningConfig,
    run_behavior_cloning_iteration,
)
from hanabi_ai.training.reinforce import build_reinforce_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain the RL policy by imitating ConventionTempo."
    )
    parser.add_argument("--players", type=int, default=2, help="Number of players.")
    parser.add_argument("--episodes", type=int, default=8, help="Teacher self-play episodes.")
    parser.add_argument("--epochs", type=int, default=4, help="Supervised epochs over collected samples.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Behavior cloning learning rate.")
    parser.add_argument("--seed-base", type=int, default=0, help="Base game seed.")
    parser.add_argument("--policy-seed", type=int, default=0, help="Initial policy seed.")
    parser.add_argument("--hidden-size", type=int, default=48, help="Hidden layer size for the policy MLP.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder, action_indexer, policy = build_reinforce_policy(
        player_count=args.players,
        seed=args.policy_seed,
        hidden_size=args.hidden_size,
    )
    stats = run_behavior_cloning_iteration(
        policy,
        encoder=encoder,
        action_indexer=action_indexer,
        config=BehaviorCloningConfig(
            player_count=args.players,
            episode_count=args.episodes,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            seed_base=args.seed_base,
        ),
    )
    print(
        f"Behavior cloning: avg_score={stats.average_score:.3f} "
        f"min={stats.min_score} "
        f"max={stats.max_score} "
        f"samples={stats.sample_count} "
        f"accuracy={stats.training_accuracy:.3%}"
    )


if __name__ == "__main__":
    main()
