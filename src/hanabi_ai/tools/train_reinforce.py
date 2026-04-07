from __future__ import annotations

import argparse

from hanabi_ai.training.reinforce import (
    ReinforceConfig,
    build_reinforce_policy,
    run_reinforce_iteration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal REINFORCE self-play loop for Hanabi."
    )
    parser.add_argument("--players", type=int, default=2, help="Number of players.")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per iteration.")
    parser.add_argument("--iterations", type=int, default=5, help="Training iterations.")
    parser.add_argument("--actor-learning-rate", type=float, default=0.002, help="Policy learning rate.")
    parser.add_argument("--critic-learning-rate", type=float, default=0.002, help="Value-head learning rate.")
    parser.add_argument("--discount-factor", type=float, default=0.95, help="Discount factor for shaped returns.")
    parser.add_argument("--final-score-bonus-weight", type=float, default=0.5, help="Weight of the final score bonus added to the last transition.")
    parser.add_argument("--seed-base", type=int, default=0, help="Base game seed.")
    parser.add_argument("--policy-seed", type=int, default=0, help="Policy/random seed.")
    parser.add_argument("--hidden-size", type=int, default=48, help="Hidden layer size for the policy MLP.")
    parser.add_argument(
        "--greedy-evaluation",
        action="store_true",
        help="Use greedy action selection instead of sampling during each iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder, action_indexer, policy = build_reinforce_policy(
        player_count=args.players,
        seed=args.policy_seed,
        hidden_size=args.hidden_size,
    )
    for iteration_index in range(args.iterations):
        stats = run_reinforce_iteration(
            policy,
            encoder=encoder,
            action_indexer=action_indexer,
            config=ReinforceConfig(
                player_count=args.players,
                episode_count=args.episodes,
                actor_learning_rate=args.actor_learning_rate,
                critic_learning_rate=args.critic_learning_rate,
                discount_factor=args.discount_factor,
                final_score_bonus_weight=args.final_score_bonus_weight,
                seed_base=args.seed_base + (iteration_index * args.episodes),
                policy_seed=args.policy_seed,
                greedy_evaluation=args.greedy_evaluation,
            ),
        )
        print(
            f"Iteration {iteration_index + 1}: "
            f"avg_score={stats.average_score:.3f} "
            f"min={stats.min_score} "
            f"max={stats.max_score} "
            f"avg_return={stats.average_return:.4f} "
            f"avg_shaped_return={stats.average_shaped_return:.4f} "
            f"avg_value={stats.average_value_prediction:.4f} "
            f"avg_advantage={stats.average_advantage:.4f} "
            f"transitions={stats.total_transitions}"
        )


if __name__ == "__main__":
    main()
