from __future__ import annotations

import argparse

from hanabi_ai.training.warm_start import (
    WarmStartConfig,
    run_warm_started_reinforce,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run behavior cloning and then continue with REINFORCE."
    )
    parser.add_argument("--players", type=int, default=2, help="Number of players.")
    parser.add_argument(
        "--bc-episodes",
        type=int,
        default=8,
        help="Teacher self-play episodes used for behavior cloning.",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=4,
        help="Supervised epochs over the collected demonstrations.",
    )
    parser.add_argument(
        "--bc-learning-rate",
        type=float,
        default=0.05,
        help="Behavior cloning learning rate.",
    )
    parser.add_argument(
        "--rl-iterations",
        type=int,
        default=3,
        help="Number of REINFORCE iterations after cloning.",
    )
    parser.add_argument(
        "--rl-episodes",
        type=int,
        default=10,
        help="Episodes per REINFORCE iteration.",
    )
    parser.add_argument(
        "--rl-learning-rate",
        type=float,
        default=0.002,
        help="REINFORCE learning rate.",
    )
    parser.add_argument(
        "--rl-discount-factor",
        type=float,
        default=0.95,
        help="Discount factor for shaped returns during REINFORCE.",
    )
    parser.add_argument(
        "--rl-final-score-bonus-weight",
        type=float,
        default=0.5,
        help="Final score bonus weight applied on the last transition during REINFORCE.",
    )
    parser.add_argument("--seed-base", type=int, default=0, help="Base game seed.")
    parser.add_argument("--policy-seed", type=int, default=0, help="Initial policy seed.")
    parser.add_argument("--hidden-size", type=int, default=48, help="Hidden layer size for the policy MLP.")
    parser.add_argument(
        "--greedy-evaluation",
        action="store_true",
        help="Use greedy action selection during REINFORCE iterations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_warm_started_reinforce(
        WarmStartConfig(
            player_count=args.players,
            cloning_episode_count=args.bc_episodes,
            cloning_epochs=args.bc_epochs,
            cloning_learning_rate=args.bc_learning_rate,
            reinforce_iterations=args.rl_iterations,
            reinforce_episode_count=args.rl_episodes,
            reinforce_learning_rate=args.rl_learning_rate,
            reinforce_discount_factor=args.rl_discount_factor,
            reinforce_final_score_bonus_weight=args.rl_final_score_bonus_weight,
            seed_base=args.seed_base,
            policy_seed=args.policy_seed,
            greedy_evaluation=args.greedy_evaluation,
        ),
        hidden_size=args.hidden_size,
    )

    cloning_stats = stats.cloning_stats
    print(
        f"Behavior cloning: avg_score={cloning_stats.average_score:.3f} "
        f"min={cloning_stats.min_score} "
        f"max={cloning_stats.max_score} "
        f"samples={cloning_stats.sample_count} "
        f"accuracy={cloning_stats.training_accuracy:.3%}"
    )

    for iteration_index, reinforce_stats in enumerate(stats.reinforce_stats, start=1):
        print(
            f"RL iteration {iteration_index}: "
            f"avg_score={reinforce_stats.average_score:.3f} "
            f"min={reinforce_stats.min_score} "
            f"max={reinforce_stats.max_score} "
            f"avg_return={reinforce_stats.average_return:.4f} "
            f"avg_shaped_return={reinforce_stats.average_shaped_return:.4f} "
            f"avg_value={reinforce_stats.average_value_prediction:.4f} "
            f"avg_advantage={reinforce_stats.average_advantage:.4f} "
            f"transitions={reinforce_stats.total_transitions}"
        )


if __name__ == "__main__":
    main()
