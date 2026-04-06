## Heuristic Search Closure

This note closes the current round of small heuristic tuning before starting a
reinforcement-learning track.

### Kept Change

The last accepted heuristic improvement is the 5-player hint reranking already
committed in `9ac0c9a` (`Improve 5-player convention-tempo hint ranking`).

Benchmark reference:

- baseline: `reports/baseline_5p.json`
- accepted improvement: `reports/convention_tempo_5p_reranked.json`

Observed gain for `ConventionTempoHeuristicAgent` over 100 five-player games:

- average score: `15.05 -> 15.31`
- score_at_least_15_rate: `66% -> 76%`

### Discarded Follow-Up Experiments

After that accepted change, several smaller follow-up ideas were explored:

1. Endgame hint conservation in 5-player tables.
2. Forced-finish hint priority in late game.
3. Endgame bonus for higher-rank visible playable cards.
4. Narrowed version of that bonus that only applied when the hint already had
   cooperative value.

These experiments did produce occasional positive deltas on short benchmark
windows, but the effect stayed marginal and unstable:

- one variant regressed slightly (`15.31 -> 15.30`)
- one variant was flat
- one variant improved only by `+0.01`
- the narrowed variant improved by `+0.03` on the same 100-game benchmark, with
  no change in `score_at_least_15_rate`

### Decision

These follow-up tweaks are intentionally discarded.

Reasoning:

- the signal is too small relative to the manual policy complexity added
- the gains depend on a small number of seeds and are hard to justify as a
  durable policy improvement
- the heuristic family appears to be in a diminishing-returns regime

The repository is therefore reset to the best accepted heuristic checkpoint
defined by commit `9ac0c9a` plus the benchmark artifacts already stored under
`reports/`.

### Why This Matters

This closure gives the project a stable handoff point:

- the heuristic baseline is strong and reproducible
- rejected micro-optimizations are documented instead of lingering uncommitted
- the next research step can focus on learning-based agents rather than more
  rule accretion

The intended next milestone is to prototype a reinforcement-learning path on
top of the current self-play environment, using the committed heuristic policy
as the benchmark to beat.
