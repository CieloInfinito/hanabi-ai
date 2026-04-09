# Search Foundations

This document records the first `search-foundations` branch cycle: why the
work was introduced, what changed technically, what the benchmarks showed, and
what conclusions we can honestly draw from the current results.

## Goal

The objective of this branch is not to "have search code" for its own sake.
The objective is to test whether a search-based layer can outperform the
current strongest heuristic baseline, `ConventionTempoHeuristicAgent`, and
therefore move the project closer to the real research question:

- can Hanabi be played well enough to win every game, for any player count,
  when all players act as well as possible?

At the start of this branch, the repository already had:

- a stable Hanabi engine
- legal partial observations
- a strong heuristic family
- evaluation and tracing tools

What it did not have was an online search layer over compatible hidden worlds.

## Scope Of The First Cycle

The first cycle focused on adding the minimum viable infrastructure required to
experiment with search safely:

- engine cloning and state export
- compatible-world determinization from one player observation
- short-horizon rollout planning
- a hybrid `SearchHeuristicAgent`
- better evaluation metrics for near-perfect play
- diagnostic tools for comparing search decisions against the heuristic prior

No new runtime dependencies were introduced. The project still uses only the
standard library for this search track.

## Technical Changes

### 1. Engine snapshotting for simulation

Search requires cheap copies of the game state. The engine now supports:

- `EngineState`
- `HanabiGameEngine.export_state()`
- `HanabiGameEngine.from_state(...)`
- `HanabiGameEngine.clone()`

These changes make it possible to simulate candidate lines without mutating the
real game.

Relevant files:

- [engine.py](../src/hanabi_ai/game/engine.py)
- [test_engine.py](../tests/game/test_engine.py)

### 2. Evaluation metrics closer to the real goal

Average score alone is too coarse for this project. The evaluation pipeline now
also reports:

- `score_at_least_20_rate`
- `score_at_least_24_rate`
- `perfect_game_rate`
- `average_gap_to_25`

That makes it easier to tell whether an agent is actually getting closer to
perfect play.

Relevant files:

- [self_play.py](../src/hanabi_ai/training/self_play.py)
- [evaluate_agents.py](../src/hanabi_ai/tools/evaluate_agents.py)
- [test_self_play.py](../tests/training/test_self_play.py)
- [test_evaluate_agents.py](../tests/tools/test_evaluate_agents.py)

### 3. Compatible-world determinization

A new search package was introduced to sample hidden hands and reconstruct
simulation-ready engine states compatible with the current observation.

Main pieces:

- `CompatibleWorld`
- `sample_hidden_hand(...)`
- `build_determinized_engine_state(...)`
- `sample_compatible_world(...)`
- `sample_compatible_worlds(...)`

This is intentionally conservative: it reconstructs worlds that are compatible
with public information and the observing player's legal knowledge, but it does
not yet model deeper common-knowledge recursion.

Relevant files:

- [determinization.py](../src/hanabi_ai/search/determinization.py)
- [test_determinization.py](../tests/search/test_determinization.py)

### 4. Short-horizon rollouts and planner

On top of determinization, the branch added:

- `RolloutSummary`
- `evaluate_action_rollout(...)`
- `ShortHorizonPlanner`
- `ScoredAction`

The planner samples multiple compatible worlds, evaluates candidate actions,
and ranks them by short rollouts with heuristic continuation.

Relevant files:

- [rollout.py](../src/hanabi_ai/search/rollout.py)
- [planner.py](../src/hanabi_ai/search/planner.py)
- [test_planner.py](../tests/search/test_planner.py)

### 5. Hybrid search agent

The new `SearchHeuristicAgent` does not replace the heuristic baseline. It uses
`ConventionTempoHeuristicAgent` as a prior and applies search conservatively on
top of it.

Important design choices from this first cycle:

- guaranteed own plays stay heuristic-driven
- strong baseline hints are preserved
- the search layer mostly intervenes around discard-like decisions
- if determinization or planning fails, the agent falls back to the baseline

This means the current search agent is genuinely different from the heuristic
baseline, but intentionally close to it in behavior.

Relevant files:

- [search_agent.py](../src/hanabi_ai/agents/search_agent.py)

### 6. Search-specific diagnostics

The branch also added tooling to inspect whether search is discovering useful
alternatives instead of just reproducing baseline choices:

- `hanabi-analyze-search`
- search support inside `hanabi-compare-decisions`
- search support inside `hanabi-evaluate`

Relevant files:

- [analyze_search_opportunities.py](../src/hanabi_ai/tools/analyze_search_opportunities.py)
- [compare_agent_decisions.py](../src/hanabi_ai/tools/compare_agent_decisions.py)
- [test_analyze_search_opportunities.py](../tests/tools/test_analyze_search_opportunities.py)
- [test_compare_agent_decisions.py](../tests/tools/test_compare_agent_decisions.py)
- [pyproject.toml](../pyproject.toml)

## Search Tuning That Was Tried

The branch did not stop at "search compiles". Several rounds of tuning were
run, and most of them were intentionally conservative after early regressions.

### Initial search layer

The first planner was structurally correct but too weak:

- it was legal
- it was testable
- it often underperformed badly against `ConventionTempo`

Early smoke tests in 2-player games showed severe degradation.

### Constrained hybrid behavior

The next step was to keep search from overriding obviously strong heuristic
decisions. That removed the worst regressions and brought the search agent back
to roughly baseline strength.

### Hint-value pass

The latest pass added `hint_value.py` to score hint actions more directly,
especially for:

- guaranteed-play creation
- immediate playable touches
- critical-card touches
- pressure relief for the receiver

One important lesson from that pass: rewarding generic information gain too
aggressively hurts performance. The planner now uses only a small, focused
static hint bonus that mainly acts as a tie-breaker for hints that create near-
term value.

Relevant files:

- [hint_value.py](../src/hanabi_ai/search/hint_value.py)
- [test_hint_value.py](../tests/search/test_hint_value.py)

## Benchmark Results

### Multi-table smoke benchmark

A short benchmark over `2, 3, 4, 5` players with `10` games per table was
saved to:

- [search_baseline_2to5_10g.json](../reports/search_baseline_2to5_10g.json)

That benchmark showed:

- `SearchHeuristicAgent` was competitive with `ConventionTempo`
- the clearest improvement signal appeared in `2p`
- in `3p-5p`, the search layer was mostly tied with the baseline

### Wider 2-player check

A wider 2-player benchmark over `30` games was saved to:

- [search_2p_30g.json](../reports/search_2p_30g.json)

That broader sample showed that the apparent 2-player gain was not yet robust:

- `ConventionTempoHeuristicAgent`: `15.733`
- `SearchHeuristicAgent`: `15.700`

The latest hint-value tuning improved the short `10`-game smoke check again:

- `ConventionTempoHeuristicAgent`: `16.2`
- `SearchHeuristicAgent`: `16.3`

But that is still only a light signal, not proof of a stable improvement.

## What We Can Honestly Claim

After this first cycle, the strongest honest claims are:

- the repository now has a real search infrastructure, not just a plan
- search over compatible worlds is integrated into the project end-to-end
- the search agent no longer catastrophically degrades the heuristic baseline
- the search tooling can surface concrete turns where hints appear better than
  baseline discards

What we cannot honestly claim yet:

- that `SearchHeuristicAgent` is robustly stronger than `ConventionTempo`
- that the current planner meaningfully closes the gap to near-perfect Hanabi
- that the search layer has already justified itself as a stronger policy,
  rather than as useful research infrastructure

## Interpretation

The current search agent is technically distinct from the heuristic baseline,
but not yet empirically different enough.

That matters because this branch is meant to answer a performance question, not
a code-architecture question. If a search or learning layer starts from a
heuristic and ends up matching it without beating it reliably, then that new
layer is still mostly infrastructure.

That does not make the branch a failure. It means the branch succeeded at
building the experimental platform needed for the next step:

- better short-horizon hint evaluation
- better modeling of the partner's next action after a hint
- larger and more reliable 2-player benchmarks
- eventually, a stronger planner that earns its complexity

## Recommended Next Step

The clearest next target is still 2-player Hanabi.

The recommended path is:

1. keep `ConventionTempoHeuristicAgent` as the hard baseline
2. study seeds where search and baseline diverge
3. improve hint-line evaluation, not generic information scoring
4. require search to beat the baseline on wider 2-player benchmarks before
   expanding scope further

Until that happens, the search branch should be viewed as a solid research
foundation, not yet as a new best agent.
