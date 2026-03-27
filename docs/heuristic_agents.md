# Heuristic Agents

This project has one heuristic family with a few clear variants. The easiest
way to read it is:

- `Basic` is the public-information baseline
- `Convention` adds private communication conventions
- `Tempo` changes hint spending behavior
- `ConventionTempo` combines both ideas
- `LargeTable` is currently a compatibility wrapper around the 5-player
  `ConventionTempo` policy slot

## Family Tree

```text
BaseHeuristicAgent
   |
   +-- BasicHeuristicAgent
   |      |
   |      +-- ConventionHeuristicAgent
   |      +-- TempoHeuristicAgent
   |             |
   |             +-- ConventionTempoHeuristicAgent
   |                    |
   |                    `-- LargeTableHeuristicAgent
```

## Shared Pieces

All heuristic agents:

- act only from `PlayerObservation`
- use public-information inference from `PublicBeliefState`
- try safe play first when they can prove it
- otherwise compare hints and discards
- avoid risky self-plays unless the fallback options are weak

Internal responsibilities are split like this:

- `base.py`: action-selection skeleton
- `_mixins.py`: shared belief caching and helper logic
- `_scoring.py`: hint score aliases and small helpers
- `_convention_mixin.py`: convention-only behavior

## `BasicHeuristicAgent`

Use this as the baseline public-information policy.

It tries to:

- play guaranteed safe cards
- choose useful, actionable hints
- discard the safest card when needed
- protect critical cards when discard risk is high

It also includes lightweight player-count-aware hint weighting, so 4-5 player
tables can care more about turn distance and near-term coordination without
turning those ideas into hard rules.

## `ConventionHeuristicAgent`

This is `Basic` plus private hint conventions.

Those conventions live in the agent layer, not the engine, so they can be
tested and changed without redefining Hanabi itself.

Use this variant when you want to measure whether better communication helps
more than stricter tempo policy.

## `TempoHeuristicAgent`

This is `Basic` plus a stricter hint-economy policy.

Its main question is:

- should we really spend this hint, or is it better to discard and recover
  tempo?

It is especially useful as a contrast agent in small tables, where short-horizon
hint economy matters more clearly.

## `ConventionTempoHeuristicAgent`

This is the main hybrid and usually the strongest stable heuristic.

It combines:

- the private communication conventions from `Convention`
- the hint-economy discipline from `Tempo`
- a small 5-player adjustment for hints that preserve coordination

If you want one heuristic agent to inspect first, start here.

## `LargeTableHeuristicAgent`

Right now this is a thin compatibility wrapper around the current 5-player
`ConventionTempo` behavior.

It remains in the repo for two reasons:

- it preserves a stable named benchmark slot
- it gives the project a clean place to branch future 5-player experiments

## How To Read The Heuristics

If the code feels dense, read it in this order:

1. `basic.py`
2. `tempo.py`
3. `convention.py`
4. `convention_tempo.py`
5. `base.py` only when you want the shared action-selection pipeline

That order matches the conceptual layering better than reading helper modules
first.
