# Architecture

This project is easiest to understand if you keep one rule in mind:

- the engine knows everything
- the observation knows only what one player may legally see
- the agent acts only from that observation

Everything else follows from that.

## One Turn In The System

```text
engine state
   |
   v
player observation
   |
   v
agent.act(observation)
   |
   v
chosen action
   |
   v
engine.step(action)
```

If we ever blur those layers, Hanabi evaluation stops being trustworthy.

## Source Layout

```text
src/hanabi_ai/
|- game/
|- agents/
|- training/
|- tools/
`- visualization/
```

## What Each Folder Owns

### `game/`

The game layer is the source of truth for Hanabi itself.

- `cards.py`: card types, deck, hand sizes, shared constants
- `actions.py`: typed actions such as play, discard, and hints
- `rules.py`: pure rule helpers like playability and game-end checks
- `observation.py`: legal player-visible views of the game state
- `engine.py`: the omniscient game engine

This layer should know the rules of Hanabi, but not how a policy wants to
play.

### `agents/`

The agent layer turns observations into actions.

- `random.py`: legal random baseline
- `beliefs.py`: public-information inference built on top of observations
- `heuristic/`: the rule-based agent family

This layer should not read hidden engine state directly.

### `training/`

The training layer runs complete games between agents and summarizes results.

Right now its main role is evaluation and regression checking through self-play.

### `tools/`

The tools layer is the command-line entry point for everyday work.

- demos
- benchmarking
- decision comparison

If you want to inspect or compare agents, this is usually where to start.

### `visualization/`

The visualization layer renders game state, observations, and traces in a
terminal-friendly format.

It exists to make debugging and policy iteration easier, not to own game
logic.

## Heuristic Stack

The heuristic family is intentionally split by responsibility:

- `base.py`: high-level action selection order
- `basic.py`: baseline public-information policy
- `tempo.py`: stricter hint-economy behavior
- `convention.py`: private hint-ordering conventions
- `convention_tempo.py`: convention + tempo hybrid
- `large_table.py`: compatibility wrapper for the current 5-player hybrid slot
- `_mixins.py`, `_scoring.py`, `_convention_mixin.py`: shared internal helpers

That split keeps the public agent classes small and makes experiments easier to
contain.

## Design Rules

When adding or changing code, these constraints matter most:

1. Do not let agents access hidden cards.
2. Keep rule logic in `game/`, not in visualization or tools.
3. Keep reusable public inference separate from one-off policy choices.
4. Prefer small, measurable policy changes over large opaque rewrites.
5. Use self-play and traces together: average score alone is often not enough.

## Mental Model

If you are new to the repo, the simplest path is:

1. Read `game/engine.py` and `game/observation.py`.
2. Read `agents/heuristic/basic.py` and `agents/heuristic/base.py`.
3. Run `hanabi-demo-basic`.
4. Run `hanabi-evaluate`.
5. Use `hanabi-compare-decisions` when two agents disagree.
