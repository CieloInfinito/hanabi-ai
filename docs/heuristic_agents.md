# Heuristic Agents

## Overview

The project currently includes three baseline agents:

- `RandomAgent`
- `BasicHeuristicAgent`
- `ConservativeHeuristicAgent`

The heuristic agents use only partial observations and public information.

## `RandomAgent`

Defined in `src/hanabi_ai/agents/random.py`.

This baseline samples uniformly from legal actions.

## `BasicHeuristicAgent`

Defined in `src/hanabi_ai/agents/heuristic/basic.py`.

This is the shared rule-based baseline without any private hint-ordering
conventions. It uses simple public deductions based on:

- Visible teammate hands
- Current fireworks
- Discarded cards

Shared priorities:

- Play a card that is guaranteed playable from current knowledge
- Otherwise, give the most useful legal hint to another player
- Otherwise, choose the safest available discard
- Avoid blind plays whenever a discard is legal
- If forced to play, choose the own-hand card with the highest inferred
  probability of being playable

## `ConservativeHeuristicAgent`

Defined in `src/hanabi_ai/agents/heuristic/conservative.py`.

This variant keeps the same local priorities as the basic heuristic but adds
two private communication conventions:

- Color hints are pointed in ascending rank order, so the receiver can also
  infer when two or more hinted cards share a rank
- Rank hints are grouped by immediate playability first, then non-playability

These conventions intentionally live in the agent layer rather than the engine,
so different bots can adopt different hidden signaling schemes.

## Visualization Support

The CLI renderers can optionally display how advanced heuristic agents interpret
public hint history according to their private conventions.

This currently matters most for the conservative heuristic, which can:

- refine own-hand knowledge from color-hint ordering
- refine own-hand knowledge from grouped rank hints
- attach human-readable convention notes to traces
