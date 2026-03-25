# Heuristic Agents

## Overview

The project currently includes three baseline agents:

- `RandomAgent`
- `BasicHeuristicAgent`
- `ConventionHeuristicAgent`

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
- Remaining public copy counts for each card identity

Shared priorities:

- Play a card that is guaranteed playable from current knowledge
- Otherwise, give the most useful legal hint to another player
- Otherwise, choose the safest available discard
- Avoid risky plays whenever a discard is legal
- If forced to play, choose the own-hand card with the highest inferred
  probability of being playable

The current baseline is still intentionally lightweight, but it no longer
treats every hidden-card possibility as equally likely. When a hidden card
could match several identities, the agent weights those candidates by the
number of unseen public copies still available.

This powers two practical ideas:

- Discards use approximate expected risk rather than only set-based rules
- The agent protects critical cards, meaning cards whose loss would remove the
  last remaining live copy needed to finish some future play

Hint scoring also now prefers signals that are immediately actionable and, when
possible, less noisy. In practice this means the baseline balances:

- hints that create a guaranteed safe play for the receiver
- hints that expose playable cards right now
- hints that reveal useful future information
- hints that avoid mixing one urgent signal with too many extra touched cards

When a risky self-play is the only active option left, the baseline can still
take a high-confidence probabilistic play, but only under a stricter threshold
that becomes more conservative as strike tokens increase.

## `ConventionHeuristicAgent`

Defined in `src/hanabi_ai/agents/heuristic/convention.py`.

This variant keeps the same local priorities as the basic heuristic but adds
two private communication conventions:

- Color hints are pointed in ascending rank order, so the receiver can also
  infer when two or more hinted cards share a rank
- Rank hints are grouped by immediate playability first, then non-playability

These conventions intentionally live in the agent layer rather than the engine,
so different bots can adopt different hidden signaling schemes.

Aside from those extra conventions, the convention heuristic inherits the
same remaining-copy weighting, critical-card protection, and risk-aware
discard logic from the shared heuristic base, along with the same
actionability-first hint scoring and bounded probabilistic self-play policy.

## Visualization Support

The CLI renderers can optionally display how advanced heuristic agents interpret
public hint history according to their private conventions.

This currently matters most for the convention heuristic, which can:

- refine own-hand knowledge from color-hint ordering
- refine own-hand knowledge from grouped rank hints
- attach human-readable convention notes to traces
