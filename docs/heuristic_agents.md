# Heuristic Agents

## Overview

The project currently includes three baseline agents:

- `RandomAgent`
- `BasicHeuristicAgent`
- `ConventionHeuristicAgent`
- `TempoHeuristicAgent`
- `ConventionTempoHeuristicAgent`

The heuristic agents use only partial observations and public information.

The shared heuristic stack now sits on top of `PublicBeliefState`, which means
the decision policy does not recompute every public inference from scratch in
multiple places. Instead:

- `PlayerObservation` supplies visible game facts
- `PublicBeliefState` derives reusable public hand knowledge and card beliefs
- heuristic agents focus on prioritizing actions

Internally, the heuristic implementation is now split so that:

- `base.py` owns the top-level action policy
- `_mixins.py` owns shared belief and scoring internals
- `_scoring.py` defines common score aliases and utility helpers
- `_convention_mixin.py` isolates the private convention logic used only by
  `ConventionHeuristicAgent`

Recent architecture direction:

- `BaseHeuristicAgent` owns the shared hint-priority pipeline and common action
  ordering
- `BasicHeuristicAgent` owns the baseline player-count-specific hint weights
- `ConventionHeuristicAgent` layers private communication on top of that basic
  baseline
- `TempoHeuristicAgent` keeps the basic baseline weights intact and only adds
  its own tactical hint-economy adjustments
- `ConventionTempoHeuristicAgent` combines convention-aware communication with
  tempo-aware spending rules

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

The basic baseline now also includes soft player-count-aware weighting in hint
selection. The principles stay the same across formats, but their relative
importance shifts with the table size. In practice that means larger tables can
care more about turn distance, near-term receivers, and visible follow-on play
value without turning those ideas into hard rules.

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

## `TempoHeuristicAgent`

Defined in `src/hanabi_ai/agents/heuristic/tempo.py`.

This experimental variant keeps the same public-inference stack as the basic
heuristic, but changes one policy choice: when hint economy is low, it becomes
less willing to spend the team's last hint token on a hint that does not create
an immediate actionable play.

In practice this means:

- if a hint creates a guaranteed safe play, tempo still spends the hint
- if hint tokens are low and the best hint is mostly informational, tempo
  prefers recovering the economy with a discard when legal

This makes it a good comparison point when testing whether a stronger
short-horizon tempo policy beats a more information-friendly baseline.

The current design intentionally keeps tempo-specific behavior separate from
the baseline player-count weights. That separation makes it easier to tune the
shared heuristic profile in `BasicHeuristicAgent` without accidentally baking
tempo policy decisions into every other agent.

## `ConventionTempoHeuristicAgent`

Defined in `src/hanabi_ai/agents/heuristic/convention_tempo.py`.

This hybrid variant combines the two ideas above:

- it uses the private hint-ordering conventions from `ConventionHeuristicAgent`
- it uses the hint-economy policy from `TempoHeuristicAgent`

In practice, this makes it the natural experiment for checking whether better
private communication and stricter hint spending are complementary or whether
one suppresses the other.

So far, this hybrid has generally been the strongest aggregate heuristic in
short benchmark runs across 2-5 player tables.

## Visualization Support

The CLI renderers can optionally display how advanced heuristic agents interpret
public hint history according to their private conventions.

This currently matters most for the convention heuristic, which can:

- refine own-hand knowledge from color-hint ordering
- refine own-hand knowledge from grouped rank hints
- attach human-readable convention notes to traces
