# Agent Development Notes

Short running notes about what we are learning while iterating on Hanabi
agents. This file is intentionally lightweight and should prefer concrete
observations over long prose.

## Current Observations

- The engine / observation / agent split is holding up well. Most recent
  iteration work has been about policy quality, not engine correctness.
- The baseline player-count weighting now lives in `BasicHeuristicAgent`, with
  `TempoHeuristicAgent` adding only tactical adjustments on top. That keeps the
  shared format-specific heuristics in one place.
- `TempoHeuristicAgent` is strongest in 2-player tables, where conserving hint
  economy has a clearer payoff.
- Private hint conventions matter more once the table gets larger. In 3-5
  player games, `ConventionHeuristicAgent` and the convention-tempo hybrid are
  consistently stronger than plain tempo-only play.
- A fusion approach is promising. `ConventionTempoHeuristicAgent` currently
  combines good short-horizon tempo control with better communication.
- Turn order seems to matter more in 4-5 player games than in 2-player games.
  Spending a hint on the next player can be stronger than a slightly richer
  hint to a player whose turn is farther away.
- Using turn order as a hard priority was too aggressive for plain tempo in
  larger tables. Using it as a tie-breaker or soft bonus behaved better.
- The new trace instrumentation is already paying off. Looking only at average
  scores was not enough to explain why 5-player variants won or lost; per-seed
  action comparisons exposed where policies were diverging.
- In 5-player games, local tempo improvements can be real without being net
  positive. Some seeds improved when the agent converted weak hints into
  earlier plays or discards, but the losses were often larger than the wins.
- The old `LargeTableHeuristicAgent` experiment was useful for isolating
  5-player hypotheses, but the behavior that survived that round has now been
  folded back into `ConventionTempoHeuristicAgent`.
- Private communication still seems to matter more than pure local tempo in
  5-player tables. The convention-aware agents remain the most reliable
  aggregate performers there.
- A recurring failure mode in 5-player traces is dropping shared coordination
  too early. Discarding or taking a local play one turn sooner can look good
  tactically while still weakening the table's future turns.
- `LargeTableHeuristicAgent` now remains mainly as a compatibility wrapper and
  named benchmark slot. New real 5-player policy changes should land in
  `ConventionTempoHeuristicAgent` unless the project explicitly reopens a
  separate experimental branch.
- Three recent 5-player hypotheses did not produce a stronger agent:
  "good enough hint beats discard" was behaviorally unstable and did not
  improve aggregate results; two later narrower hypotheses produced no
  effective divergence from `ConventionTempo` at all.
- When a hypothesized rule changes benchmark outcomes but not in a stable way,
  the project should treat that as evidence about the policy boundary, not as
  a failed experiment. In this case it suggests the real gap is larger than a
  single local weight or threshold tweak.

## Working Hypotheses

- In larger tables, hint timing may matter almost as much as hint content.
- A good next-player hint may deserve extra weight even when it is only
  slightly better than a discard according to raw information gain.
- In 4-5 player games, "chain value" may be approximated by combining turn
  proximity with either immediate playability or strong useful information.
- In larger tables, some players may be publicly "stuck": no guaranteed play
  and very little hint information. Hints that unlock those hands may deserve
  extra priority.
- One cheap approximation of chain value is visible follow-on unlocks: after a
  hinted playable card is played, how many visible teammate cards of the next
  rank would become immediately live?
- Hand-size differences likely matter. In 4-5 player games each player has 4
  cards, which changes both the value of a hint and the cost of a discard.
- The main 5-player decision may be "is this hint still good enough to preserve
  coordination?" rather than "is this the best immediate local tempo play?"
- Hints that are only moderately actionable can still outperform a discard if
  they preserve the table's shared plan across a long turn cycle.
- The project should treat per-seed divergence analysis as a first-class tool
  for policy iteration, not just rely on aggregate benchmark tables.
- Small 5-player policy tweaks may already be mostly subsumed by
  `ConventionTempoHeuristicAgent`; future improvements may need a more
  structural change than a local ranking or threshold adjustment.
- If the project reintroduces a dedicated `LargeTableHeuristicAgent`, it should
  again be treated as a narrow hypothesis harness: start from the strongest
  stable heuristic and introduce exactly one behavior change, then check
  whether that actually creates meaningful divergence.

## Open Questions

- Can turn-order-aware tempo rules improve plain `TempoHeuristicAgent` in 4-5
  player games without hurting its strong 2-player results?
- Is the hybrid agent strong because of communication, because of tempo, or
  because the two cover each other's weaknesses?
- Should the project eventually keep separate tuned agents for 2-player and
  4-5 player tables instead of forcing one universal heuristic?
- What is the right threshold for keeping a "good enough" hint over taking a
  local discard in 5-player games?
- Are the most valuable 5-player hints the ones that unlock the next player,
  or the ones that keep shared multi-turn coordination intact?
- Which convention-tempo trace patterns most often precede `+1` or `+2` score
  swings in larger tables?
- If local heuristic tweaks are mostly saturated, what larger policy concept is
  still missing from the current family: stronger conventions, explicit
  multi-turn plan preservation, or a better public model of teammate pressure?
