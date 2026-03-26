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

## Open Questions

- Can turn-order-aware tempo rules improve plain `TempoHeuristicAgent` in 4-5
  player games without hurting its strong 2-player results?
- Is the hybrid agent strong because of communication, because of tempo, or
  because the two cover each other's weaknesses?
- Should the project eventually keep separate tuned agents for 2-player and
  4-5 player tables instead of forcing one universal heuristic?
