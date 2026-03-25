# Architecture

## Project Principles

This project treats Hanabi as an imperfect-information cooperative game. The
most important architectural rule is to keep these three layers separate:

- Real game state: the omniscient state owned by the engine
- Player observation: the partial information visible to one player
- Agent logic: decision-making based only on the observation

If an agent can access hidden information, training and evaluation become
invalid.

## Source Layout

```text
src/hanabi_ai/
|- agents/
|  |- beliefs.py
|  |- heuristic/
|  |  |- base.py
|  |  |- basic.py
|  |  |- convention.py
|  |  `- tempo.py
|  `- random.py
|- game/
|  |- actions.py
|  |- cards.py
|  |- engine.py
|  |- observation.py
|  `- rules.py
|- tools/
|  |- demo_basic_trace.py
|  |- demo_convention_trace.py
|  `- evaluate_agents.py
|- training/
|  `- self_play.py
`- visualization/
   `- cli.py
```

## Architecture At A Glance

```text
engine (real game state)
   |
   v
observation builder (partial information)
   |
   v
agent.act(observation)
   |
   v
action
   |
   v
engine.step(action)
```

## Core Modules

### `game/cards.py`

Defines:

- `Color`
- `Rank`
- `Card`
- Standard Hanabi deck construction
- Hand-size rules by player count
- Game-wide constants such as max hint and strike tokens

Hand sizes follow standard Hanabi rules:

- 2 or 3 players: 5-card hands
- 4 or 5 players: 4-card hands

### `game/actions.py`

Defines explicit typed actions:

- `PlayAction`
- `DiscardAction`
- `HintColorAction`
- `HintRankAction`

### `game/rules.py`

Contains pure helpers for:

- Playability checks
- Score computation
- Win/loss checks
- Hint-token constraints
- Discard legality

The game uses 3 shared lives. A life is only lost when a player tries to play a
card that does not fit the current fireworks. When lives reach 0, the game is lost.

### `game/observation.py`

Builds the partial view for one player and tracks hidden-hand knowledge:

- Own real cards are not exposed
- Other players' real cards are visible
- Public state is included
- Legal actions are included for the active player
- Safe-play helpers can detect own-hand indices that are guaranteed playable
  from current knowledge alone
- Public-card counting helpers can estimate how many unseen copies of each
  card identity still remain compatible with the observer's hand

This module is still intentionally observation-side only: it does not leak
hidden cards from the engine, but it does derive stronger public-information
features from visible hands, discards, and fireworks.

The current heuristic layer builds on top of these features to estimate card
distributions, discard risk, and whether a hint would likely create an
immediately safe follow-up action for the receiving player.

### `agents/beliefs.py`

Builds derived belief views on top of a single `PlayerObservation`.

This layer is intentionally separate from both the engine and the raw
observation model:

- `PlayerObservation` contains visible facts
- `PublicBeliefState` contains reusable public inference
- Heuristic agents consume both to choose actions

The current `PublicBeliefState` centralizes:

- reconstructed public knowledge of each player's own hand
- remaining public copy counts
- weighted hidden-card distributions
- public hint updates used for hint scoring

This keeps inference reusable across one turn without turning the game engine
into a belief tracker.

### `game/engine.py`

Implements the omniscient Hanabi state and core API:

- `reset()`
- `step(action)`
- `get_legal_actions(player_id)`
- `get_observation(player_id)`
- `is_terminal()`
- `get_score()`

The engine currently supports:

- Standard deck shuffling
- Dealing
- Playing, discarding, and giving hints
- Strike tracking
- Hint-token spending and recovery
- Final-round countdown after the deck is exhausted
- Per-turn history records

### `training/self_play.py`

Runs full games between agents and exposes compact summaries and aggregate
evaluation helpers.

The current heuristic stack uses this layer mainly for regression-style
comparison between agent variants while the observation and policy models are
still evolving.

That evaluation role matters because heuristic changes are increasingly about
decision quality, not just legality, so the project uses repeated self-play as
the main guardrail for policy iteration.

### `visualization/cli.py`

Provides text-based renderers for:

- Full omniscient game state
- One player's partial observation
- Cards, fireworks, and actions in compact debug-friendly form
