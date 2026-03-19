# Hanabi AI

Public Hanabi project focused on building:

1. A fully functional game engine
2. Correct partial observations for each player
3. A self-play framework
4. Agent training pipelines, from heuristics to reinforcement learning

## Project Principles

This project treats Hanabi as an imperfect-information cooperative game. The most important architectural rule is to keep these three layers separate:

- Real game state: the omniscient state owned by the engine
- Player observation: the partial information visible to one player
- Agent logic: decision-making based only on the observation

If an agent can access hidden information, training and evaluation become invalid.

## Current Status

The repository already includes a first working vertical slice:

- Strongly typed card and action models
- Pure Hanabi rule helpers
- A playable game engine with turn progression
- Partial observation building with hidden own-hand cards
- A baseline `RandomAgent`
- A self-play runner for complete games
- Automated tests for core engine behavior

## Implemented Modules

```text
src/card_game_ai/
|- agents/
|  |- heuristic_agent.py
|  `- random_agent.py
|- game/
|  |- actions.py
|  |- cards.py
|  |- engine.py
|  |- observation.py
|  `- rules.py
`- training/
   `- self_play.py
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

### `agents/random_agent.py`

Baseline agent that samples uniformly from legal actions.

### `agents/heuristic_agent.py`

Rule-based baseline that only uses partial observations. Its current priorities are:

- Play a card that is guaranteed playable from current knowledge
- Give a hint if another player's visible card is immediately playable
- Discard if possible
- Fall back to a play action if no better option exists

### `training/self_play.py`

Runs a full game between agents and returns a compact summary:

- Final score
- Turn count
- Remaining hint tokens
- Strike count
- Deck size
- Win/loss flags

### `visualization/cli.py`

Provides text-based renderers for:

- Full omniscient game state
- One player's partial observation
- Cards, fireworks, and actions in compact debug-friendly form

## Tests

The current test suite covers core invariants:

- Correct initial dealing
- Correct hand size for 2-5 players
- No hint actions when hint tokens are zero
- No discards when hint tokens are already full
- Wrong plays add a strike and discard the card
- Correct plays advance the fireworks
- Player observations do not expose own real cards
- Random agent always returns a legal action
- Self-play completes successfully

Run the full suite with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

You can also run a single test file directly:

```powershell
.venv\Scripts\python.exe tests\test_game_engine.py
```

The tests bootstrap `src/` automatically, so manual `PYTHONPATH` setup is not required.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Example: Random Self-Play

```python
from card_game_ai.agents.random_agent import RandomAgent
from card_game_ai.training.self_play import run_self_play_game

agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
result = run_self_play_game(agents, seed=3)

print(result)
```

## Example: CLI Visualization

```python
from card_game_ai.game.engine import HanabiGameEngine
from card_game_ai.visualization.cli import render_game_state, render_player_observation

engine = HanabiGameEngine(player_count=2, seed=7)

print(render_game_state(engine))
print()
print(render_player_observation(engine.get_observation(0)))
```

## Example: Self-Play Trace

```python
from card_game_ai.agents.random_agent import RandomAgent
from card_game_ai.training.self_play import run_self_play_game_with_trace

agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
traced_game = run_self_play_game_with_trace(agents, seed=3)

print(traced_game.trace)
```

You can also run the reusable demo script:

```powershell
.venv\Scripts\python.exe scripts\demo_trace.py
```

Use the heuristic baseline instead of the random one:

```powershell
.venv\Scripts\python.exe scripts\demo_trace.py --agent heuristic
```

Example with more players and explicit seeds:

```powershell
.venv\Scripts\python.exe scripts\demo_trace.py --agent heuristic --players 4 --game-seed 7 --agent-seed-base 10
```

## Next Steps

Planned next milestones:

- Improve observation-side card knowledge modeling
- Add more engine and edge-case tests
- Build training utilities on top of self-play
- Introduce RL agents once the environment is stable
