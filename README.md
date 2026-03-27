# Hanabi AI

Public Hanabi project focused on building:

1. A fully functional game engine
2. Correct partial observations for each player
3. A self-play framework
4. Agent training pipelines, from heuristics to reinforcement learning

## Current Status

The repository already includes a first working vertical slice:

- Strongly typed card and action models
- Pure Hanabi rule helpers
- A playable game engine with turn progression
- Partial observation building with hidden own-hand cards
- Baseline random and heuristic agents
- Observation-side inference based on remaining public card copies
- Heuristic policies that score discard risk and hint actionability
- Self-play evaluation utilities
- Automated tests for core behavior

Recent heuristic and evaluation progress:

- Multi-table benchmarking with JSON export and report-to-report comparison
- A dedicated `TempoHeuristicAgent` focused on hint economy
- A `ConventionTempoHeuristicAgent` hybrid that combines private conventions
  with tempo-aware hint spending
- A `LargeTableHeuristicAgent` variant tuned for 5-player communication
- Shared hint-priority infrastructure across heuristic agents
- Player-count-aware baseline weighting in `BasicHeuristicAgent`
- Lightweight development notes for tracking strategy discoveries over time

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

Important: activating `.venv` only switches Python environments. The
`hanabi-demo-*` and `hanabi-evaluate` commands are created by
`python -m pip install -e .`.

For notebook work and other local development tooling:

```powershell
python -m pip install -e .[dev]
```

Dependency split:

- `pyproject.toml` is the source of truth for package metadata and dependencies
- `requirements.txt` remains intentionally empty because runtime currently uses
  only the Python standard library
- `requirements-dev.txt` is a compatibility shim for environments that still
  prefer requirements files during local setup

## Quick Start

Run the 2-player basic heuristic demo:

```powershell
hanabi-demo-basic
```

Run the 2-player convention heuristic demo:

```powershell
hanabi-demo-convention
```

Run a batched comparison between the heuristics and the random baseline:

```powershell
hanabi-evaluate --players 2 --games 200
```

Run a broader benchmark across multiple table sizes and save a machine-readable
report:

```powershell
hanabi-evaluate --players 2 3 4 5 --games 200 --json-output reports\benchmark.json
```

The benchmark includes the basic, convention, tempo, convention-tempo hybrid,
and large-table heuristic agents.

Compare a new run against a previous saved report:

```powershell
hanabi-evaluate --players 2 3 4 5 --games 200 --compare-json reports\benchmark_previous.json --json-output reports\benchmark_current.json
```

Equivalent module form:

```powershell
python -m hanabi_ai.tools.demo_basic_trace
python -m hanabi_ai.tools.demo_convention_trace --game-seed 7
python -m hanabi_ai.tools.evaluate_agents --players 2 --games 200
python -m hanabi_ai.tools.evaluate_agents --players 2 3 4 5 --games 200 --json-output reports\benchmark.json
python -m hanabi_ai.tools.evaluate_agents --players 2 3 4 5 --games 200 --compare-json reports\benchmark_previous.json --json-output reports\benchmark_current.json
```

If you want to run the tools without installing the package first, point Python
at the `src` tree explicitly:

```powershell
$env:PYTHONPATH = "src"
python -m hanabi_ai.tools.demo_convention_trace --game-seed 7
```

## Examples

Random self-play:

```python
from hanabi_ai.agents.random import RandomAgent
from hanabi_ai.training.self_play import run_self_play_game

agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
result = run_self_play_game(agents, seed=3)

print(result)
```

CLI visualization:

```python
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.visualization.cli import render_game_state, render_player_observation

engine = HanabiGameEngine(player_count=2, seed=7)

print(render_game_state(engine))
print()
print(render_player_observation(engine.get_observation(0)))
```

Self-play trace:

```python
from hanabi_ai.agents.random import RandomAgent
from hanabi_ai.training.self_play import run_self_play_game_with_trace

agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
traced_game = run_self_play_game_with_trace(agents, seed=3)

print(traced_game.trace)
```

## Documentation

- [Architecture](docs/architecture.md)
- [Heuristic Agents](docs/heuristic_agents.md)
- [Agent Notes](docs/agent_notes.md)
- [Testing](docs/testing.md)

## Next Steps

Planned next milestones:

- Continue improving observation-side card knowledge modeling
- Turn the newer risk-aware and player-count-aware heuristic ideas into
  stronger play and hint policies
- Keep tightening evaluation around policy quality, not just engine correctness
- Add more engine and edge-case tests
- Build training utilities on top of self-play
- Introduce RL agents once the environment is stable
