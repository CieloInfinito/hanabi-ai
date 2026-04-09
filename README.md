# "To win, or not to win. That is the question"

This is a research project with one solitary goal:

**to determine whether Hanabi can be played well enough to win every game, 
for any player count, when all players act as well as possible**

That is, to say: 
** is it possible to win every game, from start to finish, even with no luck involved?**

Everything in this repository exists to answer this "yes or no" question.

We (just me, really) don't really care whether this helps to expand the AI reaserch window,
explore new frontiers, or whatever. The engine, observation model, benchmarks,
traces, agents, and so forth are not the end goal by themselves.

They are the infrastructure needed to study the real question.

## Exodia, The Forbidden One

This is not about making a technical exhibition here. It's about a geek
trying to win the game. Every game. To do that, we'll use AI and see
what happens. No big deal, really.

Hence, *how* this question will be answered is whether or not the 
"perfect agent" can be assembled. We'll call him "Exodia". Since
it is my child, I get to name him. See Yu-Gi-Oh for reference.

## Coming back to Earth

In practice, the project currently advances toward that goal through three
immediate subgoals:

1. Model the real Hanabi game correctly.
2. Give each player only the information they are allowed to see.
3. Make it easy to compare agent policies in self-play.

## What Is In The Repo

Today the repository already includes a complete vertical slice of that
research stack:

- a typed Hanabi engine
- partial observations for each player
- random and heuristic agents
- self-play evaluation tooling
- text-based visualization for debugging and traces
- tests for engine, agents, tooling, and rendering

The project is still strongest as a clean Hanabi environment plus a heuristic
research harness, but it now also includes an initial RL scaffold.

Current lightweight RL defaults are tuned for stability rather than raw speed:
the current pure Python MLP uses `hidden_size=48`, behavior cloning defaults to
`8` teacher episodes and `4` epochs, and warm-start REINFORCE defaults to a
smaller learning rate with shaped-return discounting. The RL track is still
experimental and currently remains well below `ConventionTempoHeuristicAgent`
in self-play quality.

## Start Here

Setup:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

For notebooks and local development extras:

```powershell
python -m pip install -e .[dev]
```

Important:

- Activating `.venv` only switches Python environments.
- The `hanabi-*` commands appear after `python -m pip install -e .`.

## Most Useful Commands

Run a basic demo:

```powershell
hanabi-demo-basic
```

Run a convention-aware demo:

```powershell
hanabi-demo-convention
```

Benchmark the current agents:

```powershell
hanabi-evaluate --players 2 3 4 5 --games 200
```

Run a minimal REINFORCE smoke training loop:

```powershell
hanabi-train-behavior-cloning --players 2 --episodes 8 --epochs 4
```

```powershell
hanabi-train-reinforce --players 2 --episodes 10 --iterations 3
```

```powershell
hanabi-train-warm-start --players 2 --bc-episodes 8 --bc-epochs 4 --rl-iterations 3 --rl-episodes 10
```

Save a benchmark report:

```powershell
hanabi-evaluate --players 2 3 4 5 --games 200 --json-output reports\benchmark.json
```

Compare two heuristic agents decision-by-decision on one seed:

```powershell
hanabi-compare-decisions --players 5 --game-seed 7 --left-agent convention --right-agent convention-tempo
```

If you want to run modules directly without installing the package:

```powershell
$env:PYTHONPATH = "src"
python -m hanabi_ai.tools.evaluate_agents --players 2 3 4 5 --games 50
```

## How The Project Is Organized

The codebase is built around one architectural rule:

- the engine owns the real game state
- observations expose only legal player-visible information
- agents act only from observations

That separation is the core of the project. If an agent can see hidden cards,
evaluation becomes meaningless.

Main source layout:

```text
src/hanabi_ai/
|- game/           # engine, cards, actions, rules, observations
|- agents/         # random agent, beliefs, heuristic family
|- training/       # self-play runners and summaries
|- tools/          # demos, benchmarks, decision comparison
`- visualization/  # terminal rendering and trace output
```

## Agent Family

Current agent lineup:

- `RandomAgent`: legal random baseline
- `BasicHeuristicAgent`: public-information heuristic baseline
- `ConventionHeuristicAgent`: basic + private hint conventions
- `TempoHeuristicAgent`: basic + stricter hint-economy policy
- `ConventionTempoHeuristicAgent`: convention + tempo hybrid
- `LargeTableHeuristicAgent`: compatibility wrapper for the current 5-player
  `ConventionTempo` policy

In practice, `ConventionTempoHeuristicAgent` is the main "strong heuristic"
agent right now, and `LargeTableHeuristicAgent` remains useful as a stable
named slot for larger-table experiments.

## Common Workflows

Inspect one game with full trace:

```python
from hanabi_ai.agents.random import RandomAgent
from hanabi_ai.training.self_play import run_self_play_game_with_trace

agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
result = run_self_play_game_with_trace(agents, seed=3)

print(result.trace)
```

Render game state and a player observation:

```python
from hanabi_ai.game.engine import HanabiGameEngine
from hanabi_ai.visualization.cli import render_game_state, render_player_observation

engine = HanabiGameEngine(player_count=2, seed=7)

print(render_game_state(engine))
print()
print(render_player_observation(engine.get_observation(0)))
```

## Tests

Run the full suite:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Run one module:

```powershell
.venv\Scripts\python.exe -m unittest tests.agents.heuristic.test_convention_tempo
```

## Documentation

- [Project Architecture](docs/architecture.md)
- [Heuristic Agents](docs/heuristic_agents.md)
- [Agent Development Notes](docs/agent_notes.md)
- [Heuristic Search Closure](docs/heuristic_search_closure.md)
- [Reinforcement Learning](docs/reinforcement_learning.md)
- [Testing Guide](docs/testing.md)

## Current Direction

The current focus is not "add more infrastructure at any cost". The repo
already has the core environment. The highest-value work now is:

- improving heuristic policy quality
- understanding why agents diverge on specific seeds
- preserving the clean separation between engine, observation, and policy
- preparing a stable base for future learning-based agents
