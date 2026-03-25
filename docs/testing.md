# Testing

## Test Layout

Tests are organized by domain:

- `tests/game/test_engine.py` covers core engine transitions and observation building
- `tests/agents/test_random.py` covers the random baseline agent
- `tests/agents/heuristic/_shared.py` checks the baseline decision logic that every
  heuristic agent in the family should satisfy
- `tests/agents/heuristic/test_basic.py` checks that the basic heuristic does
  not emit or interpret any private hint-ordering conventions
- `tests/agents/heuristic/test_convention.py` checks that the convention
  heuristic emits and interprets its private color-order and
  rank-playability conventions
- `tests/training/test_self_play.py` covers self-play execution and aggregate evaluation
- `tests/visualization/test_cli.py` covers the text CLI renderers

## Running Tests

Run the full suite with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Run a single module with:

```powershell
python -m unittest tests.game.test_engine
```

## Test Style

Heuristic tests are split between shared and agent-specific responsibilities:

- shared tests validate behavior expected from any heuristic in the family
- basic tests verify the absence of private conventions
- convention tests verify the presence of the agreed private conventions

Each test includes a short comment describing exactly what it verifies.
