# Testing

The test suite is organized the same way as the project:

- `tests/game/`: engine, rules, and observations
- `tests/agents/`: agents and heuristic behavior
- `tests/training/`: self-play runners and summaries
- `tests/tools/`: benchmark and comparison tooling
- `tests/visualization/`: terminal rendering

## What To Run Most Often

Run the whole suite:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Run one module:

```powershell
.venv\Scripts\python.exe -m unittest tests.agents.heuristic.test_convention_tempo
```

Run a small group:

```powershell
.venv\Scripts\python.exe -m unittest `
  tests.agents.heuristic.test_basic `
  tests.agents.heuristic.test_tempo `
  tests.agents.heuristic.test_convention_tempo
```

## How The Heuristic Tests Are Split

- shared tests check behavior expected from the whole heuristic family
- agent-specific tests check only the extra behavior that variant adds

That split matters because it keeps each test focused on one responsibility.

## Practical Rule Of Thumb

After changing policy code, the most useful loop is usually:

1. run the relevant unit tests
2. run a short benchmark
3. inspect a few traces or decision comparisons

Hanabi policy regressions are often visible in traces before they are obvious
from average score alone.
