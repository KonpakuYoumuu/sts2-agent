# Slay the Spire 2 AI agent

An RL/neural-network agent for Slay the Spire 2. See [docs/00-project-plan.md](docs/00-project-plan.md) for the plan and progress, [docs/01-interface-research.md](docs/01-interface-research.md) for how the game connection works, [docs/02-heuristic-bot.md](docs/02-heuristic-bot.md) for the rule-based bot and results, and [docs/04-imitation-learning.md](docs/04-imitation-learning.md) for the neural combat policy.

**New PC? Start with [docs/SETUP.md](docs/SETUP.md)** (game version, mod install, Python environment, how we work together).

**Python:** use the project environment `.venv\Scripts\python.exe` (PyTorch + CUDA).

## Layout

```
agent/
  interface/client.py   HTTP client for the STS2MCP mod: get_state, act, wait-until-actionable
  interface/actions.py  legal_actions(state): every legal move, in the mod's action format
  bots/base.py          Policy interface shared by all bots
  bots/random_bot.py    random legal-move bot (robustness test)
  bots/heuristic.py     rule-based bot (baseline + imitation-learning teacher)
  bots/parsing.py       card text / enemy intent -> numbers
  bots/explore.py       --explore wrapper: random moves for data collection, flagged in logs
  bots/nn_bot.py        neural combat policy (+ heuristic outside combat)
  learn/features.py     state + legal actions -> arrays (shared by imitation learning and RL)
  learn/build_dataset.py  run logs -> combat dataset
  learn/model.py        transformer policy/value network
  learn/train.py        behavior cloning
  learn/compare.py      compare bots: floors, HP lost per fight, agreement with the teacher
  harness/runner.py     plays full runs: menus, stuck detection, per-decision logging
  harness/run.py        command-line entry point
  harness/supervise.py  unattended headless collection with game relaunch
  harness/record_human.py  record a human's run as a state sequence
  data/ironclad_cards.json  card values used by the heuristic bot (tunable)
tests/                  unit tests (no game needed)
external/STS2MCP/       the game mod's source, with our patches
patches/                our changes to the mod, as a diff against upstream
logs/                   run logs (one .jsonl.gz per run + summary.jsonl)
data/                   built datasets        models/   trained networks
```

## Running

1. Start Slay the Spire 2 (mods enabled, profile 2) and leave it on the main menu.
2. Play runs:

   ```
   python -m agent.harness.run --runs 10 --bot random
   ```

   Bots: `random`, `heuristic`, `nn` (loads `models/combat_bc_v2`). Options: `--seed N` (bot randomness), `--log-dir DIR`, `--keep-going` (don't stop when a run gets stuck).
3. **Unattended, headless and ~5× faster** (no window, muted; relaunches the game if it crashes):

   ```
   python -m agent.harness.supervise --runs 600 --bot heuristic --explore 0.1 --log-dir logs/night
   ```

   Needs Steam running and `steam_appid.txt` (containing `2868840`) in the game folder; close the normal game first. It launches `SlayTheSpire2.exe --headless` with `STS2MCP_FAST=1`. Stops and waits when a Timeline reveal is needed.
4. Record yourself playing: `python -m agent.harness.record_human` (ends at game over).
5. Tests: `python -m pytest -q`
6. Imitation learning:

   ```
   python -m agent.learn.build_dataset logs/night_0927 --out data/combat_v1
   python -m agent.learn.train --data data/combat_v1 --out models/combat_bc_v1
   python -m agent.learn.compare logs/eval_heur_0927 logs/eval_nn_0927
   ```

Don't click in the game while the bot is playing. If the bot stops with "manual action needed", open **Timeline** in the main menu, reveal the new unlock, and rerun.

## Logs

Each `logs/<session>/run_NNNN.jsonl.gz` line is one decision: `state` (full game state), `action`, `ok`/`message` (the mod's response), `n_legal`, `changed`. `summary.jsonl` has one line per run: outcome, act, floor, steps, seconds. A stuck run also writes `run_NNNN_stuck_state.json`.

## Rebuilding the mod

The game must be closed to replace the DLL.

```powershell
$env:STS2_GAME_DIR = "C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"   # your game folder
cd external\STS2MCP
.\build.ps1
Copy-Item out\STS2_MCP\STS2_MCP.dll "$env:STS2_GAME_DIR\mods\STS2_MCP\" -Force
```

`external/STS2MCP` is upstream commit `55e0648` plus our fixes; `patches/sts2mcp-changes.patch` lists exactly what we changed. On the original PC the .NET SDK is project-local: put `.tools\dotnet` first on `PATH` before building.
