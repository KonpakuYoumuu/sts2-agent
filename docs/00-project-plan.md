# Project plan — Slay the Spire 2 agent with RL + neural networks

## Progress (updated 2026-09-27, evening)

| Phase | Status | Result |
|---|---|---|
| 1. Interface | ✅ Done | STS2MCP mod (patched for the Enchant screen) + Python client; random bot finished 20/20 runs with no hangs |
| 2. Harness & logging | ✅ Done | Every decision logged; headless fast mode + supervisor: **~75 runs/hour** unattended (was ~30 windowed); `--explore` flags random moves |
| 3. Heuristic baseline | 🔄 In progress | 596-run headless batch: median floor 17 (Act 1 boss), 20% beat Act 1, best floor 50 (final boss); no wins yet. **Route planner (Step 1): 54% beat Act 1 vs 22% before** (26-run early result) |
| 4. Imitation learning | ✅ Done (combat) | Transformer combat policy copies the rule-based bot's move 90% of the time and **plays about as well live** (50 vs 50 runs: floor 16.6 vs 16.9, same HP loss within noise); trains in 6 min on the GPU ([04-imitation-learning.md](04-imitation-learning.md)) |
| 5. Combat simulator | 🔄 Fight arena working | Real game headless with the developers' fast switches (~75 runs/h), and now a **fight arena**: any logged fight situation (5,999 scenarios) set up and played directly, ~750 fights/h ([05-fight-arena.md](05-fight-arena.md)). Next: RL in the arena |
| 6–7 | ⏳ Not started | |

Details: [01-interface-research.md](01-interface-research.md) (interface, mod quirks), [02-heuristic-bot.md](02-heuristic-bot.md) (rule-based bot, results, known weaknesses).

## Next steps toward a high win rate (agreed 2026-09-27)

Where the bot loses (596 runs): 80% of deaths are in Act 1 (boss 38%, elites 26%, normal fights 16%). The strongest predictor of beating the Act 1 boss is HP on arrival (15% at 40–60% HP, 40–45% above 60%); deck size, basics left and relic count barely differ between wins and losses. So combat play is weak, and HP is spent badly on the way to the boss.

| Step | What | Why |
|---|---|---|
| 1. Route & rest planning ✅ (first result) | Choose map routes and heal/upgrade by an estimated chance of beating the act boss, with HP losses and death risk measured from the logs (early elites are the costliest) | Cheap, measurable; targets HP at the boss |
| 2. Fight arena ✅ | Mod change: start any fight (enemy, deck, HP, relics) directly in the headless game; fixed fight suites for evaluation | Any fight on demand (~750/hour), comparisons on identical fights; prerequisite for RL |
| 3. Combat RL (+ lookahead) | PPO from the imitation weights in the arena; try in-game lookahead for the current turn | The step that can beat the rule-based teacher |
| 4. Strategy network | Value network predicting run progress from deck/relics/HP/floor; overnight runs with randomized strategic choices; the user's recorded wins as examples | Replaces the rule-based card/path/shop/event choices |

## Goal and scope

- **Target:** a neural-network agent that clearly beats a heuristic baseline in win rate, with **one character at Ascension 0**. Expand later.
- **Metric:** win rate and mean floor reached over ≥200 runs on fixed held-out seeds, plus per-act death causes.
- **Non-goal (for now):** end-to-end RL from scratch on the live game. It's too slow (see *Why hybrid*).

## Why hybrid

| Constraint | Consequence |
|---|---|
| The live game gives roughly hundreds to low thousands of runs/day (to be measured) | Enough for supervised learning, far too little for RL from scratch |
| Wins depend on decisions made 40+ floors earlier | Credit assignment is hard, so split combat (short horizon) from run strategy (long horizon) |
| Legal actions vary every step | Action masking plus set/attention encoders over cards and enemies |
| Early Access patches change numbers | Pin the game version; keep card/enemy data in data files; re-evaluate after updates |

So: **RL where we can simulate cheaply (combat)**, **supervised/offline learning where we can't (run strategy)**, with imitation learning to bootstrap both.

## Architecture

```
 Slay the Spire 2 + STS2MCP mod  (REST, localhost:15526)
              │
      agent/interface   ── state → typed Python objects, action execution, polling, rejection handling
              │
   ┌──────────┴───────────┐
 combat policy         strategy policy
 (PPO, masked;         (win-probability value net:
  trained in sim,       card picks, path, shop, rest, events)
  fine-tuned live)
              │
      harness/  ── batch runs, logging every (state, action, outcome), metrics
              │
      sim/      ── fast Python combat simulator (one character, Act 1 → 3)
```

### Model sketch

- **Encoding:** each card, relic, power and enemy becomes a learned embedding of its ID plus numeric features (cost, upgraded, HP, block, intent damage…). Hand, piles and enemies are sets fed through a small transformer or DeepSets encoder, pooled together with global features (HP, energy, floor, gold).
- **Combat head:** logits over `(card slot × target) + potions + end_turn`, masked to legal actions. A value head for PPO.
- **Strategy value net:** `V(deck, relics, HP, gold, floor, act) → P(win)`. Score each option by the value of the state it leads to.

## Phases

### Phase 1 — Interface ✅
- Build STS2MCP from source (the 0.4.0 release is broken on game v0.107.1; the fix is on `main`). **Pin the game version.**
- `agent/interface`: `get_state()`, `act()`, wait-until-stable polling, retry/rejection handling, re-query after every card play.
- **Milestone:** a random-legal-action bot completes 50 full runs with no hangs.

### Phase 2 — Harness and data logging
- Batch runner with seeds, instant mode and (if possible) several game instances.
- Log every decision as JSONL: `(state, legal_actions, action, run_id, floor)`, plus the final outcome per run.
- Measure throughput (runs/hour), since it sets the training budget.

### Phase 3 — Heuristic baseline
- Greedy combat (block vs. incoming damage, lethal checks, power priority) and tier-list card picks / path rules.
- **Milestone:** baseline A0 win rate on held-out seeds. Everything after this is compared against it.
- Also the data source for imitation learning.

### Phase 4 — Imitation learning
- Gymnasium-style observation/action encoding shared by every later phase.
- Train the combat policy and strategy choices by behavior cloning on heuristic (and own-play) logs.
- **Milestone:** the cloned policy plays about as well as the baseline live.

### Phase 5 — Combat simulator + RL (core RL work)
- `sim/`: deterministic, seedable Python combat engine for one character's card pool and the Act 1 enemies, with card/enemy definitions in data files.
- Validate against the live game by replaying logged fights and diffing the states.
- PPO with action masking (`sb3-contrib` MaskablePPO or CleanRL), initialized from the Phase 4 policy. Reward: `−HP lost`, `+win`, a small penalty per turn.
- Extend to Acts 2–3 enemies.
- **Milestone:** the RL combat policy loses less HP per fight than the baseline, in the simulator and live.

### Phase 6 — Strategy value network
- Train `P(win | run state)` on logged run outcomes (Monte Carlo targets). Retrain as better policies generate better data.
- Use it for card rewards, pathing, shop, rest and events.
- **Milestone:** a full NN agent beats the baseline win rate with statistical significance.

### Phase 7 — Live fine-tuning and expansion (optional)
- Small-step PPO fine-tuning of the combat policy on the live game.
- More characters, higher Ascension.

## Tooling
- Python 3.11+ (3.13 installed), PyTorch, Gymnasium, sb3-contrib or CleanRL, numpy, pydantic (state schemas), TensorBoard or W&B.
- .NET 9 SDK (to build/patch the mod).
- A GPU is optional; simulator speed matters more.

## Risks
| Risk | Mitigation |
|---|---|
| Game updates break the mod | Pin the version (Steam beta branch / disable auto-update); keep a patched fork |
| The simulator doesn't match the game | Replay-diff tests against logged live fights |
| Live throughput too low | Instant mode, several instances, move more training into the simulator |
| Reward hacking in the simulator | Always evaluate live on held-out seeds |
