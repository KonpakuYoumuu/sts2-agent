# Step 2 — Fight arena (2026-09-30)

Goal: play any single fight on demand in the real (headless) game, for combat training and exact comparisons, without playing whole runs to reach it.

## How it works

The game ships a developer console (`MegaCrit.Sts2.Core.DevConsole`) with commands like `fight <encounter>`, `card`, `relic`, `heal`. They're disabled in the release build, but the console class takes a flag that enables them, and in single-player a command runs directly through the game's own code (`fight` calls `RunManager.EnterRoomDebug`). So the mod can drive them without re-implementing any game logic.

Mod additions ([McpMod.Arena.cs](../external/STS2MCP/McpMod.Arena.cs); listed in `patches/sts2mcp-changes.patch`):

| Action | What it does |
|---|---|
| `console {command}` | Runs a developer-console command, e.g. `fight VANTOM_BOSS` |
| `arena_setup {deck, relics, max_hp, hp}` | Replaces the deck (card IDs with upgrade counts), relics and HP, outside combat. Relics are added without their pickup effects (no card choices or max-HP gains popping up) |
| `arena_status` | Whether the last `arena_setup` has finished (it runs asynchronously) |
| battle state `encounter_id` | The current fight's encounter ID, logged from now on |

Python: [agent/arena/](../agent/arena/).

- `scenarios.py` turns every logged fight start into a **scenario**: encounter, deck with upgrades, relics, HP, max HP.
  - Old logs don't record the encounter ID, so it comes from the enemy group, mapped with the game's own run history (`agent/data/encounters.json`: 60 encounters; every enemy group maps to exactly one).
  - Card IDs for draw/discard piles, which only show names, are learned from hands.
  - Result (`data/scenarios_v1.jsonl`): **5,999 scenarios** from 6,316 logged fights: 501 Act 1 boss fights, 973 Act 1 elites, 3,664 Act 1 normal fights, plus Act 2–3.
- `arena.py` plays episodes. Each one:
  1. makes sure a run is in progress (starts one after a death);
  2. runs `arena_setup` with the scenario;
  3. runs `fight <encounter>`;
  4. lets a bot play until the rewards screen or game over, and logs every decision in the normal run-log format plus a per-fight result line.

  It uses the same stuck protection as full runs.

```
python -m agent.arena.arena --episodes 150 --bot heuristic --act 1 --seed 7 --log-dir logs/arena_eval_heur
```

`--kind monster|elite|boss` and `--act N` filter the scenarios. The same `--seed` draws the same scenarios in the same order, so two bots can be compared on identical fights.

## Checks

- **Fidelity:** first test, 5 fights. Deck size, upgrades, relics, HP and max HP matched the source scenario exactly in all five. The only difference was +2 HP from Blood Vial ("heal 2 at the start of combat"), which is correct game behaviour.
- **Speed:** about **4.7 s per fight (~750 fights/hour)** including new runs after deaths. That's about the same per fight as inside full runs; the gain is choosing the fights (e.g. hundreds of boss fights) and comparing bots on identical ones.
- **Realism of hard fights:** the first paired run showed elites and especially bosses much harder in the arena (14 of 15 Act 1 boss fights lost) than in the real runs the scenarios came from (6 of 15 won). The cause was **potions**, which the scenarios didn't include; the bot saves them for elites and bosses. With potions (scenario field `potions`, set via the console's `potion` command), 40 Act 1 boss scenarios gave **17 wins in the arena and 17 in the real runs**.
- **Randomness:** the game re-rolls shuffles and enemy moves each time, so the same scenario plays out differently. Comparisons need many fights.

## Bug found with the arena

The upgrade screen (Armaments: "Confirm Card to Upgrade") never reports the picked card, and clicking again un-picks it. The bot clicked forever. In full runs the stuck detector forced a confirm after 25 repeats, so it went unnoticed: **26,976 logged states** were this loop. The bot now picks once, then confirms.

## Baseline in the arena (150 Act 1 fights, same scenarios for both bots; before the potion fix)

| | Rule-based bot | Imitation network (v2) |
|---|---|---|
| HP lost, normal fights (98) | 7.0 | 7.2 (diff +0.2 ± 2.0) |
| HP lost, elites (37) | 36.9 | 38.5 (+1.6 ± 4.9) |
| HP lost, bosses (15) | 55.7 | 55.5 (−0.2 ± 2.2) |
| Deaths | 24 | 26 |

The two play equally well, as the live runs showed. To be redone with potions as the official baseline for RL.

## RL training (started 2026-09-30)

[agent/learn/ppo.py](../agent/learn/ppo.py): PPO from `models/combat_bc_v2`.
- **Fights:** Act 1 scenarios, 40% normal fights, 30% elites, 30% bosses.
- **Reward:** HP fraction left at the end of the fight, −0.5 for dying, −0.05 per potion used.
- **Value baseline:** the network's fight-outcome head.
- **Anchoring:** a KL penalty (0.1) keeps the policy close to the imitation network.
- **Run:** `python -m agent.learn.ppo --init models/combat_bc_v2 --out models/combat_ppo_v1 --iterations 60 --fights 40`. 60 rounds of 40 fights, about 6 minutes per round. Resume after an interruption with `--init models/combat_ppo_v1 --ref models/combat_bc_v2` (same `--out`); long runs are started as a detached process (`Start-Process`) so they outlive the terminal.

### Result of the first RL run (`models/combat_ppo_v1`, finished 2026-10-01)

2,400 training fights. The game slowed down and hung after ~3.5 hours of continuous use. Training now restarts it every 10 rounds and recovers from hangs.

**Training trend, flat:** HP lost per training fight was 26.6 in rounds 1–10 and 29.4 in rounds 51–60. Distance from the imitation network (KL) only reached 0.06, so the policy barely changed.

**Paired comparison, 150 Act 1 fights with potions** (`python -m agent.arena.compare_bots`; `logs/arena_compare_1001/summary.md`):

| Fights | Rule-based | Imitation (v2) | PPO v1 |
|---|---|---|---|
| Normal (98): HP lost / deaths | 6.9 / 2 | 6.2 / 4 | 7.4 / 3 |
| Elite (37) | 29.6 / 3 | 35.6 / 10 | 32.7 / 8 |
| Boss (15) | 53.2 / 10 | 53.9 / 13 | 53.0 / 12 |
| All (150) | 17.1 / 15 | 18.2 / 27 | 18.2 / 23 |

**No improvement from RL.** PPO v1 is within noise of the imitation network. Both networks die more often than the rule-based bot, especially against elites (3 deaths vs 10 and 8). Likely reasons:
- The reward comes only once, at the end of a fight. A fight's outcome depends far more on card draws than on any one decision, so the learning signal is very noisy.
- 2,400 fights is a small budget for this.
- The KL anchor and small learning rate kept updates tiny.

Next: per-turn rewards (HP lost and enemy damage dealt each turn) with GAE, which gives much less noisy credit for each decision, and a larger step size.

## RL run 2: per-move rewards (started 2026-10-01 14:15)

Changes from run 1 ([agent/learn/ppo.py](../agent/learn/ppo.py)):

- **Reward per move**, not per fight:
  - +0.3 × the share of total enemy HP removed (potential-based shaping, so it doesn't change what's optimal);
  - − the share of own max HP lost;
  - −0.05 per potion used;
  - −0.5 on death.
- **Credit:** GAE (γ = 1, λ = 0.95) against the network's value output, now read as "reward still to come".
- **Value warm-up:** the first 5 rounds train only the value output's own layers, so its early errors can't move the policy.
- **Looser anchoring:** learning rate 1e-4 (was 3e-5), KL penalty 0.02 (was 0.1).
- **Scale:** 150 rounds × 40 fights = 6,000 fights, about 12 hours, with a game restart every 10 rounds. Snapshots in `models/combat_ppo_v2/itNNN/` every 25 rounds.
- **Evaluation:** a paired comparison on the same 150 fights is queued to run afterwards (`logs/arena_compare_ppo2/summary.md`).
