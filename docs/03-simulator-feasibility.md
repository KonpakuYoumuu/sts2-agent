# Step 4 — Combat simulator feasibility (2026-09-27)

Question: can we build a combat simulator fast and accurate enough to train RL, instead of learning only from the live game (~1 decision/s, ~30 runs/hour)?

**Short answer: yes.** The game's combat logic is cleanly separated from its graphics and even has a built-in headless test mode. There are two viable routes; try the faster-to-build, higher-fidelity one (B) first.

Method: decompiled `sts2.dll` (v0.107.1) with ILSpy into a scratch folder (3,425 source files, 37 s). The decompiled code is for reading only: it's kept out of this repo and must not be published.

## What the code looks like

| Area | Files | Observations |
|---|---|---|
| Cards (`Models.Cards`) | 578 (all characters; **87 in the Ironclad pool**) | Each card is a small class: cost/type/rarity/target, numbers as `DynamicVar`s (e.g. Bash: `DamageVar(8)`, `PowerVar<Vulnerable>(2)`), an async `OnPlay` using a few commands (`DamageCmd.Attack`, `PowerCmd.Apply`, `CreatureCmd.GainBlock`...), and `OnUpgrade` |
| Monsters (`Models.Monsters`) | 121 (~14.7k lines) | HP ranges, move numbers (with Ascension variants), and an explicit **move state machine**: named moves, intents, follow-ups, conditional/random branches. E.g. Nibbit: Butt 12 → Slice 6 + 5 block → Hiss +2 Str → repeat; the opening depends on position |
| Powers (`Models.Powers`) | 260 (~11.7k lines) | React to events via the hook system (`Hooks/Hook.cs`, ~2.6k lines) |
| Encounters / Acts | 88 encounters; 5 act definitions (Overgrowth: 22 encounters) | Which monsters appear where |
| Commands / Combat | `DamageCmd`, `CreatureCmd`, `PowerCmd`, `CardPileCmd`, `CombatManager` | The rules engine |

**Coupling to the Godot engine is low:** only 16 of 578 cards, 8 of 260 powers and 29 of 121 monsters import Godot, mostly for visual effects.

**Built-in test support:**
- `TestMode.IsOn` is checked in **264 files** to skip visuals, animations and delays.
- `TestCardSelector` answers card-choice prompts from code.
- `TestRngInjector` overrides randomness (shuffles, generated cards, relics).
- `NonInteractiveMode` and a `RiderTestRunner` exist.
- The game recognises Godot's `--headless` flag.

So MegaCrit run the real game logic without graphics in their own tests.

**Side finding:** in release builds the game **silently downgrades "Instant Mode" to "Fast"** at startup (`NGame`: `if (!OS.HasFeature("editor") && FastMode == Instant) FastMode = Fast`). A one-line mod patch that re-enables Instant could speed up live data collection right away.

## Options

### A. Re-implement combat in Python, porting the decompiled rules

- **Speed:** ~5k–50k decisions/s per core; trivially parallel.
- **Fidelity:** good *if* ported carefully; checked by replaying logged live fights.
- **Effort:** a scoped version (87 Ironclad cards + statuses/curses, ~22 Act 1 encounters, ~40–60 powers, combat-relevant relics, the hook order) is roughly **5–8k lines of Python, several weeks**. Each game patch needs manual re-syncing.
- Numbers (card values, monster HP/moves) can be **dumped automatically** by a mod that instantiates every model through `ModelDb` and writes JSON; only the *logic* has to be ported by hand.

### B. Run the real game headless and drive combat from a mod

- Launch `SlayTheSpire2.exe --headless` (no window, no rendering) with a mod that turns on `TestMode`, builds a combat directly (player deck + encounter), and exposes a fast step/reset API.
- **Fidelity: exact.** It *is* the game, including every card, relic, enemy and future patch (just rebuild the mod).
- **Speed:** unknown until measured. The logic is C#, but commands are async and tied to Godot's frame loop. Guess: hundreds to thousands of actions/s per process, plus several processes in parallel.
- **Risks:** turning on `TestMode` inside a release build is unsupported (the real test runner isn't shipped); setting up a combat without the normal run/room flow may hit code that expects scene nodes; Steam may not allow several instances.
- **Effort:** a 1–2 session spike tells us whether it works.

### C. Keep collecting live data regardless

Needed for imitation learning, for checking a Python simulator, and for final evaluation. Costs nothing overnight.

## Recommendation

1. **Spike B first** (1–2 sessions): headless launch + mod that starts a combat in TestMode, plays random legal moves, and reports actions/s. Success = a full Act 1 fight simulated with correct results at ≥100 actions/s.
2. If B works: build the Gymnasium environment on top of it. RL trains against the real rules; no porting and no drift after patches.
3. If B fails: **A**, with the numbers auto-dumped from `ModelDb` and the logic ported for Ironclad + Act 1 first, checked against logged fights.
4. Either way, patch the Instant-mode downgrade to speed up live collection.

## Spike, step 1: whole game headless + fast (2026-09-27) ✅

Before building combat-only simulation, we tested the cheapest version of B: the **unmodified game flow, but headless and with the developers' fast switches on**, driven by the existing bot over the existing API.

**How it's set up:**

| Piece | Detail |
|---|---|
| Launch | `SlayTheSpire2.exe --headless` (Godot: no window, no rendering) |
| Steam | Launching the exe directly fails Steam init ("launch from the Steam client" popup, invisible when headless). Fix: `steam_appid.txt` containing `2868840` in the game folder (standard Steamworks practice; delete the file to undo). Steam must be running |
| `STS2MCP_FAST=1` (env var, our mod patch) | Sets `NonInteractiveMode.AutoSlayerCheck = () => true`, the flag the developers' AutoSlay bot uses. This skips `Cmd.Wait` delays, sound and music, and makes the `ActionExecutor` run actions without waiting for animation frames. It also forces `FastMode = Instant` every frame (release builds downgrade Instant to Fast at startup) and mutes the master volume, since FMOD still plays audio headless. We did **not** turn on `TestMode`, because it swaps in test card selectors that bypass the screens our API drives |
| `STS2MCP_PORT` (env var) | Overrides the API port (for several instances later) |
| Without the env vars | The mod behaves exactly as before, so the normal game is unaffected |
| Supervisor | `python -m agent.harness.supervise` launches the headless game, waits for the API, runs the runner, relaunches the game if it dies, and stops cleanly when a Timeline reveal needs a human |

**Result:**

| | Windowed (Fast mode) | Headless + fast switches |
|---|---|---|
| Time per decision | ~0.9 s | **~0.2 s** |
| Time per run | 2–5 min | **15–90 s (avg ~48 s)** |
| Runs per hour | ~30 | **~75** |
| Rules | exact | exact (same game) |

The remaining ~0.2 s/decision is the game's frame loop plus the mod's main-thread queue. Faster client polling (0.1 → 0.03 s) didn't change it. Overnight this produced **596 runs** unattended ([02-heuristic-bot.md](02-heuristic-bot.md)).

**Headless quirks found:**
- The main menu reports no options for about a second after launch; the client now waits.
- The game still needs a human for **Timeline unlock reveals** (the mod refuses to automate them), so collection pauses until someone opens the normal game and reveals them.
- Ascension: after a win the game starts runs at the highest unlocked ascension (the bot played A1 overnight). The mod can't choose the ascension yet (the character select screen has `NAscensionPanel.SetAscensionLevel` to hook into).

**What this doesn't solve:** it's still whole runs at ~75/hour. That's plenty for imitation learning, but RL needs *combat-only* episodes at thousands per second. **Step 2** of the spike is still open: from the mod, start a fight directly (chosen deck + encounter, seeded) and reset it instantly, then measure actions/s. Parallel headless instances (one port each) are a further multiplier, but they share the same save profile, which needs care.
