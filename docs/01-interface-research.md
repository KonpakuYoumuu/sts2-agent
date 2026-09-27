# Step 1 — Game interface research (2026-09-26)

Goal: decide how the agent reads StS2 state and sends actions.

## Local environment

| Item | Status |
|---|---|
| Slay the Spire 2 | Installed: `C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2`, **v0.107.1** (2026-06-18) |
| Slay the Spire 1 | Also installed (`...\common\SlayTheSpire`) |
| Game tech | C# / .NET 9 on Godot 4.5.1 Mono. `sts2.dll`, `0Harmony.dll` and a built-in mod loader ship with the game (`data_sts2_windows_x86_64\`) |
| Existing mods | `mods\DamageMeter` |
| .NET 9 SDK | **Missing** (only runtimes). Needed to build/modify a C# mod |
| Python | 3.13.5 (miniconda) |

## Existing interface mods

| Project | Interface | Coverage | Notes |
|---|---|---|---|
| **[STS2MCP](https://github.com/Gennadiyev/STS2MCP)** | REST on `localhost:15526` (+ optional MCP server) | **Full run**: combat, potions, rewards, map, events, rest, shop, treasure, card selects, menus/seed selection | MIT. Latest release v0.4.0 (May 2026), last stated test on game **v0.99.1**; unverified on v0.107.1 |
| [Communication_Mod_STS2](https://github.com/Manuelbbl/Communication_Mod_STS2) | Mod POSTs state to a Python server on `:5000` | Combat only (`PlayCard`, `EndTurn`) | Needs BaseLib; no license stated |
| [auto-spire](https://github.com/vorpus/auto-spire) | REST on `:31452` | Combat state + commands | AI client not yet built |
| [sts2-mcp-bridge](https://github.com/brmarcosbr/sts2-mcp-bridge) | Built on Communication_Mod_STS2 | Combat autopilot (greedy) | No win-rate data. Useful lessons, see below |

## STS2MCP API (the candidate)

- `GET /api/v1/singleplayer` returns state; `state_type` is one of `menu, monster, elite, boss, hand_select, rewards, card_reward, map, event, rest_site, shop, fake_merchant, treasure, card_select, bundle_select, relic_select, crystal_sphere, game_over, overlay, unknown`.
- `POST /api/v1/singleplayer` sends an action, e.g. `play_card {card_index, target}`, `use_potion {slot, target}`, `end_turn`, `select_card_reward {card_index}`, `choose_map_node {index}`, `choose_event_option {index}`, `choose_rest_option {index}`, `shop_purchase {index}`, `proceed`, `menu_select {option, seed}`.
- Hand cards include `can_play` and `unplayable_reason`. Enemy intents come as `{type, label, title, description}`. Intents and cards are **text descriptions**, so combat simulation needs our own card/enemy data.
- Hand indices shift after each play, so re-query state between actions.
- There is no explicit "ready" signal: poll until the state settles (e.g. "Opening chest...").

## Lessons from others

- Handle rejected actions. If a play fails, record it and exclude the card, or the bot loops forever.
- Never block Godot's main thread in mod code; use async calls with timeouts.
- Early Access patches silently break Harmony patches. Pin the game version, and treat mod compatibility as a thing to test after every update.

## Decision

Use **STS2MCP** as the interface instead of writing our own mod. If it turns out not to work on v0.107.1, fork it (MIT) and fix it, which needs the .NET 9 SDK.

## Install log (2026-09-26)

- The 0.4.0 release is **broken on v0.107.1** (issue #114). The fix is on `main` (#123, 2026-07-29) but unreleased.
- Game v0.111 exists and breaks the mod again (issues #131/#132 open). **Do not update the game** until that's merged; turn off Steam auto-update or pin a beta branch.
- .NET 9 SDK 9.0.318 installed project-locally at `.tools\dotnet` (not on the system PATH).
- Source at `external\STS2MCP` (commit `55e0648`). Rebuild with:
  `$env:PATH="C:\slay_the_spire_project\.tools\dotnet;"+$env:PATH; .\build.ps1 -GameDir "C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"`
- Installed to `<game>\mods\STS2_MCP\` (`STS2_MCP.dll` + `STS2_MCP.json`).

## Live verification (2026-09-26, game v0.107.1)

- `GET /` → `Hello from STS2 MCP v0.4.0`. Mod loads fine on v0.107.1.
- `POST menu_select {option: "continue"}` loaded the saved run: actions work.
- Useful facts for the client and the RL encoding:
  - **Readiness:** `battle.is_play_phase` is false while cards are drawn/animations run, so poll until true. After loading, `state_type` is `unknown` for a few seconds.
  - **Overlays:** with the map open during combat, `state_type` is `map`, not `monster`.
  - **Stable IDs:** hand cards have `id` (`STRIKE_IRONCLAD`, `BASH`), relics have `id`, enemies have `entity_id` (`NIBBIT_0`). **Draw/discard/exhaust entries only have `name`/`cost`/`description`, no `id`**, so map name → id ourselves (or patch the mod).
  - Hand cards carry `type`, `rarity`, `target_type`, `can_play`, `unplayable_reason`.
  - Intent damage is a string `label` (`"12"`); multi-hit format still unknown.
  - **Save & quit, then continue, restarts the current fight from turn 1.** This could reset fights for live RL.
- Game language set to English so names/descriptions are English.

## Quirks found while running the random bot (2026-09-26)

| Problem | Cause | Fix |
|---|---|---|
| Enchant screen never confirms (mod issue #118) | The mod only knew the Upgrade preview containers, not `%EnchantSinglePreviewContainer` / `%EnchantMultiPreviewContainer`, so every confirm re-clicked the main button | **Patched the mod** (`CardGridPreviewContainers` in `McpMod.Helpers.cs`); found via the game's own `AutoSlay` `DeckEnchantScreenHandler` |
| Shop never leaves | Reading shop state reopens the inventory, which disables Proceed (`can_proceed` is always false). The first `proceed` only starts closing the inventory | Always offer `proceed` in shops; client re-POSTs `proceed`/`end_turn` a few times **without GETting state in between** |
| Event with empty options | Options load a moment after the event appears | Events count as settled only once options (or dialogue) exist; runner re-polls 10 s before calling a run stuck |
| Neow: `advance_dialogue` does nothing | Ancients keep `in_dialogue: true` while options are clickable | Options take priority over advancing dialogue |
| Main menu has no Singleplayer | New profiles must reveal Timeline unlocks by hand; the mod blocks automating it (issue #92) | Runner stops with a "manual action needed" message |
| Duplicate menu clicks after Embark | State stays `character_select` while the run loads | Wait for the screen to change after embark/confirm |

Selection toggles (`select_card`) aren't always visible in the state, so they're never blocked as no-ops.

The game ships a developer auto-play test system (`MegaCrit.Sts2.Core.AutoSlay`, one handler per screen). It's the best reference for how each screen should be driven. Decompile with `.tools\ilspy\ilspycmd.exe` (needs `DOTNET_ROOT=.tools\dotnet`, `DOTNET_ROLL_FORWARD=Major`).

Later additions (2026-09-26/27):

| Problem | Cause | Fix |
|---|---|---|
| Rest sites picked the wrong option (e.g. Smith at 40/85 HP before the boss) | First choice rejected with "Rest site room is not open" while the room loads; the runner then blocked it and took the other option | Client re-POSTs any action rejected with `is not open` / `not ready` / `currently disabled`, since these are timing errors, not illegal moves |
| Selection clicks slow | Runner waited 3 s for a state change that toggles never show | 0.4 s wait for `select_card` / `combat_select_card` |

Found during unattended headless collection (2026-09-27):

| Problem | Cause | Fix |
|---|---|---|
| Runner logged ~600 empty "stuck" runs in seconds | Headless main menu has no options for ~1 s after launch; `--keep-going` retried instantly | Menu with no options isn't "settled"; runner stops after 3 stuck runs in a row and pauses 5 s after each |
| Restarted collection overwrote earlier run logs | Run numbering restarted at 0 | Numbering continues from `summary.jsonl` |
| Final-boss fight stalled 60 s, run marked stuck | A boss phase had **no targetable enemies**; the client required a living enemy before calling combat actionable | A player turn in play phase is actionable even with an empty enemy list |
| Wins would be logged as deaths | The game-over screen reports 0 HP after a win too; the runner decided by HP | A run is a win if it reached the ending event (The Architect) that follows the final boss. Checked against the game's own run history: the bot has no wins so far, and none were mislabeled |
| Human recording blocked leaving shops | Every state read re-opens the shop inventory | Recorder polls every 10 s in shops |

## Status: done (Phase 1 complete)

- STS2MCP installed (patched build), Python client + legal-action enumeration + run harness built.
- Random bot: **20/20 runs completed with no hangs** (median floor 7, ~30 runs/hour). See [02-heuristic-bot.md](02-heuristic-bot.md) for what came next.
