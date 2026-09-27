# Step 2–3 — Rule-based bot and data collection (2026-09-26/27)

The rule-based ("heuristic") bot is the Phase 3 baseline and the teacher for imitation learning (Phase 4). Code: [agent/bots/heuristic.py](../agent/bots/heuristic.py), text parsing in [agent/bots/parsing.py](../agent/bots/parsing.py), card values in [agent/data/ironclad_cards.json](../agent/data/ironclad_cards.json).

## How it decides

**Combat** is greedy, one card at a time. Each legal play gets a score, and the bot takes the best score per energy (`score / (cost + 0.5)`). It ends the turn when nothing scores above 0.5.

| Component | Rule |
|---|---|
| Damage | HP damage after enemy block (×1.5 on Vulnerable targets); **+10 + 1.5 × that enemy's attack** for a kill |
| Block | Only block that stops incoming intent damage counts fully. Its weight grows with the enemy HP left (`1.6 + min(enemy HP / 100, 1.5)`), or 2.5 if the incoming hit would kill. Extra block is worth 0.15/pt |
| Lethal check | If some set of attacks in hand kills **every** enemy with the energy available, block is worth nothing (brute force over subsets; 20% overkill margin with several enemies) |
| Debuffs | Vulnerable is worth more with attacks left in hand; Weak is worth more against big hitters |
| Other | Draw +2.5/card, energy +4, Strength +3, HP loss −1 (−3 at low HP); Powers +8 in rounds 1–2, +4 after |
| Text the parser misses | Rage = 3 block per attack left in hand; Plating, heals (missing HP only), Prolong, One-Two Punch, The Bomb, card generators; anything else playable gets 1.5, so energy isn't wasted |
| Potions | Only in elite/boss fights, when an incoming hit would kill, or below 35% HP |

Card text is dynamic: descriptions already include the player's Strength/Dexterity/Weak, but not the target's Vulnerable. Intents like `"6x2"` are parsed to 12.

**Outside combat:**

| Screen | Rule |
|---|---|
| Map | Route planner ([route.py](../agent/bots/route.py)): picks the next node that maximizes the estimated chance of beating the act boss. Tracks HP along every route to the boss with losses and death risks measured from the logs (hallway fight 9% of max HP; elite 38% up to row 8, 25% after; death risk rising at low HP), small multipliers for rewards (elite relic ×1.08, treasure ×1.05), and a boss-win curve by arrival HP. Later choices are optimized too (dynamic programming), so a rest after an elite counts. Replaced the 4-floor lookahead with fixed node scores |
| Card reward | Tier-list value, minus 1.5 per copy already in the deck and −1 at 25+ cards; +1 for attacks early in Act 1; skip below 5 |
| Rest site | Same planner: heal if that gives a better chance at the boss than upgrading (×1.05). Right before the boss that means heal unless HP ≥ 90% (was: heal below 55%). Without map context: heal below 55% HP |
| Shop | Card removal (9) > relics (8) > cards valued ≥7 > cheap potions if the belt has room; then leave |
| Card grids | Remove/transform the worst card (curses first); upgrade/enchant/choose the best |
| Events | Keyword scoring: +Max HP, relic, upgrade, remove, rare, gold; −curse, injury; −lost HP (more when HP is low) |

The deck is only visible during combat, so the bot records it (all piles) at the start of each fight.

**Card values** come from the [Nat1 Gaming Ironclad tier list](https://nat1gaming.com/sts2/tier-list/ironclad-card-tier-list/) (Aug 13 2026 update): S=9, A=8, B=6.5, C=5, D=3, F=1, matched by card name. Break is S there but a ~31%-win-rate "trap" on [Spire Codex](https://spire-codex.com/guides/ironclad-tier-list), so it's set to 6. The mobalytics list blocks automated access (HTTP 403).

## Results so far (Ironclad; Ascension 0 unless noted)

| Batch | Bot version | Runs | Floor reached | Notes |
|---|---|---|---|---|
| `batch20` | random | 20 | median 7, max 17 | 20/20 completed without hangs |
| `heur10` | heuristic v1 | 5 | 8, 17, 17, 17, **33** | Beat Act 1 once; died to the Act 2 boss |
| `heur10_v2` | + tier list, lethal rule, rest fix | 6 | 7, 7, 17, 17, 17, 17 | 4/6 reach the Act 1 boss (stopped early to ship more fixes) |
| `overnight_0927b` | + deck tracking, Rage/unknown cards, block scaling, 10% exploration | 2 | 11, 12 | Stopped after finding the X-cost / enemy-block bugs |
| `overnight_0927c` | + X-cost damage, damage into block | 11 | median 9 | Windowed; replaced by headless |
| `night_0927` | same, **headless + fast mode**, 10% exploration, mostly **Ascension 1** | **596** | p25 9, **median 17**, p75 17, max **50** | 119 beat Act 1 (20%), 4 reached Act 3, 1 reached the final boss; 0 wins |
| `eval_heur_0927` | same as night_0927, no exploration | 50 | mean 16.9, median 17 | 22% beat Act 1; baseline for the route planner |
| `eval_route_0927` | + **route planner** (map + rest by estimated boss-win chance) | 26 (stopped early) | mean 22.6, median 21.5 | **54% beat Act 1** (vs 22% in `eval_heur_0927`, same bot without the planner); 89% reach the Act 1 boss (was 58%) with 86% HP (was 71%); early elites 0.04 per run (was 0.76) |

Floor 17 is the Act 1 boss (Ceremonial Beast, Kin Priest + Kin Followers, or Vantom). **The Act 1 boss is the main wall.**

## Bugs found by watching the bot

| Symptom | Cause | Fix |
|---|---|---|
| Upgraded a card at 40/85 HP right before the boss | Rest-site choice rejected as "room is not open" (timing), then blocked | Client retries transient errors |
| Ended turns holding Rage at 6 HP vs 20 damage | Parser only reads "Gain N Block"; Rage and 20 other cards scored 0 | Rules for Rage & co.; 1.5 default for any playable unknown card |
| Died to Bygone Effigy (127 HP, hits 23) at 94% HP | Took Bludgeon over two Defends every turn | Block weight scales with enemy HP left |
| Decks with 3–4 copies of the same card | No deck knowledge at card rewards | Deck tracked from combat; duplicate penalty |
| Never played Frantic Escape (Act 2 boss) | Its Sandpit value isn't in the text or state | Flat bonus; Sandpit counter still not exposed |
| Blocked when attacks in hand were lethal | No lethal check | Lethal check (suggested by the user) |
| One run looped 5,000 steps on "Choose 3 cards to Enchant" | Confirm is enabled with 0 selected; the bot confirmed at once, the empty confirm did nothing, then confirm/cancel ping-ponged | Bot picks the number of cards the prompt asks for before confirming; runner aborts a run after 100 visits to the same state |
| Played Defend vs a non-attacking Nibbit instead of Whirlwind | Damage into enemy block scored 0; X-cost "X times" counted as one hit | Breaking block worth 0.4/pt; X = energy spent |
| Ended turns with energy left, holding Toxic (took 5 damage per Toxic) | Status cards scored 0, so "play it to exhaust it" was never considered | Cards that hurt you at end of turn while in hand are worth the damage they prevent (× block weight) |

After the fixes, replaying all ~7,400 logged decisions through the bot finds no turn ended with a useful playable card left.

## Human play recording

`python -m agent.harness.record_human` waits for the game to start, then logs every distinct settled state while a person plays (the mod can't report which button a human pressed, so actions are inferred later from consecutive states). In shops it polls only every 10 s, because each state read re-opens the shop inventory. First recording: `logs/human/human_20260927-010201.jsonl.gz`, **floors 1–8 of a winning run** (108 states); the rest of that run wasn't recorded.

Second recording: `logs/human/human_20260927-161932.jsonl.gz`, a **complete winning Ascension 1 run** (48 floors, final boss Aeonglass), 1,471 states over 29 minutes. Arrived at each boss with high HP (80/80, 67/83, 85/85), which is the factor that best predicts the bot's Act 1 boss results. By the end the deck had no Strikes left, and every card was upgraded (Apotheosis).

## Known weaknesses (next to fix)

1. **Act 1 boss.** Decks are too weak and HP too low by floor 17. Candidates: rest more before the boss, value scaling cards (Strength, Demon Form) higher, remove Strikes earlier.
2. **Greedy combat.** One card at a time, no planning of card order within a turn (e.g. Bash → then attacks happens only through the Vulnerable bonus). A small search over card orderings for the current turn would fix most of this; it's also a stepping stone to the combat simulator (Phase 5).
3. **Events** use keyword scoring only; no per-event knowledge.
4. **Enemy mechanics** the bot can't see (Sandpit, Illusion, Slippery, Territorial...).

## Data collection

- Command (headless, supervised): `python -m agent.harness.supervise --runs 600 --bot heuristic --explore 0.1 --log-dir logs/night_0927`
- **First dataset (2026-09-27):** `logs/night_0927`: 596 complete runs, **172,829 decisions, of which 94,458 are combat decisions** (8.5% exploratory), 57 MB. 597 runs at Ascension 1 ("Swarming Elites": more elites), 2 at A0.
- Deaths: 225 of 596 at the Act 1 boss (floor 17). Most common final bosses: Ceremonial Beast 81, Vantom 79, Kin Priest (+ Followers) 65, The Insatiable 21, Knowledge Demon 13.
- `--explore 0.1`: 10% of combat moves are random (never end-turn/proceed). Each is logged with `"explore": true` and the bot's intended `greedy_action`. Imitation learning uses these as states but not as labels.
- Target: ~300–500 runs for combat imitation learning (≈50k–100k combat decisions), 1,000+ runs for the strategy value network (built up over later phases).
- Size: ~40–100 KB per run compressed; 1,000 runs ≈ 50–100 MB.
- The overnight run needs: PC plugged in (it sleeps after 3 min on battery), game open on profile 2, Steam auto-update off. Timeline unlock reveals still stop the bot until someone does them by hand.
