"""Compare bots on identical arena fights and write a paired summary.

    python -m agent.arena.compare_bots --episodes 150 --seed 7 --out logs/arena_compare \\
        --bot heuristic --bot nn=models/combat_bc_v2 --bot nn=models/combat_ppo_v1

Each bot plays the same scenarios in the same order (fresh game before each
bot). Prints, and saves to <out>/summary.md, HP lost and deaths per fight type,
with the paired difference to the first bot (± 95% interval).

--wait-pid waits for another process (e.g. training) to exit first.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import subprocess
import time
from pathlib import Path

from agent.arena.arena import Arena, ArenaError, FightResult, load_scenarios
from agent.bots.heuristic import HeuristicBot
from agent.harness.supervise import DEFAULT_GAME_DIR, restart_game
from agent.interface.client import GameClient, GameNotRunning, StateTimeout


def make_bot(spec: str, seed: int):
    if spec == "heuristic":
        return HeuristicBot(seed=seed)
    if spec.startswith("nn="):
        from agent.bots.nn_bot import NNBot
        return NNBot(Path(spec[3:]), seed=seed)
    raise SystemExit(f"Unknown bot {spec!r} (use heuristic or nn=<model dir>)")


def pid_alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in out


def play_all(spec: str, scenarios: list[dict], seed: int, log_dir: Path) -> list[dict]:
    if not restart_game(DEFAULT_GAME_DIR, log_dir / "game"):
        raise SystemExit("Game didn't start")
    policy = make_bot(spec, seed)
    arena = Arena(GameClient(), HeuristicBot(seed=seed))
    results = []
    for i, sc in enumerate(scenarios):
        if i and i % 200 == 0:
            restart_game(DEFAULT_GAME_DIR, log_dir / "game")
        try:
            r = arena.play_fight(sc, policy)
        except (ArenaError, GameNotRunning, StateTimeout) as e:
            if not isinstance(e, ArenaError):
                restart_game(DEFAULT_GAME_DIR, log_dir / "game")
            r = FightResult(sc["encounter"], sc["kind"], sc["hp"], sc["max_hp"], sc["hp"], False, False,
                            0, 0, 0.0, error=str(e)[:200])
        results.append({"encounter": r.encounter, "kind": r.kind, "hp_lost": r.hp_lost, "died": r.died,
                        "error": r.error})
        print(f"[{spec}] {i + 1}/{len(scenarios)} {r.encounter} lost {r.hp_lost}{' DIED' if r.died else ''}",
              flush=True)
    return results


def summarize(all_results: dict[str, list[dict]]) -> str:
    names = list(all_results)
    base = all_results[names[0]]
    lines = ["| Fights | Bot | HP lost | Deaths | vs " + names[0] + " (HP lost) |", "|---|---|---|---|---|"]
    # Only fights every bot finished, so all rows cover the same fights.
    clean = [i for i in range(len(base)) if not any(all_results[n][i]["error"] for n in names)]
    for kind in ("monster", "elite", "boss", "all"):
        ok = [i for i in clean if kind == "all" or base[i]["kind"] == kind]
        if not ok:
            continue
        for name in names:
            res = all_results[name]
            lost = st.mean(res[i]["hp_lost"] for i in ok)
            deaths = sum(res[i]["died"] for i in ok)
            diff = ""
            if name != names[0] and len(ok) > 1:
                d = [res[i]["hp_lost"] - base[i]["hp_lost"] for i in ok]
                diff = f"{st.mean(d):+.1f} ± {1.96 * st.stdev(d) / len(d) ** 0.5:.1f}"
            lines.append(f"| {kind} ({len(ok)}) | {name} | {lost:.1f} | {deaths} | {diff} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bot", action="append", required=True)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--act", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scenarios", type=Path, default=Path("data/scenarios_v1.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("logs/arena_compare"))
    ap.add_argument("--wait-pid", type=int, default=None)
    args = ap.parse_args()

    if args.wait_pid:
        while pid_alive(args.wait_pid):
            time.sleep(30)
    scs = [s for s in load_scenarios(args.scenarios) if s["act"] == args.act]
    rng = random.Random(args.seed)
    scenarios = [rng.choice(scs) for _ in range(args.episodes)]
    args.out.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for spec in args.bot:
        all_results[spec] = play_all(spec, scenarios, args.seed, args.out)
        (args.out / "results.json").write_text(json.dumps(all_results))
    table = summarize(all_results)
    (args.out / "summary.md").write_text(
        f"# Arena comparison ({time.strftime('%Y-%m-%d %H:%M')})\n\n{args.episodes} Act {args.act} fights, "
        f"seed {args.seed}, same scenarios for every bot (potions included).\n\n{table}\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
