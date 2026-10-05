"""Record a human (or another mod, e.g. AutoSTS2) playing, as a sequence of game states.

    python -m agent.harness.record_human --out logs/human
    python -m agent.harness.record_human --out logs/autosts2 --loop   # run after run, unattended

The mod doesn't report which button was pressed, so we log every distinct
settled state; combat moves are inferred afterwards from consecutive states
(agent/learn/infer_actions.py).

Reading shop state through the mod re-opens the shop inventory, which disables
the shop's leave button. So in shops we only poll every --shop-poll seconds,
leaving the human time to close the inventory and leave.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

from agent.bots.heuristic import HeuristicBot
from agent.harness.runner import Runner, StuckError, fingerprint
from agent.interface.actions import legal_actions
from agent.interface.client import GameClient, GameNotRunning, StateTimeout, is_settled, is_victory_event


def _wait_for_game(client: GameClient) -> None:
    if not client.ping():
        print("Waiting for the game to start...", flush=True)
        while not client.ping():
            time.sleep(2)


def record_run(client: GameClient, path: Path, args: argparse.Namespace, starter: Runner | None = None) -> str:
    """Record one run into `path`. Returns "victory", "death" or "closed".

    With a `starter`, runs are started from the menus by our harness (AutoSTS2
    turns its own run looping off at every game start); the run itself is
    played by whoever plays it.
    """
    t0 = time.monotonic()
    changed_at = t0
    map_bot = HeuristicBot(seed=0)
    last_fp = None
    n = 0
    in_run = False
    reached_ending = False
    outcome = "closed"
    # Append mode adds a new gzip member; readers see one continuous file.
    with gzip.open(path, "at", encoding="utf-8") as log:
        while True:
            try:
                state = client.get_state()
            except GameNotRunning:
                print("Game closed.", flush=True)
                break
            st = state.get("state_type")
            reached_ending = reached_ending or is_victory_event(state)
            if st == "menu" and in_run:
                # The game was restarted (or quit to the menu) mid-run.
                print("Back at the menu mid-run; ending this recording.", flush=True)
                break
            # Menus and a leftover game-over screen belong to no run yet.
            if st not in ("menu", "game_over"):
                in_run = True
            elif starter is not None and not in_run:
                try:
                    starter.start_run()
                except (StuckError, StateTimeout, GameNotRunning) as e:
                    print(f"Couldn't start a run: {e}", flush=True)
                    time.sleep(10)
                continue
            if in_run and is_settled(state):
                fp = fingerprint(state)
                if fp != last_fp:
                    # ts (wall clock) lines records up with the mod's move log.
                    log.write(json.dumps({"t": round(time.monotonic() - t0, 3), "ts": round(time.time(), 3),
                                          "state": state}) + "\n")
                    log.flush()
                    last_fp = fp
                    changed_at = time.monotonic()
                    n += 1
                    if n % 25 == 0:
                        run = state.get("run") or {}
                        print(f"  {n} states  act {run.get('act')} floor {run.get('floor')}  [{st}]", flush=True)
                elif (args.nudge_map and st == "map" and starter is not None
                      and time.monotonic() - changed_at > args.nudge_map):
                    # AutoSTS2 sometimes misses the map opening (after events); move on for it.
                    action = map_bot.choose(state, legal_actions(state))
                    res = client.act(action)
                    print(f"Map unchanged for {args.nudge_map:.0f} s; chose {action} ourselves: {res.message}",
                          flush=True)
                    log.write(json.dumps({"t": round(time.monotonic() - t0, 3), "ts": round(time.time(), 3),
                                          "state": state, "map_nudge": action}) + "\n")
                    log.flush()
                    changed_at = time.monotonic()
            if st == "game_over" and in_run:
                # Always kept: it ends the run and tells the last fight's outcome.
                if fingerprint(state) != last_fp:
                    log.write(json.dumps({"t": round(time.monotonic() - t0, 3), "ts": round(time.time(), 3),
                                          "state": state}) + "\n")
                    n += 1
                # The game-over screen shows 0 HP after a win too.
                outcome = "victory" if reached_ending else "death"
                print(f"Game over ({outcome}); {n} states recorded.", flush=True)
                break
            time.sleep(args.shop_poll if st in ("shop", "fake_merchant") else args.poll)
    if n == 0 and args.file is None:
        path.unlink(missing_ok=True)
    return outcome


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("logs/human"))
    p.add_argument("--poll", type=float, default=0.15)
    p.add_argument("--shop-poll", type=float, default=10.0)
    p.add_argument("--file", type=Path, default=None, help="append to this recording instead of a new one")
    p.add_argument("--loop", action="store_true",
                   help="keep recording run after run (one file each), e.g. while a mod plays unattended")
    p.add_argument("--start-runs", action="store_true",
                   help="start each run from the menus ourselves; the run is played by someone else")
    p.add_argument("--character", default="IRONCLAD")
    p.add_argument("--nudge-map", type=float, default=0.0,
                   help="with --start-runs: pick the next map node ourselves when the map is unchanged "
                        "this many seconds (0 = never)")
    args = p.parse_args()

    client = GameClient()
    args.out.mkdir(parents=True, exist_ok=True)
    starter = Runner(client, None, character=args.character, log_dir=args.out) if args.start_runs else None
    counts = {"victory": 0, "death": 0, "closed": 0}
    while True:
        _wait_for_game(client)
        path = args.file or args.out / time.strftime("human_%Y%m%d-%H%M%S.jsonl.gz")
        print(f"Recording to {path}. Recording ends at game over (or Ctrl+C).", flush=True)
        counts[record_run(client, path, args, starter)] += 1
        if not args.loop:
            return 0
        print(f"Runs so far: {counts['victory']} won, {counts['death']} lost, "
              f"{counts['closed']} interrupted by the game closing.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
