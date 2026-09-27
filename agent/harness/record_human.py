"""Record a human playing, as a sequence of game states.

    python -m agent.harness.record_human --out logs/human

The mod doesn't report which button a human pressed, so we log every distinct
settled state; actions are inferred afterwards from consecutive states (a card
left the hand, an enemy lost HP, the map position changed...).

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

from agent.harness.runner import fingerprint
from agent.interface.client import GameClient, GameNotRunning, is_settled, is_victory_event


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("logs/human"))
    p.add_argument("--poll", type=float, default=0.15)
    p.add_argument("--shop-poll", type=float, default=10.0)
    p.add_argument("--file", type=Path, default=None, help="append to this recording instead of a new one")
    args = p.parse_args()

    client = GameClient()
    if not client.ping():
        print("Waiting for the game to start...", flush=True)
        while not client.ping():
            time.sleep(2)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.file or args.out / time.strftime("human_%Y%m%d-%H%M%S.jsonl.gz")
    print(f"Recording to {path}. Play normally; recording ends at game over (or Ctrl+C).", flush=True)

    t0 = time.monotonic()
    last_fp = None
    n = 0
    in_run = False
    reached_ending = False
    # Append mode adds a new gzip member; readers see one continuous file.
    with gzip.open(path, "at", encoding="utf-8") as log:
        while True:
            try:
                state = client.get_state()
            except GameNotRunning:
                print("Game closed; stopping.", flush=True)
                break
            st = state.get("state_type")
            reached_ending = reached_ending or is_victory_event(state)
            if st != "menu":
                in_run = True
            if in_run and is_settled(state):
                fp = fingerprint(state)
                if fp != last_fp:
                    log.write(json.dumps({"t": round(time.monotonic() - t0, 3), "state": state}) + "\n")
                    log.flush()
                    last_fp = fp
                    n += 1
                    if n % 25 == 0:
                        run = state.get("run") or {}
                        print(f"  {n} states  act {run.get('act')} floor {run.get('floor')}  [{st}]", flush=True)
            if st == "game_over":
                # The game-over screen shows 0 HP after a win too.
                print(f"Game over ({'victory' if reached_ending else 'death'}); {n} states recorded.", flush=True)
                break
            time.sleep(args.shop_poll if st in ("shop", "fake_merchant") else args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
