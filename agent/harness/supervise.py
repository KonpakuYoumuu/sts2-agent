"""Unattended collection on a headless, fast game: relaunch the game if it dies.

    python -m agent.harness.supervise --runs 600 --bot heuristic --explore 0.1 --log-dir logs/night

Launches SlayTheSpire2.exe --headless with STS2MCP_FAST=1 (needs steam_appid.txt
in the game folder and Steam running), waits for the mod's API, then runs the
normal runner. The game folder is --game-dir, else the STS2_GAME_DIR environment
variable, else the default Steam location. If the game process dies, it's
relaunched and collection resumes in the same log directory until --runs runs
are logged or a run needs a human.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from agent.interface.client import GameClient

DEFAULT_GAME_DIR = Path(os.environ.get(
    "STS2_GAME_DIR", r"C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"))
PORT = 15526


def game_running() -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq SlayTheSpire2.exe"],
                         capture_output=True, text=True).stdout
    return "SlayTheSpire2.exe" in out


def launch_game(game_dir: Path, log_dir: Path) -> None:
    env = dict(os.environ, STS2MCP_FAST="1", STS2MCP_PORT=str(PORT))
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "game_stdout.txt", "ab") as out, open(log_dir / "game_stderr.txt", "ab") as err:
        subprocess.Popen([str(game_dir / "SlayTheSpire2.exe"), "--headless"], cwd=game_dir,
                         env=env, stdout=out, stderr=err)


def wait_for_api(timeout: float = 120) -> bool:
    client = GameClient()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.ping():
            return True
        time.sleep(2)
    return False


def logged_runs(log_dir: Path) -> list[dict]:
    path = log_dir / "summary.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, required=True)
    p.add_argument("--bot", default="heuristic")
    p.add_argument("--explore", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    args = p.parse_args()
    if not (args.game_dir / "SlayTheSpire2.exe").exists():
        print(f"SlayTheSpire2.exe not found in {args.game_dir}. Pass --game-dir or set STS2_GAME_DIR.")
        return 1

    restarts = 0
    no_progress = 0
    # Only react to manual-action stops from this session, not ones already in the log.
    start_count = len(logged_runs(args.log_dir))
    while True:
        done = logged_runs(args.log_dir)
        if len(done) >= args.runs:
            print(f"Done: {len(done)} runs logged.")
            return 0
        if len(done) > start_count and done[-1].get("extra", {}).get("manual_action"):
            print("Stopping: the game needs a manual action (Timeline reveal).")
            return 2
        if not game_running():
            print(f"Launching headless game (restart #{restarts})", flush=True)
            launch_game(args.game_dir, args.log_dir)
            restarts += 1
        if not wait_for_api():
            print("Mod API didn't come up; killing the game and retrying.", flush=True)
            subprocess.run(["taskkill", "/F", "/IM", "SlayTheSpire2.exe"], capture_output=True)
            time.sleep(5)
            continue
        # A fresh seed per restart so the bot's random choices don't repeat.
        cmd = [sys.executable, "-u", "-m", "agent.harness.run", "--runs", str(args.runs - len(done)),
               "--bot", args.bot, "--explore", str(args.explore), "--seed", str(args.seed + restarts * 1000),
               "--keep-going", "--log-dir", str(args.log_dir)]
        before = len(done)
        subprocess.run(cmd)
        if len(logged_runs(args.log_dir)) == before:
            no_progress += 1
            if no_progress >= 5:
                print("Stopping: 5 attempts in a row logged no runs.")
                return 3
            time.sleep(30)
        else:
            no_progress = 0
        if game_running() and GameClient().ping():
            # Runner exited on its own with the game fine: finished or needs a human.
            continue
        print("Game died during collection; relaunching.", flush=True)
        subprocess.run(["taskkill", "/F", "/IM", "SlayTheSpire2.exe"], capture_output=True)
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
