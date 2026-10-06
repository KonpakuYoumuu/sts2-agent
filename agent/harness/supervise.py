"""Unattended collection on a headless, fast game: relaunch the game if it dies.

    python -m agent.harness.supervise --runs 600 --bot heuristic --explore 0.1 --log-dir logs/night
    python -m agent.harness.supervise --record --log-dir logs/autosts2   # a mod plays, we record

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

from agent.interface.client import COMBAT_TYPES, GameClient
from agent.learn.infer_actions import read_jsonl

DEFAULT_GAME_DIR = Path(os.environ.get(
    "STS2_GAME_DIR", r"C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"))
PORT = 15526


def game_running() -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq SlayTheSpire2.exe"],
                         capture_output=True, text=True).stdout
    return "SlayTheSpire2.exe" in out


def launch_game(game_dir: Path, log_dir: Path, fast: str = "1", move_log: Path | None = None) -> None:
    """fast: "1" = AutoSlay-style simulation + Instant mode, "2" = Instant mode only, "0" = off.

    move_log: the mod appends every combat move (state + action) to this file.
    """
    env = dict(os.environ, STS2MCP_FAST=fast, STS2MCP_PORT=str(PORT))
    if move_log is not None:
        env["STS2MCP_MOVE_LOG"] = str(move_log.resolve())
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


def restart_game(game_dir: Path, log_dir: Path) -> bool:
    """Kill the game (if any) and start a fresh headless one; True once its API is up."""
    subprocess.run(["taskkill", "/F", "/IM", "SlayTheSpire2.exe"], capture_output=True)
    time.sleep(5)
    launch_game(game_dir, log_dir)
    return wait_for_api()


def logged_runs(log_dir: Path) -> list[dict]:
    path = log_dir / "summary.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _activity(log_dir: Path) -> tuple[str | None, float]:
    """(screen of the last recorded state, time anything was last recorded or logged).

    Read from the recorder's files rather than the game: reading a shop's state
    through the mod reopens its inventory, which blocks leaving the shop.
    """
    recordings = sorted(log_dir.glob("human_*.jsonl.gz"), key=lambda f: f.stat().st_mtime)
    files = recordings + list(log_dir.glob("moves_*.jsonl"))
    last_write = max((f.stat().st_mtime for f in files), default=0.0)
    last = read_jsonl(recordings[-1])[-1:] if recordings else []
    return (last[0]["state"].get("state_type") if last else None), last_write


def supervise_recording(args: argparse.Namespace) -> int:
    """Keep a headless game and the looping recorder alive while a mod plays.

    The game restarts if it dies, or if its state stops changing: after
    --idle-seconds outside fights (the mod acts within seconds there, so it missed
    a trigger; reloading the save re-fires it), or --stall-minutes in a fight
    (the solver may think for minutes). The recorder waits through restarts.
    """
    recorder = None
    restarts = 0
    stall_restarts: list[float] = []
    launched = time.time()
    try:
        while True:
            if not game_running():
                fast = "0" if args.no_fast else "2"
                print(f"Launching headless game (restart #{restarts}, STS2MCP_FAST={fast})", flush=True)
                launch_game(args.game_dir, args.log_dir, fast=fast,
                            move_log=args.log_dir / time.strftime("moves_%Y%m%d-%H%M%S.jsonl"))
                restarts += 1
                if not wait_for_api():
                    print("Mod API didn't come up; killing the game and retrying.", flush=True)
                    subprocess.run(["taskkill", "/F", "/IM", "SlayTheSpire2.exe"], capture_output=True)
                    time.sleep(5)
                    continue
                launched = time.time()
                recent = [t for t in stall_restarts if t > time.time() - 30 * 60]
                if len(recent) >= 3:
                    # Reloading resumes the same spot (e.g. an event that crashes headless); give up on this run.
                    _abandon_saved_run()
                    stall_restarts.clear()
            if recorder is None or recorder.poll() is not None:
                recorder = subprocess.Popen([sys.executable, "-u", "-m", "agent.harness.record_human",
                                             "--loop", "--start-runs", "--nudge-map", "8", "--nudge-fight", "180",
                                             "--out", str(args.log_dir)])
            st, last_write = _activity(args.log_dir)
            idle = time.time() - max(last_write, launched)
            limit = args.stall_minutes * 60 if st in COMBAT_TYPES or st == "hand_select" else args.idle_seconds
            if idle > limit:
                print(f"State unchanged for {idle:.0f} s on '{st}'; restarting the game.", flush=True)
                stall_restarts.append(time.time())
                # The recorder goes too: it mustn't click "continue" before a stuck run is abandoned.
                _kill_tree(recorder)
                recorder = None
                subprocess.run(["taskkill", "/F", "/IM", "SlayTheSpire2.exe"], capture_output=True)
                time.sleep(5)
                continue
            time.sleep(15)
    finally:
        _kill_tree(recorder)


def _kill_tree(proc: subprocess.Popen | None) -> None:
    """Stop a child Python and its own children (the venv's python.exe is a launcher)."""
    if proc is not None and proc.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)


def _abandon_saved_run(timeout: float = 90.0) -> None:
    client = GameClient()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            state = client.get_state()
        except Exception:
            time.sleep(2)
            continue
        options = [o if isinstance(o, str) else o.get("name") for o in state.get("options") or []]
        if state.get("menu_screen") == "main" and "abandon_run" in options:
            client.act({"action": "menu_select", "option": "abandon_run"})
        elif state.get("menu_screen") == "popup" and "yes" in options:
            client.act({"action": "menu_select", "option": "yes"})
            print("Stuck in the same run after 3 restarts; abandoned it.", flush=True)
            return
        elif state.get("menu_screen") == "main" and options:
            return  # no saved run
        time.sleep(1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, help="runs to play (bot mode)")
    p.add_argument("--bot", default="heuristic")
    p.add_argument("--explore", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    p.add_argument("--record", action="store_true",
                   help="don't play: start runs and record while a mod (e.g. AutoSTS2) plays them, until stopped")
    p.add_argument("--no-fast", action="store_true", help="headless at normal game speed")
    p.add_argument("--stall-minutes", type=float, default=15.0,
                   help="--record: restart the game when a fight's state is unchanged for this long")
    p.add_argument("--idle-seconds", type=float, default=90.0,
                   help="--record: restart the game when a non-fight screen is unchanged for this long")
    args = p.parse_args()
    if not (args.game_dir / "SlayTheSpire2.exe").exists():
        print(f"SlayTheSpire2.exe not found in {args.game_dir}. Pass --game-dir or set STS2_GAME_DIR.")
        return 1
    if args.record:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        return supervise_recording(args)
    if args.runs is None:
        p.error("--runs is required unless --record is given")

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
