"""Play full runs with a Policy and log every decision.

Responsibilities beyond the policy itself:
- navigate menus (start a run, dismiss pop-ups, return from game over);
- wait until the game is actionable after every action;
- recover from rejected or no-op actions instead of looping forever;
- write one gzipped JSONL decision log per run plus a summary line.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.bots.base import Policy
from agent.interface.actions import PROGRESS_ACTIONS, Action, action_key, legal_actions
from agent.interface.client import GameClient, State, StateTimeout

# After this many visits to an identical state, only progress actions are allowed.
MAX_VISITS_BEFORE_FORCING = 25
# Even with --keep-going, this many stuck runs in a row means something is broken.
MAX_STUCK_IN_A_ROW = 3
# Seeing the same state this often means an action loop (e.g. confirm/cancel ping-pong).
MAX_VISITS_BEFORE_ABORT = 100
# A run that takes more steps than this is considered stuck.
MAX_STEPS_PER_RUN = 5000
# How long to wait for an action to visibly change the state.
CHANGE_TIMEOUT = 3.0
# Selection toggles aren't always visible in the state, so an "ok" with no
# visible change isn't proof the action did nothing.
NOOP_EXEMPT = {"select_card", "combat_select_card"}
# Actions allowed once a state has been visited too often.
FORCED_ACTIONS = PROGRESS_ACTIONS | {"confirm_selection", "combat_confirm_selection",
                                     "confirm_bundle_selection"}


class StuckError(RuntimeError):
    pass


class ManualActionRequired(StuckError):
    """Something in the game needs a human (e.g. Timeline unlock reveals)."""


@dataclass
class RunResult:
    run_index: int
    policy: str
    character: str
    outcome: str  # "victory", "death", "stuck", "error"
    act: int | None = None
    floor: int | None = None
    hp: int | None = None
    steps: int = 0
    seconds: float = 0.0
    rejected_actions: int = 0
    noop_actions: int = 0
    error: str | None = None
    log_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def fingerprint(state: State) -> str:
    return hashlib.sha1(json.dumps(state, sort_keys=True).encode()).hexdigest()


def _option_names(state: State) -> list[str]:
    names = []
    for opt in state.get("options", []):
        if isinstance(opt, str):
            names.append(opt.lower())
        elif opt.get("enabled", True):
            names.append(str(opt["name"]).lower())
    return names


class Runner:
    def __init__(self, client: GameClient, policy: Policy, character: str = "IRONCLAD",
                 log_dir: Path = Path("logs"), verbose: bool = True):
        self.client = client
        self.policy = policy
        self.character = character.lower()
        self.log_dir = log_dir
        self.verbose = verbose

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ---- menus ------------------------------------------------------------

    def _menu_action(self, state: State, picked_character: bool) -> str:
        screen = state.get("menu_screen")
        options = _option_names(state)
        if screen == "main":
            if "continue" in options:
                return "continue"
            if "singleplayer" in options:
                return "singleplayer"
            blocked = [b.get("reason") for b in state.get("blocked_options", [])]
            if "manual_epoch_reveal_required" in blocked:
                raise ManualActionRequired(
                    "The game wants a new unlock revealed on the Timeline. Open Timeline in "
                    "the main menu, reveal it by hand, return to the main menu, then rerun.")
            raise StuckError(f"Main menu offers neither continue nor singleplayer: {options}")
        if screen == "singleplayer":
            return "standard"
        if screen == "character_select":
            if not picked_character:
                return self.character
            return "embark" if "embark" in options else "confirm"
        if screen == "tutorial_prompt":
            return "no"
        if screen == "popup":
            for pref in ("ignore", "ok", "continue", "confirm", "back"):
                if pref in options:
                    return pref
        if "back" in options:
            return "back"
        if options:
            return options[0]
        raise StuckError(f"No usable option on menu screen {screen!r}: {state}")

    def start_run(self, timeout: float = 120.0) -> State:
        """From wherever the game is (menus), get into a run and return its first state."""
        deadline = time.monotonic() + timeout
        picked_character = False
        while time.monotonic() < deadline:
            state = self.client.wait_for_input(timeout=60)
            st = state.get("state_type")
            if st == "game_over":
                self.client.act({"action": "menu_select", "option": "main_menu"})
            elif st == "menu":
                option = self._menu_action(state, picked_character)
                res = self.client.act({"action": "menu_select", "option": option})
                self._say(f"  menu[{state.get('menu_screen')}] -> {option}: {res.message}")
                if state.get("menu_screen") == "character_select":
                    # A failed embark means the selection was lost (e.g. after the tutorial prompt).
                    picked_character = res.ok if option == self.character else res.ok and picked_character
                if res.ok and option in ("embark", "confirm"):
                    self._await_screen_change(state, timeout=15.0)
                    continue
            else:
                return state
            time.sleep(0.5)
        raise StuckError("Couldn't start a run from the menus")

    # ---- one run ----------------------------------------------------------

    def _await_screen_change(self, before: State, timeout: float) -> None:
        """Wait until we leave `before`'s menu screen (e.g. the run starts loading)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self.client.get_state()
            if (s.get("state_type"), s.get("menu_screen")) != ("menu", before.get("menu_screen")):
                return
            time.sleep(0.25)

    def _await_change(self, before_fp: str, timeout: float = CHANGE_TIMEOUT) -> State:
        """After an action: wait for the state to change, then for it to settle."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if fingerprint(self.client.get_state()) != before_fp:
                break
            time.sleep(self.client.poll_interval)
        return self.client.wait_for_input(timeout=60)

    def _wait_for_actions(self, state: State, timeout: float = 10.0) -> State:
        """Some screens are briefly empty while loading; re-poll before giving up."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.25)
            state = self.client.wait_for_input(timeout=60)
            if legal_actions(state) or state.get("state_type") in ("game_over", "menu"):
                return state
        raise StuckError(f"No legal actions in state_type={state.get('state_type')!r} "
                         f"after {timeout:.0f}s")

    def play_run(self, run_index: int) -> RunResult:
        self.policy.on_run_start()
        result = RunResult(run_index=run_index, policy=self.policy.name,
                           character=self.character, outcome="error")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"run_{run_index:04d}.jsonl.gz"
        result.log_path = str(log_path)
        t0 = time.monotonic()

        visits: Counter[str] = Counter()
        banned: set[tuple[str, str]] = set()
        state: State | None = None
        try:
            state = self.start_run()
            with gzip.open(log_path, "wt", encoding="utf-8") as log:
                for step in range(MAX_STEPS_PER_RUN):
                    st = state.get("state_type")
                    if st == "game_over":
                        self._finish(result, state)
                        log.write(json.dumps({"step": step, "state": state, "terminal": True}) + "\n")
                        self.client.act({"action": "menu_select", "option": "main_menu"})
                        break
                    if st == "menu":
                        raise StuckError("Returned to the main menu mid-run")

                    if not legal_actions(state):
                        state = self._wait_for_actions(state)
                        continue

                    fp = fingerprint(state)
                    visits[fp] += 1
                    actions = [a for a in legal_actions(state) if (fp, action_key(a)) not in banned]
                    if visits[fp] > MAX_VISITS_BEFORE_ABORT:
                        raise StuckError(f"Same state seen {visits[fp]} times in state_type={st!r}")
                    if visits[fp] > MAX_VISITS_BEFORE_FORCING:
                        forced = [a for a in actions if a["action"] in FORCED_ACTIONS]
                        actions = forced or actions
                    if not actions:
                        raise StuckError(f"No legal actions in state_type={st!r}")

                    action = self.policy.choose(state, actions)
                    res = self.client.act(action)
                    new_state = self._await_change(
                        fp, 0.4 if action["action"] in NOOP_EXEMPT else CHANGE_TIMEOUT)
                    changed = fingerprint(new_state) != fp
                    if not res.ok and not changed:
                        banned.add((fp, action_key(action)))
                        result.rejected_actions += 1
                    elif res.ok and not changed and action["action"] not in NOOP_EXEMPT:
                        banned.add((fp, action_key(action)))
                        result.noop_actions += 1

                    log.write(json.dumps({
                        "step": step, "t": round(time.monotonic() - t0, 3),
                        "state": state, "n_legal": len(actions), "action": action,
                        "ok": res.ok, "message": res.message, "changed": changed,
                        **getattr(self.policy, "last_info", {}),
                    }) + "\n")
                    result.steps = step + 1
                    if step % 25 == 0 or st in ("map", "game_over"):
                        run = new_state.get("run") or {}
                        hp = (new_state.get("player") or {}).get("hp")
                        self._say(f"  step {step:4d}  act {run.get('act')} floor {run.get('floor')} "
                                  f"hp {hp}  [{st}] {action_key(action)}"
                                  + ("" if res.ok else f"  REJECTED: {res.message}"))
                    state = new_state
                else:
                    raise StuckError(f"Exceeded {MAX_STEPS_PER_RUN} steps")
        except (StuckError, StateTimeout) as e:
            result.outcome = "stuck"
            result.error = str(e)
            if isinstance(e, ManualActionRequired):
                result.extra["manual_action"] = True
                return result
            last = getattr(e, "last_state", None) or state
            if last is not None:
                dump = self.log_dir / f"run_{run_index:04d}_stuck_state.json"
                dump.write_text(json.dumps(last, indent=1, ensure_ascii=False), encoding="utf-8")
                result.extra["stuck_state"] = str(dump)
        finally:
            result.seconds = round(time.monotonic() - t0, 1)
        return result

    @staticmethod
    def _finish(result: RunResult, state: State) -> None:
        run = state.get("run") or {}
        player = state.get("player") or {}
        result.act, result.floor, result.hp = run.get("act"), run.get("floor"), player.get("hp")
        result.extra["ascension"] = run.get("ascension")
        result.outcome = "victory" if (player.get("hp") or 0) > 0 else "death"
        result.extra["game_over"] = state.get("game_over")

    # ---- many runs --------------------------------------------------------

    def play(self, n_runs: int, stop_on_stuck: bool = True) -> list[RunResult]:
        results = []
        summary_path = self.log_dir / "summary.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Continue numbering after runs already in this directory, so a restarted
        # collection never overwrites earlier run logs.
        offset = len(summary_path.read_text(encoding="utf-8").splitlines()) if summary_path.exists() else 0
        stuck_in_a_row = 0
        for i in range(n_runs):
            self._say(f"=== run {i + 1}/{n_runs} ({self.policy.name}) ===")
            r = self.play_run(offset + i)
            results.append(r)
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(r)) + "\n")
            self._say(f"=== run {i + 1}: {r.outcome}  act {r.act} floor {r.floor}  "
                      f"{r.steps} steps  {r.seconds}s  rejected {r.rejected_actions}"
                      + (f"  error: {r.error}" if r.error else ""))
            if r.extra.get("manual_action"):
                self._say(f"Stopping: manual action needed in the game. {r.error}")
                break
            if r.outcome == "stuck" and stop_on_stuck:
                self._say("Stopping: run got stuck. See the stuck-state dump.")
                break
            stuck_in_a_row = stuck_in_a_row + 1 if r.outcome == "stuck" else 0
            if stuck_in_a_row >= MAX_STUCK_IN_A_ROW:
                self._say(f"Stopping: {stuck_in_a_row} stuck runs in a row.")
                break
            if r.outcome == "stuck":
                time.sleep(5)
        return results
