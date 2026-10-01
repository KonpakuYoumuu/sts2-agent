"""The fight arena: play single fights set up directly in the (headless) game.

Each episode: make sure a run is in progress, set the deck/relics/HP of a
scenario with the mod's `arena_setup`, start its encounter with the developer
console's `fight` command, and let a policy play until the fight is over. The
game's own rules run everything, so results match real runs; there is no map,
menu or reward screen in between, so fights come much faster.

    python -m agent.arena.arena --scenarios data/scenarios_v1.jsonl --episodes 5 --bot heuristic
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.bots.base import Policy
from agent.harness.runner import FORCED_ACTIONS, MAX_VISITS_BEFORE_FORCING, NOOP_EXEMPT, Runner, fingerprint
from agent.interface.actions import action_key, legal_actions
from agent.interface.client import COMBAT_TYPES, GameClient, State

FIGHT_OVER = {"rewards", "card_reward", "game_over", "map"}
MAX_STEPS_PER_FIGHT = 600


class ArenaError(RuntimeError):
    pass


@dataclass
class FightResult:
    encounter: str
    kind: str
    hp_start: int
    max_hp: int
    hp_end: int
    died: bool
    won: bool
    steps: int
    rounds: int
    seconds: float
    error: str | None = None

    @property
    def hp_lost(self) -> int:
        return self.hp_start - self.hp_end


class Arena:
    def __init__(self, client: GameClient, runner_policy: Policy, character: str = "IRONCLAD"):
        self.client = client
        # Used only to navigate menus when a new run has to be started.
        self.runner = Runner(client, runner_policy, character=character, log_dir=Path("logs/arena_runs"),
                             verbose=False)

    def _act(self, action: dict) -> dict:
        res = self.client.act(action)
        if not res.ok:
            raise ArenaError(f"{action.get('action')} failed: {res.message}")
        return res.raw

    def ensure_in_run(self) -> State:
        state = self.client.wait_for_input(timeout=60)
        if state.get("state_type") == "game_over":
            self.client.act({"action": "menu_select", "option": "main_menu"})
            state = self.client.wait_for_input(timeout=60)
        if state.get("state_type") == "menu":
            state = self.runner.start_run()
        if state.get("state_type") in COMBAT_TYPES or state.get("state_type") == "hand_select":
            # Left over from an aborted episode: end that fight first.
            self._act({"action": "console", "command": "win"})
            state = self._wait_until(lambda s: s.get("state_type") not in COMBAT_TYPES | {"hand_select"})
        return state

    def _wait_until(self, cond, timeout: float = 30.0) -> State:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.client.get_state()
            if cond(state):
                return state
            time.sleep(self.client.poll_interval)
        raise ArenaError("Timed out waiting for the game")

    def setup(self, scenario: dict) -> None:
        self._act({"action": "arena_setup", "deck": scenario["deck"], "relics": scenario["relics"],
                   "max_hp": scenario["max_hp"], "hp": scenario["hp"]})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            data = self._act({"action": "arena_status"})
            if data.get("setup_error"):
                raise ArenaError(f"arena_setup failed: {data['setup_error']}")
            if data.get("setup_done"):
                return
            time.sleep(self.client.poll_interval)
        raise ArenaError("arena_setup timed out")

    def set_potions(self, potions: list[str]) -> None:
        """Replace the potion belt (the console's `potion` command adds one)."""
        belt = (self.client.get_state().get("player") or {}).get("potions", [])
        for p in sorted(belt, key=lambda p: -p["slot"]):
            self._act({"action": "discard_potion", "slot": p["slot"]})
        for pid in potions:
            self._act({"action": "console", "command": f"potion {pid}"})

    def play_fight(self, scenario: dict, policy: Policy, log=None) -> FightResult:
        t0 = time.monotonic()
        self.ensure_in_run()
        self.setup(scenario)
        self.set_potions(scenario.get("potions", []))
        self._act({"action": "console", "command": f"fight {scenario['encounter']}"})
        state = self._wait_until(lambda s: s.get("state_type") in COMBAT_TYPES)
        state = self.client.wait_for_input(timeout=60)
        hp_start = (state.get("player") or {}).get("hp", scenario["hp"])
        max_hp = (state.get("player") or {}).get("max_hp", scenario["max_hp"])
        banned: set[tuple[str, str]] = set()
        visits: Counter[str] = Counter()
        steps = rounds = 0
        while state.get("state_type") not in FIGHT_OVER:
            if steps >= MAX_STEPS_PER_FIGHT:
                raise ArenaError(f"Fight exceeded {MAX_STEPS_PER_FIGHT} steps")
            fp = fingerprint(state)
            actions = [a for a in legal_actions(state) if (fp, action_key(a)) not in banned]
            visits[fp] += 1
            if visits[fp] > MAX_VISITS_BEFORE_FORCING:  # going in circles: confirm / move on
                actions = [a for a in actions if a["action"] in FORCED_ACTIONS] or actions
            if not actions:
                state = self.client.wait_for_input(timeout=10)
                if fingerprint(state) == fp:
                    raise ArenaError(f"No legal actions in state_type={state.get('state_type')!r}")
                continue
            action = policy.choose(state, actions)
            res = self.client.act(action)
            new_state = self.runner._await_change(fp, 0.4 if action["action"] in NOOP_EXEMPT else 3.0)
            if fingerprint(new_state) == fp and action["action"] not in NOOP_EXEMPT:
                banned.add((fp, action_key(action)))
            if log is not None:
                log.write(json.dumps({"step": steps, "state": state, "action": action, "ok": res.ok,
                                      "arena": scenario["encounter"],
                                      **getattr(policy, "last_info", {})}) + "\n")
            rounds = max(rounds, (state.get("battle") or {}).get("round") or 0)
            steps += 1
            state = new_state
        player = state.get("player") or {}
        died = state.get("state_type") == "game_over"
        return FightResult(
            encounter=scenario["encounter"], kind=scenario["kind"], hp_start=hp_start, max_hp=max_hp,
            hp_end=0 if died else player.get("hp", 0), died=died, won=not died, steps=steps, rounds=rounds,
            seconds=round(time.monotonic() - t0, 2))


def load_scenarios(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main() -> int:
    from agent.harness.run import BOTS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", type=Path, default=Path("data/scenarios_v1.jsonl"))
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--bot", default="heuristic", choices=sorted(BOTS))
    ap.add_argument("--kind", default=None, help="only monster / elite / boss fights")
    ap.add_argument("--act", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", type=Path, default=Path("logs/arena"))
    args = ap.parse_args()

    scenarios = [s for s in load_scenarios(args.scenarios)
                 if (args.kind is None or s["kind"] == args.kind) and (args.act is None or s["act"] == args.act)]
    rng = random.Random(args.seed)
    client = GameClient()
    if not client.ping():
        print("Can't reach the STS2MCP mod. Start the game (headless is fine) first.")
        return 1
    policy = BOTS[args.bot](seed=args.seed)
    arena = Arena(client, policy)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with gzip.open(args.log_dir / time.strftime("fights_%Y%m%d-%H%M%S.jsonl.gz"), "wt", encoding="utf-8") as log, \
            open(args.log_dir / "results.jsonl", "a", encoding="utf-8") as res_file:
        for i in range(args.episodes):
            sc = rng.choice(scenarios)
            try:
                r = arena.play_fight(sc, policy, log)
            except ArenaError as e:
                r = FightResult(sc["encounter"], sc["kind"], sc["hp"], sc["max_hp"], sc["hp"], False, False,
                                0, 0, 0.0, error=str(e))
            results.append(r)
            res_file.write(json.dumps({**asdict(r), "bot": args.bot, "floor": sc.get("floor")}) + "\n")
            res_file.flush()
            print(f"{i + 1:4d} {r.encounter:32s} hp {r.hp_start:3d} -> {r.hp_end:3d}"
                  f"{'  DIED' if r.died else ''}  {r.rounds} rounds  {r.seconds:.1f}s"
                  + (f"  ERROR {r.error}" if r.error else ""), flush=True)
    ok = [r for r in results if not r.error]
    if ok:
        print(f"\n{len(ok)} fights: mean HP lost {sum(r.hp_lost for r in ok) / len(ok):.1f}, "
              f"deaths {sum(r.died for r in ok)}, {sum(r.seconds for r in ok) / len(ok):.1f}s per fight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
