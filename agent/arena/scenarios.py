"""Fight scenarios for the arena, taken from real logged fights.

A scenario is the player's situation at the start of a fight: the deck (with
upgrades), relics, potions, HP, and which encounter it was. Replaying real situations
keeps arena training close to what the agent meets in actual runs.

    python -m agent.arena.scenarios logs/night_0927 logs/eval_route_0927 --out data/scenarios_v1.jsonl
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import re
from pathlib import Path

from agent.interface.client import COMBAT_TYPES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UPGRADE = re.compile(r"^(.*?)\+(\d*)$")


def load_encounters() -> dict[str, dict]:
    return json.loads((DATA_DIR / "encounters.json").read_text(encoding="utf-8"))


def monsters_key(entity_ids: list[str]) -> tuple[str, ...]:
    return tuple(sorted(re.sub(r"_\d+$", "", e) for e in entity_ids))


def split_upgrades(name: str) -> tuple[str, int]:
    """'Bash+' -> ('Bash', 1); 'Wither+2' -> ('Wither', 2); 'Strike' -> ('Strike', 0)."""
    m = UPGRADE.match(name.strip())
    if not m:
        return name.strip(), 0
    return m.group(1), int(m.group(2) or 1)


class ScenarioBuilder:
    def __init__(self) -> None:
        self.encounters = load_encounters()
        self.by_monsters: dict[tuple[str, ...], str] = {}
        for enc, info in self.encounters.items():
            self.by_monsters.setdefault(tuple(sorted(info["monsters"])), enc)
        self.card_ids: dict[str, str] = {}   # card name (no "+") -> card ID
        self.skipped = collections.Counter()

    def learn_card_ids(self, state: dict) -> None:
        for c in (state.get("player") or {}).get("hand", []):
            if c.get("id") and c.get("name"):
                self.card_ids.setdefault(split_upgrades(c["name"])[0], c["id"])

    def encounter_for(self, state: dict) -> str | None:
        battle = state.get("battle") or {}
        if battle.get("encounter_id"):
            return battle["encounter_id"]
        key = monsters_key([e["entity_id"] for e in battle.get("enemies", [])])
        if key in self.by_monsters:
            return self.by_monsters[key]
        # Monsters that spawn later aren't in the first state: accept the one encounter
        # whose monster list contains exactly these plus others.
        matches = [enc for m, enc in self.by_monsters.items()
                   if collections.Counter(key) <= collections.Counter(m)]
        return matches[0] if len(matches) == 1 else None

    def scenario(self, state: dict, source: str) -> dict | None:
        player = state.get("player") or {}
        enc = self.encounter_for(state)
        if enc is None:
            self.skipped["unknown encounter"] += 1
            return None
        deck = []
        for pile in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
            for c in player.get(pile, []):
                name, ups = split_upgrades(c.get("name", ""))
                cid = c.get("id") or self.card_ids.get(name)
                if cid is None:
                    self.skipped["unknown card"] += 1
                    return None
                deck.append({"id": cid, "upgrades": ups})
        info = self.encounters[enc]
        return {
            "encounter": enc, "kind": info["kind"], "act": info["act"],
            "deck": deck, "relics": [r["id"] for r in player.get("relics", [])],
            "potions": [q["id"] for q in player.get("potions", []) if q.get("id")],
            "hp": player.get("hp"), "max_hp": player.get("max_hp"),
            "floor": (state.get("run") or {}).get("floor"), "source": source,
        }


def fight_starts(path: str):
    """The first combat state of each fight in a run log."""
    in_fight = False
    for line in gzip.open(path, "rt", encoding="utf-8"):
        s = json.loads(line)["state"]
        st = s.get("state_type")
        if st in COMBAT_TYPES:
            if not in_fight and (s.get("battle") or {}).get("round") == 1:
                yield s
            in_fight = True
        elif st != "hand_select":
            in_fight = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dirs", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("data/scenarios_v1.jsonl"))
    args = ap.parse_args()
    b = ScenarioBuilder()
    files = sorted(f for d in args.log_dirs for f in glob.glob(str(Path(d) / "run_*.jsonl.gz")))
    # First pass: learn card name -> ID from hands (piles only have names).
    starts = [(f, s) for f in files for s in fight_starts(f)]
    for _, s in starts:
        b.learn_card_ids(s)
    scenarios = [sc for f, s in starts if (sc := b.scenario(s, Path(f).name))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for sc in scenarios:
            out.write(json.dumps(sc) + "\n")
    kinds = collections.Counter((sc["act"], sc["kind"]) for sc in scenarios)
    print(f"{len(scenarios)} scenarios from {len(starts)} fights; skipped {dict(b.skipped)}")
    print(dict(sorted(kinds.items())))


if __name__ == "__main__":
    main()
