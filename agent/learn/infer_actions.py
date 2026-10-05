"""Infer combat moves from state-only recordings (record_human), for imitation.

The mod doesn't report what another player (a human, or a mod like AutoSTS2)
did, so moves are read off consecutive combat states:

- a card that left the hand (and wasn't just upgraded) was played;
- a potion slot that emptied was used;
- a new round with nothing played means the turn was ended.

Targets of single-target moves are the enemy that lost the most HP + block, or
whose powers changed, before the next move. Steps that can't be read
unambiguously (two cards gone at once, no visible target) get no label, since a
wrong label teaches a wrong move.

Output: one run log per recorded run, in the runner's format, so
build_dataset reads it unchanged.

    python -m agent.learn.infer_actions logs/autosts2 --out logs/autosts2_labeled
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import Counter
from pathlib import Path

from agent.interface.client import COMBAT_TYPES, State

# How many states after a move to look for its effect on an enemy.
TARGET_LOOKAHEAD = 6
# Card-choice screens inside a fight (some relics and cards open one every turn).
FIGHT_TYPES = set(COMBAT_TYPES) | {"hand_select"}


def _card_key(card: dict) -> tuple[str, bool]:
    return card.get("id") or card.get("name"), bool(card.get("is_upgraded"))


def _round(state: State) -> int | None:
    return (state.get("battle") or {}).get("round")


def _enemies(state: State) -> dict[str, dict]:
    return {e["entity_id"]: e for e in (state.get("battle") or {}).get("enemies", []) if e.get("hp", 0) > 0}


def _alive_ids(state: State) -> list[str]:
    return list(_enemies(state))


def _played_cards(before: State, after: State) -> list[dict]:
    """Cards of `before`'s hand that are gone in `after`, ignoring in-hand upgrades."""
    hand_b = (before.get("player") or {}).get("hand", [])
    hand_a = (after.get("player") or {}).get("hand", [])
    left = Counter(_card_key(c) for c in hand_b) - Counter(_card_key(c) for c in hand_a)
    added = Counter(_card_key(c) for c in hand_a) - Counter(_card_key(c) for c in hand_b)
    gone = []
    for (cid, up), n in left.items():
        # Armaments-style upgrades: the base card "leaves" and its upgraded copy "arrives".
        n -= min(n, added.get((cid, True), 0)) if not up else 0
        gone += [next(c for c in hand_b if _card_key(c) == (cid, up))] * n
    return gone


def _used_potions(before: State, after: State) -> list[dict]:
    slots_a = {p["slot"] for p in (after.get("player") or {}).get("potions", [])}
    return [p for p in (before.get("player") or {}).get("potions", []) if p["slot"] not in slots_a]


def _target(states: list[State], i: int) -> str | None:
    """The enemy a single-target move at states[i] hit, judged from what follows."""
    alive = _alive_ids(states[i])
    if len(alive) == 1:
        return alive[0]
    before = _enemies(states[i])
    best, best_drop = None, 0
    for after in states[i + 1:i + 1 + TARGET_LOOKAHEAD]:
        if after.get("state_type") not in FIGHT_TYPES or _round(after) != _round(states[i]):
            break
        if after.get("state_type") not in COMBAT_TYPES:
            continue
        now = _enemies(after)
        for eid, e in before.items():
            a = now.get(eid)
            drop = (e["hp"] + (e.get("block") or 0)) - ((a["hp"] + (a.get("block") or 0)) if a else 0)
            changed = a is not None and a.get("status") != e.get("status")
            score = drop if drop > 0 else (0.5 if changed else 0)
            if score > best_drop:
                best, best_drop = eid, score
        if best is not None:
            return best
    return None


def _move(states: list[State], i: int) -> tuple[dict | None, str]:
    """The move made in states[i], and a reason code for the statistics."""
    before = states[i]
    # The next combat state of this turn, looking past card-choice screens.
    for after in states[i + 1:]:
        if after.get("state_type") not in FIGHT_TYPES:
            return None, "fight_ended"
        if _round(after) != _round(before):
            # The hand is discarded at the end of the turn, so departures say nothing here.
            return {"action": "end_turn"}, "end_turn"
        if after.get("state_type") in COMBAT_TYPES:
            break
    else:
        return None, "fight_ended"
    cards, potions = _played_cards(before, after), _used_potions(before, after)
    if len(cards) + len(potions) > 1:
        return None, "ambiguous"
    if not cards and not potions:
        return None, "no_move"
    if cards:
        card = cards[0]
        action = {"action": "play_card", "card_index": card["index"]}
        target_type = card.get("target_type")
    else:
        potion = potions[0]
        action = {"action": "use_potion", "slot": potion["slot"]}
        target_type = potion.get("target_type")
    if target_type == "AnyEnemy":
        target = _target(states, i)
        if target is None:
            return None, "no_target"
        action["target"] = target
    return action, action["action"]


def label_records(records: list[dict]) -> tuple[list[dict], Counter]:
    """Add an inferred `action` to combat records; returns the records and stats."""
    stats: Counter = Counter()
    states = [r["state"] for r in records]
    out = []
    for i, rec in enumerate(records):
        rec = dict(rec)
        if states[i].get("state_type") in COMBAT_TYPES and i + 1 < len(states):
            action, why = _move(states, i)
            stats[why] += 1
            if action is not None:
                rec["action"] = action
                rec["inferred"] = True
        out.append(rec)
    return out, stats


def split_runs(records: list[dict]) -> list[list[dict]]:
    """A recording can hold several runs (looped recording, or appended files)."""
    runs, cur = [], []
    for rec in records:
        cur.append(rec)
        if rec["state"].get("state_type") == "game_over":
            runs.append(cur)
            cur = []
    if any(r["state"].get("state_type") in COMBAT_TYPES for r in cur):
        runs.append(cur)  # unfinished run: its fights are still usable
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total: Counter = Counter()
    n = 0
    for path in sorted(glob.glob(str(Path(args.recording_dir) / "*.jsonl.gz"))):
        records = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
        for run in split_runs(records):
            labeled, stats = label_records(run)
            total += stats
            with gzip.open(out / f"run_{n:04d}.jsonl.gz", "wt", encoding="utf-8") as f:
                for rec in labeled:
                    f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"{n} runs written to {out}")
    print(json.dumps(dict(total.most_common()), indent=1))


if __name__ == "__main__":
    main()
