"""Build the combat imitation dataset from run logs.

Each example is one combat decision: the encoded state, the encoded legal
actions, the index of the teacher's action, and how the fight ended (HP left as
a fraction of max HP, and whether the player died), for the value head.
Exploratory (random) moves keep their state but use the bot's intended
`greedy_action` as the label. With --relabel, labels come from re-running the
current rule-based bot on each logged state instead, so fixes to the bot reach
the network without collecting new games.

    python -m agent.learn.build_dataset logs/night_0927 --out data/combat_v1
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import pickle
from pathlib import Path

from agent.bots.heuristic import HeuristicBot
from agent.interface.actions import legal_actions
from agent.interface.client import COMBAT_TYPES
from agent.learn.features import Vocab, encode_actions, encode_state, same_action

FIGHT_TYPES = set(COMBAT_TYPES) | {"hand_select"}
COMBAT_ACTIONS = {"play_card", "use_potion", "end_turn"}


def _fight_outcomes(records: list[dict]) -> list[tuple[float, float] | None]:
    """Per record: (HP after the fight / max HP, died) if it's inside a fight."""
    out: list[tuple[float, float] | None] = [None] * len(records)
    i = 0
    while i < len(records):
        if records[i]["state"].get("state_type") not in FIGHT_TYPES:
            i += 1
            continue
        j = i
        while j < len(records) and records[j]["state"].get("state_type") in FIGHT_TYPES:
            j += 1
        if j < len(records):  # the run log ends mid-fight when a run was aborted
            after = records[j]["state"]
            p = after.get("player") or {}
            died = after.get("state_type") == "game_over" and (p.get("hp") or 0) <= 0
            frac = 0.0 if died else (p.get("hp") or 0) / max(p.get("max_hp") or 1, 1)
            for k in range(i, j):
                out[k] = (frac, float(died))
        i = j
    return out


def build(log_dirs: list[str], vocab: Vocab, val_every: int = 10,
          relabel: bool = False) -> tuple[list, list, dict]:
    train, val = [], []
    stats = {"runs": 0, "examples": 0, "explore_labels": 0, "label_missing": 0, "relabeled_changed": 0}
    teacher = HeuristicBot(seed=0)
    files = sorted(f for d in log_dirs for f in glob.glob(str(Path(d) / "run_*.jsonl.gz")))
    for n, path in enumerate(files):
        records = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
        outcomes = _fight_outcomes(records)
        stats["runs"] += 1
        split = val if n % val_every == 0 else train
        for rec, outcome in zip(records, outcomes):
            state, action = rec["state"], rec.get("action") or {}
            if state.get("state_type") not in COMBAT_TYPES or action.get("action") not in COMBAT_ACTIONS:
                continue
            label_action = rec.get("greedy_action") if rec.get("explore") else action
            actions = legal_actions(state)
            if relabel:
                new = teacher.choose(state, actions)
                stats["relabeled_changed"] += not same_action(new, label_action)
                label_action = new
            label = next((i for i, a in enumerate(actions) if same_action(a, label_action)), None)
            if label is None:
                stats["label_missing"] += 1
                continue
            stats["explore_labels"] += bool(rec.get("explore"))
            split.append({
                "x": encode_state(state, vocab),
                "a": encode_actions(state, actions),
                "y": label,
                "v": outcome if outcome is not None else (-1.0, -1.0),  # -1 = unknown
                "run": path,
            })
            stats["examples"] += 1
        if n % 100 == 0:
            print(f"{n}/{len(files)} files, {stats['examples']} examples", flush=True)
    return train, val, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dirs", nargs="+")
    ap.add_argument("--out", default="data/combat_v1")
    ap.add_argument("--relabel", action="store_true", help="labels from the current rule-based bot")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = Vocab()
    train, val, stats = build(args.log_dirs, vocab, relabel=args.relabel)
    vocab.save(out / "vocab.json")
    with open(out / "dataset.pkl", "wb") as f:
        pickle.dump({"train": train, "val": val}, f)
    stats.update(train=len(train), val=len(val), vocab=vocab.sizes())
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
