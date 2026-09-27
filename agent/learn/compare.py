"""Compare bots on their run logs: floors reached and HP lost per fight.

    python -m agent.learn.compare logs/eval_heur_0927 logs/eval_nn_0927

HP lost per fight is far less noisy than floor reached, so it shows combat
differences with a few dozen runs. It is measured from the first combat state
of a fight to the first state after it (before any rewards/healing), and a
death counts as losing all HP left.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics
from pathlib import Path

from agent.learn.build_dataset import FIGHT_TYPES


def fights(path: str) -> list[dict]:
    records = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
    out, i = [], 0
    while i < len(records):
        s = records[i]["state"]
        if s.get("state_type") not in FIGHT_TYPES:
            i += 1
            continue
        start, kind = s["player"]["hp"], s.get("state_type")
        j = i
        agree = []
        while j < len(records) and records[j]["state"].get("state_type") in FIGHT_TYPES:
            if "agree" in records[j]:
                agree.append(records[j]["agree"])
            j += 1
        if j < len(records):
            after = records[j]["state"]
            end = 0 if after.get("state_type") == "game_over" else (after.get("player") or {}).get("hp", 0)
            out.append({"kind": kind, "lost": start - end, "died": end <= 0,
                        "act": (s.get("run") or {}).get("act"), "agree": agree})
        i = j
    return out


def report(log_dir: str) -> dict:
    summary = [json.loads(line) for line in open(Path(log_dir) / "summary.jsonl", encoding="utf-8")]
    runs = [r for r in summary if r["outcome"] in ("death", "victory")]
    floors = sorted(r["floor"] or 0 for r in runs)
    all_fights = [f for path in sorted(glob.glob(str(Path(log_dir) / "run_*.jsonl.gz"))) for f in fights(path)]
    by_kind = {}
    for kind in ("monster", "elite", "boss"):
        fs = [f for f in all_fights if f["kind"] == kind and f["act"] == 1]
        if fs:
            by_kind[f"act1_{kind}_hp_lost"] = round(statistics.mean(f["lost"] for f in fs), 1)
            by_kind[f"act1_{kind}_n"] = len(fs)
    agree = [a for f in all_fights for a in f["agree"]]
    return {
        "runs": len(runs), "wins": sum(r["outcome"] == "victory" for r in runs),
        "floor_mean": round(statistics.mean(floors), 1) if floors else None,
        "floor_median": statistics.median(floors) if floors else None,
        "beat_act1": round(sum(f > 17 for f in floors) / max(len(floors), 1), 3),
        **by_kind,
        "teacher_agreement": round(sum(agree) / len(agree), 3) if agree else None,
        "stuck": sum(r["outcome"] == "stuck" for r in summary),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dirs", nargs="+")
    args = ap.parse_args()
    for d in args.log_dirs:
        print(d, json.dumps(report(d)))


if __name__ == "__main__":
    main()
