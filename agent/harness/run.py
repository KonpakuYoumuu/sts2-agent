"""CLI: play N runs with a bot and log them.

    python -m agent.harness.run --runs 5 --bot random
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from agent.bots.explore import Explore
from agent.bots.heuristic import HeuristicBot
from agent.bots.random_bot import RandomBot
from agent.harness.runner import Runner
from agent.interface.client import GameClient


def _nn_bot(seed=None):
    from agent.bots.nn_bot import NNBot  # imports torch, so only when asked for

    return NNBot(seed=seed)


BOTS = {"random": RandomBot, "heuristic": HeuristicBot, "nn": _nn_bot}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--bot", choices=sorted(BOTS), default="random")
    p.add_argument("--character", default="IRONCLAD")
    p.add_argument("--seed", type=int, default=None, help="bot RNG seed")
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--keep-going", action="store_true", help="don't stop when a run gets stuck")
    p.add_argument("--explore", type=float, default=0.0,
                   help="probability of a random combat move (data collection), e.g. 0.1")
    args = p.parse_args()

    client = GameClient()
    if not client.ping():
        print("Can't reach the STS2MCP mod on localhost:15526. Start the game with the mod enabled.")
        return 1

    log_dir = args.log_dir or Path("logs") / time.strftime(f"%Y%m%d-%H%M%S-{args.bot}")
    policy = BOTS[args.bot](seed=args.seed)
    if args.explore > 0:
        policy = Explore(policy, epsilon=args.explore, seed=args.seed)
    runner = Runner(client, policy, character=args.character, log_dir=log_dir)
    results = runner.play(args.runs, stop_on_stuck=not args.keep_going)

    outcomes = Counter(r.outcome for r in results)
    total_time = sum(r.seconds for r in results)
    print(f"\n{len(results)} runs: {dict(outcomes)}")
    if results:
        print(f"avg floor {sum(r.floor or 0 for r in results) / len(results):.1f}, "
              f"avg {total_time / len(results):.0f}s/run, logs in {log_dir}")
    return 0 if outcomes.get("stuck", 0) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
