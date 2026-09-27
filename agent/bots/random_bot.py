"""Random legal-action bot: the Phase 1 robustness test.

Uniformly random, except that "progress" actions (end turn, proceed, skip)
get a low weight, so the bot actually plays cards and claims rewards instead
of skipping through everything.
"""

from __future__ import annotations

import random

from agent.bots.base import Policy
from agent.interface.actions import PROGRESS_ACTIONS, Action
from agent.interface.client import State


class RandomBot(Policy):
    name = "random"

    def __init__(self, seed: int | None = None, progress_weight: float = 0.15):
        self.rng = random.Random(seed)
        self.progress_weight = progress_weight

    def choose(self, state: State, actions: list[Action]) -> Action:
        weights = [self.progress_weight if a["action"] in PROGRESS_ACTIONS else 1.0
                   for a in actions]
        return self.rng.choices(actions, weights=weights, k=1)[0]
